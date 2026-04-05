#!/usr/bin/env python3
"""Harvest VizieR catalog metadata into entity_dictionary.

Queries TAPVizieR (http://tapvizier.cds.unistra.fr/TAPVizieR/tap) via the
TAP sync endpoint with ADQL to retrieve full catalog metadata, parses the
VOTable XML response, and bulk-loads entries via scix.dictionary.bulk_load()
with entity_type='dataset', source='vizier'.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load

logger = logging.getLogger(__name__)

TAP_SYNC_URL = "http://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"

ADQL_QUERY = (
    "SELECT table_name, description, utype "
    "FROM TAP_SCHEMA.tables "
    "WHERE schema_name NOT IN ('TAP_SCHEMA', 'ivoa')"
)

# VOTable XML namespace
_VOTABLE_NS = {"vot": "http://www.ivoa.net/xml/VOTable/v1.3"}


def query_tap_vizier(
    url: str = TAP_SYNC_URL,
    query: str = ADQL_QUERY,
) -> bytes:
    """Query TAPVizieR sync endpoint and return raw VOTable XML bytes.

    Posts an ADQL query to the TAP sync endpoint with retry and exponential
    backoff.

    Args:
        url: TAPVizieR sync endpoint URL.
        query: ADQL query string.

    Returns:
        Raw VOTable XML response bytes.

    Raises:
        urllib.error.URLError: If the request fails after retries.
    """
    params = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "votable",
            "MAXREC": "200000",
            "QUERY": query,
        }
    ).encode("utf-8")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=params,
                headers={"User-Agent": "scix-experiments/1.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()

            logger.info("TAP query returned %d bytes", len(data))
            return data

        except (urllib.error.URLError, OSError) as exc:
            if attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    "TAP query attempt %d/%d failed: %s — retrying in %ds",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Failed to query TAPVizieR after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise

    # Unreachable, but satisfies type checker
    raise RuntimeError("query_tap_vizier: unexpected exit from retry loop")


def parse_votable_catalogs(xml_bytes: bytes) -> list[dict[str, str]]:
    """Parse VOTable XML into a list of catalog dicts.

    Extracts table_name, description, and utype from VOTable TABLEDATA rows.
    Uses FIELD element names to determine column order dynamically.

    Args:
        xml_bytes: Raw VOTable XML bytes from TAPVizieR.

    Returns:
        List of dicts with keys: table_name, description, utype.
    """
    root = ET.fromstring(xml_bytes)

    # Try with namespace first, fall back to no namespace
    resource = root.find("vot:RESOURCE", _VOTABLE_NS)
    if resource is None:
        resource = root.find("RESOURCE")
    if resource is None:
        # Try the root itself as resource container
        resource = root

    table = resource.find("vot:TABLE", _VOTABLE_NS)
    if table is None:
        table = resource.find("TABLE")
    if table is None:
        table = resource

    # Determine column order from FIELD elements
    fields_ns = table.findall("vot:FIELD", _VOTABLE_NS)
    fields_bare = table.findall("FIELD")
    fields = fields_ns if fields_ns else fields_bare

    col_names: list[str] = []
    for field in fields:
        name = field.get("name", "")
        col_names.append(name.lower())

    logger.debug("VOTable columns: %s", col_names)

    # Find TABLEDATA rows
    data_elem = table.find(".//vot:TABLEDATA", _VOTABLE_NS)
    if data_elem is None:
        data_elem = table.find(".//TABLEDATA")
    if data_elem is None:
        logger.warning("No TABLEDATA element found in VOTable")
        return []

    tr_elements = data_elem.findall("vot:TR", _VOTABLE_NS)
    if not tr_elements:
        tr_elements = data_elem.findall("TR")

    catalogs: list[dict[str, str]] = []
    for tr in tr_elements:
        td_elements = tr.findall("vot:TD", _VOTABLE_NS)
        if not td_elements:
            td_elements = tr.findall("TD")

        values: list[str] = []
        for td in td_elements:
            values.append(td.text.strip() if td.text else "")

        row: dict[str, str] = {}
        for idx, col_name in enumerate(col_names):
            if idx < len(values):
                row[col_name] = values[idx]

        catalogs.append(row)

    logger.info("Parsed %d catalog entries from VOTable", len(catalogs))
    return catalogs


def build_dictionary_entries(
    catalogs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Convert raw catalog dicts into entity_dictionary records.

    Each returned dict has keys compatible with dictionary.bulk_load():
    canonical_name, entity_type, source, external_id, aliases, metadata.

    Args:
        catalogs: List of dicts with keys table_name, description, utype.

    Returns:
        List of entity dictionary entry dicts.
    """
    entries: list[dict[str, Any]] = []
    skipped = 0

    for cat in catalogs:
        table_name = cat.get("table_name", "").strip()
        description = cat.get("description", "").strip()
        utype = cat.get("utype", "").strip()

        if not table_name:
            skipped += 1
            continue

        # Use description as canonical name; fall back to table_name
        canonical_name = description if description else table_name

        metadata: dict[str, Any] = {}
        if utype:
            metadata["utype"] = utype

        entries.append(
            {
                "canonical_name": canonical_name,
                "entity_type": "dataset",
                "source": "vizier",
                "external_id": table_name,
                "aliases": [],
                "metadata": metadata,
            }
        )

    if skipped:
        logger.warning("Skipped %d entries missing table_name", skipped)

    logger.info("Built %d dictionary entries from VizieR catalogs", len(entries))
    return entries


def run_harvest(dsn: str | None = None) -> int:
    """Run the full VizieR catalog harvest pipeline.

    Queries TAPVizieR, parses the VOTable response, builds dictionary
    entries, and loads them into entity_dictionary.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.

    Returns:
        Number of entries loaded.
    """
    t0 = time.monotonic()

    xml_bytes = query_tap_vizier()
    catalogs = parse_votable_catalogs(xml_bytes)
    entries = build_dictionary_entries(catalogs)

    conn = get_connection(dsn)
    try:
        count = bulk_load(conn, entries)
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "VizieR harvest complete: %d entries loaded in %.1fs",
        count,
        elapsed,
    )
    return count


def main() -> None:
    """Parse arguments and run the VizieR harvest pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest VizieR catalog metadata into entity_dictionary",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    count = run_harvest(dsn=args.dsn)
    print(f"Loaded {count} VizieR catalog entries into entity_dictionary")


if __name__ == "__main__":
    main()
