#!/usr/bin/env python3
"""Harvest the ASCL (Astrophysics Source Code Library) catalog into entity_dictionary.

Downloads the full ASCL JSON catalog from https://ascl.net/code/json, parses
each entry into a dictionary record (entity_type='software', source='ascl'),
and bulk-loads them via scix.dictionary.bulk_load().
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load

logger = logging.getLogger(__name__)

ASCL_URL = "https://ascl.net/code/json"


def download_ascl_catalog(url: str = ASCL_URL) -> list[dict[str, Any]]:
    """Download the ASCL JSON catalog and return parsed entries.

    Uses urllib.request with retry and exponential backoff.

    Args:
        url: URL of the ASCL JSON endpoint.

    Returns:
        List of raw ASCL entry dicts.

    Raises:
        urllib.error.URLError: If the download fails after retries.
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "scix-experiments/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()

            entries = json.loads(data)
            logger.info("Downloaded ASCL catalog: %d entries", len(entries))
            return entries

        except (urllib.error.URLError, OSError) as exc:
            if attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    "Download attempt %d/%d failed: %s — retrying in %ds",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Failed to download ASCL catalog after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise

    # Unreachable, but satisfies type checker
    raise RuntimeError("download_ascl_catalog: unexpected exit from retry loop")


def parse_ascl_entries(raw_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse raw ASCL JSON entries into entity_dictionary records.

    Each returned dict has keys compatible with dictionary.bulk_load():
    canonical_name, entity_type, source, external_id, aliases, metadata.

    Args:
        raw_entries: List of raw dicts from the ASCL JSON API.

    Returns:
        List of entity dictionary entry dicts.
    """
    entries: list[dict[str, Any]] = []
    skipped = 0

    for raw in raw_entries:
        title = raw.get("title", "").strip()
        ascl_id = raw.get("ascl_id", "").strip()

        if not title or not ascl_id:
            skipped += 1
            continue

        # Build aliases: include lowercase variant if different from title
        aliases: list[str] = []
        title_lower = title.lower()
        if title_lower != title:
            aliases.append(title_lower)

        # Build metadata with available fields
        metadata: dict[str, Any] = {}
        bibcode = raw.get("bibcode", "").strip()
        if bibcode:
            metadata["bibcode"] = bibcode

        credit = raw.get("credit", "").strip()
        if credit:
            metadata["credit"] = credit

        entries.append(
            {
                "canonical_name": title,
                "entity_type": "software",
                "source": "ascl",
                "external_id": ascl_id,
                "aliases": aliases,
                "metadata": metadata,
            }
        )

    if skipped:
        logger.warning("Skipped %d entries missing title or ascl_id", skipped)

    logger.info("Parsed %d ASCL entries into dictionary records", len(entries))
    return entries


def run_harvest(dsn: str | None = None) -> int:
    """Run the full ASCL harvest pipeline.

    Downloads the catalog, parses entries, and loads them into entity_dictionary.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.

    Returns:
        Number of entries loaded.
    """
    t0 = time.monotonic()

    raw_entries = download_ascl_catalog()
    entries = parse_ascl_entries(raw_entries)

    conn = get_connection(dsn)
    try:
        count = bulk_load(conn, entries)
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "ASCL harvest complete: %d entries loaded in %.1fs",
        count,
        elapsed,
    )
    return count


def main() -> None:
    """Parse arguments and run the ASCL harvest pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest ASCL software catalog into entity_dictionary",
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
    print(f"Loaded {count} ASCL entries into entity_dictionary")


if __name__ == "__main__":
    main()
