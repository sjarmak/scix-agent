#!/usr/bin/env python3
"""Enrich AAS instrument entries with Wikidata aliases and metadata.

Reads existing instrument entries from entity_dictionary (source='aas',
entity_type='instrument'), queries the Wikidata SPARQL endpoint for
matching entities, and enriches aliases[] and metadata with Wikidata
QIDs, labels, and instrument sub-components.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import upsert_entry

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "scix-experiments/1.0 (https://github.com/scix; mailto:scix@example.org)"


def build_sparql_query(name: str) -> str:
    """Build a SPARQL query to find a Wikidata entity by English label.

    Searches for an exact rdfs:label match and harvests:
    - The Wikidata QID
    - All English alternative labels (skos:altLabel)
    - Sub-components via P527 ("has part")

    Args:
        name: The canonical name to search for.

    Returns:
        SPARQL query string.
    """
    # Escape double quotes in the name for safe embedding in SPARQL
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f"""\
SELECT ?item ?altLabel ?partLabel WHERE {{
  ?item rdfs:label "{escaped}"@en .
  OPTIONAL {{ ?item skos:altLabel ?altLabel . FILTER(LANG(?altLabel) = "en") }}
  OPTIONAL {{
    ?item wdt:P527 ?part .
    ?part rdfs:label ?partLabel . FILTER(LANG(?partLabel) = "en")
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}"""


def execute_sparql(query: str, endpoint: str = WIKIDATA_SPARQL_URL) -> dict[str, Any]:
    """Execute a SPARQL query against the Wikidata endpoint.

    Args:
        query: SPARQL query string.
        endpoint: SPARQL endpoint URL.

    Returns:
        Parsed JSON response dict.

    Raises:
        urllib.error.URLError: On network failure after retries.
        ValueError: On malformed JSON response.
    """
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    url = f"{endpoint}?{params}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
        },
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            return json.loads(data.decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            if attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    "SPARQL attempt %d/%d failed: %s — retrying in %ds",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "SPARQL query failed after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Malformed SPARQL response: {exc}") from exc

    raise RuntimeError("execute_sparql: unexpected exit from retry loop")


def parse_sparql_results(
    sparql_json: dict[str, Any],
) -> tuple[str | None, list[str], list[str]]:
    """Parse SPARQL JSON results into a QID, aliases, and sub-components.

    Args:
        sparql_json: Parsed Wikidata SPARQL JSON response.

    Returns:
        Tuple of (wikidata_qid or None, list of alias strings, list of
        sub-component names).
    """
    bindings = sparql_json.get("results", {}).get("bindings", [])
    if not bindings:
        return None, [], []

    # Extract QID from the first binding's item URI
    qid: str | None = None
    first_item = bindings[0].get("item", {}).get("value", "")
    if "/entity/" in first_item:
        qid = first_item.rsplit("/", 1)[-1]

    aliases: set[str] = set()
    parts: set[str] = set()

    for binding in bindings:
        alt = binding.get("altLabel", {}).get("value", "")
        if alt:
            aliases.add(alt)
        part = binding.get("partLabel", {}).get("value", "")
        if part:
            parts.add(part)

    return qid, sorted(aliases), sorted(parts)


def merge_aliases(existing: list[str], new: list[str]) -> list[str]:
    """Merge new aliases into an existing list, deduplicating case-insensitively.

    Preserves the original casing of existing entries. New entries are added
    only if no case-insensitive match exists.

    Args:
        existing: Current alias list.
        new: Aliases to merge in.

    Returns:
        Deduplicated merged alias list.
    """
    lower_set: set[str] = {a.lower() for a in existing}
    merged = list(existing)
    for alias in new:
        if alias.lower() not in lower_set:
            merged.append(alias)
            lower_set.add(alias.lower())
    return merged


def fetch_instrument_entries(
    conn: Any,
) -> list[dict[str, Any]]:
    """Fetch all AAS instrument entries from entity_dictionary.

    Args:
        conn: Database connection (psycopg).

    Returns:
        List of row dicts with id, canonical_name, aliases, metadata.
    """
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT id, canonical_name, entity_type, source,
                   external_id, aliases, metadata
            FROM entity_dictionary
            WHERE source = 'aas' AND entity_type = 'instrument'
            ORDER BY canonical_name
            """)
        return [dict(row) for row in cur.fetchall()]


def enrich_entry(
    conn: Any,
    entry: dict[str, Any],
    *,
    endpoint: str = WIKIDATA_SPARQL_URL,
) -> bool:
    """Enrich a single entity_dictionary entry with Wikidata data.

    Args:
        conn: Database connection.
        entry: Dict with canonical_name, entity_type, source, aliases, metadata.
        endpoint: SPARQL endpoint URL.

    Returns:
        True if the entry was enriched, False if no match was found.
    """
    name = entry["canonical_name"]
    query = build_sparql_query(name)

    try:
        sparql_json = execute_sparql(query, endpoint=endpoint)
    except (urllib.error.URLError, ValueError) as exc:
        logger.warning("Skipping '%s': SPARQL query failed: %s", name, exc)
        return False

    qid, new_aliases, sub_components = parse_sparql_results(sparql_json)

    if qid is None:
        logger.debug("No Wikidata match for '%s'", name)
        return False

    # Merge aliases
    existing_aliases = entry.get("aliases") or []
    merged_aliases = merge_aliases(existing_aliases, new_aliases)

    # Update metadata
    existing_metadata = entry.get("metadata") or {}
    updated_metadata = {**existing_metadata, "wikidata_qid": qid}
    if sub_components:
        updated_metadata["wikidata_sub_components"] = sub_components

    upsert_entry(
        conn,
        canonical_name=name,
        entity_type=entry["entity_type"],
        source=entry["source"],
        external_id=entry.get("external_id"),
        aliases=merged_aliases,
        metadata=updated_metadata,
    )

    logger.info(
        "Enriched '%s' with QID=%s, %d new aliases, %d sub-components",
        name,
        qid,
        len(merged_aliases) - len(existing_aliases),
        len(sub_components),
    )
    return True


def run_enrich(
    dsn: str | None = None,
    *,
    batch_size: int = 50,
    delay: float = 1.0,
    endpoint: str = WIKIDATA_SPARQL_URL,
) -> tuple[int, int]:
    """Run the Wikidata enrichment pipeline for AAS instruments.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        batch_size: Number of entries to process before logging progress.
        delay: Seconds to wait between SPARQL requests.
        endpoint: SPARQL endpoint URL.

    Returns:
        Tuple of (total_processed, total_enriched).
    """
    t0 = time.monotonic()

    conn = get_connection(dsn)
    try:
        entries = fetch_instrument_entries(conn)
        logger.info("Found %d AAS instrument entries to enrich", len(entries))

        enriched = 0
        for idx, entry in enumerate(entries):
            if idx > 0:
                time.sleep(delay)

            if enrich_entry(conn, entry, endpoint=endpoint):
                enriched += 1

            if (idx + 1) % batch_size == 0:
                logger.info("Progress: %d/%d processed", idx + 1, len(entries))
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "Wikidata enrichment complete: %d/%d entries enriched in %.1fs",
        enriched,
        len(entries),
        elapsed,
    )
    return len(entries), enriched


def main() -> None:
    """Parse arguments and run the Wikidata enrichment pipeline."""
    parser = argparse.ArgumentParser(
        description="Enrich AAS instrument entries with Wikidata aliases and metadata",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Log progress every N entries (default: 50)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between SPARQL requests (default: 1.0)",
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

    total, enriched = run_enrich(
        dsn=args.dsn,
        batch_size=args.batch_size,
        delay=args.delay,
    )
    print(f"Enriched {enriched}/{total} AAS instrument entries with Wikidata data")


if __name__ == "__main__":
    main()
