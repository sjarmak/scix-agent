#!/usr/bin/env python3
"""Harvest unique dataset source labels from the papers.data[] column.

Queries the local PostgreSQL papers table to extract all unique values from
the data TEXT[] array column, deduplicates and counts paper occurrences per
data source, and loads the top entries into entity_dictionary with
entity_type='dataset', source='ads_data'.

Usage:
    python scripts/harvest_ads_data_field.py [--dsn DSN] [--min-count N]
                                              [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import psycopg
from psycopg.rows import dict_row

# Allow running from repo root: add src/ to path
sys.path.insert(0, "src")

from scix.db import get_connection  # noqa: E402
from scix.dictionary import bulk_load  # noqa: E402

logger = logging.getLogger(__name__)

ENTITY_TYPE = "dataset"
SOURCE = "ads_data"


def fetch_data_sources(
    conn: psycopg.Connection,
    *,
    min_count: int = 1,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Query papers.data[] to get deduplicated dataset source labels with counts.

    Args:
        conn: Database connection.
        min_count: Minimum paper count to include a source (default 1).
        limit: Maximum number of sources to return (None = all).

    Returns:
        List of dicts with keys 'source_label' and 'paper_count',
        ordered by paper_count descending.
    """
    query = """
        SELECT unnest(data) AS source, count(*) AS paper_count
        FROM papers
        GROUP BY 1
        HAVING count(*) >= %(min_count)s
        ORDER BY 2 DESC
    """
    params: dict[str, Any] = {"min_count": min_count}

    if limit is not None:
        query += " LIMIT %(limit)s"
        params["limit"] = limit

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [{"source_label": row["source"], "paper_count": row["paper_count"]} for row in rows]


def build_entries(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert fetched data source rows into entity_dictionary entry dicts.

    Args:
        sources: List of dicts with 'source_label' and 'paper_count'.

    Returns:
        List of entry dicts ready for dictionary.bulk_load().
    """
    return [
        {
            "canonical_name": src["source_label"],
            "entity_type": ENTITY_TYPE,
            "source": SOURCE,
            "metadata": {"paper_count": src["paper_count"]},
        }
        for src in sources
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Harvest dataset source labels from papers.data[]",
    )
    parser.add_argument("--dsn", default=None, help="PostgreSQL DSN (default: SCIX_DSN env)")
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum paper count to include a source (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of sources to load (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without loading into entity_dictionary",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = get_connection(args.dsn)
    try:
        logger.info("Querying papers.data[] for dataset source labels...")
        sources = fetch_data_sources(conn, min_count=args.min_count, limit=args.limit)
        logger.info("Found %d unique data source labels", len(sources))

        if not sources:
            logger.info("No data sources found. Nothing to load.")
            return 0

        # Print top entries
        for src in sources[:20]:
            logger.info("  %-40s %d papers", src["source_label"], src["paper_count"])
        if len(sources) > 20:
            logger.info("  ... and %d more", len(sources) - 20)

        if args.dry_run:
            logger.info("Dry run — skipping entity_dictionary load")
            return 0

        entries = build_entries(sources)
        loaded = bulk_load(conn, entries)
        logger.info(
            "Loaded %d entries into entity_dictionary (entity_type=%s, source=%s)",
            loaded,
            ENTITY_TYPE,
            SOURCE,
        )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
