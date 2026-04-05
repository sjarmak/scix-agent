#!/usr/bin/env python3
"""CLI script to compute entity specificity scores from the extractions table.

Queries the database for entity document frequencies, scores each entity
using IDF-like specificity, and outputs JSON to stdout.

Usage:
    SCIX_DSN="dbname=scix" python scripts/score_entities.py [--top 100] [--threshold 2.3]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict

from scix.db import get_connection
from scix.specificity import DEFAULT_THRESHOLD, score_entities

logger = logging.getLogger(__name__)


def fetch_entity_frequencies(dsn: str | None) -> tuple[list[tuple[str, int]], int]:
    """Query the DB for entity document frequencies and total paper count.

    Returns:
        Tuple of (entity_freq_pairs, total_papers) where entity_freq_pairs
        is a list of (entity_string, distinct_bibcode_count).
    """
    conn = get_connection(dsn)
    try:
        with conn.cursor() as cur:
            # Total distinct bibcodes in extractions
            cur.execute("SELECT COUNT(DISTINCT bibcode) FROM extractions")
            row = cur.fetchone()
            total_papers: int = row[0] if row else 0

            # Count distinct bibcodes per entity from payload->'entities'
            # payload->'entities' is a JSONB array of strings
            cur.execute("""
                SELECT entity, COUNT(DISTINCT bibcode) AS df
                FROM extractions,
                     jsonb_array_elements_text(payload->'entities') AS entity
                WHERE extraction_type = 'entities'
                GROUP BY entity
                ORDER BY df DESC
                """)
            entity_freqs: list[tuple[str, int]] = [(row[0], row[1]) for row in cur.fetchall()]
    finally:
        conn.close()

    return entity_freqs, total_papers


def main(argv: list[str] | None = None) -> None:
    """Entry point for the entity specificity scoring CLI."""
    parser = argparse.ArgumentParser(
        description="Compute entity specificity scores from the extractions table."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of top entities to output (default: 100)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Specificity threshold for keep/filter (default: {DEFAULT_THRESHOLD:.4f})",
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Fetching entity frequencies from database...")
    entity_freqs, total_papers = fetch_entity_frequencies(args.dsn)
    logger.info("Found %d unique entities across %d papers", len(entity_freqs), total_papers)

    if total_papers == 0:
        logger.warning("No papers found in extractions table")
        json.dump([], sys.stdout, indent=2)
        print()
        return

    scored = score_entities(entity_freqs, N=total_papers, threshold=args.threshold)

    top_scored = scored[: args.top]

    output = [asdict(s) for s in top_scored]
    json.dump(output, sys.stdout, indent=2)
    print()  # trailing newline

    keep_count = sum(1 for s in top_scored if s.classification == "keep")
    filter_count = sum(1 for s in top_scored if s.classification == "filter")
    logger.info(
        "Output %d entities: %d keep, %d filter (threshold=%.4f)",
        len(top_scored),
        keep_count,
        filter_count,
        args.threshold,
    )


if __name__ == "__main__":
    main()
