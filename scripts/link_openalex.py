#!/usr/bin/env python3
"""Batch-link papers to OpenAlex by DOI.

Fetches OpenAlex work IDs and topics for papers that have a DOI
but no openalex_id yet. Supports resumption by skipping already-linked rows.

Usage:
    python scripts/link_openalex.py --mailto user@example.com
    python scripts/link_openalex.py --mailto user@example.com --batch-size 500 --limit 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.openalex import link_papers_batch

logger = logging.getLogger(__name__)


def fetch_unlinked_papers(
    conn,
    batch_size: int = 100,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    """Fetch papers that have a DOI but no openalex_id (resumable).

    Returns list of (doi, bibcode) tuples.
    Uses the first element of the doi TEXT[] array.
    """
    sql = """
        SELECT doi[1], bibcode
        FROM papers
        WHERE doi IS NOT NULL
          AND array_length(doi, 1) > 0
          AND doi[1] IS NOT NULL
          AND doi[1] != ''
          AND openalex_id IS NULL
        ORDER BY bibcode
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    return [(row[0], row[1]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Link papers to OpenAlex by DOI")
    parser.add_argument(
        "--mailto",
        required=True,
        help="Email address for OpenAlex polite pool (required)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of papers per batch (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum total papers to process (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    conn = get_connection(dsn=args.dsn)

    try:
        papers = fetch_unlinked_papers(conn, batch_size=args.batch_size, limit=args.limit)
        total = len(papers)
        logger.info("Found %d unlinked papers with DOIs", total)

        if total == 0:
            logger.info("Nothing to do — all papers with DOIs are already linked.")
            return

        linked_total = 0
        for i in range(0, total, args.batch_size):
            batch = papers[i : i + args.batch_size]
            linked = link_papers_batch(conn, batch, mailto=args.mailto)
            linked_total += linked
            logger.info(
                "Progress: %d/%d processed, %d linked so far",
                min(i + args.batch_size, total),
                total,
                linked_total,
            )

        logger.info("Done. Linked %d / %d papers to OpenAlex.", linked_total, total)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
