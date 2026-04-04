#!/usr/bin/env python3
"""Backfill the body column from raw JSONB for existing papers.

Processes in batches to avoid WAL bloat and long-running locks.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import psycopg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 50_000


def backfill_body(dsn: str, batch_size: int = BATCH_SIZE) -> None:
    """Update papers.body from raw->>'body' in batches keyed by bibcode."""
    with psycopg.connect(dsn) as conn:
        conn.autocommit = False

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM papers WHERE raw ? 'body' AND body IS NULL")
            total = cur.fetchone()[0]
            logger.info("Papers to backfill: %d", total)

        if total == 0:
            logger.info("Nothing to backfill.")
            return

        updated = 0
        last_bibcode = ""
        t0 = time.monotonic()

        while True:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE papers
                    SET body = raw->>'body',
                        raw  = raw - 'body'
                    WHERE bibcode IN (
                        SELECT bibcode FROM papers
                        WHERE raw ? 'body' AND body IS NULL AND bibcode > %s
                        ORDER BY bibcode
                        LIMIT %s
                    )
                    RETURNING bibcode
                    """,
                    (last_bibcode, batch_size),
                )
                rows = cur.fetchall()

            if not rows:
                conn.commit()
                break

            # RETURNING order is not guaranteed; use max() for cursor pagination
            last_bibcode = max(r[0] for r in rows)
            updated += len(rows)
            conn.commit()

            elapsed = time.monotonic() - t0
            rate = updated / elapsed if elapsed > 0 else 0
            logger.info(
                "Backfilled %d / %d (%.0f rec/s, last: %s)",
                updated,
                total,
                rate,
                last_bibcode,
            )

    logger.info("Done. Updated %d papers in %.1f s.", updated, time.monotonic() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill body column from raw JSONB")
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_DSN", "dbname=scix"),
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Rows per batch (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()
    backfill_body(args.dsn, args.batch_size)


if __name__ == "__main__":
    main()
