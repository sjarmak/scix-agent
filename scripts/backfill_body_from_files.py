#!/usr/bin/env python3
"""Backfill papers.body from on-disk JSONL files.

Reads JSONL (plain, .gz, .xz) files, extracts bibcode + body for records
that have body text, and updates papers.body in the database for rows
where body IS NULL.

No ADS API calls — purely local file reads + DB writes.

Usage:
    # Backfill from all files in the harvest directory:
    python scripts/backfill_body_from_files.py ads_metadata_by_year_picard/

    # Backfill from specific files:
    python scripts/backfill_body_from_files.py ads_metadata_by_year_picard/ads_metadata_2000_full.jsonl.gz

    # Dry run — just count how many bodies are in the files:
    python scripts/backfill_body_from_files.py --dry-run ads_metadata_by_year_picard/
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import lzma
import os
import time
from pathlib import Path

import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DB_BATCH = 1000


def open_jsonl(path: str):
    """Open a JSONL file, handling .gz and .xz transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    elif path.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "r", encoding="utf-8")


def find_jsonl_files(path: str) -> list[str]:
    """Find all JSONL files in a directory, sorted by name."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    files = []
    for ext in ("*.jsonl", "*.jsonl.gz", "*.jsonl.xz"):
        files.extend(str(f) for f in p.glob(ext))
    return sorted(files)


def backfill_file(
    conn: psycopg.Connection,
    filepath: str,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Process one JSONL file. Returns (total_records, records_with_body, updated)."""
    total = 0
    with_body = 0
    updated = 0
    batch: list[tuple[str, str]] = []

    with open_jsonl(filepath) as f:
        for line in f:
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            bibcode = rec.get("bibcode")
            body = rec.get("body")
            if not bibcode or not body:
                continue

            # Strip NUL bytes — PostgreSQL text columns reject them
            body = body.replace("\x00", "")
            if not body:
                continue

            with_body += 1

            if not dry_run:
                batch.append((body, bibcode))
                if len(batch) >= DB_BATCH:
                    updated += flush_batch(conn, batch)
                    batch.clear()

            if total % 100_000 == 0:
                logger.info(
                    "  %s: %d records, %d with body, %d updated",
                    os.path.basename(filepath),
                    total,
                    with_body,
                    updated,
                )

    if batch and not dry_run:
        updated += flush_batch(conn, batch)

    return total, with_body, updated


def flush_batch(conn: psycopg.Connection, batch: list[tuple[str, str]]) -> int:
    """Write a batch of (body, bibcode) pairs. Returns rows updated."""
    updated = 0
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE papers SET body = %s WHERE bibcode = %s AND body IS NULL",
            batch,
        )
        updated = cur.rowcount
    conn.commit()
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill papers.body from on-disk JSONL files")
    parser.add_argument(
        "path",
        help="JSONL file or directory containing JSONL files",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_DSN", "dbname=scix"),
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count bodies in files without writing to DB",
    )
    args = parser.parse_args()

    files = find_jsonl_files(args.path)
    if not files:
        logger.error("No JSONL files found at %s", args.path)
        return

    logger.info("Found %d JSONL files", len(files))

    conn = None
    if not args.dry_run:
        conn = psycopg.connect(args.dsn)
        conn.autocommit = False

    grand_total = 0
    grand_body = 0
    grand_updated = 0
    t0 = time.monotonic()

    try:
        for filepath in files:
            logger.info("Processing %s", os.path.basename(filepath))
            total, with_body, updated = backfill_file(conn, filepath, dry_run=args.dry_run)
            grand_total += total
            grand_body += with_body
            grand_updated += updated
            logger.info(
                "  Done: %d records, %d with body, %d updated",
                total,
                with_body,
                updated,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted. Progress so far is committed.")
    finally:
        if conn:
            conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "Complete: %d records scanned, %d with body (%.1f%%), %d DB rows updated in %.0fs",
        grand_total,
        grand_body,
        100.0 * grand_body / grand_total if grand_total else 0,
        grand_updated,
        elapsed,
    )


if __name__ == "__main__":
    main()
