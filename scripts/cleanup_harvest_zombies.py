#!/usr/bin/env python3
"""Mark stuck harvest_runs rows as aborted_zombie.

A "zombie" is a row in harvest_runs with status='running' whose started_at is
older than the stale threshold (default 6 hours). GCMD and SPASE harvesters
have been observed to leave rows in this state when the worker process
crashes without updating the row.

Usage:
    python scripts/cleanup_harvest_zombies.py
    python scripts/cleanup_harvest_zombies.py --dsn "dbname=scix_test"
    python scripts/cleanup_harvest_zombies.py --hours 12

Exits 0 after printing a summary of which rows were updated.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import psycopg

logger = logging.getLogger("cleanup_harvest_zombies")

DEFAULT_STALE_HOURS = 6


@dataclass(frozen=True)
class ZombieRow:
    """A harvest_runs row that was marked aborted_zombie by this run."""

    id: int
    source: str
    started_at: str


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_DSN", "dbname=scix"),
        help="PostgreSQL DSN (default: $SCIX_DSN or 'dbname=scix')",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_STALE_HOURS,
        help=f"Age threshold in hours (default: {DEFAULT_STALE_HOURS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report but do not update any rows",
    )
    return parser.parse_args(argv)


def find_zombies(conn: psycopg.Connection, hours: int) -> list[tuple[int, str, str]]:
    """Return harvest_runs rows that look like zombies (status='running' and old)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source, started_at::text
              FROM harvest_runs
             WHERE status = 'running'
               AND started_at < now() - make_interval(hours => %s)
             ORDER BY id
            """,
            (hours,),
        )
        return [(row[0], row[1], row[2]) for row in cur.fetchall()]


def mark_zombies(conn: psycopg.Connection, hours: int) -> list[ZombieRow]:
    """Flip zombie rows to status='aborted_zombie' and return what was updated."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE harvest_runs
               SET status       = 'aborted_zombie',
                   finished_at  = now(),
                   error_message = COALESCE(error_message, '')
                                  || CASE WHEN error_message IS NULL OR error_message = ''
                                          THEN 'cleanup_harvest_zombies: stuck >'
                                               || %s || 'h'
                                          ELSE '; cleanup_harvest_zombies: stuck >'
                                               || %s || 'h'
                                     END
             WHERE status = 'running'
               AND started_at < now() - make_interval(hours => %s)
            RETURNING id, source, started_at::text
            """,
            (hours, hours, hours),
        )
        updated = [ZombieRow(id=row[0], source=row[1], started_at=row[2]) for row in cur.fetchall()]
    conn.commit()
    return updated


def _print_summary(zombies: list[ZombieRow], hours: int, dry_run: bool) -> None:
    verb = "Would abort" if dry_run else "Aborted"
    print(f"{verb} {len(zombies)} zombie harvest_runs (threshold: {hours}h)")
    for z in zombies:
        print(f"  id={z.id:<8} source={z.source:<20} started_at={z.started_at}")
    if not zombies:
        print("  (no zombies found)")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    logger.info("Connecting to %s", args.dsn)
    with psycopg.connect(args.dsn) as conn:
        if args.dry_run:
            rows = find_zombies(conn, args.hours)
            zombies = [ZombieRow(id=row[0], source=row[1], started_at=row[2]) for row in rows]
        else:
            zombies = mark_zombies(conn, args.hours)

    _print_summary(zombies, args.hours, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
