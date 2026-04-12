#!/usr/bin/env python3
"""TTL cleanup + partition maintenance for ``document_entities_jit_cache``.

The u10 JIT cache is partitioned by ``expires_at``. In steady state this
script runs daily via cron and performs two jobs:

1. **Create forward-dated range partitions** so today's and tomorrow's
   writes never fall into the DEFAULT partition. We create the next
   ``--forward-days`` daily partitions idempotently.
2. **Drop expired partitions** whose entire range ended more than
   ``--grace-days`` ago. For bootstrap / tests we can also DELETE rows
   from the DEFAULT partition whose ``expires_at`` is already in the
   past; partition DROPs are the preferred eviction path but DELETE is
   the catch-all.

The script is idempotent and safe to run under ``set -e``: every DDL
statement uses ``IF NOT EXISTS`` / ``IF EXISTS`` guards.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import psycopg

logger = logging.getLogger(__name__)


DEFAULT_PARTITION = "document_entities_jit_cache_default"


def _partition_name(day: date) -> str:
    return f"document_entities_jit_cache_p{day.strftime('%Y%m%d')}"


def _create_partition(conn: psycopg.Connection, day: date) -> None:
    name = _partition_name(day)
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    sql = (
        f"CREATE TABLE IF NOT EXISTS {name} "
        f"PARTITION OF document_entities_jit_cache "
        f"FOR VALUES FROM ('{start.isoformat()}') TO ('{end.isoformat()}')"
    )
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("ensured partition %s", name)


def _drop_expired_partitions(conn: psycopg.Connection, cutoff: datetime) -> list[str]:
    """Drop daily partitions whose upper bound is strictly before cutoff."""
    dropped: list[str] = []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT child.relname
            FROM pg_inherits
            JOIN pg_class child  ON pg_inherits.inhrelid  = child.oid
            JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
            WHERE parent.relname = 'document_entities_jit_cache'
              AND child.relname LIKE 'document_entities_jit_cache_p%'
            """)
        partitions = [row[0] for row in cur.fetchall()]

    cutoff_tag = cutoff.strftime("%Y%m%d")
    for name in partitions:
        tag = name.rsplit("_p", 1)[-1]
        if len(tag) == 8 and tag.isdigit() and tag < cutoff_tag:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {name}")
            dropped.append(name)
            logger.info("dropped expired partition %s", name)
    conn.commit()
    return dropped


def _delete_expired_default_rows(conn: psycopg.Connection, cutoff: datetime) -> int:
    """DELETE catch-all — rows in the DEFAULT partition that have already
    expired. Partition DROPs are the preferred eviction path; this only
    runs for rows that fell into the DEFAULT partition before the daily
    partitions existed.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {DEFAULT_PARTITION} WHERE expires_at < %s",  # noqa: resolver-lint
            (cutoff,),
        )
        deleted = cur.rowcount
    conn.commit()
    if deleted:
        logger.info("deleted %d expired rows from %s", deleted, DEFAULT_PARTITION)
    return deleted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    dsn: str,
    *,
    forward_days: int,
    grace_days: int,
    today: date,
) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    created: list[str] = []
    dropped: list[str] = []

    with psycopg.connect(dsn) as conn:
        for offset in range(forward_days + 1):
            day = today + timedelta(days=offset)
            _create_partition(conn, day)
            created.append(_partition_name(day))

        cutoff_day = today - timedelta(days=grace_days)
        cutoff = datetime(cutoff_day.year, cutoff_day.month, cutoff_day.day, tzinfo=timezone.utc)
        dropped = _drop_expired_partitions(conn, cutoff)
        deleted = _delete_expired_default_rows(conn, datetime.now(timezone.utc))

    return {
        "created_partitions": created,
        "dropped_partitions": dropped,
        "deleted_default_rows": deleted,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_DSN") or os.environ.get("SCIX_TEST_DSN") or "dbname=scix",
        help="Postgres DSN (defaults to $SCIX_DSN, $SCIX_TEST_DSN, or dbname=scix).",
    )
    parser.add_argument(
        "--forward-days",
        type=int,
        default=14,
        help="How many forward-dated partitions to pre-create (default 14).",
    )
    parser.add_argument(
        "--grace-days",
        type=int,
        default=14,
        help="Drop partitions whose upper bound is older than this many days (default 14).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    result = run(
        args.dsn,
        forward_days=args.forward_days,
        grace_days=args.grace_days,
        today=datetime.now(timezone.utc).date(),
    )
    logger.info("jit_cache_cleanup done: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
