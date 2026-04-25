#!/usr/bin/env python3
"""Operator script: refresh the v_claim_edges materialized view.

Implements the daily-cron half of MH-2 (SciX Deep Search v1 PRD,
docs/prd/scix_deep_search_v1.md): runs ``REFRESH MATERIALIZED VIEW
CONCURRENTLY v_claim_edges`` and records a row in ``ingest_log`` so
operators can confirm the refresh ran. Concurrent refresh requires the
unique index installed by migrations/057_v_claim_edges.sql.

Usage:
    python scripts/refresh_v_claim_edges.py [--dsn DSN] [--allow-prod]

Production safety: refuses to run against the production DSN unless
``--allow-prod`` is passed. daily_sync.sh wraps this script with
``$SCIX_BATCH`` per the memory-isolation rule in CLAUDE.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import psycopg

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn
from scix.views import refresh_view

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

VIEW_NAME = "v_claim_edges"
# Logical "filename" used as the ingest_log key for this refresh job. Keeps
# the row idempotent across daily cron runs (ON CONFLICT DO UPDATE).
INGEST_LOG_KEY = f"refresh::{VIEW_NAME}"


def _record_refresh(
    conn: psycopg.Connection,
    *,
    duration_s: float,
    success: bool,
    error: str | None,
) -> None:
    """Upsert a row into ingest_log marking this refresh attempt."""
    status = "complete" if success else "failed"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ingest_log (
                filename, records_loaded, errors_skipped, edges_loaded,
                status, started_at, finished_at
            )
            VALUES (
                %(key)s, 0, %(errors)s, 0,
                %(status)s, NOW() - (%(duration)s || ' seconds')::interval, NOW()
            )
            ON CONFLICT (filename) DO UPDATE SET
                errors_skipped = EXCLUDED.errors_skipped,
                status = EXCLUDED.status,
                started_at = EXCLUDED.started_at,
                finished_at = EXCLUDED.finished_at
            """,
            {
                "key": INGEST_LOG_KEY,
                "errors": 0 if success else 1,
                "status": status,
                "duration": f"{duration_s:.3f}",
            },
        )
    if error:
        logger.error("refresh failed: %s", error)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN")
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to run against production DSN",
    )
    args = parser.parse_args()

    if is_production_dsn(args.dsn) and not args.allow_prod:
        logger.error(
            "Refusing to run against production DSN %s — pass --allow-prod to override",
            redact_dsn(args.dsn),
        )
        return 2

    logger.info("Refreshing %s on %s", VIEW_NAME, redact_dsn(args.dsn))

    # CONCURRENT refresh cannot run inside a transaction block, so the
    # connection must be in autocommit. We open a separate non-autocommit
    # connection for the ingest_log write so that side-channel logging
    # commits cleanly even if the earlier refresh failed.
    t0 = time.monotonic()
    with psycopg.connect(args.dsn, autocommit=True) as refresh_conn:
        result = refresh_view(refresh_conn, VIEW_NAME)
    elapsed = time.monotonic() - t0

    with psycopg.connect(args.dsn) as log_conn:
        _record_refresh(
            log_conn,
            duration_s=result.duration_s,
            success=result.success,
            error=result.error,
        )
        log_conn.commit()

    if result.success:
        logger.info(
            "Refreshed %s in %.2fs (wall %.2fs)",
            VIEW_NAME,
            result.duration_s,
            elapsed,
        )
        return 0

    logger.error("Refresh of %s failed: %s", VIEW_NAME, result.error)
    return 1


if __name__ == "__main__":
    sys.exit(main())
