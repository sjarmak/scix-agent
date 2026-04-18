#!/usr/bin/env python3
"""Operator script: refresh document_entities_canonical MV and validate results.

Usage:
    python scripts/refresh_fusion_mv.py [--dsn DSN] [--allow-prod]

Calls fusion_mv.refresh_if_due() with min_interval_seconds=0 to force an
immediate refresh, then validates the acceptance criteria:
  - refresh completes within PRD budget (<15 min)
  - sample entity top-k query returns in <100ms
  - fusion_mv_state.last_refresh_at is updated
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import psycopg

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn
from scix.fusion_mv import refresh_if_due

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BUDGET_SECONDS = 15 * 60
QUERY_LATENCY_MS = 100
MIN_MV_ROW_COUNT = 10_000


def validate(dsn: str) -> bool:
    ok = True
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT dirty, last_refresh_at FROM fusion_mv_state WHERE id = 1")
        row = cur.fetchone()
        if row is None:
            logger.error("FAIL: fusion_mv_state has no row (id=1) — was migration 033 applied?")
            return False
        dirty, last_refresh_at = row
        logger.info("fusion_mv_state: dirty=%s, last_refresh_at=%s", dirty, last_refresh_at)
        if dirty:
            logger.error("FAIL: dirty bit still set after refresh")
            ok = False
        if last_refresh_at is None:
            logger.error("FAIL: last_refresh_at is NULL after refresh")
            ok = False

        cur.execute("SELECT COUNT(*) FROM document_entities_canonical")
        row_count = cur.fetchone()[0]
        logger.info("MV row count: %s", f"{row_count:,}")
        if row_count < MIN_MV_ROW_COUNT:
            logger.error("FAIL: MV row count %d is suspiciously low", row_count)
            return False

        cur.execute(
            "SELECT entity_id FROM document_entities_canonical "
            "ORDER BY fused_confidence DESC LIMIT 1"
        )
        top_row = cur.fetchone()
        if top_row is None:
            logger.error("FAIL: document_entities_canonical is empty")
            return False
        top_entity = top_row[0]

        t0 = time.perf_counter()
        cur.execute(
            "SELECT bibcode, fused_confidence "
            "FROM document_entities_canonical "
            "WHERE entity_id = %s "
            "ORDER BY fused_confidence DESC LIMIT 20",
            (top_entity,),
        )
        cur.fetchall()
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Sample top-k query: entity_id=%d, %.1f ms",
            top_entity, latency_ms,
        )
        if latency_ms > QUERY_LATENCY_MS:
            logger.error(
                "FAIL: query latency %.1f ms exceeds %d ms budget",
                latency_ms, QUERY_LATENCY_MS,
            )
            ok = False

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN")
    parser.add_argument(
        "--allow-prod", action="store_true",
        help="Required to run against production DSN",
    )
    args = parser.parse_args()

    if is_production_dsn(args.dsn) and not args.allow_prod:
        logger.error(
            "Refusing to run against production DSN %s — pass --allow-prod to override",
            redact_dsn(args.dsn),
        )
        return 2

    logger.info("Refreshing document_entities_canonical on %s", redact_dsn(args.dsn))

    with psycopg.connect(args.dsn) as conn:
        t0 = time.perf_counter()
        refreshed = refresh_if_due(conn, min_interval_seconds=0)
        elapsed = time.perf_counter() - t0

    if not refreshed:
        logger.error("refresh_if_due() returned False — check fusion_mv_state")
        return 1

    logger.info("Refresh completed in %.1f s (budget: %d s)", elapsed, BUDGET_SECONDS)
    if elapsed > BUDGET_SECONDS:
        logger.error(
            "FAIL: refresh took %.1f s (%.1f min), exceeds budget of %d s",
            elapsed, elapsed / 60, BUDGET_SECONDS,
        )
        return 1

    if not validate(args.dsn):
        return 1

    logger.info("All acceptance criteria passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
