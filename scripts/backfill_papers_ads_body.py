#!/usr/bin/env python3
"""Backfill papers_ads_body from existing papers.body column.

This is a one-time bulk operation to populate papers_ads_body with body text
already stored in the papers table (papers.body, populated by migration 010
and various ADS body backfills).

The script:
  1. Optionally drops the GIN index for faster bulk inserts.
  2. INSERT INTO papers_ads_body ... SELECT FROM papers WHERE body IS NOT NULL
     in batches keyed by bibcode cursor, with ON CONFLICT DO NOTHING.
  3. Recreates the GIN index.
  4. Runs ANALYZE.
  5. Verifies counts and a sample tsquery.

Usage:
    # Against test DB:
    SCIX_TEST_DSN="dbname=scix_test" python scripts/backfill_papers_ads_body.py \\
        --dsn "dbname=scix_test" --batch-size 1000

    # Against production (requires explicit opt-in):
    python scripts/backfill_papers_ads_body.py --yes-production --batch-size 50000

SAFETY:
    * Refuses production DSN without --yes-production (same guard as ads_body.py).
    * Uses ON CONFLICT DO NOTHING so re-runs are safe (idempotent).
    * Sets statement_timeout = 0 for the session to avoid timeouts on long inserts.
    * Never uses UNLOGGED tables.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg
    from typing import Any

# Add src/ to path for direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import (  # noqa: E402
    DEFAULT_DSN,
    IndexManager,
    get_connection,
    is_production_dsn,
    redact_dsn,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + stats (both frozen — immutable data structures per project rules)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackfillConfig:
    """Immutable configuration for the papers_ads_body backfill."""

    dsn: str | None = None
    batch_size: int = 50_000
    dry_run: bool = False
    yes_production: bool = False


@dataclass(frozen=True)
class BackfillStats:
    """Immutable result record from run_backfill()."""

    rows_inserted: int
    rows_already_present: int
    total_body_rows: int
    elapsed_seconds: float
    dry_run: bool


# ---------------------------------------------------------------------------
# Production guard
# ---------------------------------------------------------------------------


class ProductionGuardError(RuntimeError):
    """Raised when targeting production without explicit opt-in."""


def _check_production_guard(cfg: BackfillConfig) -> None:
    """Refuse production DSN unless yes_production is set.

    Resolves cfg.dsn -> DEFAULT_DSN so a None dsn cannot bypass the guard.
    """
    effective_dsn = cfg.dsn or DEFAULT_DSN
    if is_production_dsn(effective_dsn) and not cfg.yes_production:
        raise ProductionGuardError(
            f"Refusing to run backfill against production DSN "
            f"({redact_dsn(effective_dsn)}). Pass --yes-production to override."
        )


# ---------------------------------------------------------------------------
# Core backfill logic
# ---------------------------------------------------------------------------

# PostgreSQL tsvector has a hard 1 MB limit.  A handful of papers have body
# text exceeding this (e.g. 1.49 MB) — the GENERATED tsv column would fail
# on INSERT.  Filter them out; they can still be searched via the raw
# papers.body column if needed.
_TSVECTOR_MAX_BYTES = 1_048_575

# Count papers with body text (total source rows).
_COUNT_SOURCE_SQL = (
    "SELECT count(*) FROM papers"
    " WHERE body IS NOT NULL AND body != ''"
    f" AND length(body) <= {_TSVECTOR_MAX_BYTES}"
)

# Count existing rows in papers_ads_body (for reporting already-present).
_COUNT_TARGET_SQL = "SELECT count(*) FROM papers_ads_body"

# Batch insert using bibcode cursor pagination.
# harvested_at is set to now() since this is a backfill from an already-stored
# column — the original harvest time is not tracked separately.
_BATCH_INSERT_SQL = """
    INSERT INTO papers_ads_body (bibcode, body_text, body_length, harvested_at)
    SELECT bibcode, body, length(body), now()
    FROM papers
    WHERE body IS NOT NULL AND body != ''
          AND length(body) <= {max_bytes} AND bibcode > %s
    ORDER BY bibcode
    LIMIT %s
    ON CONFLICT (bibcode) DO NOTHING
""".format(max_bytes=_TSVECTOR_MAX_BYTES)

# Fetch the max bibcode from the batch we just tried to insert, for cursor
# pagination. We query papers directly (not papers_ads_body) because
# ON CONFLICT DO NOTHING means some rows may not appear in the target.
_CURSOR_SQL = """
    SELECT max(bibcode) FROM (
        SELECT bibcode FROM papers
        WHERE body IS NOT NULL AND body != ''
              AND length(body) <= {max_bytes} AND bibcode > %s
        ORDER BY bibcode
        LIMIT %s
    ) sub
""".format(max_bytes=_TSVECTOR_MAX_BYTES)


def run_backfill(cfg: BackfillConfig) -> BackfillStats:
    """Execute the backfill. Returns stats."""
    _check_production_guard(cfg)

    effective_dsn = cfg.dsn or DEFAULT_DSN
    t_start = time.monotonic()

    conn = get_connection(cfg.dsn)
    try:
        # Disable statement timeout for this session — the insert is long.
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = 0")

        # Get source count
        with conn.cursor() as cur:
            cur.execute(_COUNT_SOURCE_SQL)
            total_body_rows = cur.fetchone()[0]

        # Get pre-existing target count
        with conn.cursor() as cur:
            cur.execute(_COUNT_TARGET_SQL)
            pre_existing = cur.fetchone()[0]

        logger.info(
            "Backfill: %d papers with body, %d already in papers_ads_body, dsn=%s",
            total_body_rows,
            pre_existing,
            redact_dsn(effective_dsn),
        )

        if cfg.dry_run:
            logger.info("Dry run — no rows will be inserted.")
            return BackfillStats(
                rows_inserted=0,
                rows_already_present=pre_existing,
                total_body_rows=total_body_rows,
                elapsed_seconds=time.monotonic() - t_start,
                dry_run=True,
            )

        # Drop GIN index for bulk insert performance (if more than 10K rows to insert)
        idx_mgr = IndexManager(conn, "papers_ads_body")
        saved_indexes = []
        rows_to_insert = total_body_rows - pre_existing
        if rows_to_insert > 10_000:
            saved_indexes = idx_mgr.drop_indexes()
            if saved_indexes:
                logger.info(
                    "Dropped %d indexes on papers_ads_body for bulk load",
                    len(saved_indexes),
                )

        try:
            _run_batched_insert(conn, cfg.batch_size, total_body_rows)
        except Exception:
            logger.exception("Backfill failed during batched insert")
            conn.rollback()
            raise
        finally:
            # Always recreate indexes, even on failure
            if saved_indexes:
                logger.info("Recreating %d indexes on papers_ads_body", len(saved_indexes))
                idx_mgr.recreate_indexes(saved_indexes)
                logger.info("Index rebuild complete")

        # ANALYZE for planner statistics
        with conn.cursor() as cur:
            cur.execute("ANALYZE papers_ads_body")
        conn.commit()
        logger.info("ANALYZE papers_ads_body complete")

        # Get final target count
        with conn.cursor() as cur:
            cur.execute(_COUNT_TARGET_SQL)
            final_count = cur.fetchone()[0]

        elapsed = time.monotonic() - t_start
        rows_inserted = final_count - pre_existing

        logger.info(
            "Backfill complete: inserted=%d already_present=%d total_target=%d elapsed=%.1fs",
            rows_inserted,
            pre_existing,
            final_count,
            elapsed,
        )

        return BackfillStats(
            rows_inserted=rows_inserted,
            rows_already_present=pre_existing,
            total_body_rows=total_body_rows,
            elapsed_seconds=elapsed,
            dry_run=False,
        )
    finally:
        conn.close()


def _run_batched_insert(
    conn: "psycopg.Connection[Any]",
    batch_size: int,
    total: int,
) -> int:
    """Insert in batches using bibcode cursor pagination. Returns total inserted."""
    total_inserted = 0
    last_bibcode = ""
    batch_num = 0
    t0 = time.monotonic()

    while True:
        with conn.cursor() as cur:
            # Find the max bibcode for this batch window
            cur.execute(_CURSOR_SQL, (last_bibcode, batch_size))
            row = cur.fetchone()
            max_bib = row[0] if row else None

            if max_bib is None:
                # No more rows to process
                conn.commit()
                break

            # Insert the batch
            cur.execute(_BATCH_INSERT_SQL, (last_bibcode, batch_size))
            inserted = cur.rowcount

        conn.commit()
        total_inserted += inserted
        last_bibcode = max_bib
        batch_num += 1

        elapsed = time.monotonic() - t0
        rate = total_inserted / elapsed if elapsed > 0 else 0
        logger.info(
            "Batch %d: inserted=%d cumulative=%d/%d (%.0f rec/s, last=%s)",
            batch_num,
            inserted,
            total_inserted,
            total,
            rate,
            last_bibcode[:30] if last_bibcode else "",
        )

    return total_inserted


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_backfill(dsn: str | None = None, *, yes_production: bool = False) -> bool:
    """Verify the backfill: counts match and tsquery works.

    Read-only — but still requires ``yes_production`` for consistency
    with :func:`run_backfill`.
    """
    cfg = BackfillConfig(dsn=dsn, yes_production=yes_production)
    _check_production_guard(cfg)
    conn = get_connection(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(_COUNT_SOURCE_SQL)
            source_count = cur.fetchone()[0]

            cur.execute(_COUNT_TARGET_SQL)
            target_count = cur.fetchone()[0]

            logger.info(
                "Verification: papers.body=%d, papers_ads_body=%d",
                source_count,
                target_count,
            )

            if target_count < source_count:
                logger.warning(
                    "Target count (%d) < source count (%d) — %d rows missing",
                    target_count,
                    source_count,
                    source_count - target_count,
                )

            # Sample tsquery — just verify it doesn't error
            cur.execute(
                "SELECT bibcode FROM papers_ads_body "
                "WHERE tsv @@ to_tsquery('english', 'observation') LIMIT 1"
            )
            sample = cur.fetchone()
            if sample:
                logger.info("Sample tsquery hit: %s", sample[0])
            else:
                logger.warning("Sample tsquery returned no results")

        return target_count >= source_count
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill papers_ads_body from papers.body column.",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix').",
    )

    def _positive_int(value: str) -> int:
        n = int(value)
        if n < 1:
            raise argparse.ArgumentTypeError(f"batch-size must be >= 1, got {n}")
        return n

    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=50_000,
        help="Rows per INSERT batch (default: 50000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts but do not insert any rows.",
    )
    parser.add_argument(
        "--yes-production",
        action="store_true",
        help="Explicitly allow targeting a production DSN.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification (count check + tsquery).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.verify_only:
        try:
            ok = verify_backfill(args.dsn, yes_production=args.yes_production)
        except ProductionGuardError as exc:
            logging.error("%s", exc)
            return 2
        return 0 if ok else 1

    cfg = BackfillConfig(
        dsn=args.dsn,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        yes_production=args.yes_production,
    )

    try:
        stats = run_backfill(cfg)
    except ProductionGuardError as exc:
        logging.error("%s", exc)
        return 2

    logging.info(
        "Done: inserted=%d already_present=%d total_body=%d dry_run=%s elapsed=%.1fs",
        stats.rows_inserted,
        stats.rows_already_present,
        stats.total_body_rows,
        stats.dry_run,
        stats.elapsed_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
