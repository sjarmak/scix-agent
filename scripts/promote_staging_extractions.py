#!/usr/bin/env python3
"""Promote rows from staging.* into public.* for the M4 NER pipeline.

Batches:
    * ``staging.extractions``            -> ``public.extractions``
    * ``staging.extraction_entity_links`` -> ``public.extraction_entity_links``

Both INSERTs use ``ON CONFLICT DO NOTHING`` against the unique constraints
declared in migrations 009 and 049, so re-running the promotion is a no-op
for already-promoted rows.

Usage::

    # live promotion using SCIX_DSN
    python scripts/promote_staging_extractions.py --batch-size 5000

    # dry run — SELECTs and INSERTs run but the transaction rolls back
    python scripts/promote_staging_extractions.py --dry-run

    # only promote a specific producer
    python scripts/promote_staging_extractions.py --source-filter ner_v1

Tests drive the :func:`promote` function directly with a test DSN; see
``tests/test_promote_staging_extractions.py``.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional

import psycopg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.db import DEFAULT_DSN, get_connection, redact_dsn  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 10_000


@dataclass(frozen=True)
class PromotionCounts:
    """Number of rows promoted per table in a single invocation."""

    extractions: int
    entity_links: int

    @property
    def total(self) -> int:
        return self.extractions + self.entity_links


def _promote_extractions(
    cur: psycopg.Cursor,
    batch_size: int,
    source_filter: Optional[str],
) -> int:
    """Promote up to ``batch_size`` rows from staging.extractions into public.

    Only the columns present on public.extractions (migration 001 + 009)
    are selected; provenance columns on staging (source, confidence_tier)
    are preserved in staging for audit but do not travel to public.
    """
    params: list[object] = []
    where = ""
    if source_filter is not None:
        where = "WHERE s.source = %s"
        params.append(source_filter)
    params.append(batch_size)

    sql = f"""
        WITH picked AS (
            SELECT
                s.bibcode,
                s.extraction_type,
                s.extraction_version,
                s.payload,
                s.created_at
              FROM staging.extractions s
              {where}
              ORDER BY s.id
              LIMIT %s
        ),
        inserted AS (
            INSERT INTO public.extractions
                (bibcode, extraction_type, extraction_version, payload, created_at)
            SELECT bibcode, extraction_type, extraction_version, payload, created_at
              FROM picked
            ON CONFLICT (bibcode, extraction_type, extraction_version)
            DO NOTHING
            RETURNING 1
        )
        SELECT count(*) FROM inserted
    """
    cur.execute(sql, tuple(params))
    row = cur.fetchone()
    return int(row[0]) if row is not None else 0


def _promote_entity_links(
    cur: psycopg.Cursor,
    batch_size: int,
    source_filter: Optional[str],
) -> int:
    """Promote up to ``batch_size`` rows from staging.extraction_entity_links."""
    params: list[object] = []
    where = ""
    if source_filter is not None:
        where = "WHERE s.source = %s"
        params.append(source_filter)
    params.append(batch_size)

    sql = f"""
        WITH picked AS (
            SELECT
                s.extraction_id,
                s.bibcode,
                s.entity_type,
                s.entity_id,
                s.entity_surface,
                s.entity_canonical,
                s.span_start,
                s.span_end,
                s.source,
                s.confidence_tier,
                s.confidence,
                s.extraction_version,
                s.payload,
                s.created_at
              FROM staging.extraction_entity_links s
              {where}
              ORDER BY s.id
              LIMIT %s
        ),
        inserted AS (
            INSERT INTO public.extraction_entity_links (
                extraction_id, bibcode, entity_type, entity_id,
                entity_surface, entity_canonical, span_start, span_end,
                source, confidence_tier, confidence, extraction_version,
                payload, created_at
            )
            SELECT
                extraction_id, bibcode, entity_type, entity_id,
                entity_surface, entity_canonical, span_start, span_end,
                source, confidence_tier, confidence, extraction_version,
                payload, created_at
              FROM picked
            ON CONFLICT (bibcode, entity_type, entity_surface,
                         extraction_version, source)
            DO NOTHING
            RETURNING 1
        )
        SELECT count(*) FROM inserted
    """
    cur.execute(sql, tuple(params))
    row = cur.fetchone()
    return int(row[0]) if row is not None else 0


def promote(
    conn: psycopg.Connection,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
    source_filter: Optional[str] = None,
) -> PromotionCounts:
    """Promote one batch of staging rows into public.

    Runs both INSERTs inside a single transaction.  When ``dry_run`` is
    true the transaction is rolled back instead of committed — useful for
    sizing batches without touching public.

    Parameters
    ----------
    conn:
        An already-open psycopg connection.  The caller manages the
        connection lifetime.
    batch_size:
        Per-table ceiling on the number of rows to promote in this call.
    dry_run:
        When true, all work is rolled back.  The returned counts still
        reflect what *would* have been promoted.
    source_filter:
        Optional ``staging.*.source`` value to restrict the promotion to
        a single producer.

    Returns
    -------
    PromotionCounts with per-table row counts.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    previous_autocommit = conn.autocommit
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            n_ext = _promote_extractions(cur, batch_size, source_filter)
            n_eel = _promote_entity_links(cur, batch_size, source_filter)

        if dry_run:
            conn.rollback()
            logger.info(
                "promote(dry_run): would have promoted %d extractions, %d entity_links",
                n_ext,
                n_eel,
            )
        else:
            conn.commit()
            logger.info(
                "promote: committed %d extractions, %d entity_links",
                n_ext,
                n_eel,
            )
        return PromotionCounts(extractions=n_ext, entity_links=n_eel)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = previous_autocommit


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=DEFAULT_DSN,
        help="PostgreSQL DSN (default: $SCIX_DSN or 'dbname=scix')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Max rows to promote per table in this invocation (default: 10000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the INSERTs then roll back instead of committing",
    )
    parser.add_argument(
        "--source-filter",
        default=None,
        help="Only promote rows whose staging.*.source matches this value",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    logger.info(
        "Promoting staging -> public (dsn=%s, batch_size=%d, dry_run=%s, source_filter=%s)",
        redact_dsn(args.dsn),
        args.batch_size,
        args.dry_run,
        args.source_filter,
    )

    with get_connection(args.dsn, autocommit=False) as conn:
        counts = promote(
            conn,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            source_filter=args.source_filter,
        )

    print(
        f"promoted: extractions={counts.extractions} "
        f"entity_links={counts.entity_links} total={counts.total}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
