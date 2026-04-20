"""Backfill paper_embeddings.embedding_hv (halfvec(768)) from embedding (vector).

Bead: scix_experiments-0vy.

This is the long-running leg of the halfvec migration (Option B, shadow
column). It batches over paper_embeddings in bibcode-order for
model_name='indus', casts the stored vector to halfvec(768), and writes the
result to embedding_hv. Progress is persisted in halfvec_backfill_progress
so restarts are cheap.

MUST be invoked via scix-batch so that user@1000.service OOM pressure does
not target the gascity supervisor:

    scix-batch python scripts/backfill_halfvec.py \
        --dsn "$SCIX_DSN" --model indus --batch-size 20000

Design notes
------------
- The UPDATE is guarded by `embedding_hv IS NULL` so re-runs skip already-
  migrated rows. This also makes the script safe to run concurrently with
  scripts/embed.py (new INDUS rows land with embedding_hv already set by
  the updated embed.py writer path; legacy rows get backfilled here).
- Batches are bounded by bibcode range (not LIMIT/OFFSET) so each batch
  takes a narrow, predictable lock window and does not rely on a stable
  row order across transactions.
- We commit after every batch to keep WAL pressure bounded. A batch of
  20k halfvec(768) rows writes ~30 MB of TOAST which is comfortably under
  a single checkpoint cycle.
- No paid-API dependencies. This is pure SQL: the cast happens inside
  postgres via `embedding::halfvec(768)`.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass

import psycopg

logger = logging.getLogger("halfvec_backfill")


@dataclass(frozen=True)
class BackfillConfig:
    dsn: str
    model_name: str
    batch_size: int
    max_batches: int | None
    resume: bool
    dry_run: bool


_STOP = False


def _install_signal_handler() -> None:
    def _handle(signum: int, _frame: object) -> None:  # noqa: ARG001
        global _STOP
        logger.warning("Received signal %s — will stop after current batch", signum)
        _STOP = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _fetch_progress_cursor(conn: psycopg.Connection, model_name: str) -> tuple[int, str | None]:
    """Return (progress_row_id, last_bibcode) for the most recent open run."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, last_bibcode
              FROM halfvec_backfill_progress
             WHERE model_name = %s AND finished_at IS NULL
             ORDER BY started_at DESC
             LIMIT 1
            """,
            (model_name,),
        )
        row = cur.fetchone()
    if row is None:
        return (_open_progress_row(conn, model_name), None)
    return (row[0], row[1])


def _open_progress_row(conn: psycopg.Connection, model_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO halfvec_backfill_progress (model_name)
            VALUES (%s)
            RETURNING id
            """,
            (model_name,),
        )
        row = cur.fetchone()
    conn.commit()
    assert row is not None
    return int(row[0])


def _update_progress(
    conn: psycopg.Connection,
    progress_id: int,
    last_bibcode: str,
    delta_rows: int,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE halfvec_backfill_progress
               SET last_bibcode = %s,
                   rows_updated = rows_updated + %s,
                   updated_at   = now()
             WHERE id = %s
            """,
            (last_bibcode, delta_rows, progress_id),
        )


def _close_progress_row(conn: psycopg.Connection, progress_id: int, note: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE halfvec_backfill_progress
               SET finished_at = now(),
                   note        = %s
             WHERE id = %s
            """,
            (note, progress_id),
        )
    conn.commit()


def _count_remaining(conn: psycopg.Connection, model_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*)
              FROM paper_embeddings
             WHERE model_name = %s
               AND embedding     IS NOT NULL
               AND embedding_hv  IS NULL
            """,
            (model_name,),
        )
        row = cur.fetchone()
    return int(row[0]) if row else 0


def _run_batch(
    conn: psycopg.Connection,
    model_name: str,
    cursor_bibcode: str | None,
    batch_size: int,
    dry_run: bool,
) -> tuple[int, str | None]:
    """Update one batch. Returns (rows_updated, new_cursor_bibcode)."""
    # Use a CTE so we know the exact max(bibcode) touched — this is our new
    # cursor. Without RETURNING we'd have to re-query for it.
    sql = """
        WITH batch AS (
            SELECT bibcode
              FROM paper_embeddings
             WHERE model_name = %(model)s
               AND embedding     IS NOT NULL
               AND embedding_hv  IS NULL
               AND (%(cursor)s::text IS NULL OR bibcode > %(cursor)s::text)
             ORDER BY bibcode
             LIMIT %(limit)s
        ),
        upd AS (
            UPDATE paper_embeddings pe
               SET embedding_hv = pe.embedding::halfvec(768)
              FROM batch b
             WHERE pe.bibcode    = b.bibcode
               AND pe.model_name = %(model)s
            RETURNING pe.bibcode
        )
        SELECT count(*)::bigint AS updated,
               max(bibcode)     AS max_bibcode
          FROM upd;
    """
    params = {
        "model": model_name,
        "cursor": cursor_bibcode,
        "limit": batch_size,
    }

    if dry_run:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*)::bigint, max(bibcode)
                  FROM (
                    SELECT bibcode
                      FROM paper_embeddings
                     WHERE model_name = %(model)s
                       AND embedding     IS NOT NULL
                       AND embedding_hv  IS NULL
                       AND (%(cursor)s::text IS NULL OR bibcode > %(cursor)s::text)
                     ORDER BY bibcode
                     LIMIT %(limit)s
                  ) q
                """,
                params,
            )
            row = cur.fetchone()
        conn.rollback()
        return (int(row[0]), row[1]) if row else (0, None)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    conn.commit()
    if row is None or row[0] == 0:
        return (0, None)
    return (int(row[0]), row[1])


def run(cfg: BackfillConfig) -> int:
    logger.info(
        "halfvec backfill starting — model=%s batch=%d dry_run=%s",
        cfg.model_name,
        cfg.batch_size,
        cfg.dry_run,
    )

    with psycopg.connect(cfg.dsn) as conn:
        remaining_before = _count_remaining(conn, cfg.model_name)
        logger.info("rows pending backfill: %d", remaining_before)
        if remaining_before == 0:
            logger.info("nothing to do — all rows already have embedding_hv")
            return 0

        progress_id, cursor_bibcode = _fetch_progress_cursor(conn, cfg.model_name)
        logger.info(
            "progress row id=%d resume_cursor=%s",
            progress_id,
            cursor_bibcode or "(start)",
        )

        batches = 0
        total_updated = 0
        t_start = time.perf_counter()

        while True:
            if _STOP:
                logger.warning("stop signal received — flushing and exiting")
                break
            if cfg.max_batches is not None and batches >= cfg.max_batches:
                logger.info("reached --max-batches=%d, stopping", cfg.max_batches)
                break

            t_batch = time.perf_counter()
            updated, new_cursor = _run_batch(
                conn, cfg.model_name, cursor_bibcode, cfg.batch_size, cfg.dry_run
            )
            batch_ms = (time.perf_counter() - t_batch) * 1000.0

            if updated == 0 or new_cursor is None:
                logger.info("no more rows to backfill")
                break

            total_updated += updated
            cursor_bibcode = new_cursor
            batches += 1

            if not cfg.dry_run:
                _update_progress(conn, progress_id, new_cursor, updated)
                conn.commit()

            if batches % 10 == 0 or batches == 1:
                rate = total_updated / max(1.0, time.perf_counter() - t_start)
                logger.info(
                    "batch=%d updated=%d cumulative=%d rate=%.0f rows/s "
                    "batch_ms=%.0f cursor=%s",
                    batches,
                    updated,
                    total_updated,
                    rate,
                    batch_ms,
                    new_cursor,
                )

        note = "stopped" if _STOP else "finished"
        if not cfg.dry_run:
            _close_progress_row(conn, progress_id, note)

        remaining_after = _count_remaining(conn, cfg.model_name)
        logger.info(
            "halfvec backfill %s — updated=%d elapsed=%.1fs remaining=%d",
            note,
            total_updated,
            time.perf_counter() - t_start,
            remaining_after,
        )
    return 0 if not _STOP else 130


def parse_args(argv: list[str] | None = None) -> BackfillConfig:
    p = argparse.ArgumentParser(description="Backfill paper_embeddings.embedding_hv from embedding.")
    p.add_argument("--dsn", default=os.environ.get("SCIX_DSN", "dbname=scix"))
    p.add_argument("--model", dest="model_name", default="indus")
    p.add_argument("--batch-size", type=int, default=20000)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return BackfillConfig(
        dsn=args.dsn,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _install_signal_handler()
    sys.exit(run(parse_args()))
