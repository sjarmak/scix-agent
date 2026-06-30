#!/usr/bin/env python3
"""Lafia-style informal-reference pass over paper bodies (bead dbl.18).

Streams ``papers.body`` (OA/preprint-gated by default), runs the pure
cue-phrase detector in ``scix.extract.lafia`` over each body, and upserts the
detected software/dataset references into ``entities`` + ``document_entities``
under the ``match_method='lafia_informal'`` namespace. These are the
context-disambiguated mentions GLiNER misses (e.g. "data from the LAMOST
survey", "implemented in TOPCAT"), so they sit alongside — never on top of —
the GLiNER ``mentions`` rows (distinct ``link_type='informal_mention'``).

Always wrap heavy production runs in scix-batch (CLAUDE.md systemd-oomd rule):

    scix-batch python scripts/run_lafia_pass.py --max-papers 2000 --dry-run

    scix-batch --mem-high 8G --mem-max 16G \\
        python scripts/run_lafia_pass.py --allow-prod

The pass is pure-Python (no model, no GPU) and resumable: it walks bibcodes in
``bibcode`` order, checkpoints each batch in ``ingest_log``, and ``--since-bibcode``
continues from a watermark. The full-corpus production run is an operator action
(``--allow-prod`` under scix-batch); this script ships the pipeline plus a
``--dry-run`` smoke/estimate mode.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make src/ importable when running from a worktree without an editable
# install — same pattern as scripts/run_ner_bodies.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import (  # noqa: E402
    DEFAULT_DSN,
    get_connection,
    is_production_dsn,
    redact_dsn,
)
from scix.extract.lafia import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    detect_informal_references,
    insert_mentions,
)
from scix.extract.ner_pass import iter_paper_batches  # noqa: E402

logger = logging.getLogger("run_lafia_pass")

_CHECKPOINT_SQL = """
    INSERT INTO ingest_log
        (filename, records_loaded, edges_loaded, status, finished_at)
    VALUES (%s, %s, %s, 'complete', now())
    ON CONFLICT (filename) DO UPDATE
        SET records_loaded = EXCLUDED.records_loaded,
            edges_loaded   = EXCLUDED.edges_loaded,
            status         = EXCLUDED.status,
            finished_at    = now()
"""


def _checkpoint_key(first_bibcode: str) -> str:
    return f"lafia_pass:{first_bibcode}"


def _is_batch_done(conn, key: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM ingest_log WHERE filename = %s", (key,))
        row = cur.fetchone()
    return bool(row and row[0] == "complete")


def run_pipeline(
    conn,
    *,
    batch_size: int,
    since_bibcode: str | None,
    max_papers: int | None,
    min_confidence: float,
    dry_run: bool,
    oa_only: bool,
    log_every: int = 50,
) -> dict[str, int]:
    """Drive the pass; return totals. Commits + checkpoints per batch."""
    totals = {"papers_seen": 0, "papers_with_mentions": 0, "rows_written": 0}
    n_batches = 0
    for batch in iter_paper_batches(
        conn,
        target="body",
        batch_size=batch_size,
        since_bibcode=since_bibcode,
        max_papers=max_papers,
        oa_only=oa_only,
    ):
        key = _checkpoint_key(batch[0].bibcode)
        if not dry_run and _is_batch_done(conn, key):
            logger.info("skip checkpointed batch %s", key)
            totals["papers_seen"] += len(batch)
            continue

        batch_rows = 0
        for paper in batch:
            totals["papers_seen"] += 1
            mentions = [
                m for m in detect_informal_references(paper.text) if m.confidence >= min_confidence
            ]
            if not mentions:
                continue
            totals["papers_with_mentions"] += 1
            if dry_run:
                batch_rows += len(mentions)
                for m in mentions:
                    logger.debug(
                        "%s [%.2f %s/%s] %s",
                        paper.bibcode,
                        m.confidence,
                        m.entity_type,
                        m.cue_id,
                        m.surface,
                    )
            else:
                batch_rows += insert_mentions(conn, paper.bibcode, mentions)

        totals["rows_written"] += batch_rows
        if not dry_run:
            _record_checkpoint(conn, key, totals["papers_with_mentions"], batch_rows)
            conn.commit()

        n_batches += 1
        if n_batches % log_every == 0:
            logger.info(
                "batches=%d papers_seen=%d papers_with_mentions=%d rows_written=%d",
                n_batches,
                totals["papers_seen"],
                totals["papers_with_mentions"],
                totals["rows_written"],
            )
    return totals


def _record_checkpoint(conn, key: str, records_loaded: int, edges_loaded: int) -> None:
    with conn.cursor() as cur:
        cur.execute(_CHECKPOINT_SQL, (key, records_loaded, edges_loaded))


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--batch-size", type=int, default=500, help="Papers per cursor batch (default: 500)."
    )
    p.add_argument(
        "--since-bibcode",
        default=None,
        help="Resume watermark — process bibcodes strictly greater than this.",
    )
    p.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Cap total papers processed (for sample / smoke runs).",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Cue confidence floor for emitted mentions " f"(default: {DEFAULT_MIN_CONFIDENCE}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and count but skip all DB writes (smoke / yield estimate).",
    )
    p.add_argument(
        "--allow-prod", action="store_true", help="Required to write to the production DSN."
    )
    p.add_argument(
        "--include-closed",
        action="store_true",
        help="Process closed-access papers too (default: OA/preprint only).",
    )
    p.add_argument("--dsn", default=None, help="Database DSN; defaults to SCIX_DSN.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    dsn = args.dsn or DEFAULT_DSN
    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "Refusing to write to production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        return 2
    if args.allow_prod and not args.dry_run and not os.environ.get("INVOCATION_ID"):
        logger.error(
            "Refusing to run --allow-prod outside a systemd scope. "
            "Invoke via: scix-batch python %s <args...>",
            os.path.basename(sys.argv[0] or __file__),
        )
        return 2

    logger.info(
        "Lafia informal-reference pass on %s "
        "(dry_run=%s, since=%s, max=%s, min_conf=%.2f, oa_only=%s)",
        redact_dsn(dsn),
        args.dry_run,
        args.since_bibcode,
        args.max_papers,
        args.min_confidence,
        not args.include_closed,
    )
    if args.include_closed:
        logger.warning(
            "--include-closed is ACTIVE: pass will read closed-access bodies. "
            "Pattern detection is abstract-safe but body access carries TDM risk."
        )

    conn = get_connection(dsn)
    try:
        totals = run_pipeline(
            conn,
            batch_size=args.batch_size,
            since_bibcode=args.since_bibcode,
            max_papers=args.max_papers,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run,
            oa_only=not args.include_closed,
        )
    finally:
        conn.close()

    seen = totals["papers_seen"] or 1
    logger.info(
        "DONE: papers_seen=%d papers_with_mentions=%d rows_written=%d "
        "(%.3f rows/paper, %.1f%% papers hit)",
        totals["papers_seen"],
        totals["papers_with_mentions"],
        totals["rows_written"],
        totals["rows_written"] / seen,
        100.0 * totals["papers_with_mentions"] / seen,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
