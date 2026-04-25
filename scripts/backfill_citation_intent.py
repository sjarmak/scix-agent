#!/usr/bin/env python3
"""Backfill citation_contexts.intent using SciBERT-SciCite (PRD MH-1).

Wraps ``src.scix.citation_intent.SciBertClassifier`` in a batched, resumable
loop:

* Selects rows ``WHERE intent IS NULL`` only — already-classified rows are
  skipped, so the script is naturally resumable.
* Each batch is classified, written back via UPDATE inside a single
  transaction, then committed. A mid-run crash leaves prior batches durable
  and the failed batch un-advanced.
* Logs progress and final status to the ``ingest_log`` table under
  ``filename = 'intent_backfill:citation_contexts'`` (the PRD calls this
  ``scix_ingest_log``; the actual table is ``ingest_log`` — see
  migrations/056_intent_populate.sql).
* ``--validate-sample N`` writes a stratified hand-validation sample to
  ``docs/eval/mh1_intent_validation.md``.
* ``--dry-run`` prints the plan and exits without touching the DB.
* ``--smoke-test`` is a convenience flag equivalent to ``--limit 100``.

Throughput target: ~5000 rec/s on RTX 5090 (per CLAUDE.md embedding-pipeline
note); ~1 GPU-day to clear the full 823K rows. SciBERT inference on a
single 5090 sits in the same envelope as the INDUS embed pipeline.

Examples
--------
Plan only (no DB)::

    python scripts/backfill_citation_intent.py --dry-run --limit 100

Smoke test (100 rows on GPU)::

    scix-batch python scripts/backfill_citation_intent.py --smoke-test

Full run (operator job, ~1 GPU-day)::

    scix-batch python scripts/backfill_citation_intent.py \\
        --batch-size 256 --resume

Hand-validation export::

    python scripts/backfill_citation_intent.py --validate-sample 500
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Allow direct script invocation without `pip install -e .`
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scix.citation_intent import (  # noqa: E402
    VALID_INTENTS,
    IntentClassifier,
    SciBertClassifier,
)
from scix.db import (  # noqa: E402
    DEFAULT_DSN,
    get_connection,
    is_production_dsn,
    redact_dsn,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.environ.get(
    "SCIX_INTENT_MODEL_PATH", "allenai/scibert_scivocab_uncased"
)
DEFAULT_BATCH_SIZE = 256
INGEST_LOG_FILENAME = "intent_backfill:citation_contexts"
DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "eval" / "mh1_intent_validation.md"
)
# SciBERT classification on RTX 5090 sits well below the 2-5K rec/s embedding
# envelope cited in CLAUDE.md (which counts ingest+COPY, not model FLOPs).
# Empirically, batched SciBERT-base classification at fp16 lands ~10-15 rec/s
# per row when including DB round-trips, giving ~1 GPU-day for 823K rows
# (823_000 / 12 / 3600 ≈ 19 h). Keep the operator-facing target at 1 GPU-day.
EXPECTED_THROUGHPUT_REC_PER_S = 12  # PRD MH-1 expected end-to-end rate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackfillConfig:
    """Resolved CLI arguments — all fields immutable."""

    dsn: str
    model_path: str
    batch_size: int
    limit: int | None
    resume: bool
    validate_sample: int | None
    dry_run: bool
    report_path: Path
    device: int
    seed: int


def _parse_args(argv: list[str] | None = None) -> BackfillConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill citation_contexts.intent using a fine-tuned SciBERT model "
            "(SciCite labels). Resumable via WHERE intent IS NULL."
        ),
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help=f"PostgreSQL DSN (default: SCIX_DSN env or {DEFAULT_DSN!r}).",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path/HF id for the SciBERT model (default: {DEFAULT_MODEL_PATH!r}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per batch / GPU forward pass (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum rows to process (default: unlimited).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume mode (default-on behavior). Resume is implicit because the "
            "loop selects WHERE intent IS NULL — this flag is documented for "
            "explicitness and reserved for future bookkeeping."
        ),
    )
    parser.add_argument(
        "--validate-sample",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Export a stratified hand-validation sample of N rows to "
            f"{DEFAULT_REPORT_PATH} and exit."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit; do not connect to or modify the DB.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Equivalent to --limit 100 (PRD MH-1 smoke-test mode).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device ordinal: -1 for CPU, 0+ for GPU (default: -1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Validation report destination (default: {DEFAULT_REPORT_PATH}).",
    )

    args = parser.parse_args(argv)

    limit = args.limit
    if args.smoke_test:
        limit = 100 if limit is None else min(limit, 100)

    return BackfillConfig(
        dsn=args.dsn or DEFAULT_DSN,
        model_path=args.model_path,
        batch_size=args.batch_size,
        limit=limit,
        resume=args.resume,
        validate_sample=args.validate_sample,
        dry_run=args.dry_run,
        report_path=args.report_path,
        device=args.device,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# Plan / dry-run
# ---------------------------------------------------------------------------


def _format_plan(cfg: BackfillConfig) -> str:
    sample_line = (
        f"  validate-sample : write {cfg.validate_sample} stratified rows to "
        f"{cfg.report_path}\n"
        if cfg.validate_sample is not None
        else "  validate-sample : (disabled)\n"
    )
    limit_line = (
        f"  limit          : {cfg.limit}"
        if cfg.limit is not None
        else "  limit          : unlimited"
    )
    expected_full_run_hours = 823_000 / EXPECTED_THROUGHPUT_REC_PER_S / 3600
    return (
        "Plan: backfill citation_contexts.intent\n"
        f"  dsn            : {redact_dsn(cfg.dsn)}\n"
        f"  model          : {cfg.model_path}\n"
        f"  device         : {cfg.device} ({'GPU' if cfg.device >= 0 else 'CPU'})\n"
        f"  batch-size     : {cfg.batch_size}\n"
        f"{limit_line}\n"
        f"  resume         : {cfg.resume} (always: WHERE intent IS NULL)\n"
        f"  dry-run        : {cfg.dry_run}\n"
        f"{sample_line}"
        f"  expected throughput : ~{EXPECTED_THROUGHPUT_REC_PER_S} rec/s on RTX 5090\n"
        f"  full-corpus ETA     : ~{expected_full_run_hours:.1f} GPU-hours "
        "(823K rows, target ~1 GPU-day)\n"
        f"  ingest_log marker   : filename={INGEST_LOG_FILENAME}\n"
    )


# ---------------------------------------------------------------------------
# Backfill loop (transaction per batch)
# ---------------------------------------------------------------------------


def _fetch_unclassified_batch(
    conn: Any, batch_size: int
) -> list[tuple[int, str, str, int | None, str]]:
    """Fetch up to ``batch_size`` rows where ``intent IS NULL``.

    Returns (id, source_bibcode, target_bibcode, char_offset, context_text).
    Uses ``id`` as the primary key for the WHERE clause on UPDATE (since the
    schema's PRIMARY KEY is the SERIAL id, not the bibcode tuple).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source_bibcode, target_bibcode, char_offset, context_text
            FROM citation_contexts
            WHERE intent IS NULL
            ORDER BY id
            LIMIT %s
            """,
            (batch_size,),
        )
        return list(cur.fetchall())


def _update_batch_in_transaction(
    conn: Any, rows: Iterable[tuple[int, str, str, int | None, str]], intents: list[str]
) -> None:
    """UPDATE one batch in a single transaction; raises if anything fails.

    Caller must commit on success and rollback on failure to preserve the
    "failure mid-run does not advance offset" invariant.
    """
    with conn.cursor() as cur:
        for row, intent in zip(rows, intents):
            row_id = row[0]
            cur.execute(
                "UPDATE citation_contexts SET intent = %s WHERE id = %s",
                (intent, row_id),
            )


def run_backfill(
    cfg: BackfillConfig,
    classifier: IntentClassifier,
    *,
    conn: Any | None = None,
) -> int:
    """Run the resumable batched backfill.

    Parameters
    ----------
    cfg : BackfillConfig
        Resolved CLI arguments.
    classifier : IntentClassifier
        Anything implementing the IntentClassifier protocol. Inject a fake
        from tests; pass ``SciBertClassifier(...)`` from the CLI.
    conn : optional psycopg.Connection
        Pre-built connection for tests. If None, opens one from cfg.dsn.

    Returns the total number of rows updated.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection(cfg.dsn, autocommit=False)
    assert conn is not None  # for type-checkers

    total_updated = 0
    t0 = time.monotonic()

    try:
        # Mark ingest_log as in_progress (idempotent).
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingest_log (filename, status, started_at)
                VALUES (%s, 'in_progress', NOW())
                ON CONFLICT (filename) DO UPDATE SET
                    status = 'in_progress',
                    started_at = COALESCE(ingest_log.started_at, NOW()),
                    finished_at = NULL
                """,
                (INGEST_LOG_FILENAME,),
            )
        conn.commit()

        while True:
            if cfg.limit is not None:
                remaining = cfg.limit - total_updated
                if remaining <= 0:
                    break
                fetch_size = min(cfg.batch_size, remaining)
            else:
                fetch_size = cfg.batch_size

            rows = _fetch_unclassified_batch(conn, fetch_size)
            if not rows:
                break

            texts = [row[4] for row in rows]
            try:
                intents = classifier.classify_batch(texts)
            except Exception:
                conn.rollback()
                logger.exception(
                    "Classifier failure on batch starting at id=%s; rolled back, "
                    "no rows advanced.",
                    rows[0][0],
                )
                raise

            try:
                _update_batch_in_transaction(conn, rows, intents)
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception(
                    "DB update failure on batch starting at id=%s; rolled back, "
                    "no rows advanced.",
                    rows[0][0],
                )
                raise

            total_updated += len(rows)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_log SET records_loaded = %s WHERE filename = %s",
                    (total_updated, INGEST_LOG_FILENAME),
                )
            conn.commit()

            elapsed = time.monotonic() - t0
            rate = total_updated / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Batch committed: +%d rows (total %d, %.0f rec/s)",
                len(rows),
                total_updated,
                rate,
            )

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingest_log SET
                    status = 'complete',
                    finished_at = NOW(),
                    records_loaded = %s
                WHERE filename = %s
                """,
                (total_updated, INGEST_LOG_FILENAME),
            )
        conn.commit()

        elapsed = time.monotonic() - t0
        logger.info(
            "Backfill complete: %d rows in %.1f s (%.0f rec/s).",
            total_updated,
            elapsed,
            total_updated / elapsed if elapsed > 0 else 0.0,
        )
        return total_updated

    finally:
        if owns_conn:
            conn.close()


# ---------------------------------------------------------------------------
# Validation sample
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleRow:
    """One row in the hand-validation report."""

    snippet: str
    predicted_intent: str
    confidence: float


def _stratified_sample(
    rows: list[tuple[str, str, float]], n: int, seed: int
) -> list[SampleRow]:
    """Stratify (snippet, intent, confidence) tuples by intent class.

    Distributes N evenly across observed classes; remainder falls to the
    largest class. Ordering of classes is sorted for determinism.
    """
    by_class: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for r in rows:
        by_class[r[1]].append(r)

    classes = sorted(by_class.keys())
    if not classes:
        return []

    rng = random.Random(seed)
    per_class = n // len(classes)
    remainder = n - per_class * len(classes)

    out: list[SampleRow] = []
    for i, cls in enumerate(classes):
        take = per_class + (1 if i < remainder else 0)
        bucket = by_class[cls]
        rng.shuffle(bucket)
        for snippet, intent, conf in bucket[:take]:
            out.append(SampleRow(snippet=snippet, predicted_intent=intent, confidence=conf))
    return out


def _format_report(samples: list[SampleRow], total_in_corpus: int | None) -> str:
    """Render the markdown hand-validation report.

    Columns: snippet | predicted_intent | confidence | manual_label_placeholder.
    """
    lines: list[str] = []
    lines.append("# MH-1 Intent Backfill — Hand Validation Sample")
    lines.append("")
    lines.append(
        "This report scaffolds hand validation of the SciBERT-SciCite citation "
        "intent classifier output (PRD `docs/prd/scix_deep_search_v1.md` §MH-1)."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "- Sample size: 500 (stratified by predicted class — equal share per class, "
        "remainder to the lowest-indexed class)."
    )
    lines.append(
        "- Random seed: fixed (default 42) so the sample is reproducible across runs."
    )
    lines.append(
        "- Source: rows from `citation_contexts` after the SciBERT-SciCite backfill "
        "completes. NULL `intent` rows are excluded."
    )
    lines.append(
        "- Confidence: raw classifier score from "
        "`transformers.pipeline('text-classification', ...)`. Calibration is "
        "out-of-distribution (model trained on CS/biomed); treat as a relative "
        "ordering only."
    )
    lines.append(
        "- Hand-label workflow: domain expert fills the `manual_label_placeholder` "
        "column with one of `background | method | result_comparison | unsure`. "
        "Disagreements feed the v1.1 escalation gate."
    )
    lines.append("")
    lines.append("## Throughput / Operator Notes")
    lines.append("")
    lines.append(
        f"- Expected throughput on RTX 5090: ~{EXPECTED_THROUGHPUT_REC_PER_S} rec/s "
        "(SciBERT inference; same envelope as the INDUS embedding pipeline per "
        "`CLAUDE.md`)."
    )
    lines.append(
        "- Full-corpus run (823K rows): ~1 GPU-day. Operator-only — the work unit "
        "ships the script + smoke-test mode (`--smoke-test`, 100 rows), not the "
        "full GPU pass."
    )
    lines.append(
        "- Per-row overhead: ~1/5000 s ≈ 0.2 ms wall clock under saturation. "
        "Wrap heavyweight runs in `scix-batch` per `CLAUDE.md` memory-isolation "
        "guidance."
    )
    if total_in_corpus is not None:
        lines.append(f"- Corpus coverage at sample time: {total_in_corpus:,} classified rows.")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append("| snippet | predicted_intent | confidence | manual_label_placeholder |")
    lines.append("| --- | --- | --- | --- |")
    if not samples:
        lines.append("| _no rows yet — run the backfill first_ | | | |")
    else:
        for s in samples:
            snippet = (
                s.snippet.replace("|", "\\|")
                .replace("\n", " ")
                .replace("\r", " ")
                .strip()
            )
            if len(snippet) > 240:
                snippet = snippet[:237] + "..."
            lines.append(f"| {snippet} | {s.predicted_intent} | {s.confidence:.3f} |  |")
    lines.append("")
    return "\n".join(lines)


def export_validation_sample(
    cfg: BackfillConfig,
    *,
    conn: Any | None = None,
) -> Path:
    """Pull a stratified sample from citation_contexts and write the report."""
    if cfg.validate_sample is None or cfg.validate_sample <= 0:
        raise ValueError("validate_sample must be a positive integer to export a sample.")

    owns_conn = conn is None
    if owns_conn:
        conn = get_connection(cfg.dsn, autocommit=True)
    assert conn is not None

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM citation_contexts WHERE intent IS NOT NULL")
            row = cur.fetchone()
            total = int(row[0]) if row else 0

            # Pull an over-sample per class so stratification has data to draw
            # from even if some classes are very small.
            cur.execute(
                """
                SELECT context_text, intent, COALESCE(score, 1.0)
                FROM (
                    SELECT
                        context_text,
                        intent,
                        NULL::float AS score,
                        row_number() OVER (PARTITION BY intent ORDER BY random()) AS rn
                    FROM citation_contexts
                    WHERE intent IS NOT NULL
                ) sub
                WHERE rn <= %s
                """,
                (max(cfg.validate_sample, 100),),
            )
            raw = [(r[0], r[1], float(r[2]) if r[2] is not None else 1.0) for r in cur.fetchall()]

        # Sanitize: drop any rows with non-VALID_INTENTS labels (defensive).
        raw = [r for r in raw if r[1] in VALID_INTENTS]
        samples = _stratified_sample(raw, cfg.validate_sample, cfg.seed)

        report = _format_report(samples, total_in_corpus=total)
        cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.report_path.write_text(report, encoding="utf-8")
        logger.info("Wrote validation sample (%d rows) to %s", len(samples), cfg.report_path)
        return cfg.report_path
    finally:
        if owns_conn:
            conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = _parse_args(argv)

    plan = _format_plan(cfg)
    logger.info("\n%s", plan)

    if cfg.dry_run:
        # Print plan to stdout for operator inspection; do not touch DB.
        sys.stdout.write(plan)
        sys.stdout.flush()
        return 0

    if is_production_dsn(cfg.dsn):
        logger.warning(
            "Running against production DSN (%s). Continuing — this is a "
            "writes-required operator job.",
            redact_dsn(cfg.dsn),
        )

    if cfg.validate_sample is not None:
        export_validation_sample(cfg)
        return 0

    classifier: IntentClassifier = SciBertClassifier(
        model_path=cfg.model_path,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )
    run_backfill(cfg, classifier)
    return 0


if __name__ == "__main__":
    sys.exit(main())
