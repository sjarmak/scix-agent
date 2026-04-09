#!/usr/bin/env python3
"""Manual entity extraction quality gate.

Samples N papers from the extractions table, joins to abstracts, and dumps a
JSONL file containing each paper's bibcode, abstract, and extracted entities
grouped by extraction type. A human (or agent acting as one) then annotates
gold entities per paper and feeds the annotations back through this module to
compute precision, recall, F1 per extraction type and overall, then renders a
gate-decision markdown report.

Usage:
    # Step 1 - dump 50 sampled papers to JSONL
    scripts/manual_extraction_eval.py dump --sample-size 50 \
        --output build-artifacts/manual-eval-sample.jsonl

    # Step 2 - score annotated paper file (annotated JSONL has a "gold" key)
    scripts/manual_extraction_eval.py score \
        --annotated build-artifacts/manual-eval-annotated.jsonl \
        --output build-artifacts/extraction-quality-gate.md
"""

from __future__ import annotations

import argparse
import enum
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection  # noqa: E402

logger = logging.getLogger(__name__)

EXTRACTION_TYPES: tuple[str, ...] = ("datasets", "instruments", "materials", "methods")
DEFAULT_F1_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperSample:
    """A sampled paper with its abstract and extractions grouped by type."""

    bibcode: str
    abstract: str
    extracted: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class PerTypeMetrics:
    """Micro precision/recall/F1 for one extraction type or the overall total."""

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


class GateDecision(str, enum.Enum):
    PROCEED = "PROCEED"
    REDESIGN = "REDESIGN"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def fetch_samples(
    conn: psycopg.Connection,
    sample_size: int,
    seed: int | None = None,
) -> list[PaperSample]:
    """Sample papers with abstracts and group their extractions by type.

    Uses a single SQL query that:
      1) selects DISTINCT bibcodes from extractions joined to papers (so the
         abstract exists),
      2) ORDER BY RANDOM() and LIMIT N to draw the sample,
      3) joins extractions and aggregates entities into a JSONB object keyed
         by extraction_type.

    Args:
        conn: psycopg connection.
        sample_size: number of papers to sample.
        seed: optional integer for setseed() so the sample is reproducible.

    Returns:
        List of PaperSample.
    """
    with conn.cursor() as cur:
        if seed is not None:
            # Map int seed to [-1, 1] for setseed.
            cur.execute("SELECT setseed(%s)", (((seed % 2000) - 1000) / 1000.0,))
        cur.execute(
            """
            WITH sampled AS (
                SELECT p.bibcode, p.abstract
                FROM papers p
                WHERE p.abstract IS NOT NULL
                  AND p.bibcode IN (SELECT DISTINCT bibcode FROM extractions)
                ORDER BY RANDOM()
                LIMIT %s
            )
            SELECT
                s.bibcode,
                s.abstract,
                COALESCE(
                    jsonb_object_agg(
                        e.extraction_type,
                        COALESCE(e.payload->'entities', '[]'::jsonb)
                    ) FILTER (WHERE e.extraction_type IS NOT NULL),
                    '{}'::jsonb
                ) AS extractions
            FROM sampled s
            LEFT JOIN extractions e ON e.bibcode = s.bibcode
            GROUP BY s.bibcode, s.abstract
            """,
            (sample_size,),
        )
        rows = cur.fetchall()

    samples: list[PaperSample] = []
    for bibcode, abstract, ext in rows:
        if isinstance(ext, str):
            ext = json.loads(ext)
        grouped = {t: tuple(ext.get(t, []) or []) for t in EXTRACTION_TYPES}
        samples.append(
            PaperSample(bibcode=bibcode, abstract=abstract or "", extracted=grouped)
        )

    logger.info("Fetched %d sampled papers from the database", len(samples))
    return samples


def dump_samples_jsonl(samples: list[PaperSample], output: Path) -> Path:
    """Write sampled papers as one JSONL record per paper.

    Each record has bibcode, abstract, and an extracted dict keyed by type.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for s in samples:
            record = {
                "bibcode": s.bibcode,
                "abstract": s.abstract,
                "extracted": {t: list(s.extracted[t]) for t in EXTRACTION_TYPES},
                "gold": {t: [] for t in EXTRACTION_TYPES},  # to be filled by annotator
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d samples to %s", len(samples), output)
    return output


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    return s.strip().lower()


def score_paper(
    extracted: dict[str, tuple[str, ...]],
    gold: dict[str, tuple[str, ...]],
) -> dict[str, tuple[int, int, int]]:
    """Compare extracted vs gold per type, return (tp, fp, fn) per type."""
    out: dict[str, tuple[int, int, int]] = {}
    for t in EXTRACTION_TYPES:
        ext = {_normalize(x) for x in extracted.get(t, ())}
        gld = {_normalize(x) for x in gold.get(t, ())}
        tp = len(ext & gld)
        fp = len(ext - gld)
        fn = len(gld - ext)
        out[t] = (tp, fp, fn)
    return out


def aggregate_metrics(
    per_paper: list[dict[str, tuple[int, int, int]]],
) -> dict[str, PerTypeMetrics]:
    """Aggregate per-paper (tp, fp, fn) tuples into micro metrics per type."""
    totals: dict[str, list[int]] = {t: [0, 0, 0] for t in EXTRACTION_TYPES}
    overall = [0, 0, 0]

    for paper in per_paper:
        for t in EXTRACTION_TYPES:
            tp, fp, fn = paper.get(t, (0, 0, 0))
            totals[t][0] += tp
            totals[t][1] += fp
            totals[t][2] += fn
            overall[0] += tp
            overall[1] += fp
            overall[2] += fn

    metrics: dict[str, PerTypeMetrics] = {
        t: PerTypeMetrics(tp=totals[t][0], fp=totals[t][1], fn=totals[t][2])
        for t in EXTRACTION_TYPES
    }
    metrics["overall"] = PerTypeMetrics(tp=overall[0], fp=overall[1], fn=overall[2])
    return metrics


def decide_gate(
    metrics: dict[str, PerTypeMetrics],
    threshold: float = DEFAULT_F1_THRESHOLD,
) -> GateDecision:
    """Return PROCEED if overall F1 >= threshold else REDESIGN."""
    overall = metrics["overall"]
    return GateDecision.PROCEED if overall.f1 >= threshold else GateDecision.REDESIGN


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_gate_report(
    *,
    papers_evaluated: int,
    metrics: dict[str, PerTypeMetrics],
    decision: GateDecision,
    threshold: float,
    notes: str = "",
    failure_modes: list[str] | None = None,
) -> str:
    """Render the quality-gate markdown report."""
    overall = metrics["overall"]

    lines: list[str] = []
    lines.append("# Extraction Quality Gate")
    lines.append("")
    lines.append(f"**Decision**: `{decision.value}`")
    lines.append("")
    lines.append(f"**Overall F1**: {overall.f1:.2f} (threshold {threshold:.2f})")
    lines.append(f"**Papers evaluated**: {papers_evaluated}")
    lines.append("")

    lines.append("## Metrics by Extraction Type")
    lines.append("")
    lines.append("| Type | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|------|---:|---:|---:|----------:|-------:|---:|")
    for t in EXTRACTION_TYPES:
        m = metrics[t]
        lines.append(
            f"| {t} | {m.tp} | {m.fp} | {m.fn} | "
            f"{m.precision:.2f} | {m.recall:.2f} | {m.f1:.2f} |"
        )
    lines.append(
        f"| **overall** | {overall.tp} | {overall.fp} | {overall.fn} | "
        f"{overall.precision:.2f} | {overall.recall:.2f} | {overall.f1:.2f} |"
    )
    lines.append("")

    if failure_modes:
        lines.append("## Failure Modes Observed")
        lines.append("")
        for mode in failure_modes:
            lines.append(f"- {mode}")
        lines.append("")

    if notes:
        lines.append("## Notes")
        lines.append("")
        lines.append(notes)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_dump(args: argparse.Namespace) -> None:
    conn = get_connection(args.dsn)
    try:
        samples = fetch_samples(conn, sample_size=args.sample_size, seed=args.seed)
    finally:
        conn.close()
    dump_samples_jsonl(samples, Path(args.output))
    print(f"Wrote {len(samples)} samples to {args.output}")


def _load_annotated(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _cmd_score(args: argparse.Namespace) -> None:
    records = _load_annotated(Path(args.annotated))

    per_paper: list[dict[str, tuple[int, int, int]]] = []
    for r in records:
        extracted = {t: tuple(r.get("extracted", {}).get(t, [])) for t in EXTRACTION_TYPES}
        gold = {t: tuple(r.get("gold", {}).get(t, [])) for t in EXTRACTION_TYPES}
        per_paper.append(score_paper(extracted, gold))

    metrics = aggregate_metrics(per_paper)
    decision = decide_gate(metrics, threshold=args.threshold)

    report = format_gate_report(
        papers_evaluated=len(records),
        metrics=metrics,
        decision=decision,
        threshold=args.threshold,
        notes=args.notes or "",
        failure_modes=args.failure_mode,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote gate report to {out_path}")
    print(f"Decision: {decision.value}  (overall F1 = {metrics['overall'].f1:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    dump_p = sub.add_parser("dump", help="Sample papers and dump JSONL for annotation")
    dump_p.add_argument("--dsn", default=None)
    dump_p.add_argument("--sample-size", type=int, default=50)
    dump_p.add_argument("--seed", type=int, default=42)
    dump_p.add_argument(
        "--output",
        default="build-artifacts/manual-eval-sample.jsonl",
    )
    dump_p.set_defaults(func=_cmd_dump)

    score_p = sub.add_parser("score", help="Score an annotated JSONL and write gate report")
    score_p.add_argument("--annotated", required=True)
    score_p.add_argument(
        "--output",
        default="build-artifacts/extraction-quality-gate.md",
    )
    score_p.add_argument("--threshold", type=float, default=DEFAULT_F1_THRESHOLD)
    score_p.add_argument("--notes", default="")
    score_p.add_argument(
        "--failure-mode",
        action="append",
        default=[],
        help="Repeatable: failure-mode bullet to include in the report.",
    )
    score_p.set_defaults(func=_cmd_score)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
