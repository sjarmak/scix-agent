#!/usr/bin/env python3
"""Operator-precision sample helper for tier-3 discipline-gated links.

Bead xz4.p2v gates the linker on ≥0.90 manual precision over a 100-row
random sample. This script pulls the sample, formats each row as a
human-readable judgment card, and writes both:

* ``build-artifacts/tier3_target_gated_eval_<N>.tsv`` — machine-readable
  table with columns ``bibcode | canonical_name | confidence |
  matched_surface | judgment`` ready for manual annotation in a
  spreadsheet.
* ``build-artifacts/tier3_target_gated_eval_<N>.md`` — human-readable
  cards (title, abstract excerpt, matched surface, link evidence).

After the operator fills in the ``judgment`` column (TP/FP/SKIP), run
the same script with ``--score`` pointing at the annotated TSV; it
reports precision and aborts non-zero if precision < 0.90.

Usage::

    # Sample 100 random tier-3 links (default size)
    python scripts/eval_target_gated_precision.py --sample

    # Score an annotated TSV
    python scripts/eval_target_gated_precision.py \\
        --score build-artifacts/tier3_target_gated_eval_100.tsv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import pathlib
import sys
import textwrap
from dataclasses import dataclass
from typing import Optional, Sequence

import psycopg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.db import DEFAULT_DSN, get_connection  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE: int = 100
DEFAULT_OUT_DIR: pathlib.Path = REPO_ROOT / "build-artifacts"
PRECISION_GATE: float = 0.90


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgmentRow:
    bibcode: str
    entity_id: int
    canonical_name: str
    confidence: float
    matched_surface: str
    is_alias: bool
    title: str
    abstract: str
    arxiv_class: list[str]
    bibstem: list[str]
    evidence: dict[str, object]


_SAMPLE_SQL = """
    SELECT de.bibcode,
           de.entity_id,
           e.canonical_name,
           de.confidence,
           de.evidence,
           p.title,
           p.abstract,
           COALESCE(p.arxiv_class, ARRAY[]::text[]) AS arxiv_class,
           COALESCE(p.bibstem, ARRAY[]::text[]) AS bibstem
      FROM document_entities de
      JOIN entities e ON e.id = de.entity_id
      JOIN papers   p ON p.bibcode = de.bibcode
     WHERE de.link_type = 'target_gated_match'
       AND de.tier = 3
     ORDER BY random()
     LIMIT %s
"""


def sample_judgment_rows(conn: psycopg.Connection, n: int) -> list[JudgmentRow]:
    """Pull ``n`` random tier-3 links with paper context."""
    rows: list[JudgmentRow] = []
    with conn.cursor() as cur:
        cur.execute(_SAMPLE_SQL, (n,))
        for r in cur.fetchall():
            (
                bibcode,
                entity_id,
                canonical,
                confidence,
                evidence,
                title,
                abstract,
                arxiv_class,
                bibstem,
            ) = r
            ev = dict(evidence or {})
            rows.append(
                JudgmentRow(
                    bibcode=bibcode,
                    entity_id=int(entity_id),
                    canonical_name=canonical,
                    confidence=float(confidence or 0.0),
                    matched_surface=str(ev.get("matched_surface", canonical)),
                    is_alias=bool(ev.get("is_alias", False)),
                    title=title or "",
                    abstract=abstract or "",
                    arxiv_class=list(arxiv_class or []),
                    bibstem=list(bibstem or []),
                    evidence=ev,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_tsv(rows: Sequence[JudgmentRow], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(
            [
                "bibcode",
                "entity_id",
                "canonical_name",
                "matched_surface",
                "confidence",
                "is_alias",
                "judgment",  # operator fills this column: TP / FP / SKIP
                "comment",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.bibcode,
                    r.entity_id,
                    r.canonical_name,
                    r.matched_surface,
                    f"{r.confidence:.4f}",
                    "true" if r.is_alias else "false",
                    "",  # judgment
                    "",  # comment
                ]
            )
    logger.info("Wrote %d rows to %s", len(rows), out_path)


def write_cards(rows: Sequence[JudgmentRow], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = [
        "# Tier-3 Discipline-Gated Target Linker — Manual Precision Sample",
        "",
        f"Sample size: **{len(rows)}**.  Gate: precision ≥ **{PRECISION_GATE:.2f}**.",
        "",
        "Annotate each row in the companion TSV with `TP` (true positive), "
        "`FP` (false positive), or `SKIP` (cannot judge — no abstract, etc.). ",
        "Re-run this script with `--score` to compute precision.",
        "",
        "---",
        "",
    ]
    for idx, r in enumerate(rows, start=1):
        abstract_preview = textwrap.shorten(
            r.abstract.replace("\n", " "), width=600, placeholder=" […]"
        )
        out.extend(
            [
                f"## {idx}. `{r.bibcode}` → entity {r.entity_id} (`{r.canonical_name}`)",
                "",
                f"- matched surface: **`{r.matched_surface}`** "
                f"({'alias' if r.is_alias else 'canonical'})",
                f"- confidence: `{r.confidence:.3f}`",
                f"- arxiv_class: {r.arxiv_class}",
                f"- bibstem: {r.bibstem}",
                f"- evidence: `{r.evidence}`",
                "",
                f"**title:** {r.title or '_(none)_'}",
                "",
                f"**abstract:** {abstract_preview or '_(none)_'}",
                "",
                "---",
                "",
            ]
        )
    out_path.write_text("\n".join(out), encoding="utf-8")
    logger.info("Wrote judgment cards to %s", out_path)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreReport:
    total: int
    tp: int
    fp: int
    skip: int

    @property
    def judged(self) -> int:
        return self.tp + self.fp

    @property
    def precision(self) -> float:
        if self.judged == 0:
            return 0.0
        return self.tp / self.judged


def score_tsv(path: pathlib.Path) -> ScoreReport:
    tp = fp = skip = total = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            total += 1
            judgment = (row.get("judgment") or "").strip().upper()
            if judgment == "TP":
                tp += 1
            elif judgment == "FP":
                fp += 1
            elif judgment in ("SKIP", "SKP", ""):
                skip += 1
            else:
                # Unrecognized labels behave like SKIP but logged.
                logger.warning(
                    "row %s: unknown judgment %r — counting as SKIP",
                    row.get("bibcode"),
                    judgment,
                )
                skip += 1
    return ScoreReport(total=total, tp=tp, fp=fp, skip=skip)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db-url", type=str, default=None)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true", help="Pull a random sample")
    g.add_argument(
        "--score",
        type=pathlib.Path,
        default=None,
        help="Score an annotated TSV (judgment column filled with TP/FP/SKIP)",
    )
    p.add_argument(
        "-n",
        "--size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Sample size (default {DEFAULT_SAMPLE_SIZE})",
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default {DEFAULT_OUT_DIR})",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.score:
        report = score_tsv(args.score)
        print(
            f"judged {report.judged}/{report.total} (skip={report.skip})  "
            f"TP={report.tp}  FP={report.fp}  precision={report.precision:.3f}"
        )
        if report.judged == 0:
            print(
                "ERROR: no TP/FP rows found. Annotate the judgment column.",
                file=sys.stderr,
            )
            return 2
        if report.precision < PRECISION_GATE:
            print(
                f"FAIL: precision {report.precision:.3f} < gate {PRECISION_GATE:.2f}",
                file=sys.stderr,
            )
            return 1
        print(f"PASS: precision {report.precision:.3f} >= gate {PRECISION_GATE:.2f}")
        return 0

    # --sample
    dsn = args.db_url or os.environ.get("SCIX_TEST_DSN") or DEFAULT_DSN
    conn = get_connection(dsn)
    try:
        rows = sample_judgment_rows(conn, args.size)
    finally:
        conn.close()

    if not rows:
        print(
            "no tier-3 target_gated_match rows found — "
            "run scripts/link_targets_discipline_gated.py first",
            file=sys.stderr,
        )
        return 2

    tsv_path = args.out_dir / f"tier3_target_gated_eval_{args.size}.tsv"
    md_path = args.out_dir / f"tier3_target_gated_eval_{args.size}.md"
    write_tsv(rows, tsv_path)
    write_cards(rows, md_path)
    print(f"sampled {len(rows)} rows; annotate {tsv_path} then re-run with --score")
    return 0


if __name__ == "__main__":
    sys.exit(main())
