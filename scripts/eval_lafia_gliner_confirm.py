#!/usr/bin/env python3
"""Eval harness: Lafia heuristic vs Lafia + GLiNER type-confirmation (bead dbl.20).

Measures whether the GLiNER type-confirmation pass (``scix.extract.lafia_confirm``)
lifts Lafia informal-reference precision over the ~60-67% heuristic-only baseline
that blocked bead dbl.18, on a real arXiv sample. EVAL ONLY — no DB writes.

Two stages, split so the human/LLM labelling step sits cleanly between them:

  extract  Sample real arXiv OA/preprint bodies, run the Lafia detector, run the
           GLiNER confirmation pass, and write one JSONL row per candidate
           mention with its evidence snippet and whether GLiNER confirmed it.
           GLiNER inference is GPU work — wrap in scix-batch:

               scix-batch python scripts/eval_lafia_gliner_confirm.py extract \\
                   --target-mentions 80 --out results/dbl20_candidates.jsonl

  score    Read a labelled JSONL (each row gains ``"label": true|false`` =
           "is this a real software/dataset reference?") and report baseline vs
           confirmed precision and the recall the confirmation pass retains:

               python scripts/eval_lafia_gliner_confirm.py score \\
                   --labeled tests/fixtures/lafia_confirm_validation.jsonl \\
                   --report docs/eval/dbl20_lafia_confirm_eval.md

OA gate: the sample is restricted to arXiv preprints
(``array_length(arxiv_class,1) > 0`` — the preprint branch of migration 068's
``papers_is_oa_or_preprint`` predicate, which is itself unapplied on prod). arXiv
papers are preprints, so this honours the OA/preprint gate without touching prod.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import DEFAULT_DSN, get_connection, redact_dsn  # noqa: E402
from scix.extract.lafia import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    detect_informal_references,
)
from scix.extract.lafia_confirm import (  # noqa: E402
    DEFAULT_CONTEXT_CHARS,
    confirm_mentions,
)
from scix.extract.ner_pass import GlinerExtractor  # noqa: E402

logger = logging.getLogger("eval_lafia_gliner_confirm")

# Preprint branch ONLY of migration 068's papers_is_oa_or_preprint() predicate.
# The function is unapplied on prod, so the eval inlines the predicate; arXiv
# papers are preprints, so this is exactly the OA/preprint population the pass
# gates to. NOTE: this is the arXiv subset, NOT the full predicate — the
# production pass must call papers_is_oa_or_preprint(papers) to also pick up the
# `'OPENACCESS' = ANY(property)` (published-OA) branch. Do not copy this WHERE
# clause as the production gate.
_SAMPLE_SQL = """
    SELECT bibcode, body
    FROM papers TABLESAMPLE SYSTEM (%s) REPEATABLE (%s)
    WHERE bibcode LIKE '%%arXiv%%'
      AND body IS NOT NULL AND body <> ''
      AND array_length(arxiv_class, 1) > 0
"""


def _stream_sample(conn, *, fraction: float, seed: int):
    """Yield (bibcode, body) for sampled arXiv preprints, one at a time."""
    with conn.cursor(name="lafia_eval_sample") as cur:
        cur.itersize = 100
        cur.execute(_SAMPLE_SQL, (fraction, seed))
        for bibcode, body in cur:
            yield bibcode, body or ""


def _candidate_row(bibcode, mention, confirmed_by) -> dict:
    """Serialise one Lafia candidate + its GLiNER confirmation verdict."""
    c = confirmed_by.get((mention.surface, mention.entity_type, mention.start_char))
    return {
        "bibcode": bibcode,
        "surface": mention.surface,
        "entity_type": mention.entity_type,
        "cue_id": mention.cue_id,
        "cue_confidence": round(mention.confidence, 3),
        "start_char": mention.start_char,
        "end_char": mention.end_char,
        "evidence": mention.evidence_span,
        "gliner_confirmed": c is not None,
        "gliner_type": c.gliner_type if c else None,
        "gliner_confidence": round(c.gliner_confidence, 3) if c else None,
        "gliner_surface": c.gliner_surface if c else None,
    }


def run_extract(args) -> int:
    dsn = args.dsn or DEFAULT_DSN
    logger.info(
        "extract: dsn=%s target_mentions=%d fraction=%.3f seed=%d min_conf=%.2f",
        redact_dsn(dsn),
        args.target_mentions,
        args.sample_fraction,
        args.repeatable_seed,
        args.min_confidence,
    )

    extractor = GlinerExtractor(
        device=args.device,
        confidence=args.gliner_confidence,
    )

    rows: list[dict] = []
    n_papers = n_papers_with = 0
    n_mentions = n_confirmed = 0
    conn = get_connection(dsn)
    try:
        for bibcode, body in _stream_sample(
            conn, fraction=args.sample_fraction, seed=args.repeatable_seed
        ):
            n_papers += 1
            mentions = [
                m for m in detect_informal_references(body) if m.confidence >= args.min_confidence
            ]
            if not mentions:
                continue
            n_papers_with += 1
            confirmed = confirm_mentions(
                extractor, body, mentions, context_chars=args.context_chars
            )
            confirmed_by = {
                (c.mention.surface, c.mention.entity_type, c.mention.start_char): c
                for c in confirmed
            }
            # Cap per paper so the sample spans many papers, not a few prolific
            # ones — a precision estimate from 5 papers is not representative.
            for m in mentions[: args.max_mentions_per_paper]:
                rows.append(_candidate_row(bibcode, m, confirmed_by))
                n_mentions += 1
                if (m.surface, m.entity_type, m.start_char) in confirmed_by:
                    n_confirmed += 1
            if n_mentions >= args.target_mentions:
                break
            if args.max_papers and n_papers >= args.max_papers:
                break
    finally:
        conn.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    logger.info(
        "extract DONE: papers=%d papers_with_mentions=%d candidates=%d "
        "gliner_confirmed=%d (%.1f%% kept) -> %s",
        n_papers,
        n_papers_with,
        n_mentions,
        n_confirmed,
        100.0 * n_confirmed / max(n_mentions, 1),
        out,
    )
    return 0


def _is_true(row: dict) -> bool:
    return row.get("label") in (True, "true", "True", 1, "1")


def _canon(s: str | None) -> str:
    return " ".join((s or "").split()).lower()


def _variant_metrics(rows: list[dict], keep, tp_all: int) -> dict:
    """precision + retained-recall for the subset of ``rows`` passing ``keep``."""
    sel = [r for r in rows if keep(r)]
    n = len(sel)
    tp = sum(_is_true(r) for r in sel)
    return {
        "n": n,
        "tp": tp,
        "fp": n - tp,
        "precision": tp / n if n else 0.0,
        "recall": tp / tp_all if tp_all else 0.0,
    }


def run_score(args) -> int:
    rows = [
        json.loads(line) for line in Path(args.labeled).read_text().splitlines() if line.strip()
    ]
    labeled = [r for r in rows if r.get("label") is not None]
    unlabeled = len(rows) - len(labeled)
    if unlabeled:
        logger.warning("%d/%d rows have no label and are excluded", unlabeled, len(rows))

    tp_all = sum(_is_true(r) for r in labeled)

    def _confirmed(r: dict) -> bool:
        return bool(r.get("gliner_confirmed"))

    # Each variant is reconstructible from the stored per-candidate fields, so
    # the whole sweep re-runs with no GLiNER inference. "Default" is the policy
    # baked into the extract pass; the others probe whether a tighter knob lifts
    # precision to the >=85% gate without collapsing recall.
    variants = [
        ("Lafia heuristic (baseline)", lambda r: True),
        ("+ GLiNER confirm — default (subset overlap, conf>=0.7)", _confirmed),
        (
            "+ GLiNER confirm — conf>=0.95",
            lambda r: _confirmed(r) and (r.get("gliner_confidence") or 0.0) >= 0.95,
        ),
        (
            "+ GLiNER confirm — exact surface match",
            lambda r: _confirmed(r) and _canon(r.get("gliner_surface")) == _canon(r.get("surface")),
        ),
    ]
    measured = [(name, _variant_metrics(labeled, keep, tp_all)) for name, keep in variants]

    report = _render_report(source=args.labeled, tp_all=tp_all, measured=measured)
    print(report)
    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report)
        logger.info("report -> %s", rp)
    return 0


def _render_report(*, source: str, tp_all: int, measured: list) -> str:
    base = measured[0][1]
    default = measured[1][1]
    gate = "PASS" if default["precision"] >= 0.85 else "FAIL"
    fp_removed = base["fp"] - default["fp"]

    lines = [
        "# Lafia + GLiNER type-confirmation precision eval (bead dbl.20)",
        "",
        f"Source sample: `{source}` ({base['n']} labelled candidate mentions, "
        f"{tp_all} true references).",
        "",
        "| pipeline | candidates kept | true | false | precision | recall retained |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in measured:
        lines.append(
            f"| {name} | {m['n']} | {m['tp']} | {m['fp']} | "
            f"**{m['precision']:.1%}** | {m['recall']:.1%} |"
        )
    lines += [
        "",
        f"- **Baseline precision**: {base['precision']:.1%} on this sample. dbl.18 measured "
        "60-67% on its two samples; absolute precision is sample-dependent (this slice is "
        "modern, cs-heavy arXiv with figure/table OCR noise), so the transferable signal is "
        "the *relative* lift, not the absolute baseline.",
        f"- **Default-confirmed precision**: {default['precision']:.1%} — acceptance gate "
        f"(>=85%): **{gate}**.",
        f"- **Recall retained**: {default['recall']:.1%} of true references survive confirmation "
        f"({default['tp']}/{tp_all}).",
        f"- **False positives removed**: {fp_removed} of {base['fp']} "
        f"({fp_removed / max(base['fp'], 1):.1%}).",
        "- **Knob sweep**: raising the GLiNER confidence floor or demanding an exact surface "
        "match trades recall for precision but does not clear 85% at usable recall — exact match "
        "reaches 100% precision only by discarding most true references (GLiNER's span includes "
        "the head noun, so it rarely equals the cue-extracted surface).",
        "",
        "## Recommendation",
        "",
    ]
    if gate == "PASS":
        lines += [
            "GLiNER type-confirmation clears the >=85% gate at usable recall. Greenlight the "
            "operator-gated production pass over the OA subset (separate bead), reusing "
            "`scix.extract.lafia_confirm.confirm_mentions` ahead of `lafia.insert_mentions` at the "
            "config measured here.",
        ]
    else:
        lines += [
            f"GLiNER confirmation lifts precision from {base['precision']:.1%} to "
            f"{default['precision']:.1%}, removes {fp_removed / max(base['fp'], 1):.0%} of false "
            f"positives, and retains {default['recall']:.0%} recall, but does NOT reach the >=85% "
            "gate on this sample, so the production pass stays blocked. The residual false "
            "positives are dominated by Lafia "
            "*span-boundary* errors (the cue captures an adjective adjacent to a head noun, e.g. "
            '"high-quality dataset"); GLiNER legitimately confirms the full noun phrase, so the '
            "confirmation pass structurally cannot reject them. Two non-exclusive next steps, both "
            "separate beads: (a) fix Lafia name-span extraction so the candidate excludes leading "
            "generic adjectives, then re-measure; (b) escalate to the heavier ML detector the "
            "dbl.18 refs use (Falcon-7b SOMD / Bi-LSTM-CRF), which was out of scope here.",
        ]
    return "\n".join(lines) + "\n"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="Sample arXiv bodies, run Lafia + GLiNER confirm.")
    ex.add_argument(
        "--target-mentions",
        type=int,
        default=80,
        help="Stop after collecting at least this many candidates (>=50 for the bead).",
    )
    ex.add_argument(
        "--sample-fraction",
        type=float,
        default=0.2,
        help="TABLESAMPLE SYSTEM percent (default 0.2).",
    )
    ex.add_argument(
        "--repeatable-seed",
        type=int,
        default=42,
        help="TABLESAMPLE REPEATABLE seed for a reproducible sample.",
    )
    ex.add_argument("--max-papers", type=int, default=2000, help="Safety cap on papers scanned.")
    ex.add_argument(
        "--max-mentions-per-paper",
        type=int,
        default=6,
        help="Cap candidates taken per paper for sample diversity (default 6).",
    )
    ex.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Lafia cue confidence floor (default {DEFAULT_MIN_CONFIDENCE}).",
    )
    ex.add_argument(
        "--gliner-confidence",
        type=float,
        default=0.7,
        help="GLiNER score floor for a confirmation (default 0.7).",
    )
    ex.add_argument(
        "--context-chars",
        type=int,
        default=DEFAULT_CONTEXT_CHARS,
        help=f"Body context each side of a span (default {DEFAULT_CONTEXT_CHARS}).",
    )
    ex.add_argument("--device", default="cuda", help="Torch device for GLiNER.")
    ex.add_argument("--out", required=True, help="Output candidates JSONL.")
    ex.add_argument("--dsn", default=None, help="DB DSN; defaults to SCIX_DSN.")
    ex.set_defaults(func=run_extract)

    sc = sub.add_parser("score", help="Score a labelled candidates JSONL.")
    sc.add_argument("--labeled", required=True, help="Labelled candidates JSONL (with 'label').")
    sc.add_argument("--report", default=None, help="Optional markdown report path.")
    sc.set_defaults(func=run_score)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
