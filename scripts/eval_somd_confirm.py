#!/usr/bin/env python3
"""Score the SOMD software-mention detector as a second confirmation stage (bead dbl.22).

The cheap Lafia + GLiNER pass plateaus at 78.9% precision on the dbl.21 labelled
sample (``results/dbl21_lafia_confirm_postfix.jsonl``); the 8 surviving false
positives split into 5 software-typed ("Adam solver", "IBM framework", ...) and 3
dataset-typed. GLiNER cannot reject the software ones — they *are* software-typed.
This harness measures whether the heavier SOMD detector (``scix.extract.somd_detect``)
rejects them without dropping the genuine software references, by replaying it over
the SAME labelled sample — apples-to-apples with the GLiNER-only numbers, no
re-labelling.

Pipeline scored:

    GLiNER + SOMD  =  gliner_confirmed AND (dataset-typed OR a SOMD software span
                      overlaps the candidate surface in its evidence window)

Dataset-typed candidates pass through on GLiNER's verdict alone — SOMD/SoMeSci
annotates software only, so it is out of domain for datasets (documented limit).

EVAL ONLY: reads the labelled JSONL + the trained detector, writes a markdown
report. No DB access, no prod writes.

    .venv/bin/python scripts/eval_somd_confirm.py \
        --labeled results/dbl21_lafia_confirm_postfix.jsonl \
        --model models/somd_scibert \
        --report results/dbl22_somd_eval.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("eval_somd_confirm")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.extract.somd_detect import SomdDetector, _overlaps  # noqa: E402


def _locate_surface(evidence: str, surface: str) -> tuple[int, int] | None:
    """Char span of ``surface`` inside ``evidence``, the occurrence nearest centre.

    The evidence window is centred on the candidate, so when a short surface
    occurs more than once we want the central one. Case-insensitive; returns None
    if the surface is not found verbatim (logged by the caller, never silently
    dropped).
    """
    hay = evidence.lower()
    needle = surface.lower()
    centre = len(evidence) / 2
    best: tuple[int, int] | None = None
    best_dist = float("inf")
    start = hay.find(needle)
    while start != -1:
        mid = start + len(needle) / 2
        dist = abs(mid - centre)
        if dist < best_dist:
            best_dist, best = dist, (start, start + len(surface))
        start = hay.find(needle, start + 1)
    return best


def _prf(rows: list[dict], kept_key: str) -> tuple[int, int, int, float, float]:
    """(kept, true_kept, total_true, precision, recall_retained) for a pipeline flag."""
    total_true = sum(1 for r in rows if r["label"])
    kept = [r for r in rows if r[kept_key]]
    true_kept = sum(1 for r in kept if r["label"])
    precision = true_kept / len(kept) if kept else 0.0
    recall = true_kept / total_true if total_true else 0.0
    return len(kept), true_kept, total_true, precision, recall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeled", type=Path, default=Path("results/dbl21_lafia_confirm_postfix.jsonl"))
    ap.add_argument("--model", type=Path, default=Path("models/somd_scibert"))
    ap.add_argument("--report", type=Path, default=Path("results/dbl22_somd_eval.md"))
    ap.add_argument("--gate", type=float, default=0.85)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = [json.loads(line) for line in args.labeled.read_text().splitlines() if line.strip()]
    detector = SomdDetector(args.model)

    evidences = [r["evidence"] for r in rows]
    per_row_spans = detector.detect_batch(evidences)

    unlocatable = 0
    for r, spans in zip(rows, per_row_spans, strict=True):
        gliner_ok = bool(r.get("gliner_confirmed"))
        is_software = r.get("entity_type") == "software"
        somd_overlap = False
        loc = _locate_surface(r["evidence"], r["surface"])
        if loc is None and is_software:
            unlocatable += 1
            logger.warning(
                "unlocatable software surface %r in row %s — keeping GLiNER verdict",
                r["surface"], r.get("bibcode"),
            )
        if loc is not None:
            clo, chi = loc
            somd_overlap = any(_overlaps(clo, chi, s.start_char, s.end_char) for s in spans)
        r["_baseline"] = True
        r["_gliner"] = gliner_ok
        # Software must clear SOMD; datasets pass through on GLiNER's verdict.
        # An unlocatable software surface keeps the GLiNER verdict (no silent drop).
        somd_pass = somd_overlap or (loc is None)
        r["_gliner_somd"] = gliner_ok and (somd_pass if is_software else True)
        r["_somd_overlap"] = somd_overlap

    pipelines = [
        ("Lafia heuristic (baseline)", "_baseline"),
        ("+ GLiNER confirm (dbl.20/21)", "_gliner"),
        ("+ GLiNER + SOMD detector (dbl.22)", "_gliner_somd"),
    ]
    lines = [
        "# Lafia + SOMD software-mention detector precision eval (bead dbl.22)",
        "",
        f"Source sample: `{args.labeled}` "
        f"({len(rows)} labelled candidate mentions, {sum(1 for r in rows if r['label'])} true references).",
        f"Detector: SciBERT fine-tuned on SOMD-2024 subtask-1 (`{args.model}`).",
        "",
        "| pipeline | kept | true | false | precision | recall retained |",
        "|---|---|---|---|---|---|",
    ]
    results = {}
    for name, key in pipelines:
        kept, true_kept, total_true, prec, rec = _prf(rows, key)
        results[key] = (kept, true_kept, prec, rec)
        lines.append(
            f"| {name} | {kept} | {true_kept} | {kept - true_kept} | "
            f"**{prec * 100:.1f}%** | {rec * 100:.1f}% |"
        )

    g_kept, g_true, g_prec, g_rec = results["_gliner"]
    gs_kept, gs_true, gs_prec, gs_rec = results["_gliner_somd"]
    gate_pass = gs_prec >= args.gate
    lines += [
        "",
        f"- **GLiNER-only precision**: {g_prec * 100:.1f}% (recall {g_rec * 100:.1f}%).",
        f"- **GLiNER + SOMD precision**: {gs_prec * 100:.1f}% (recall {gs_rec * 100:.1f}%) "
        f"— acceptance gate (>={args.gate * 100:.0f}%): **{'PASS' if gate_pass else 'FAIL'}**.",
        f"- SOMD dropped {g_kept - gs_kept} of the {g_kept} GLiNER-confirmed candidates "
        f"({g_true - gs_true} true, {(g_kept - g_true) - (gs_kept - gs_true)} false).",
    ]
    if unlocatable:
        lines.append(
            f"- NOTE: {unlocatable} software candidate(s) had no verbatim surface in their "
            "evidence window; these kept the GLiNER verdict (not silently dropped)."
        )

    # Per-FP breakdown: which GLiNER-surviving false positives did SOMD reject?
    surviving_fp = [r for r in rows if r["_gliner"] and not r["label"]]
    lines += ["", "## GLiNER-surviving false positives — SOMD verdict", "",
              "| entity_type | surface | gliner_surface | SOMD rejected? |",
              "|---|---|---|---|"]
    for r in surviving_fp:
        rejected = r["_gliner"] and not r["_gliner_somd"]
        verdict = "✅ rejected" if rejected else ("— (dataset, out of domain)" if r["entity_type"] != "software" else "❌ kept")
        lines.append(f"| {r['entity_type']} | {r['surface']!r} | {r.get('gliner_surface')!r} | {verdict} |")

    args.report.write_text("\n".join(lines) + "\n")
    sys.stderr.write(
        f"GLiNER {g_prec*100:.1f}% -> GLiNER+SOMD {gs_prec*100:.1f}% "
        f"(recall {g_rec*100:.1f}% -> {gs_rec*100:.1f}%), gate {'PASS' if gate_pass else 'FAIL'}; "
        f"report -> {args.report}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
