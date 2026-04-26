"""Tool-surface eval report generator.

Reads results/tool_surface_eval/{summary.json, scored.jsonl} and writes
a human-readable Markdown report to docs/eval/tool_surface_eval.md with:

- Headline per-variant table (tool accuracy, param accuracy, avg calls, consistency)
- Per-intent-cluster breakdown showing where each variant wins/loses
- Notable per-query disagreements (queries where v0 was correct but v1 wasn't, or vice versa)
- A summary verdict suggesting which variant to ship
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]


def _pct(x: float | None) -> str:
    return f"{x:.1%}" if x is not None else "—"


def _fmt(x: float | None) -> str:
    return f"{x:.2f}" if x is not None else "—"


def render(summary: dict[str, Any], scored: list[dict[str, Any]], queries: list[dict[str, Any]]) -> str:
    variants = sorted(summary["variants"].keys())
    if not variants:
        return "# Tool-Surface Eval\n\n_No data._\n"

    out: list[str] = []
    out.append("# Tool-Surface Eval — V0 (current 18) vs V1 (consolidated 8) vs V2 (terse v1)")
    out.append("")
    out.append(
        "Compares agent tool-selection across three MCP surface variants. Each "
        "variant returns identical canned data so we measure *selection* "
        "independently from retrieval quality. Agent: claude `-p` (Sonnet via OAuth subagent), "
        "5 runs per (variant, query) at default temperature."
    )
    out.append("")

    # ---- Headline table
    out.append("## Headline")
    out.append("")
    out.append("| variant | n_runs | tool_accuracy | param_accuracy | avg_mcp_calls | selection_consistency |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for v in variants:
        agg = summary["variants"][v]
        out.append(
            f"| {v} | {agg['n_runs']} | {_pct(agg['tool_accuracy'])} | "
            f"{_pct(agg['param_accuracy'])} | {_fmt(agg['avg_mcp_calls'])} | "
            f"{_pct(agg['selection_consistency'])} |"
        )
    out.append("")

    # ---- Per-intent breakdown
    intents = sorted({k for v in variants for k in summary["variants"][v]["by_intent"].keys()})
    out.append("## Per-intent cluster")
    out.append("")
    header = "| intent | n_q | " + " | ".join(f"{v} tool / param" for v in variants) + " |"
    sep = "|---|---:|" + "|".join("---:" for _ in variants) + "|"
    out.append(header)
    out.append(sep)
    for intent in intents:
        # N queries for this intent (count unique query_ids in the per-intent rows of any variant)
        sample_variant = next((v for v in variants if intent in summary["variants"][v]["by_intent"]), None)
        if sample_variant is None:
            continue
        n_q = summary["variants"][sample_variant]["by_intent"][intent]["n"]
        cells = []
        for v in variants:
            ib = summary["variants"][v]["by_intent"].get(intent)
            if ib is None:
                cells.append("—")
                continue
            cells.append(f"{_pct(ib['tool_correct'])} / {_pct(ib['params_correct'])}")
        out.append(f"| {intent} | {n_q} | " + " | ".join(cells) + " |")
    out.append("")

    # ---- Per-query disagreement table (variant pairs where one was right and the other wrong)
    # Aggregate per-query: was each variant majority-correct (≥3/5)?
    by_q_v: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for s in scored:
        by_q_v[(s["query_id"], s["variant"])].append(bool(s["tool_correct"]))
    majority_correct: dict[tuple[str, str], bool] = {}
    for (qid, v), flags in by_q_v.items():
        majority_correct[(qid, v)] = sum(flags) > len(flags) / 2

    q_text = {q["id"]: q["query"] for q in queries}
    q_intent = {q["id"]: q.get("intent", "") for q in queries}
    disagreements: list[tuple[str, str, dict[str, bool]]] = []
    all_qids = sorted({k[0] for k in majority_correct.keys()})
    for qid in all_qids:
        per_v = {v: majority_correct.get((qid, v), False) for v in variants}
        if len(set(per_v.values())) > 1:  # disagreement
            disagreements.append((qid, q_text.get(qid, ""), per_v))

    out.append("## Disagreements (queries where variants split)")
    out.append("")
    if not disagreements:
        out.append("_No disagreements — all variants agreed on every query._")
        out.append("")
    else:
        out.append(f"_{len(disagreements)} of {len(all_qids)} queries had at least one variant disagree on majority correctness._")
        out.append("")
        out.append("| query_id | intent | query | " + " | ".join(variants) + " |")
        out.append("|---|---|---|" + "|".join("---" for _ in variants) + "|")
        for qid, qtext, per_v in disagreements[:25]:
            cells = ["✓" if per_v[v] else "✗" for v in variants]
            qtext_short = qtext if len(qtext) <= 60 else qtext[:57] + "..."
            out.append(
                f"| {qid} | {q_intent.get(qid, '')} | {qtext_short} | " + " | ".join(cells) + " |"
            )
        if len(disagreements) > 25:
            out.append(f"| _… {len(disagreements) - 25} more rows omitted; see scored.jsonl_ |||" + "|" * len(variants))
        out.append("")

    # ---- Verdict
    out.append("## Verdict")
    out.append("")
    best = max(variants, key=lambda v: summary["variants"][v]["tool_accuracy"])
    deltas = []
    base = summary["variants"][variants[0]]["tool_accuracy"]
    for v in variants[1:]:
        d = summary["variants"][v]["tool_accuracy"] - base
        deltas.append(f"{v} vs {variants[0]}: {d:+.1%}")
    out.append(f"- Highest tool accuracy: **{best}** ({_pct(summary['variants'][best]['tool_accuracy'])})")
    out.append(f"- Deltas vs {variants[0]}: " + ", ".join(deltas))
    consistency_best = max(variants, key=lambda v: summary["variants"][v]["selection_consistency"])
    out.append(f"- Most consistent across runs: **{consistency_best}** ({_pct(summary['variants'][consistency_best]['selection_consistency'])})")
    avg_calls_min = min(variants, key=lambda v: summary["variants"][v]["avg_mcp_calls"])
    out.append(f"- Fewest MCP calls per query: **{avg_calls_min}** ({_fmt(summary['variants'][avg_calls_min]['avg_mcp_calls'])})")
    out.append("")
    out.append(
        "_Interpretation: a >5pt advantage on tool_accuracy with comparable or "
        "better consistency suggests the consolidated surface is at least as "
        "selectable. A drop in v2 vs v1 isolates the description-quality effect._"
    )
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render tool-surface eval report")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "results/tool_surface_eval/summary.json")
    parser.add_argument("--scored", type=Path, default=REPO_ROOT / "results/tool_surface_eval/scored.jsonl")
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "eval/tool_surface/queries.jsonl")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs/eval/tool_surface_eval.md")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    scored = [json.loads(l) for l in args.scored.read_text().splitlines() if l.strip()]
    queries = [json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()]

    md = render(summary, scored, queries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
