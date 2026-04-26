"""Tool-surface eval scorer.

Reads results/tool_surface_eval/runs.jsonl + the oracle from
eval/tool_surface/queries.jsonl. Computes per-(variant, query) and per-variant
aggregate metrics:

- tool_correct        first MCP call's tool name matches the oracle (with alt_ok)
- params_correct      first MCP call's args contain the oracle's required subset
- mcp_call_count      number of MCP tool calls (excluding ToolSearch resolution)
- selection_consistency  same-query × R runs all picked the same first tool

Writes:
  results/tool_surface_eval/scored.jsonl    per-run scoring rows
  results/tool_surface_eval/summary.json    aggregate per-variant + per-cluster
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]


def _strip_mcp_prefix(name: str) -> str:
    """mcp__scixstub_v1__search → search."""
    if name.startswith("mcp__"):
        parts = name.split("__")
        if len(parts) >= 3:
            return "__".join(parts[2:])
    return name


def _mcp_calls_only(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out Claude Code internals (ToolSearch, etc) and keep only the
    actual MCP tool invocations, with the mcp__ prefix stripped."""
    out = []
    for tc in tool_calls:
        name = tc.get("name", "")
        if not name.startswith("mcp__"):
            continue
        out.append({"name": _strip_mcp_prefix(name), "input": tc.get("input", {})})
    return out


def _args_subset_match(actual: dict[str, Any], required: dict[str, Any]) -> bool:
    """True iff every key in `required` is present in `actual` with an equal value.
    Extra keys in `actual` are fine."""
    for k, v in required.items():
        if k not in actual:
            return False
        if actual[k] != v:
            return False
    return True


def _candidate_oracles(oracle_for_variant: dict[str, Any]) -> list[dict[str, Any]]:
    """The oracle entry for a variant has a primary tool/args plus optional
    alt_ok list of equally-acceptable alternatives. Return the full
    candidate list."""
    primary = {"tool": oracle_for_variant["tool"], "args_subset": oracle_for_variant.get("args_subset", {})}
    out = [primary]
    for alt in oracle_for_variant.get("alt_ok", []):
        if isinstance(alt, str):
            # Bare tool name with no required args
            out.append({"tool": alt, "args_subset": {}})
        elif isinstance(alt, dict):
            out.append({"tool": alt["tool"], "args_subset": alt.get("args_subset", {})})
    return out


def score_run(run: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    variant = run["variant"]
    mcp_calls = _mcp_calls_only(run["tool_calls"])
    var_oracle = oracle.get(variant)
    if var_oracle is None:
        return {
            **{k: run.get(k) for k in ("session_id", "variant", "query_id", "run_idx")},
            "mcp_call_count": len(mcp_calls),
            "tool_correct": None,
            "params_correct": None,
            "first_tool": (mcp_calls[0]["name"] if mcp_calls else None),
            "no_oracle": True,
        }
    candidates = _candidate_oracles(var_oracle)

    if not mcp_calls:
        return {
            **{k: run.get(k) for k in ("session_id", "variant", "query_id", "run_idx")},
            "mcp_call_count": 0,
            "tool_correct": False,
            "params_correct": False,
            "first_tool": None,
        }

    first = mcp_calls[0]
    tool_correct = any(first["name"] == c["tool"] for c in candidates)
    # Params: only check args against the candidate(s) that matched on tool name
    matching = [c for c in candidates if c["tool"] == first["name"]]
    params_correct = (
        tool_correct
        and any(_args_subset_match(first["input"], c["args_subset"]) for c in matching)
    )

    return {
        **{k: run.get(k) for k in ("session_id", "variant", "query_id", "run_idx")},
        "mcp_call_count": len(mcp_calls),
        "tool_correct": tool_correct,
        "params_correct": params_correct,
        "first_tool": first["name"],
        "first_args": first["input"],
    }


def aggregate(scored: list[dict[str, Any]], oracle: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-variant aggregates + per-(variant, intent_cluster) breakdown."""
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scored:
        by_variant[s["variant"]].append(s)

    summary: dict[str, Any] = {"variants": {}}
    for variant, rows in by_variant.items():
        n = len(rows)
        if n == 0:
            continue
        tool_acc = sum(1 for r in rows if r["tool_correct"]) / n
        param_acc = sum(1 for r in rows if r["params_correct"]) / n
        avg_calls = statistics.fmean([r["mcp_call_count"] for r in rows])

        # selection consistency: per-query, are all R runs the same first_tool?
        per_query: dict[str, list[str | None]] = defaultdict(list)
        for r in rows:
            per_query[r["query_id"]].append(r.get("first_tool"))
        consistent_q = sum(1 for tools in per_query.values() if len(set(tools)) == 1)
        consistency = consistent_q / len(per_query) if per_query else 0.0

        # per-intent cluster from oracle metadata (if available)
        by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            intent = oracle.get(r["query_id"], {}).get("intent", "unknown")
            by_intent[intent].append(r)
        intent_breakdown = {}
        for intent, irows in by_intent.items():
            in_n = len(irows)
            intent_breakdown[intent] = {
                "n": in_n,
                "tool_correct": sum(1 for r in irows if r["tool_correct"]) / in_n,
                "params_correct": sum(1 for r in irows if r["params_correct"]) / in_n,
                "avg_mcp_calls": statistics.fmean([r["mcp_call_count"] for r in irows]),
            }

        summary["variants"][variant] = {
            "n_runs": n,
            "n_queries": len(per_query),
            "tool_accuracy": tool_acc,
            "param_accuracy": param_acc,
            "avg_mcp_calls": avg_calls,
            "selection_consistency": consistency,
            "by_intent": intent_breakdown,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score tool-surface eval runs")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "results/tool_surface_eval/runs.jsonl")
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "eval/tool_surface/queries.jsonl")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/tool_surface_eval")
    args = parser.parse_args()

    # Load oracle: id → {intent, oracle (per variant)}
    oracle = {}
    for line in args.queries.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        oracle[q["id"]] = {
            "intent": q.get("intent", "unknown"),
            **q["oracle"],
        }

    # Load + score runs
    runs = [json.loads(l) for l in args.runs.read_text().splitlines() if l.strip()]
    scored = [score_run(r, oracle.get(r["query_id"], {})) for r in runs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.out_dir / "scored.jsonl"
    with scored_path.open("w") as fh:
        for s in scored:
            fh.write(json.dumps(s) + "\n")

    summary = aggregate(scored, oracle)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Pretty print
    print("Per-variant summary:")
    for variant, agg in summary["variants"].items():
        print(f"\n  {variant}: n={agg['n_runs']} runs across {agg['n_queries']} queries")
        print(f"    tool_accuracy:        {agg['tool_accuracy']:.2%}")
        print(f"    param_accuracy:       {agg['param_accuracy']:.2%}")
        print(f"    avg_mcp_calls:        {agg['avg_mcp_calls']:.2f}")
        print(f"    selection_consistency: {agg['selection_consistency']:.2%}")

    print(f"\nscored runs: {scored_path}")
    print(f"summary:     {summary_path}")


if __name__ == "__main__":
    main()
