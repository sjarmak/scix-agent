from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scix.eval.mcp_tool_surface.tasks import load_tasks

REPO_ROOT = Path(__file__).resolve().parents[4]


def strip_mcp_prefix(name: str) -> str:
    if name.startswith("mcp__"):
        parts = name.split("__")
        if len(parts) >= 3:
            return "__".join(parts[2:])
    return name


def score_run(run: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    base = {
        "session_id": run.get("session_id"),
        "model": run["model"],
        "task_id": run["task_id"],
        "intent": task["intent"],
    }
    if task["execution"] == "capability_gap":
        return {
            **base,
            "capability_gap": True,
            "gap_reason": task["gap_reason"],
            "tool_correct": None,
            "params_correct": None,
            "first_tool": None,
        }
    mcp_calls = [
        call for call in run.get("tool_calls", []) if call.get("name", "").startswith("mcp__")
    ]
    if not mcp_calls:
        return {
            **base,
            "capability_gap": False,
            "tool_correct": False,
            "params_correct": False,
            "first_tool": None,
        }
    first = mcp_calls[0]
    first_tool = strip_mcp_prefix(first["name"])
    arguments = first.get("input", {})
    oracle = task["oracle"]
    tool_correct = first_tool == oracle["tool"]
    required_keys = oracle.get("required_keys", [])
    subset = oracle.get("args_subset", {})
    keys_correct = all(key in arguments for key in required_keys)
    values_correct = all(arguments.get(key) == value for key, value in subset.items())
    return {
        **base,
        "capability_gap": False,
        "tool_correct": tool_correct,
        "params_correct": tool_correct and keys_correct and values_correct,
        "first_tool": first_tool,
        "first_args": arguments,
    }


def aggregate(scored: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_model[row["model"]].append(row)
    models: dict[str, Any] = {}
    for model, rows in by_model.items():
        measured = [row for row in rows if not row["capability_gap"]]
        models[model] = {
            "total_tasks": len(rows),
            "scored_tasks": len(measured),
            "capability_gaps": len(rows) - len(measured),
            "tool_accuracy": (
                sum(bool(row["tool_correct"]) for row in measured) / len(measured)
                if measured
                else None
            ),
            "param_accuracy": (
                sum(bool(row["params_correct"]) for row in measured) / len(measured)
                if measured
                else None
            ),
        }
    return {"models": models}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=REPO_ROOT / "eval/mcp_tool_surface/scix_mcp_tasks.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/mcp_tool_surface")
    args = parser.parse_args()
    tasks = {task["id"]: task for task in load_tasks(args.tasks)}
    runs = [json.loads(line) for line in args.runs.read_text().splitlines() if line.strip()]
    scored = [score_run(run, tasks[run["task_id"]]) for run in runs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "scored.jsonl").open("w") as handle:
        for row in scored:
            handle.write(json.dumps(row) + "\n")
    summary = aggregate(scored)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
