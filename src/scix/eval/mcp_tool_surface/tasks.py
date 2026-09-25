from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXECUTION_MODES = frozenset({"live", "intercept", "capability_gap"})


def validate_task(task: dict[str, Any]) -> None:
    required = ("id", "prompt", "intent", "execution")
    missing = [key for key in required if not task.get(key)]
    if missing:
        raise ValueError(f"task is missing required fields: {', '.join(missing)}")
    if task["execution"] not in EXECUTION_MODES:
        raise ValueError(f"task {task['id']} has invalid execution mode")
    if task["execution"] == "capability_gap":
        if not task.get("gap_reason"):
            raise ValueError(f"task {task['id']} needs gap_reason")
        return
    oracle = task.get("oracle")
    if not isinstance(oracle, dict) or not oracle.get("tool"):
        raise ValueError(f"task {task['id']} needs an oracle tool")


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen: set[str] = set()
    for task in tasks:
        validate_task(task)
        task_id = task["id"]
        if task_id in seen:
            raise ValueError(f"duplicate task id: {task_id}")
        seen.add(task_id)
    return tasks
