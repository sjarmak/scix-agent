from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scix.eval.mcp_tool_surface.proxy import DEFAULT_INTERCEPTED_TOOLS, parse_server_command
from scix.eval.mcp_tool_surface.tasks import load_tasks

REPO_ROOT = Path(__file__).resolve().parents[4]
SYSTEM_PROMPT = (
    "Answer the researcher's request using the available MCP tools. Select the single tool whose "
    "description and schema best match the intent. Supply every parameter the request makes explicit. "
    "Do not ask for confirmation during this evaluation. Account-writing calls are safely intercepted."
)


@dataclass(frozen=True)
class RunResult:
    session_id: str
    model: str
    task_id: str
    prompt: str
    execution: str
    tool_calls: list[dict[str, Any]]
    final_text: str
    started_at: str
    finished_at: str
    exit_code: int
    stderr_tail: str


def build_mcp_config(
    server_command: list[str],
    log_path: Path,
    intercepted_tools: frozenset[str],
) -> dict[str, Any]:
    return {
        "mcpServers": {
            "eval_target": {
                "command": sys.executable,
                "args": [
                    "-m",
                    "scix.eval.mcp_tool_surface.proxy",
                    "--server-command-json",
                    json.dumps(server_command),
                    "--intercepted-tools-json",
                    json.dumps(sorted(intercepted_tools)),
                    "--log-file",
                    str(log_path),
                ],
                "env": {"PYTHONPATH": str(REPO_ROOT / "src")},
            }
        }
    }


def parse_stream_json(stdout: str) -> tuple[list[dict[str, Any]], str]:
    tool_calls: list[dict[str, Any]] = []
    final_parts: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "assistant":
            continue
        for block in (event.get("message") or {}).get("content", []):
            if block.get("type") == "tool_use":
                tool_calls.append({"name": block.get("name", ""), "input": block.get("input", {})})
            if block.get("type") == "text" and block.get("text", "").strip():
                final_parts.append(block["text"].strip())
    return tool_calls, "\n".join(final_parts)


async def invoke_claude(
    command: list[str],
    prompt: str,
    config_dir: Path,
    timeout_seconds: int,
) -> tuple[bytes, bytes, int]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=config_dir,
        env=dict(os.environ),
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(prompt.encode()),
            timeout=timeout_seconds,
        )
        return stdout_bytes, stderr_bytes, process.returncode or 0
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return b"", b"timeout", -1


async def run_one(
    task: dict[str, Any],
    model: str,
    server_command: list[str],
    out_dir: Path,
    timeout_seconds: int,
) -> RunResult:
    session_id = uuid.uuid4().hex[:12]
    proxy_log = out_dir / "proxy_logs" / f"{session_id}.jsonl"
    config = build_mcp_config(server_command, proxy_log, DEFAULT_INTERCEPTED_TOOLS)
    config_dir = Path(tempfile.mkdtemp(prefix="mcp-surface-eval-"))
    config_path = config_dir / "mcp.json"
    config_path.write_text(json.dumps(config))
    command = [
        "claude",
        "-p",
        "--model",
        model,
        "--strict-mcp-config",
        "--mcp-config",
        str(config_path),
        "--output-format",
        "stream-json",
        "--verbose",
        "--append-system-prompt",
        SYSTEM_PROMPT,
        "--allowedTools",
        "mcp__eval_target__*",
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        stdout_bytes, stderr_bytes, exit_code = await invoke_claude(
            command,
            task["prompt"],
            config_dir,
            timeout_seconds,
        )
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)
    tool_calls, final_text = parse_stream_json(stdout_bytes.decode(errors="replace"))
    finished_at = datetime.now(timezone.utc).isoformat()
    return RunResult(
        session_id=session_id,
        model=model,
        task_id=task["id"],
        prompt=task["prompt"],
        execution=task["execution"],
        tool_calls=tool_calls,
        final_text=final_text,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stderr_tail="\n".join(stderr_bytes.decode(errors="replace").splitlines()[-10:]),
    )


async def run_matrix(
    tasks: list[dict[str, Any]],
    models: list[str],
    server_command: list[str],
    out_dir: Path,
    concurrency: int,
    timeout_seconds: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    runs_path.write_text("")
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    completed = 0
    total = len(tasks) * len(models)

    async def worker(task: dict[str, Any], model: str) -> None:
        nonlocal completed
        async with semaphore:
            result = await run_one(task, model, server_command, out_dir, timeout_seconds)
        async with write_lock:
            with runs_path.open("a") as handle:
                handle.write(json.dumps(asdict(result)) + "\n")
            completed += 1
            marker = "OK" if result.exit_code == 0 and result.tool_calls else "WARN"
            print(
                f"[{completed}/{total}] {marker} {model} {task['id']} "
                f"calls={len(result.tool_calls)} exit={result.exit_code}",
                file=sys.stderr,
                flush=True,
            )

    await asyncio.gather(*(worker(task, model) for model in models for task in tasks))
    return runs_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=REPO_ROOT / "eval/mcp_tool_surface/scix_mcp_tasks.jsonl",
    )
    parser.add_argument("--models", nargs="+", default=["sonnet", "haiku"])
    parser.add_argument("--server-command-json", required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/mcp_tool_surface")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    server_command = parse_server_command(args.server_command_json)
    path = asyncio.run(
        run_matrix(
            tasks,
            args.models,
            server_command,
            args.out_dir,
            args.concurrency,
            args.timeout_seconds,
        )
    )
    print(path)


if __name__ == "__main__":
    main()
