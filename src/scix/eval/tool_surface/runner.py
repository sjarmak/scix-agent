"""Tool-surface eval runner.

For each (variant, query, run_idx) tuple:
1. Generate a session_id and a per-session log path
2. Write an MCP config pointing claude -p at the stub server with that log path
3. Spawn `claude -p --bare --strict-mcp-config --mcp-config <cfg> --output-format
   stream-json --append-system-prompt <prompt> --allowedTools mcp__scixstub_<v>__*
   <query>`
4. Parse stream-json events to extract tool_use blocks
5. Persist {session_id, variant, query_id, run_idx, tool_calls, final_text} to
   results/tool_surface_eval/runs.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"

SYSTEM_PROMPT = (
    "You are answering a researcher's question about NASA scientific literature. "
    "You have access to MCP tools for retrieving papers, sections, claims, and "
    "graph context from a corpus of ~32M papers across astrophysics, planetary, "
    "heliophysics, biological/physical, and earth science. "
    "Use the tools to answer the question. Pick the tool whose schema and "
    "description best fits the user's intent. After your tool calls, give a "
    "concise final answer (1-3 sentences). Do not call more than 3 tools."
)


@dataclass
class RunResult:
    session_id: str
    variant: str
    query_id: str
    query: str
    run_idx: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    started_at: str = ""
    finished_at: str = ""
    exit_code: int = 0
    stderr_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "variant": self.variant,
            "query_id": self.query_id,
            "query": self.query,
            "run_idx": self.run_idx,
            "tool_calls": self.tool_calls,
            "final_text": self.final_text,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "stderr_tail": self.stderr_tail,
        }


def _build_mcp_config(variant: str, log_path: Path) -> dict[str, Any]:
    """One MCP server entry per variant. The server name becomes the prefix
    Claude uses in tool names (mcp__scixstub_v0__search etc), so it must be
    unique enough to not collide with other configured servers."""
    server_name = f"scixstub_{variant}"
    return {
        "mcpServers": {
            server_name: {
                "command": str(PYTHON_BIN),
                "args": [
                    "-m",
                    "scix.eval.tool_surface.stubs",
                    "--variant",
                    variant,
                    "--log-file",
                    str(log_path),
                ],
                "env": {"PYTHONPATH": str(REPO_ROOT / "src")},
            }
        }
    }


def _parse_stream_json(stdout: str) -> tuple[list[dict[str, Any]], str]:
    """Parse claude -p stream-json output.

    Returns (tool_calls, final_assistant_text). Each event is one JSON object
    per line; we look for assistant messages with tool_use content blocks
    and the terminal text content block."""
    tool_calls: list[dict[str, Any]] = []
    final_text_parts: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") != "assistant":
            continue
        msg = evt.get("message") or {}
        for block in msg.get("content", []) or []:
            btype = block.get("type")
            if btype == "tool_use":
                tool_calls.append(
                    {
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    }
                )
            elif btype == "text":
                txt = block.get("text", "").strip()
                if txt:
                    final_text_parts.append(txt)
    return tool_calls, "\n".join(final_text_parts)


async def run_one(
    variant: str,
    query_id: str,
    query: str,
    run_idx: int,
    out_dir: Path,
    timeout_s: int = 120,
) -> RunResult:
    session_id = uuid.uuid4().hex[:12]
    log_dir = out_dir / "stub_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{session_id}.jsonl"

    cfg = _build_mcp_config(variant, log_path)
    cfg_dir = Path(tempfile.mkdtemp(prefix=f"tseval-{variant}-"))
    cfg_path = cfg_dir / "mcp.json"
    cfg_path.write_text(json.dumps(cfg))

    server_name = f"scixstub_{variant}"
    allowed_glob = f"mcp__{server_name}__*"

    cmd = [
        "claude",
        "-p",
        "--strict-mcp-config",
        "--mcp-config",
        str(cfg_path),
        "--output-format",
        "stream-json",
        "--verbose",
        "--append-system-prompt",
        SYSTEM_PROMPT,
        "--allowedTools",
        allowed_glob,
    ]

    # cwd = a fresh tmp dir so claude -p doesn't auto-discover the scix
    # repo's CLAUDE.md and pollute the agent's context with project-specific
    # instructions. We can't use --bare because it strips OAuth auth.
    started = datetime.now(timezone.utc).isoformat()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cfg_dir),
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(input=query.encode("utf-8")), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        finished = datetime.now(timezone.utc).isoformat()
        return RunResult(
            session_id=session_id,
            variant=variant,
            query_id=query_id,
            query=query,
            run_idx=run_idx,
            started_at=started,
            finished_at=finished,
            exit_code=-1,
            stderr_tail="timeout",
        )
    finally:
        shutil.rmtree(cfg_dir, ignore_errors=True)
    finished = datetime.now(timezone.utc).isoformat()

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    tool_calls, final_text = _parse_stream_json(stdout)

    return RunResult(
        session_id=session_id,
        variant=variant,
        query_id=query_id,
        query=query,
        run_idx=run_idx,
        tool_calls=tool_calls,
        final_text=final_text,
        started_at=started,
        finished_at=finished,
        exit_code=proc.returncode or 0,
        stderr_tail="\n".join(stderr.splitlines()[-10:]),
    )


async def run_matrix(
    queries: list[dict[str, Any]],
    variants: list[str],
    runs: int,
    out_dir: Path,
    concurrency: int,
) -> Path:
    """Schedule (variant, query, run) jobs with bounded concurrency. Append
    each result to runs.jsonl as it completes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    sem = asyncio.Semaphore(concurrency)
    results_fh = runs_path.open("a", buffering=1)
    completed = 0
    total = len(queries) * len(variants) * runs

    async def _worker(variant: str, q: dict[str, Any], run_idx: int) -> None:
        nonlocal completed
        async with sem:
            res = await run_one(variant, q["id"], q["query"], run_idx, out_dir)
        results_fh.write(json.dumps(res.to_dict()) + "\n")
        completed += 1
        marker = "OK" if res.exit_code == 0 and res.tool_calls else "WARN"
        print(
            f"[{completed}/{total}] {marker} {variant} {q['id']} run={run_idx} "
            f"calls={len(res.tool_calls)} exit={res.exit_code}",
            file=sys.stderr,
            flush=True,
        )

    tasks = [
        _worker(v, q, r)
        for v in variants
        for q in queries
        for r in range(runs)
    ]
    await asyncio.gather(*tasks)
    results_fh.close()
    return runs_path


def load_queries(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tool-surface eval matrix")
    parser.add_argument("--queries", type=Path, default=REPO_ROOT / "eval/tool_surface/queries.jsonl")
    parser.add_argument("--variants", nargs="+", default=["v0"], choices=["v0", "v1", "v2"])
    parser.add_argument("--runs", type=int, default=1, help="Repeats per (variant, query)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/tool_surface_eval",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Limit query count for smoke testing")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]

    runs_path = asyncio.run(
        run_matrix(queries, args.variants, args.runs, args.out_dir, args.concurrency)
    )
    print(f"\nresults written to {runs_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
