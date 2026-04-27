"""Agent harness: runs ``claude -p`` against a question with a chosen MCP set.

Per ``feedback_claude_judge_via_oauth.md`` we use Claude Code subagents via
``claude -p`` (OAuth) rather than importing the ``anthropic`` SDK.

Two MCP variants:
  * ``control`` — production scix MCP server only (the 15 existing tools).
  * ``treatment`` — production scix + the experimental graph server with
    the day-2 primitives + graph_query_log escape hatch.

The harness captures:
  * the agent's final answer (from --output-format=stream-json)
  * the trace JSONL written by the experimental server (treatment only)
  * total tokens / cost / wall-clock time

Run-time isolation: each variant gets its own ``SCIX_GRAPH_EXP_SESSION``
so traces don't collide across runs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


Variant = Literal["control", "treatment"]


@dataclass
class HarnessConfig:
    """Static configuration for a benchmark run.

    ``snapshot_path`` and ``trace_dir`` are passed to the experimental MCP
    server via env vars at spawn time. ``budget_usd`` caps each agent run.
    """

    snapshot_path: Path
    trace_dir: Path
    production_mcp_url: str | None = None
    production_mcp_token: str | None = None
    budget_usd: float = 0.50
    model: str = "claude-sonnet-4-6"
    max_turns: int = 12

    def mcp_config_for(self, variant: Variant, session_id: str) -> dict[str, Any]:
        """Build an MCP config dict suitable for ``--mcp-config``."""
        servers: dict[str, Any] = {}
        if self.production_mcp_url:
            entry: dict[str, Any] = {
                "type": "http",
                "url": self.production_mcp_url,
            }
            if self.production_mcp_token:
                entry["headers"] = {
                    "Authorization": f"Bearer {self.production_mcp_token}"
                }
            servers["scix"] = entry
        if variant == "treatment":
            servers["scix-graph-experiment"] = {
                "type": "stdio",
                "command": ".venv/bin/python",
                "args": ["-m", "scix.mcp_graph_experiment"],
                "env": {
                    "SCIX_GRAPH_EXP_SNAPSHOT": str(self.snapshot_path),
                    "SCIX_GRAPH_EXP_TRACE_DIR": str(self.trace_dir),
                    "SCIX_GRAPH_EXP_SESSION": session_id,
                },
            }
        return {"mcpServers": servers}


@dataclass
class RunResult:
    """Outcome of one ``(question, variant)`` run."""

    question_id: str
    variant: Variant
    session_id: str
    final_answer: str
    duration_seconds: float
    cost_usd: float | None
    total_tokens: int | None
    trace_path: Path | None
    raw_event_count: int
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["trace_path"] = str(self.trace_path) if self.trace_path else None
        return out


_SYSTEM_PROMPT = (
    "You are a research assistant for the SciX scientific corpus. "
    "Use the tools available to answer the question concisely and "
    "accurately. Prefer the most direct tool. When the experimental "
    "graph tools are available you may use them, but choose the right "
    "tool for the question — do not over-use multi-hop tools when a "
    "single search would suffice. Cite bibcodes when stating facts. "
    "When you have a final answer, output it as a single concluding "
    "paragraph."
)


def run_question(
    question_prompt: str,
    question_id: str,
    variant: Variant,
    config: HarnessConfig,
) -> RunResult:
    """Spawn ``claude -p`` for one (question, variant) pair and capture outputs."""
    session_id = f"{question_id}-{variant}-{uuid.uuid4().hex[:6]}"
    config.trace_dir.mkdir(parents=True, exist_ok=True)
    mcp_config = config.mcp_config_for(variant, session_id)
    mcp_config_path = config.trace_dir / f"mcp_{session_id}.json"
    mcp_config_path.write_text(json.dumps(mcp_config))

    cmd = [
        "claude",
        "-p",
        "--mcp-config",
        str(mcp_config_path),
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--max-budget-usd",
        f"{config.budget_usd:.2f}",
        "--model",
        config.model,
        "--append-system-prompt",
        _SYSTEM_PROMPT,
        question_prompt,
    ]

    started = time.time()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            question_id=question_id,
            variant=variant,
            session_id=session_id,
            final_answer="",
            duration_seconds=time.time() - started,
            cost_usd=None,
            total_tokens=None,
            trace_path=None,
            raw_event_count=0,
            error=f"timeout: {exc}",
        )

    duration = time.time() - started
    parsed = _parse_stream_json(completed.stdout)
    trace_path = config.trace_dir / f"trace_{session_id}.jsonl"
    return RunResult(
        question_id=question_id,
        variant=variant,
        session_id=session_id,
        final_answer=parsed.get("final_answer", ""),
        duration_seconds=duration,
        cost_usd=parsed.get("cost_usd"),
        total_tokens=parsed.get("total_tokens"),
        trace_path=trace_path if trace_path.exists() else None,
        raw_event_count=parsed.get("event_count", 0),
        error=parsed.get("error") if completed.returncode != 0 else None,
        extra={
            "returncode": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-500:],
        },
    )


def _parse_stream_json(stdout: str) -> dict[str, Any]:
    """Extract final answer + cost/tokens from claude -p stream-json output.

    Stream format: one JSON object per line. Final assistant message
    appears as ``{"type": "assistant", ...}``; the closing ``"type":
    "result"`` event carries cost/usage.
    """
    final_answer_parts: list[str] = []
    cost_usd: float | None = None
    total_tokens: int | None = None
    event_count = 0
    error: str | None = None

    for raw in stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event_count += 1
        etype = event.get("type")
        if etype == "assistant":
            msg = event.get("message", {})
            for block in msg.get("content", []):
                if block.get("type") == "text":
                    final_answer_parts.append(block.get("text", ""))
        elif etype == "result":
            cost_usd = event.get("total_cost_usd") or event.get("cost_usd")
            usage = event.get("usage", {}) or {}
            total_tokens = (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            ) or None
            if event.get("subtype") == "error":
                error = event.get("result") or "unknown_error"

    return {
        "final_answer": "\n".join(p for p in final_answer_parts if p).strip(),
        "cost_usd": cost_usd,
        "total_tokens": total_tokens,
        "event_count": event_count,
        "error": error,
    }
