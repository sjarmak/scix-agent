"""JSONL trace logger for experimental MCP tool calls.

The trace log IS the experiment data — every tool invocation, its arguments,
result size, and latency end up here. Day 4 analysis reads these JSONL files
to compute query-depth distribution and tool-call patterns.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    event_id: str
    session_id: str
    tool_name: str
    args: dict[str, Any]
    started_at: float
    duration_ms: float
    ok: bool
    result_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class TraceLogger:
    """Append-only JSONL writer keyed by session id.

    Thread-safe via a single lock around append; the experimental server is
    single-process and low-throughput so this is sufficient.
    """

    def __init__(self, log_dir: Path, session_id: str | None = None) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id or os.environ.get(
            "SCIX_GRAPH_EXP_SESSION", uuid.uuid4().hex[:12]
        )
        self._path = self._log_dir / f"trace_{self._session_id}.jsonl"
        self._lock = threading.Lock()
        logger.info("TraceLogger writing to %s", self._path)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def path(self) -> Path:
        return self._path

    def _append(self, event: TraceEvent) -> None:
        line = json.dumps(asdict(event), default=str)
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    @contextmanager
    def record(
        self, tool_name: str, args: dict[str, Any]
    ) -> Generator["TraceEventBuilder", None, None]:
        builder = TraceEventBuilder(tool_name=tool_name, args=args)
        try:
            yield builder
        except Exception as exc:
            builder.fail(repr(exc))
            self._append(builder.finalize(self._session_id))
            raise
        else:
            self._append(builder.finalize(self._session_id))


@dataclass
class TraceEventBuilder:
    tool_name: str
    args: dict[str, Any]
    _started_at: float = field(default_factory=time.time)
    _result_summary: dict[str, Any] = field(default_factory=dict)
    _ok: bool = True
    _error: str | None = None

    def summarize(self, **kwargs: Any) -> None:
        self._result_summary.update(kwargs)

    def fail(self, error: str) -> None:
        self._ok = False
        self._error = error

    def finalize(self, session_id: str) -> TraceEvent:
        return TraceEvent(
            event_id=uuid.uuid4().hex[:16],
            session_id=session_id,
            tool_name=self.tool_name,
            args=self.args,
            started_at=self._started_at,
            duration_ms=(time.time() - self._started_at) * 1000.0,
            ok=self._ok,
            result_summary=self._result_summary,
            error=self._error,
        )
