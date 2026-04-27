"""Analysis of trace-logger JSONL output.

Reads ``trace_<session>.jsonl`` files (written by ``TraceLogger``) and
produces aggregates for the day-4 report:

  * tool-call distribution (which tools the agent picked, how often)
  * query-depth distribution (max ``hop`` / ``hops`` / len(``edge_sequence``)
    seen in args across a session)
  * graph_query_log payloads (the freeform queries agents wanted to issue)
  * latency and error stats per tool

The central experiment metric is whether the depth distribution shifts
upward when the experimental tools are present — that's the signal that
agents reach for multi-hop patterns when offered the surface for them.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

logger = logging.getLogger(__name__)


_DEPTH_ARG_KEYS: tuple[str, ...] = ("depth", "hops")
_DEPTH_ARG_LIST_KEYS: tuple[str, ...] = ("edge_sequence",)


@dataclass
class TraceSummary:
    """Aggregate stats for a single trace session (or merged set)."""

    session_ids: list[str]
    event_count: int
    tool_call_counts: dict[str, int]
    error_counts: dict[str, int]
    depth_per_event: list[int]
    depth_distribution: dict[int, int]
    latency_ms_per_tool: dict[str, list[float]]
    freeform_queries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def max_depth(self) -> int:
        return max(self.depth_per_event, default=0)

    @property
    def median_depth(self) -> float:
        return median(self.depth_per_event) if self.depth_per_event else 0.0

    @property
    def latency_summary_ms(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for tool, samples in self.latency_ms_per_tool.items():
            if not samples:
                continue
            out[tool] = {
                "n": float(len(samples)),
                "mean": mean(samples),
                "median": median(samples),
                "max": max(samples),
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_ids": self.session_ids,
            "event_count": self.event_count,
            "tool_call_counts": self.tool_call_counts,
            "error_counts": self.error_counts,
            "depth_distribution": self.depth_distribution,
            "max_depth": self.max_depth,
            "median_depth": self.median_depth,
            "latency_summary_ms": self.latency_summary_ms,
            "freeform_query_count": len(self.freeform_queries),
        }


def _depth_for_event(event: dict[str, Any]) -> int:
    """Best-effort hop depth for a single trace event.

    For multi_hop_neighbors / subgraph_around the request itself carries the
    intended depth. For pattern_match it's len(edge_sequence). For
    shortest_path / personalized_pagerank the request doesn't carry depth
    so we treat it as 1 (semantic 1-hop primitive). Other tools score 0.
    """
    args = event.get("args", {}) or {}
    tool = event.get("tool_name", "")

    for k in _DEPTH_ARG_KEYS:
        v = args.get(k)
        if isinstance(v, int) and v > 0:
            return v

    for k in _DEPTH_ARG_LIST_KEYS:
        v = args.get(k)
        if isinstance(v, list) and v:
            return len(v)

    if tool in {"shortest_path", "personalized_pagerank", "multi_hop_neighbors", "subgraph_around"}:
        return 1
    return 0


def summarize_traces(paths: Iterable[Path]) -> TraceSummary:
    """Read JSONL trace files and produce a ``TraceSummary``.

    Multiple files are merged into a single summary. Events with malformed
    JSON are skipped with a warning rather than aborting the analysis.
    """
    session_ids: list[str] = []
    event_count = 0
    tool_calls: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    depths: list[int] = []
    depth_dist: Counter[int] = Counter()
    latencies: dict[str, list[float]] = defaultdict(list)
    freeform: list[dict[str, Any]] = []

    seen_sessions: set[str] = set()

    for path in paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed line in %s", path)
                    continue
                event_count += 1
                sid = event.get("session_id") or ""
                if sid and sid not in seen_sessions:
                    seen_sessions.add(sid)
                    session_ids.append(sid)
                tool = event.get("tool_name", "<unknown>")
                tool_calls[tool] += 1
                if not event.get("ok", True):
                    errors[tool] += 1
                depth = _depth_for_event(event)
                depths.append(depth)
                depth_dist[depth] += 1
                duration = event.get("duration_ms")
                if isinstance(duration, (int, float)):
                    latencies[tool].append(float(duration))
                if tool == "graph_query_log":
                    freeform.append(event.get("args", {}) or {})

    return TraceSummary(
        session_ids=session_ids,
        event_count=event_count,
        tool_call_counts=dict(tool_calls),
        error_counts=dict(errors),
        depth_per_event=depths,
        depth_distribution=dict(depth_dist),
        latency_ms_per_tool=dict(latencies),
        freeform_queries=freeform,
    )


def compare_summaries(
    control: TraceSummary, treatment: TraceSummary
) -> dict[str, Any]:
    """Diff two summaries — typically (no-experimental-tools, with-tools).

    The central metrics for the AGE go/no-go decision:

      * ``depth_shift``: median treatment depth − median control depth.
        Positive = agents reach deeper when given the tools.
      * ``new_tool_usage``: experimental tools that appeared in treatment
        but never in control.
      * ``freeform_query_emergence``: number of graph_query_log invocations
        in treatment (control has zero by construction).
    """
    new_tools = sorted(set(treatment.tool_call_counts) - set(control.tool_call_counts))
    return {
        "depth_shift_median": treatment.median_depth - control.median_depth,
        "depth_shift_max": treatment.max_depth - control.max_depth,
        "new_tool_usage": {t: treatment.tool_call_counts[t] for t in new_tools},
        "freeform_query_emergence": len(treatment.freeform_queries),
        "control_summary": control.to_dict(),
        "treatment_summary": treatment.to_dict(),
    }
