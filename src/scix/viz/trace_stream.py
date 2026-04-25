"""In-process pub/sub event bus for agent-trace events.

Publishes :class:`TraceEvent` records to any number of async subscribers.
Exposes an SSE endpoint at ``GET /viz/api/trace/stream`` so browser clients
can tail the event stream in real time.

Design notes
------------
* Publication is fire-and-forget. ``publish(event)`` is safe to call from
  sync hot paths (MCP tool wrappers, etc.) — it never awaits, never blocks,
  and never raises.
* Subscribers are backed by bounded :class:`asyncio.Queue` instances
  (``maxsize=256``). When a subscriber is slow the publisher drops the
  event rather than stalling; a DEBUG log records the drop.
* Each subscriber records the event loop it was created on, and
  ``publish()`` dispatches ``put_nowait`` via
  :meth:`asyncio.loop.call_soon_threadsafe`. That way a publisher running
  on a different thread (sync MCP handler, test helper, etc.) does not
  corrupt the queue's internal state. If the publisher is already running
  inside the owning loop we call ``put_nowait`` directly.
* The module-level subscriber list is guarded by a :class:`threading.Lock`.
* The router is exported but NOT registered on the viz app here; a later
  unit wires it via ``app.include_router(trace_stream.router)``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Deque

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# Bounded per-subscriber queue size. 256 events is enough to absorb bursts
# without growing memory unbounded when a subscriber is slow.
_QUEUE_MAXSIZE: int = 256

# Disk-backed ring buffer: keep at most ``_HISTORY_CAP`` events live in
# memory; rewrite the on-disk jsonl whenever it exceeds ``_ROTATE_AT`` lines
# so the file never grows unboundedly between restarts.
_HISTORY_CAP: int = 500
_ROTATE_AT: int = 1000

_history: Deque["TraceEvent"] = deque(maxlen=_HISTORY_CAP)
_history_lock: threading.Lock = threading.Lock()
_history_file: Path | None = None
_history_lines_on_disk: int = 0


@dataclass(frozen=True)
class TraceEvent:
    """A single agent-trace event.

    Field order puts the required positional arguments (``tool_name``,
    ``latency_ms``) first so callers can construct events with just those
    two; everything else defaults.
    """

    tool_name: str
    latency_ms: float
    ts: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    params: dict[str, Any] = field(default_factory=dict)
    result_summary: str | None = None
    bibcodes: tuple[str, ...] = ()


# A subscriber is the queue it owns plus the event loop the queue lives on.
# Cross-thread ``publish`` must dispatch the ``put_nowait`` via
# ``loop.call_soon_threadsafe`` — plain ``put_nowait`` from another thread is
# not safe against an asyncio.Queue.
@dataclass
class _Subscriber:
    queue: asyncio.Queue[TraceEvent]
    loop: asyncio.AbstractEventLoop


_subscribers: list[_Subscriber] = []
_subscribers_lock: threading.Lock = threading.Lock()


def _register(sub: _Subscriber) -> None:
    with _subscribers_lock:
        _subscribers.append(sub)


def _unregister(sub: _Subscriber) -> None:
    with _subscribers_lock:
        with suppress(ValueError):
            _subscribers.remove(sub)


def _deliver(queue: asyncio.Queue[TraceEvent], event: TraceEvent) -> None:
    """Inner delivery — must run on the queue's owning loop."""
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        logger.debug(
            "trace_stream subscriber queue full; dropping event %s",
            event.event_id,
        )


def _default_history_path() -> Path:
    env = os.environ.get("SCIX_VIZ_TRACE_HISTORY")
    if env:
        return Path(env)
    # __file__ -> <repo>/src/scix/viz/trace_stream.py
    #   parents[0]=viz, [1]=scix, [2]=src, [3]=repo_root
    return Path(__file__).resolve().parents[3] / "data" / "viz" / "trace_history.jsonl"


def _event_from_dict(d: dict[str, Any]) -> TraceEvent:
    bibcodes = d.get("bibcodes") or []
    if not isinstance(bibcodes, (list, tuple)):
        bibcodes = []
    params = d.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    summary = d.get("result_summary")
    return TraceEvent(
        tool_name=str(d.get("tool_name", "unknown")),
        latency_ms=float(d.get("latency_ms", 0.0)),
        ts=float(d.get("ts", time.time())),
        event_id=str(d.get("event_id") or uuid.uuid4().hex),
        params=params,
        result_summary=summary if isinstance(summary, str) else None,
        bibcodes=tuple(str(b) for b in bibcodes),
    )


def init_history(path: Path | None = None) -> None:
    """Install the on-disk ring buffer file and pre-load the in-memory deque.

    Idempotent: call again with a different ``path`` to re-point the buffer
    (used by tests). Reads up to the last ``_HISTORY_CAP`` events from the
    file on disk so a uvicorn restart leaves the panel pre-populated.
    """
    global _history_file, _history_lines_on_disk
    target = path or _default_history_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("trace history: cannot create directory %s", target.parent)
        return
    with _history_lock:
        _history.clear()
        line_count = 0
        if target.exists():
            try:
                with target.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                logger.exception("trace history: failed to read %s", target)
                lines = []
            line_count = len(lines)
            for line in lines[-_HISTORY_CAP:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    _history.append(_event_from_dict(json.loads(line)))
                except Exception:
                    logger.debug("trace history: skipping corrupt line", exc_info=True)
        _history_file = target
        _history_lines_on_disk = line_count


def _rotate_history_file_locked() -> None:
    """Rewrite the on-disk file from the in-memory deque. Caller holds lock."""
    if _history_file is None:
        return
    try:
        tmp = _history_file.with_suffix(_history_file.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for event in _history:
                f.write(json.dumps(asdict(event), default=str) + "\n")
        tmp.replace(_history_file)
    except Exception:
        logger.exception("trace history: rotation failed")


def _persist(event: TraceEvent) -> None:
    """Append ``event`` to the in-memory deque and the on-disk jsonl ring."""
    global _history_lines_on_disk
    with _history_lock:
        _history.append(event)
        if _history_file is None:
            return
        try:
            with _history_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")
            _history_lines_on_disk += 1
            if _history_lines_on_disk > _ROTATE_AT:
                _rotate_history_file_locked()
                _history_lines_on_disk = len(_history)
        except Exception:
            logger.exception("trace history: persist failed")


def snapshot_history(limit: int = _HISTORY_CAP) -> list[TraceEvent]:
    """Return up to ``limit`` most recent events from the in-memory deque."""
    limit = max(0, min(int(limit), _HISTORY_CAP))
    with _history_lock:
        if limit == 0:
            return []
        return list(_history)[-limit:]


def publish(event: TraceEvent) -> None:
    """Fire-and-forget broadcast to all active subscribers.

    Safe to call from any thread and any context (sync or async). Never
    raises — failures are logged and swallowed so trace publication can
    never break the caller's hot path.
    """

    # Persist first so events survive across restarts even when no
    # subscriber is currently attached.
    try:
        _persist(event)
    except Exception:  # pragma: no cover — defensive, publish must not raise
        logger.exception("trace_stream persist raised unexpectedly")

    # Snapshot the list under the lock so a concurrent register / unregister
    # cannot be observed mid-mutation. The snapshot (a shallow copy) is
    # then iterated without holding the lock.
    with _subscribers_lock:
        snapshot = list(_subscribers)

    for sub in snapshot:
        try:
            # Determine whether we are already running on the owning loop.
            # If so, deliver inline; otherwise hop threads safely.
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None

            if running is sub.loop:
                _deliver(sub.queue, event)
            else:
                sub.loop.call_soon_threadsafe(_deliver, sub.queue, event)
        except RuntimeError:
            # Loop is closed (subscriber teardown racing publish). Skip.
            logger.debug("trace_stream subscriber loop closed; skipping")
        except Exception:  # pragma: no cover — defensive, publish must not raise
            logger.exception("trace_stream publish failed")


async def subscribe() -> AsyncIterator[TraceEvent]:
    """Async iterator of :class:`TraceEvent`s.

    Registers a fresh bounded queue, yields events as they arrive, and
    deregisters the queue on cleanup (generator close / cancellation).
    """

    queue: asyncio.Queue[TraceEvent] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    sub = _Subscriber(queue=queue, loop=asyncio.get_running_loop())
    _register(sub)
    try:
        while True:
            event = await queue.get()
            yield event
    finally:
        _unregister(sub)


router = APIRouter(prefix="/viz/api/trace", tags=["viz-trace"])


@router.get("/stream")
async def stream_events() -> StreamingResponse:
    """SSE endpoint: emit one ``data: {json}\\n\\n`` block per event."""

    async def gen() -> AsyncIterator[str]:
        async for event in subscribe():
            payload = json.dumps(asdict(event), default=str)
            yield f"data: {payload}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/history")
def history(limit: int = _HISTORY_CAP) -> dict[str, Any]:
    """Return the tail of the persistent trace ring buffer.

    Served straight from the in-memory deque — no disk read on the hot
    path. A freshly started uvicorn worker has the deque pre-populated by
    :func:`init_history` so the frontend can render prior events before
    subscribing to ``/stream``.
    """
    events = snapshot_history(limit)
    return {"events": [asdict(e) for e in events]}


@router.post("/publish")
async def publish_event(payload: dict) -> dict:
    """Demo-only HTTP hook so external processes can inject trace events
    into this uvicorn worker's subscriber list.

    Accepts a JSON body shaped like ``TraceEvent`` fields. Missing fields
    fall back to the dataclass defaults. Returns ``{"published": true,
    "event_id": ...}`` on success.

    Not authenticated — intended strictly for local demo driving over
    localhost. Do not expose this endpoint on a public network.
    """
    tool_name = str(payload.get("tool_name") or "unknown")
    latency_ms = float(payload.get("latency_ms") or 0.0)
    bibcodes = payload.get("bibcodes") or []
    if not isinstance(bibcodes, (list, tuple)):
        bibcodes = []
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    result_summary = payload.get("result_summary")
    event = TraceEvent(
        tool_name=tool_name,
        latency_ms=latency_ms,
        params=params,
        result_summary=result_summary if isinstance(result_summary, str) else None,
        bibcodes=tuple(str(b) for b in bibcodes),
    )
    publish(event)
    return {"published": True, "event_id": event.event_id}
