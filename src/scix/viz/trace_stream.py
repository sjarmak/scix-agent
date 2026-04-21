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
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# Bounded per-subscriber queue size. 256 events is enough to absorb bursts
# without growing memory unbounded when a subscriber is slow.
_QUEUE_MAXSIZE: int = 256


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


def publish(event: TraceEvent) -> None:
    """Fire-and-forget broadcast to all active subscribers.

    Safe to call from any thread and any context (sync or async). Never
    raises — failures are logged and swallowed so trace publication can
    never break the caller's hot path.
    """

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
