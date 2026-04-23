"""Tests for :mod:`scix.viz.trace_stream`.

Covers:
    * publish-without-subscriber is a no-op
    * publish latency headroom (p99 < 5 ms with 1000 calls, no subscribers)
    * single subscriber receives a published event
    * multiple subscribers each receive the event
    * SSE endpoint emits ``data:`` lines over the wire

No ``pytest-asyncio`` dependency — each async scenario runs inside a
fresh event loop via :func:`asyncio.run`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI

from scix.viz import trace_stream
from scix.viz.trace_stream import TraceEvent, publish, router, subscribe


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _UvicornServer:
    """Run a FastAPI app on a real uvicorn server in a background thread.

    Starlette's TestClient and httpx.ASGITransport both buffer the full
    response body before returning, so neither can exercise a true
    streaming endpoint driven by an infinite async generator. A real
    uvicorn server on a loopback socket avoids that trap.
    """

    def __init__(self, app: FastAPI) -> None:
        self.port = _free_port()
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "_UvicornServer":
        self._thread.start()
        # Wait for the server to start accepting connections.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("uvicorn server did not start in time")

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _make_event(tool_name: str = "search") -> TraceEvent:
    return TraceEvent(
        tool_name=tool_name,
        latency_ms=12.34,
        params={"q": "black hole"},
        result_summary="5 hits",
        bibcodes=("2024ApJ...1..1A",),
    )


def test_publish_no_subscriber_is_noop() -> None:
    """With zero subscribers, publish must be a silent no-op."""

    # Sanity: no lingering subscribers from a previous test.
    assert trace_stream._subscribers == []

    # Should not raise.
    publish(_make_event())

    # And should not leak any subscriber state.
    assert trace_stream._subscribers == []


def test_publish_latency_headroom() -> None:
    """Micro-benchmark: 1000 publishes with no subscribers, p99 < 5 ms."""

    event = _make_event()
    samples: list[float] = []
    for _ in range(1000):
        t0 = time.perf_counter()
        publish(event)
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    p99 = samples[989]  # index 989 of 1000 sorted samples ≈ 99th percentile
    assert p99 < 5.0, f"publish p99 latency too high: {p99:.3f} ms"


def test_single_subscriber_receives_event() -> None:
    """One subscriber sees the event published after it registers."""

    async def scenario() -> TraceEvent:
        agen = subscribe()

        # Wait until the subscriber's queue is registered so the publish
        # below has somewhere to land. ``subscribe`` registers synchronously
        # before its first ``await``, so the first scheduled step is
        # enough.
        task = asyncio.create_task(agen.__anext__())
        # Give the subscribe generator a chance to register its queue.
        for _ in range(10):
            if trace_stream._subscribers:
                break
            await asyncio.sleep(0)
        assert trace_stream._subscribers, "subscriber did not register"

        publish(_make_event("single"))

        try:
            received = await asyncio.wait_for(task, timeout=1.0)
        finally:
            await agen.aclose()
        return received

    received = asyncio.run(scenario())
    assert received.tool_name == "single"
    # And subscriber list should be empty again after aclose().
    assert trace_stream._subscribers == []


def test_multiple_subscribers_each_get_event() -> None:
    """Two subscribers both see the same published event."""

    async def scenario() -> tuple[TraceEvent, TraceEvent]:
        agen_a = subscribe()
        agen_b = subscribe()
        task_a = asyncio.create_task(agen_a.__anext__())
        task_b = asyncio.create_task(agen_b.__anext__())

        for _ in range(20):
            if len(trace_stream._subscribers) >= 2:
                break
            await asyncio.sleep(0)
        assert len(trace_stream._subscribers) == 2

        publish(_make_event("fanout"))

        try:
            a = await asyncio.wait_for(task_a, timeout=1.0)
            b = await asyncio.wait_for(task_b, timeout=1.0)
        finally:
            await agen_a.aclose()
            await agen_b.aclose()
        return a, b

    a, b = asyncio.run(scenario())
    assert a.tool_name == "fanout"
    assert b.tool_name == "fanout"
    assert a.event_id == b.event_id  # same event, broadcast to both
    assert trace_stream._subscribers == []


def test_sse_endpoint_serves_event_stream() -> None:
    """GET /viz/api/trace/stream yields a ``data:`` line for a published event.

    Uses a real uvicorn server on a loopback socket so the response body
    is actually streamed (Starlette's TestClient buffers the full body
    before returning, which would hang on an infinite generator).
    """

    app = FastAPI()
    app.include_router(router)

    # Ensure a clean registry before and after this test.
    assert trace_stream._subscribers == []

    with _UvicornServer(app) as server:
        with httpx.Client(timeout=5.0) as client:
            with client.stream(
                "GET",
                f"{server.base_url}/viz/api/trace/stream",
                headers={"Accept": "text/event-stream"},
            ) as response:
                assert response.status_code == 200
                ctype = response.headers["content-type"]
                assert ctype.startswith("text/event-stream"), ctype

                # Publisher thread waits for the subscriber queue to show
                # up, then fires one event. ``publish`` is cross-thread
                # here — exercises the ``call_soon_threadsafe`` dispatch.
                publish_done = threading.Event()

                def _publish_when_ready() -> None:
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        if trace_stream._subscribers:
                            time.sleep(0.02)
                            publish(_make_event("sse"))
                            publish_done.set()
                            return
                        time.sleep(0.01)

                threading.Thread(target=_publish_when_ready, daemon=True).start()

                buf = b""
                frame: str | None = None
                for chunk in response.iter_bytes():
                    buf += chunk
                    if b"\n\n" in buf:
                        head, _, _rest = buf.partition(b"\n\n")
                        frame = head.decode("utf-8")
                        break

                assert publish_done.wait(timeout=3.0), "publisher never fired"
                assert frame is not None, "no SSE frame received"
                assert frame.startswith("data: "), f"unexpected frame: {frame!r}"

                payload = json.loads(frame[len("data: ") :])
                assert payload["tool_name"] == "sse"
                assert payload["latency_ms"] == 12.34
                assert "event_id" in payload

    # Subscriber deregistered when the stream closed. Give the server
    # loop a moment to run the generator's finally block after shutdown.
    for _ in range(100):
        if not trace_stream._subscribers:
            break
        time.sleep(0.02)
    assert trace_stream._subscribers == []
