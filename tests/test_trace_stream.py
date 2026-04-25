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


# ---------------------------------------------------------------------------
# Persistent ring-buffer tests (init_history / _persist / /history route).
#
# Each test installs a fresh history file in a tmp dir via ``init_history``,
# then manually clears the module-level deque in a ``finally`` block so
# leftover state doesn't leak into unrelated tests.
# ---------------------------------------------------------------------------


import pytest  # noqa: E402  — kept with ring-buffer tests
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def history_path(tmp_path):
    """Install an isolated trace history file; reset state after each test."""
    path = tmp_path / "trace_history.jsonl"
    trace_stream.init_history(path)
    try:
        yield path
    finally:
        # Leave subsequent tests with no persistence (module-level state).
        with trace_stream._history_lock:
            trace_stream._history.clear()
            trace_stream._history_file = None
            trace_stream._history_lines_on_disk = 0


def test_init_history_loads_existing_file(tmp_path) -> None:
    """A pre-existing jsonl is replayed into the in-memory deque on init."""
    path = tmp_path / "trace_history.jsonl"
    seed_events = [
        {
            "tool_name": f"tool_{i}",
            "latency_ms": float(i),
            "ts": 1700000000.0 + i,
            "event_id": f"ev{i:04d}",
            "params": {"q": "x"},
            "result_summary": None,
            "bibcodes": ["2024A", "2024B"],
        }
        for i in range(3)
    ]
    with path.open("w") as f:
        for ev in seed_events:
            f.write(json.dumps(ev) + "\n")

    trace_stream.init_history(path)
    try:
        loaded = trace_stream.snapshot_history()
        assert [e.tool_name for e in loaded] == ["tool_0", "tool_1", "tool_2"]
        assert loaded[0].bibcodes == ("2024A", "2024B")
    finally:
        with trace_stream._history_lock:
            trace_stream._history.clear()
            trace_stream._history_file = None
            trace_stream._history_lines_on_disk = 0


def test_init_history_skips_corrupt_lines(tmp_path) -> None:
    """Malformed lines are logged and skipped, not raised."""
    path = tmp_path / "trace_history.jsonl"
    with path.open("w") as f:
        f.write('{"tool_name": "ok", "latency_ms": 1.0}\n')
        f.write("not json at all\n")
        f.write("\n")  # blank line also tolerated
        f.write('{"tool_name": "second", "latency_ms": 2.0}\n')

    trace_stream.init_history(path)
    try:
        loaded = trace_stream.snapshot_history()
        assert [e.tool_name for e in loaded] == ["ok", "second"]
    finally:
        with trace_stream._history_lock:
            trace_stream._history.clear()
            trace_stream._history_file = None
            trace_stream._history_lines_on_disk = 0


def test_publish_persists_to_file(history_path) -> None:
    """publish() writes one jsonl line per event and populates the deque."""
    for i in range(5):
        publish(_make_event(f"persist_{i}"))

    assert history_path.exists()
    with history_path.open() as f:
        lines = [ln for ln in f.read().splitlines() if ln]
    assert len(lines) == 5
    parsed = [json.loads(ln) for ln in lines]
    assert [p["tool_name"] for p in parsed] == [f"persist_{i}" for i in range(5)]

    snapshot = trace_stream.snapshot_history()
    assert [e.tool_name for e in snapshot] == [f"persist_{i}" for i in range(5)]


def test_rotation_trims_file_to_cap(history_path) -> None:
    """Once the file exceeds _ROTATE_AT lines, rotation keeps only the deque."""
    total = trace_stream._ROTATE_AT + 50  # 1050 publishes → at least one rotation
    for i in range(total):
        publish(_make_event(f"r{i}"))

    with history_path.open() as f:
        lines = [ln for ln in f.read().splitlines() if ln]

    # After rotation, file is rewritten from the in-memory deque (≤ cap).
    # Post-rotation writes may add up to _ROTATE_AT more lines, but never
    # more than the final counter allows.
    assert len(lines) <= trace_stream._ROTATE_AT
    assert len(lines) >= trace_stream._HISTORY_CAP

    # The deque is the authoritative recent window — it tracks the last
    # _HISTORY_CAP events published, regardless of rotation.
    snapshot = trace_stream.snapshot_history()
    expected_tail = [f"r{i}" for i in range(total - trace_stream._HISTORY_CAP, total)]
    assert [e.tool_name for e in snapshot] == expected_tail


def test_history_endpoint_returns_events(history_path) -> None:
    """GET /viz/api/trace/history serves the in-memory deque."""
    for i in range(7):
        publish(_make_event(f"hist_{i}"))

    app = FastAPI()
    app.include_router(trace_stream.router)
    client = TestClient(app)

    resp = client.get("/viz/api/trace/history")
    assert resp.status_code == 200
    body = resp.json()
    tools = [ev["tool_name"] for ev in body["events"]]
    assert tools == [f"hist_{i}" for i in range(7)]


def test_history_endpoint_respects_limit(history_path) -> None:
    """?limit=N returns at most N most recent events."""
    for i in range(20):
        publish(_make_event(f"lim_{i}"))

    app = FastAPI()
    app.include_router(trace_stream.router)
    client = TestClient(app)

    resp = client.get("/viz/api/trace/history?limit=5")
    assert resp.status_code == 200
    tools = [ev["tool_name"] for ev in resp.json()["events"]]
    assert tools == [f"lim_{i}" for i in range(15, 20)]
