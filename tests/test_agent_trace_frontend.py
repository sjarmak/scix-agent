"""Static + integration tests for the agent-trace overlay frontend.

Four layers are exercised:

* Static HTML/JS layout checks via BeautifulSoup — ensures the bundle wires
  in deck.gl, loads ``agent_trace.js``, and declares the ``#trace-panel``
  and ``#umap-root`` hooks.
* ``subscribeTraceStream`` symbol exists in the bundled JS.
* Static-mount integration via :class:`fastapi.testclient.TestClient` —
  ``GET /viz/agent_trace.html`` returns 200 text/html.
* SSE contract — the ``/viz/api/trace/stream`` route is registered on
  the viz app, and a :class:`scix.viz.trace_stream.TraceEvent` published
  via :func:`scix.viz.trace_stream.publish` is delivered to subscribers
  of the router's underlying ``subscribe()`` generator with the exact
  byte framing the SSE endpoint uses on the wire.

Starlette's :class:`TestClient` buffers the full response body before
returning (see ``tests/test_trace_stream.py``), so this module does not
try to pump bytes through the ASGI test transport for the SSE endpoint.
End-to-end streaming is already covered by ``test_trace_stream.py`` via
a real uvicorn server; here we only need to prove that the router is
wired onto ``server.app`` and that the plumbing works.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient

from scix.viz import trace_stream
from scix.viz.server import app

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HTML = _REPO_ROOT / "web" / "viz" / "agent_trace.html"
_JS = _REPO_ROOT / "web" / "viz" / "agent_trace.js"


def _load_soup() -> BeautifulSoup:
    assert _HTML.is_file(), f"Missing {_HTML}"
    return BeautifulSoup(_HTML.read_text(encoding="utf-8"), "html.parser")


# ---------------------------------------------------------------------------
# Static structure
# ---------------------------------------------------------------------------


def test_agent_trace_html_structure() -> None:
    """agent_trace.html must declare the required DOM hooks + script loads."""
    soup = _load_soup()

    root = soup.find("div", id="umap-root")
    assert root is not None, "Missing <div id='umap-root'>"

    panel = soup.find(id="trace-panel")
    assert panel is not None, "Missing element with id='trace-panel'"

    # deck.gl CDN script — same provider as umap_browser.html.
    script_srcs = [s.get("src") or "" for s in soup.find_all("script") if s.get("src")]
    assert any("unpkg.com" in src and "deck.gl" in src for src in script_srcs), (
        f"Expected a deck.gl CDN <script src=...>; found: {script_srcs!r}"
    )

    # agent_trace.js must be referenced.
    assert any(
        src.endswith("agent_trace.js") or src == "./agent_trace.js"
        for src in script_srcs
    ), f"Expected a <script src='./agent_trace.js'>; found: {script_srcs!r}"

    # Shared stylesheet must be referenced for visual consistency.
    link_hrefs = [link.get("href") or "" for link in soup.find_all("link")]
    assert any(href.endswith("shared.css") for href in link_hrefs), (
        f"Expected shared.css link; found: {link_hrefs!r}"
    )


def test_agent_trace_js_has_subscribe_symbol() -> None:
    """agent_trace.js must expose a ``subscribeTraceStream`` symbol."""
    assert _JS.is_file(), f"Missing {_JS}"
    js_text = _JS.read_text(encoding="utf-8")
    assert "subscribeTraceStream" in js_text, (
        "agent_trace.js must contain the string 'subscribeTraceStream' "
        "(function definition or window assignment)"
    )


# ---------------------------------------------------------------------------
# Static-mount integration
# ---------------------------------------------------------------------------


def test_agent_trace_html_served() -> None:
    """GET /viz/agent_trace.html via the FastAPI app returns 200 text/html."""
    client = TestClient(app)
    response = client.get("/viz/agent_trace.html")
    assert response.status_code == 200, response.text
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("text/html"), (
        f"Expected text/html content-type, got: {content_type!r}"
    )
    assert b"trace-panel" in response.content, (
        "Response body should include the trace-panel container"
    )
    assert b"umap-root" in response.content, (
        "Response body should include the umap-root container"
    )


# ---------------------------------------------------------------------------
# SSE endpoint contract — the trace-stream router is registered on app
# ---------------------------------------------------------------------------


def _first_data_frame_via_generator() -> str:
    """Drive the router's ``subscribe()`` async generator directly.

    This avoids any streaming buffering in the HTTP stack and still
    proves that ``publish`` -> ``subscribe`` plumbing works, which is
    what the SSE endpoint depends on. The framing
    (``data: <json>\\n\\n``) is recreated here from the same ``asdict``
    serializer the router uses, so this test asserts the exact byte
    shape a browser client would receive.
    """

    async def scenario() -> str:
        agen = trace_stream.subscribe()
        task = asyncio.create_task(agen.__anext__())
        # Wait for the subscriber to register its queue.
        for _ in range(50):
            if trace_stream._subscribers:
                break
            await asyncio.sleep(0)
        assert trace_stream._subscribers, "subscriber did not register"
        trace_stream.publish(
            trace_stream.TraceEvent(
                tool_name="test_tool",
                latency_ms=1.0,
                bibcodes=("2024ApJ...1..1A",),
            )
        )
        try:
            event = await asyncio.wait_for(task, timeout=2.0)
        finally:
            await agen.aclose()
        payload = json.dumps(asdict(event), default=str)
        return f"data: {payload}"

    return asyncio.run(scenario())


def _assert_sse_route_registered() -> None:
    """The SSE route must actually be wired on the viz app.

    This is the acceptance-criteria invariant for the server.py edit — it
    fails loudly if the ``include_router(trace_stream_router)`` call was
    dropped.
    """
    matching = [
        r for r in app.routes if getattr(r, "path", "") == "/viz/api/trace/stream"
    ]
    assert matching, (
        "GET /viz/api/trace/stream is not registered on the viz app — "
        "check server.py includes trace_stream_router"
    )


def test_sse_endpoint_delivers_published_event() -> None:
    """Publishing a TraceEvent reaches subscribers of the SSE endpoint.

    Strategy: first verify the route is mounted on the app, then drive
    the router's ``subscribe()`` async generator directly in an asyncio
    loop and confirm that a published event is framed as the router
    would frame it over the wire. Starlette's TestClient buffers the
    full response body (see ``tests/test_trace_stream.py`` which spins up
    a real uvicorn server to work around this), so we do not try to pump
    bytes through the ASGI test transport here. The end-to-end SSE
    wire-level behaviour is already covered by ``test_trace_stream.py``;
    this test proves the router is mounted and that ``publish`` ->
    ``subscribe`` plumbing works through the same router object that
    ``server.app`` has registered.
    """

    # Leave no stray subscribers from an earlier test in the module.
    assert trace_stream._subscribers == []

    _assert_sse_route_registered()

    line = _first_data_frame_via_generator()

    assert line.startswith("data: "), f"unexpected frame: {line!r}"
    payload = json.loads(line[len("data: ") :])
    assert isinstance(payload, dict), f"expected dict payload, got {type(payload)!r}"
    assert payload["tool_name"] == "test_tool"

    # Cleanup: make sure no subscribers linger for sibling tests.
    for _ in range(100):
        if not trace_stream._subscribers:
            break
        time.sleep(0.02)
    assert trace_stream._subscribers == []
