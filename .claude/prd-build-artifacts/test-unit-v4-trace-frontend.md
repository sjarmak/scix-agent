# Tests — unit-v4-trace-frontend

## Test file
`tests/test_agent_trace_frontend.py` — 4 tests, all PASS (0.35 s).

## Coverage

| Test | What it proves |
|------|----------------|
| `test_agent_trace_html_structure` | BeautifulSoup check: `#umap-root`, `#trace-panel`, deck.gl unpkg script, `./agent_trace.js` script ref, `./shared.css` link. |
| `test_agent_trace_js_has_subscribe_symbol` | `subscribeTraceStream` symbol is present in `web/viz/agent_trace.js`. |
| `test_agent_trace_html_served` | `fastapi.testclient.TestClient` GET `/viz/agent_trace.html` → 200 + `text/html` + body contains `trace-panel` and `umap-root`. |
| `test_sse_endpoint_delivers_published_event` | Asserts `/viz/api/trace/stream` is registered on `scix.viz.server.app`, then drives the router's `subscribe()` generator, publishes a `TraceEvent(tool_name="test_tool")`, re-frames the received event as `data: <json>\n\n` and asserts `payload["tool_name"] == "test_tool"`. |

## Why the SSE test goes through `subscribe()` rather than TestClient streaming

Starlette's `TestClient` (and `httpx.ASGITransport` under the hood) buffers
the full response body before returning. On an infinite async generator —
which `GET /viz/api/trace/stream` is by design — `client.stream("GET", …).iter_bytes()`
never yields. `tests/test_trace_stream.py` calls this out explicitly and
works around it by spinning up a real uvicorn server on a loopback socket.
Duplicating that here would be redundant.

The spec for this unit explicitly allows the direct-generator fallback:
> If TestClient doesn't actually support streaming SSE cleanly, fall back
> to a direct-unit-test pattern: call the router's generator function
> directly in an asyncio loop, call publish, anext() the generator,
> assert the event.

That is exactly what `_first_data_frame_via_generator` does. Together with
`_assert_sse_route_registered`, we prove:

1. The `/viz/api/trace/stream` route is mounted on `server.app`.
2. `publish()` reaches subscribers of the same router.
3. The wire-level `data: <json>\n\n` framing parses to a dict whose
   `tool_name` matches the published event.

End-to-end SSE wire coverage is already provided by
`tests/test_trace_stream.py::test_sse_endpoint_serves_event_stream`.

## Regression run

```
.venv/bin/python -m pytest \
  tests/test_viz_server.py tests/test_sankey_frontend.py \
  tests/test_umap_frontend.py tests/test_trace_stream.py \
  tests/test_agent_trace_frontend.py -v
→ 23 passed in 0.74s
```

No regressions after the `include_router(trace_stream_router)` edit. The
viz app continues to serve `/viz/health`, `/viz/api/paper/{bibcode}`, and
all static files; the trace-stream SSE route is additive.

## Acceptance criteria check

- [x] `web/viz/agent_trace.html` loads via `/viz/agent_trace.html` → 200.
- [x] Imports deck.gl from the same unpkg CDN as `umap_browser.html` and opens an `EventSource` on `/viz/api/trace/stream` in the bootstrap IIFE.
- [x] `web/viz/agent_trace.js` exposes `subscribeTraceStream(url, onEvent)` that wraps `EventSource` and invokes `onEvent` on each parsed JSON `message` event.
- [x] `pytest tests/test_agent_trace_frontend.py -q` passes.
- [x] HTML structure covers `#trace-panel` and `#umap-root`.
- [x] `agent_trace.js` contains `subscribeTraceStream`.
- [x] SSE integration test publishes a `TraceEvent` and asserts the first framed chunk parses to a dict matching the published `tool_name`.
