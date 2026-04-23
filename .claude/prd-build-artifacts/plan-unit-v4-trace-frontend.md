# Plan — unit-v4-trace-frontend

## Files

### 1. `src/scix/viz/server.py` — EDIT
Add:
```python
from scix.viz.trace_stream import router as trace_stream_router
...
app.include_router(viz_api_router)
app.include_router(trace_stream_router)   # NEW — SSE at /viz/api/trace/stream
app.mount("/viz", StaticFiles(...), name="viz")
```
Include before the static mount so the SSE path wins over static-file routing.

### 2. `web/viz/agent_trace.html` — NEW
Mirrors `umap_browser.html` structure:
- Same `<link rel="stylesheet" href="./shared.css">`.
- Same `<script src="https://unpkg.com/deck.gl@9/dist.min.js"></script>`.
- Layout: main viz container, a two-pane grid with `<div id="umap-root">` and an aside `<div id="trace-panel">`.
- Bootstrap IIFE:
  - Fetch UMAP dataset (same candidates as umap_browser).
  - `window.umapInstance = window.renderUMAP(data, root)` — capture the deck.gl instance for the trace layer injection.
  - `window.subscribeTraceStream('/viz/api/trace/stream', handleEvent)`.
- `handleEvent(event)`: append a `<div class="trace-line">` to `#trace-panel` with `tool_name`, timestamp, bibcode count. If `event.bibcodes.length >= 2` call `flashTraceSegments(event.bibcodes)` to drop a transient `LineLayer` on the deck.gl instance for 1.2 s.

Script loads:
```html
<script src="./umap_browser.js"></script>
<script src="./agent_trace.js"></script>
```

### 3. `web/viz/agent_trace.js` — NEW
```js
function subscribeTraceStream(url, onEvent) {
  const es = new EventSource(url)
  es.addEventListener('message', (ev) => {
    try { onEvent(JSON.parse(ev.data)) } catch (e) { console.error(e) }
  })
  return es
}
window.subscribeTraceStream = subscribeTraceStream
```
Also export a tiny `flashTraceSegments(deckInstance, bibcodes, positionsLookup)` helper used by the bootstrap for visual polish.

### 4. `tests/test_agent_trace_frontend.py` — NEW
Four cases:
1. `test_agent_trace_html_structure` — BeautifulSoup: `#trace-panel`, `#umap-root`, deck.gl unpkg script, `./agent_trace.js` reference, `./shared.css`.
2. `test_agent_trace_js_has_subscribe_symbol` — file-content check for `subscribeTraceStream`.
3. `test_agent_trace_html_served` — TestClient `GET /viz/agent_trace.html` → 200 + text/html + contains `trace-panel`.
4. `test_sse_endpoint_delivers_published_event` — Starlette `TestClient` (with `.stream("GET", url)`) + background publisher thread fires `trace_stream.publish(...)` after 0.2 s. Read lines via `resp.iter_lines()`, assert first `data: ...` parses to JSON with the expected `tool_name`. If TestClient buffers too aggressively, fall back to exercising the router's `subscribe()` generator directly via `asyncio.run`.

## SSE test strategy
Plan A: Use `TestClient(app)` as a context manager, `with client.stream("GET", "/viz/api/trace/stream") as resp`, then iterate `resp.iter_lines()`. Publish from a daemon thread after a short sleep so the ASGI subscribe-generator has registered its queue.

Fallback (Plan B): call `trace_stream.subscribe()` directly inside an `asyncio.run` helper, publish, `await agen.__anext__()` and assert — this proves the plumbing without going through HTTP.

Start with Plan A. If it hangs / buffers, swap to Plan B.

## Acceptance mapping
- HTML loads: static mount serves `/viz/agent_trace.html`.
- deck.gl + EventSource: inline bootstrap.
- `subscribeTraceStream` exported: `agent_trace.js`.
- pytest passes: all 4 tests.
- No regressions: preserve existing `include_router(viz_api_router)` and static mount.

## Commit
```
prd-build: unit-v4-trace-frontend — Agent-trace overlay frontend (SSE + deck.gl)
```
Stage: html, js, server.py edit, test file, 3 artifact docs.
