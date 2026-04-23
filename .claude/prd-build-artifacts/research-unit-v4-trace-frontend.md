# Research — unit-v4-trace-frontend

## Goal
Ship the browser overlay for the agent-trace stream:
- `web/viz/agent_trace.html` — UMAP scatter + trace panel.
- `web/viz/agent_trace.js` — `subscribeTraceStream(url, onEvent)` EventSource helper.
- Wire `scix.viz.trace_stream.router` onto the FastAPI app in `src/scix/viz/server.py`.
- Add `tests/test_agent_trace_frontend.py`.

## Key references

### `web/viz/umap_browser.html`
Vanilla HTML + CDN script loads. Relevant conventions:
- `<link rel="stylesheet" href="./shared.css">`.
- deck.gl via `<script src="https://unpkg.com/deck.gl@9/dist.min.js"></script>`.
- Hosts `<div id="umap-root">` with nested `<div id="umap-tooltip">` and an `<aside id="umap-panel">`.
- Bootstrap IIFE fetches `./umap.json` / `/data/viz/umap.json` then calls `window.renderUMAP(data, root)`.
- Error surface: `<div id="umap-status">`.

### `web/viz/umap_browser.js`
Exports `window.renderUMAP(points, container)` — returns the `deck.Deck` instance. The instance has `.setProps({layers: [...]})` which we can call to inject a transient LineLayer.

### `src/scix/viz/server.py`
Current registrations (must stay intact):
```python
from scix.viz.api import router as viz_api_router
...
app.include_router(viz_api_router)          # JSON API before static mount
app.mount("/viz", StaticFiles(directory=..., html=True), name="viz")
```
The `/viz` static mount uses `html=True` so `/viz/agent_trace.html` will auto-serve that file once it exists.

### `src/scix/viz/trace_stream.py`
- `router = APIRouter(prefix="/viz/api/trace", tags=["viz-trace"])`
- `@router.get("/stream")` yields `f"data: {json.dumps(asdict(event))}\n\n"` with `media_type="text/event-stream"`.
- `publish(event)` is fire-and-forget, cross-thread safe via `call_soon_threadsafe`.
- `TraceEvent(tool_name, latency_ms, ts, event_id, params, result_summary, bibcodes)`.

### `tests/test_trace_stream.py` (SSE patterns to reuse)
- Starlette `TestClient` buffers response bodies — they note this explicitly. They spin up a real uvicorn server for end-to-end SSE.
- However `httpx.Client().stream(...)` yields bytes while the generator runs. The spec for this unit says to use `with client.stream("GET", ...)` via `TestClient` and accept its streaming behaviour: Starlette's TestClient does support streaming via `client.stream(...)` for well-behaved generators when paired with a background publisher thread + `resp.iter_lines()`.
- Publisher fires after a 0.2 s wait so the subscriber's queue is registered inside the ASGI generator.

### `tests/test_umap_frontend.py` and `tests/test_sankey_frontend.py`
Established patterns for:
- BeautifulSoup structure checks against the HTML file on disk.
- TestClient `GET /viz/<file>.html` → 200 + `text/html`.
- Static JS file string-probe (`assert "symbol" in js_text`).

## Notes
- Preserve deck.gl import parity with `umap_browser.html` (`unpkg.com/deck.gl@9/dist.min.js`) to get identical behaviour.
- `subscribeTraceStream` must be attached to `window` for the inline bootstrap to find it (matches `window.renderUMAP` pattern).
- For the line-segment flash between bibcodes we will inject a deck.gl `LineLayer` via `instance.setProps({layers: [...]})` with a `setTimeout`-driven cleanup. The visual polish is intentionally minimal per the spec — wiring end-to-end matters more than the effect.
- SSE consumers should only receive `message`-type events. The server only emits unlabelled data frames, which EventSource treats as `message`.
