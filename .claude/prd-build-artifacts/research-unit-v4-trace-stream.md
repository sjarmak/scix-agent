# Research — unit-v4-trace-stream

## Context

This unit creates an in-process pub/sub event bus for agent-trace events with
an SSE endpoint at `/viz/api/trace/stream`.

## Findings

### Existing viz app shape (`src/scix/viz/server.py`)

- `FastAPI(title="SciX Viz", docs_url=None, redoc_url=None)` — single top-level app.
- Routes: `/viz/health` for the viz health probe, `/viz/` static mount for `web/viz/`.
- Per unit instructions we do NOT touch server.py — a separate unit
  (unit-v3-umap-frontend) also modifies it, and wiring the trace routes is
  deferred to a later unit. We export an `APIRouter` and leave registration
  to the caller.

### SSE pattern with FastAPI

Standard pattern for Server-Sent Events with FastAPI:

```python
from fastapi.responses import StreamingResponse

async def gen():
    async for event in source():
        yield f"data: {json.dumps(payload)}\n\n"

return StreamingResponse(gen(), media_type="text/event-stream")
```

- `text/event-stream` triggers SSE semantics in browsers / EventSource.
- Each "event" is a block terminated by an empty line (`\n\n`). Minimal form
  uses only the `data:` field.
- When the client disconnects, FastAPI cancels the generator — `finally`
  blocks must deregister the subscriber queue.

### Pub/Sub design

- Subscribers are `asyncio.Queue[TraceEvent]` instances, one per open SSE
  connection. A fresh queue is appended to a module-level `_subscribers`
  list on subscribe and removed on teardown.
- `publish()` iterates a snapshot of `_subscribers` and calls the queue's
  `put_nowait(event)`:
  - Fire-and-forget — no `await`, safe from sync contexts.
  - On `asyncio.QueueFull` we log at DEBUG and drop (back-pressure safety).
  - On any other exception we log at ERROR and continue — publish must
    never raise.
- Bounded queue size (256) protects memory against slow subscribers.
- Thread safety: `asyncio.Queue.put_nowait` is **not** thread-safe, so
  each subscriber records the event loop it was created on and
  cross-thread publishes dispatch via `loop.call_soon_threadsafe`. The
  `_subscribers` list itself is guarded by a `threading.Lock`.

### Testing

- No `pytest-asyncio` in the env. Async scenarios run inside
  `asyncio.run(...)` with `asyncio.wait_for(...)` guards on the awaits.
- Starlette's `TestClient` and httpx's `ASGITransport` both buffer the
  full response body before returning, so neither can test an infinite
  streaming generator. The SSE endpoint test stands up a real uvicorn
  server on a loopback ephemeral port.
- Micro-benchmark uses `time.perf_counter` to measure 1000 `publish()` calls
  with no subscribers; assert p99 (sorted[989]) < 5 ms — the no-subscriber
  path is a `for q in []:` loop so this is trivial but the assertion
  guards against future regressions.

### References

- Starlette SSE patterns: https://www.starlette.io/responses/#streamingresponse
- FastAPI StreamingResponse docs: https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
- MDN EventSource / SSE spec: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
