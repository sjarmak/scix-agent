# Plan — unit-v4-trace-stream

## Deliverables

1. `src/scix/viz/trace_stream.py` — module exports:
   - `@dataclass(frozen=True) class TraceEvent` with fields
     `event_id`, `ts`, `tool_name`, `params`, `result_summary`, `latency_ms`,
     `bibcodes`. Default factories for `ts`, `event_id`, `params`,
     `result_summary=None`, `bibcodes=()`.
   - Module-level `_subscribers: list[_Subscriber]` (where
     `_Subscriber = (queue, loop)`) guarded by a `threading.Lock`.
   - `_QUEUE_MAXSIZE = 256` for bounded queues.
   - `def publish(event: TraceEvent) -> None` — fire-and-forget: iterate a
     snapshot of `_subscribers`, dispatch `put_nowait` on the owning loop
     (via `call_soon_threadsafe` if the caller is on a different thread),
     swallow all exceptions.
   - `async def subscribe() -> AsyncIterator[TraceEvent]` — creates a
     fresh queue, records the running event loop, appends a
     `_Subscriber` record under the lock, yields events forever, removes
     the record in `finally`.
   - `router: APIRouter(prefix="/viz/api/trace")` with GET `/stream`
     returning `StreamingResponse` of `data: {json}\n\n` lines.

2. `tests/test_trace_stream.py`:
   - `test_publish_no_subscriber_is_noop` — call publish; no exception;
     `_subscribers` still empty.
   - `test_publish_latency_headroom` — 1000 iterations, p99 < 5 ms.
   - `test_single_subscriber_receives_event` — async test via
     `asyncio.run`; spawn a task that iterates `subscribe()`, publish
     inside the loop once the subscriber is registered.
   - `test_multiple_subscribers_each_get_event` — two subscribers; publish
     one event; both receive it.
   - `test_sse_endpoint_serves_event_stream` — run a real uvicorn server
     on a loopback ephemeral port (TestClient and ASGITransport both
     buffer the full body and would hang on an infinite generator),
     open a streaming GET on `/viz/api/trace/stream`, publish from a
     background thread once the subscriber is registered, assert the
     first chunk is a `data: {json}` SSE frame.

## Design decisions

- Keep the module self-contained — the router is exported but not wired.
  The caller (a later unit) does `app.include_router(trace_stream.router)`.
- Use `threading.Lock` (not `asyncio.Lock`) because `publish()` is
  callable from sync code paths (MCP tool handlers).
- `asyncio.Queue.put_nowait` is not thread-safe, so publish records the
  owning event loop for each subscriber and uses `call_soon_threadsafe`
  for cross-thread delivery. Same-loop delivery skips the thread hop
  and calls `put_nowait` directly.
- Use `asdict` + `default=str` for JSON serialization to tolerate any
  non-trivial field values without crashing the stream.
- No timeouts on `subscribe()` — cancellation is driven by the client
  disconnecting, which FastAPI propagates as a `CancelledError` into the
  generator.

## Risks / mitigations

- **Slow subscriber blocking publish**: bounded queue (256) + `put_nowait`
  means we drop events rather than stall the publisher. DEBUG log on drop.
- **Test flakiness around timing**: generous timeouts (3-5 s) and short
  poll loops when asserting subscribers have registered before publishing.
- **Race on subscriber list mutation**: `threading.Lock` around
  append/remove. `publish()` snapshots the list without holding the lock
  to avoid blocking.
- **Cross-thread queue.put**: explicit `call_soon_threadsafe` dispatch
  avoids the asyncio.Queue thread-safety trap.
