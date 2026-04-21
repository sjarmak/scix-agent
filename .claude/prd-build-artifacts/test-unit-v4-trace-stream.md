# Test Results — unit-v4-trace-stream

## Command

```
.venv/bin/python -m pytest tests/test_trace_stream.py -q
```

## Outcome

```
.....                                                                    [100%]
5 passed in 0.47s
```

## Test list

| Test | Covers |
|---|---|
| `test_publish_no_subscriber_is_noop` | AC3 — fire-and-forget semantics with zero subscribers; no exception, no state leaks |
| `test_publish_latency_headroom` | AC4 — p99 of 1000 publishes < 5 ms (no-subscriber path) |
| `test_single_subscriber_receives_event` | Single-subscriber delivery via the async iterator |
| `test_multiple_subscribers_each_get_event` | Fan-out: two subscribers both receive the same event (same `event_id`) |
| `test_sse_endpoint_serves_event_stream` | AC2 — GET `/viz/api/trace/stream` returns `text/event-stream` and a `data: {...}\n\n` frame per published event, verified against a real uvicorn server |

## Implementation notes

* httpx's `ASGITransport` and Starlette's `_TestClientTransport` both
  buffer the full response body before returning, so neither supports a
  true-streaming test against an infinite async generator. The SSE test
  therefore stands up a real uvicorn server on a loopback ephemeral
  port. The `_UvicornServer` helper is small (~40 LOC) and avoids any
  additional test-framework dependencies.
* `asyncio.Queue.put_nowait` is not thread-safe. Each subscriber records
  the event loop it was created on, and cross-thread publishes dispatch
  delivery via `loop.call_soon_threadsafe`. Same-loop publishes call
  `put_nowait` directly to keep the hot path cheap.
* Lint: `ruff check` clean, `black --check` clean.

## Acceptance criteria cross-check

1. **Module exports** — `TraceEvent` (frozen dataclass with the seven
   required fields), module-level `publish(event)`, async `subscribe()`
   generator, `router` with the SSE route. Verified via a smoke import
   in test and manual REPL check (`routes: ['/viz/api/trace/stream']`).
2. **SSE endpoint returns text/event-stream with `data: ...\n\n`** —
   exercised end-to-end by `test_sse_endpoint_serves_event_stream`.
3. **Publish is a no-op with no subscribers** — verified by
   `test_publish_no_subscriber_is_noop`.
4. **p99 < 5 ms over 1000 calls** — verified by
   `test_publish_latency_headroom`.
5. **Coverage** — all five listed scenarios pass.
