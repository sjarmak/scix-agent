# Research — unit-v4-mcp-instrumentation

## Baseline

Run before any changes:

```
.venv/bin/python -m pytest tests/test_mcp_server.py -q
# ...................................................                      [100%]
# 51 passed in 0.48s
```

Baseline: **51 passed**. Regression gate: must still pass with 51 after the change.

## mcp_server.py — relevant sections

### Imports (line 17-34)

Top-level imports use `from __future__ import annotations`, then `dataclasses`, `json`, `logging`, `os`, `time`, `uuid`, plus `psycopg`, `scix.search`, `scix.db`, `scix.embed`, `scix.entity_resolver`, `scix.session`. Logger is module-scoped.

### Existing helpers (lines 191-231)

Two helpers already exist in the exact style we need to match:

- `_extract_query_text(params: dict) -> str | None` — iterates over a `_QUERY_ARG_KEYS` tuple to find the first matching key.
- `_extract_result_count(result_json: str) -> int` — parses the JSON, returns 0 on any parse failure. Tries `total`, then `len(papers)`, then `len(results)`.

**No existing `_extract_bibcodes` helper** — we need to add one. The project already has `_auto_track_bibcodes` (line 389) which implements the exact extraction logic we need (handles `papers` list + single `bibcode` shape + swallows `JSONDecodeError/TypeError`). We should mirror this to extract bibcodes into a tuple capped at 20.

### `_log_query` helper (line 234-279)

Fire-and-forget DB writer. Takes `(conn, tool_name, params, latency_ms, success, error_msg, result_json=, session_id=, is_test=)`. Entire body wrapped in try/except with `logger.warning`. This is the pattern our `_emit_trace_event` should mirror.

### `call_tool` dispatcher (lines 1143-1173)

```python
@server.call_tool()
async def call_tool(name, arguments):
    with _get_conn() as conn:
        resolved_name = _DEPRECATED_ALIASES.get(name, name)
        _set_timeout(conn, resolved_name)
        t0 = time.monotonic()
        success = True
        error_msg = None
        result_json = "{}"
        try:
            result_json = _dispatch_tool(conn, name, arguments)
        except Exception as exc:
            success = False
            error_msg = str(exc)
            result_json = json.dumps({"error": error_msg})
            raise
        finally:
            latency_ms = (time.monotonic() - t0) * 1000
            _log_query(conn, name, arguments, latency_ms, success, error_msg,
                       result_json=result_json, session_id=..., is_test=...)
        return [TextContent(type="text", text=result_json)]
```

Key timing pattern to reuse: `t0 = time.monotonic()` and `latency_ms = (time.monotonic() - t0) * 1000`.

Emission point: **after** the existing `_log_query(...)` call, still inside the `finally` block, so every tool call (success or failure) publishes exactly one TraceEvent.

## trace_stream module (src/scix/viz/trace_stream.py)

- `TraceEvent` is a `@dataclass(frozen=True)`.
- Required positional args: `tool_name: str`, `latency_ms: float`.
- Defaulted: `ts` (time.time()), `event_id` (uuid4 hex), `params: dict`, `result_summary: str | None`, `bibcodes: tuple[str, ...]`.
- `publish(event) -> None` — fire-and-forget, explicitly documented as never raises.
- `publish()` is already thread-safe (uses `call_soon_threadsafe` for cross-thread delivery).

Our lazy import should gracefully handle FastAPI's absence at import time by catching `ImportError`.

## Existing test patterns (tests/test_mcp_server.py)

- Uses `MagicMock`, `patch` from `unittest.mock`.
- `_dispatch_tool(mock_conn, name, args)` is invoked directly with a mocked connection — no real DB needed.
- Inner search functions mocked via `@patch("scix.search.lexical_search")` etc.

We'll follow the same approach: mock `_dispatch_tool`, mock the inner tool, and drive through `call_tool`. We can reach `call_tool` by constructing the server or by calling the helper directly; but the simpler route is to test the `_emit_trace_event` helper directly + do one end-to-end test driving through `call_tool` with mocks.

## Plan summary

1. Add optional lazy import of `scix.viz.trace_stream` at the top of `mcp_server.py` — set `_trace_stream = None` on `ImportError`.
2. Add `_extract_bibcodes_from_result(result_json) -> tuple[str, ...]` helper (caps at 20, swallows errors).
3. Add `_emit_trace_event(tool_name, latency_ms, params, result_json, success)` helper that wraps publish in a try/except and logs at debug level.
4. In `call_tool`'s `finally` block, immediately after `_log_query(...)`, call `_emit_trace_event(...)`.
5. Tests in `tests/test_mcp_trace_instrumentation.py` cover: success path, failure path, import-failure swallowed, overhead budget.

## Files in scope

- EDIT: `src/scix/mcp_server.py`
- NEW: `tests/test_mcp_trace_instrumentation.py`
