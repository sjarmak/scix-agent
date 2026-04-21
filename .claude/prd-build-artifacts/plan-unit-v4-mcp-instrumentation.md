# Plan — unit-v4-mcp-instrumentation

Step-by-step implementation of the MCP trace-emission hook.

## Step 1 — Imports (src/scix/mcp_server.py, top of file)

After the `from scix.session import SessionState, WorkingSetEntry` line (around line 34), add a lazy-optional import:

```python
try:
    from scix.viz import trace_stream as _trace_stream
except ImportError:  # pragma: no cover — viz extras not installed
    _trace_stream = None
```

Do NOT add any other new top-level imports.

## Step 2 — Add `_MAX_TRACE_BIBCODES` constant

Near the other module constants, add:

```python
_MAX_TRACE_BIBCODES = 20
```

## Step 3 — Add `_extract_bibcodes_from_result` helper

Just below `_extract_result_count` (around line 232), add:

```python
def _extract_bibcodes_from_result(result_json: str | None) -> tuple[str, ...]:
    """Best-effort bibcode extraction from a tool's JSON result.

    Handles:
      - ``{"papers": [{"bibcode": ...}, ...]}`` — multi-paper result.
      - ``{"bibcode": "..."}`` — single-paper shape.

    Returns an empty tuple on any parse failure. Caps at
    :data:`_MAX_TRACE_BIBCODES` to keep TraceEvents small.
    """
    if not result_json:
        return ()
    try:
        data = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return ()
    if not isinstance(data, dict):
        return ()
    bibcodes: list[str] = []
    papers = data.get("papers")
    if isinstance(papers, list):
        for p in papers:
            if isinstance(p, dict):
                bc = p.get("bibcode")
                if isinstance(bc, str):
                    bibcodes.append(bc)
                    if len(bibcodes) >= _MAX_TRACE_BIBCODES:
                        break
    if not bibcodes:
        bc = data.get("bibcode")
        if isinstance(bc, str):
            bibcodes.append(bc)
    return tuple(bibcodes)
```

## Step 4 — Add `_emit_trace_event` helper

Immediately after `_log_query` (around line 280), add:

```python
def _emit_trace_event(
    tool_name: str,
    latency_ms: float,
    params: dict[str, Any],
    result_json: str | None,
    success: bool,
) -> None:
    """Fire-and-forget TraceEvent emission.

    Publishes a ``TraceEvent`` to ``scix.viz.trace_stream`` if the viz
    module is importable. All exceptions are swallowed — trace emission
    must never break the MCP tool-call hot path.
    """
    if _trace_stream is None:
        return
    try:
        bibcodes = _extract_bibcodes_from_result(result_json)
        result_summary: str | None = None
        if not success and result_json:
            # Surface the error string as the summary so consumers can see it
            try:
                parsed = json.loads(result_json)
                if isinstance(parsed, dict) and "error" in parsed:
                    result_summary = f"error: {parsed['error']}"
            except (json.JSONDecodeError, TypeError):
                result_summary = None
        event = _trace_stream.TraceEvent(
            tool_name=tool_name,
            latency_ms=latency_ms,
            params=dict(params) if params else {},
            result_summary=result_summary,
            bibcodes=bibcodes,
        )
        _trace_stream.publish(event)
    except Exception:
        logger.debug("trace emission failed for tool=%s", tool_name, exc_info=True)
```

## Step 5 — Wire into `call_tool`

In the `finally` block of `call_tool` (line 1161-1172), immediately after the existing `_log_query(...)` call, add:

```python
_emit_trace_event(
    name,
    latency_ms,
    arguments,
    result_json,
    success,
)
```

No other changes to `call_tool`. Keep the `raise` inside the `except` (so failures propagate).

## Step 6 — Tests (tests/test_mcp_trace_instrumentation.py)

Structure:

```python
def test_extract_bibcodes_from_result_multi_paper(): ...
def test_extract_bibcodes_from_result_single_paper(): ...
def test_extract_bibcodes_from_result_bad_json_returns_empty(): ...
def test_extract_bibcodes_caps_at_20(): ...

def test_emit_trace_event_success(monkeypatch): ...
def test_emit_trace_event_failure_path(monkeypatch): ...
def test_emit_trace_event_swallows_import_failure(monkeypatch): ...

def test_call_tool_publishes_one_event_on_success(monkeypatch): ...
def test_call_tool_still_publishes_event_on_failure(monkeypatch): ...
def test_call_tool_swallows_missing_trace_stream(monkeypatch): ...
def test_overhead_under_100ms_for_100_calls(monkeypatch): ...
```

- Drive the `call_tool` end-to-end tests via `asyncio.run(handler(CallToolRequest(...)))` against a mocked-conn `_dispatch_tool`. Or more simply: call the inner function via the server's registered handler, with `_get_conn` monkeypatched to yield a `MagicMock` connection and `_log_query` monkeypatched to a no-op.
- For the overhead test: mock `_dispatch_tool` to return a small JSON string, run 100 iterations, assert total diff < 100 ms between `_trace_stream = <mock with publish>` and `_trace_stream = None`.

## Step 7 — Verification

```
.venv/bin/python -m pytest tests/test_mcp_trace_instrumentation.py -q     # all green
.venv/bin/python -m pytest tests/test_mcp_server.py -q                    # still 51 passed
```

## Step 8 — Commit

```
git add src/scix/mcp_server.py \
        tests/test_mcp_trace_instrumentation.py \
        .claude/prd-build-artifacts/research-unit-v4-mcp-instrumentation.md \
        .claude/prd-build-artifacts/plan-unit-v4-mcp-instrumentation.md \
        .claude/prd-build-artifacts/test-unit-v4-mcp-instrumentation.md
git commit -m "prd-build: unit-v4-mcp-instrumentation — MCP-server trace emission hook"
```
