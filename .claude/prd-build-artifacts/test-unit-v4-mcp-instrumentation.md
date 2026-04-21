# Test Report — unit-v4-mcp-instrumentation

## Summary

| Suite                                      | Before | After | Delta   |
| ------------------------------------------ | ------ | ----- | ------- |
| `tests/test_mcp_server.py`                 | 51     | 51    | 0       |
| `tests/test_mcp_trace_instrumentation.py`  | n/a    | 19    | +19 new |
| Other `tests/test_mcp_*.py` (5 files)      | 76     | 76    | 0       |

No regressions. 19 new passing tests cover the trace-emission hook.

## New tests (`tests/test_mcp_trace_instrumentation.py`)

### `_extract_bibcodes_from_result` (10 tests)
* `test_none_returns_empty` — `None` input returns `()`.
* `test_empty_string_returns_empty` — `""` returns `()`.
* `test_invalid_json_returns_empty` — malformed JSON returns `()`.
* `test_non_dict_root_returns_empty` — JSON array returns `()`.
* `test_multi_paper_result` — extracts all bibcodes from `{"papers":[...]}`.
* `test_single_paper_shape` — extracts single bibcode from `{"bibcode":"..."}`.
* `test_paper_without_bibcode_skipped` — robust to heterogeneous papers list.
* `test_non_string_bibcode_skipped` — integer-valued bibcode is ignored.
* `test_caps_at_max` — capped at `_MAX_TRACE_BIBCODES` (20) entries.
* `test_error_payload_returns_empty` — `{"error":...}` returns `()`.

### `_emit_trace_event` direct tests (5 tests)
* `test_success_publishes_one_event` — success path publishes exactly one
  `TraceEvent` with correct tool_name/latency/params/bibcodes.
* `test_failure_path_publishes_with_error_summary` — failure path
  emits an event with `result_summary="error: ..."` surfacing the error.
* `test_missing_trace_stream_is_noop` — when
  `mcp_server._trace_stream is None` the emitter is a silent no-op.
* `test_publish_exception_is_swallowed` — publish raising
  `RuntimeError` is caught and logged; caller is not affected.
* `test_none_params_handled` — empty-dict params are not mishandled.

### End-to-end through `call_tool` (3 tests)
* `test_successful_tool_call_publishes_trace_event` — drives a real
  `call_tool` handler via the server's `request_handlers` registry,
  asserts one TraceEvent was published.
* `test_failed_tool_call_still_publishes_trace_event` — inner tool
  raises `RuntimeError`; emission still happens in the `finally` block.
* `test_trace_stream_import_failure_is_swallowed` — sets
  `_trace_stream = None`, confirms `call_tool` still succeeds and
  `_log_query` still runs.

### Overhead budget (1 test)
* `test_overhead_under_100ms_for_100_calls` — times 100 invocations of
  `_emit_trace_event` with a stubbed publish vs 100 with
  `_trace_stream = None`. Measured overhead on this host is ~0.2 ms
  total — well under the 100 ms budget. Run 3x to confirm stability.

## Commands

```
.venv/bin/python -m pytest tests/test_mcp_trace_instrumentation.py -v
# 19 passed

.venv/bin/python -m pytest tests/test_mcp_server.py -q
# 51 passed (matches pre-change baseline)

.venv/bin/python -m pytest tests/test_mcp_smoke.py tests/test_mcp_paper_tools.py \
    tests/test_mcp_entity_tool.py tests/test_mcp_session.py \
    tests/test_mcp_citation_context.py -q
# 76 passed, 6 skipped
```

## Acceptance criteria — check

1. [x] `src/scix/mcp_server.py` imports `scix.viz.trace_stream` lazily
   inside a `try/except ImportError` — `_trace_stream` is `None` on failure.
2. [x] After every tool dispatch in `call_tool`, `_emit_trace_event`
   publishes a `TraceEvent` with `ts` (auto), `tool_name`, `latency_ms`,
   `bibcodes` extracted from the result.
3. [x] `pytest tests/test_mcp_server.py -q` still 51 passing (zero
   regressions).
4. [x] `pytest tests/test_mcp_trace_instrumentation.py -q` — 19 passing:
      (a) success publishes one event — covered.
      (b) failure still publishes — covered.
      (c) import failure swallowed — covered.
      (d) 100 calls < 100 ms overhead — covered (measured ~0.2 ms).
5. [x] No existing public API of `mcp_server.py` changed — only
   additions (new constant, two new helpers, optional module attribute).
