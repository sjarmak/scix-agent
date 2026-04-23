"""Tests for MCP-server trace-emission hook (unit-v4-mcp-instrumentation).

Covers:
  * :func:`_extract_bibcodes_from_result` — JSON shape parsing and caps.
  * :func:`_emit_trace_event` — success, failure, and
    import-failure paths.
  * End-to-end through :func:`call_tool` — publishes exactly one
    :class:`TraceEvent` per tool dispatch.
  * Overhead budget: 100 stubbed tool calls add < 100 ms total.

All tests mock the DB connection and the inner tool dispatch so no
database or real embedding model is required.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from scix import mcp_server
from scix.mcp_server import (
    _emit_trace_event,
    _extract_bibcodes_from_result,
    _MAX_TRACE_BIBCODES,
)
from scix.viz.trace_stream import TraceEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_events(monkeypatch: pytest.MonkeyPatch) -> list[TraceEvent]:
    """Swap the module-level trace_stream with a capturing stub."""
    events: list[TraceEvent] = []

    stub = MagicMock()
    stub.TraceEvent = TraceEvent  # use the real frozen dataclass
    stub.publish.side_effect = lambda ev: events.append(ev)

    monkeypatch.setattr(mcp_server, "_trace_stream", stub)
    return events


@pytest.fixture
def mock_conn(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock ``_get_conn`` to yield a plain MagicMock connection."""
    conn = MagicMock()

    @contextmanager
    def _fake_get_conn() -> Any:
        yield conn

    monkeypatch.setattr(mcp_server, "_get_conn", _fake_get_conn)
    # Also no-op out _log_query so we don't need a real DB cursor.
    monkeypatch.setattr(mcp_server, "_log_query", lambda *a, **k: None)
    # And _set_timeout.
    monkeypatch.setattr(mcp_server, "_set_timeout", lambda *a, **k: None)
    return conn


# ---------------------------------------------------------------------------
# _extract_bibcodes_from_result
# ---------------------------------------------------------------------------


class TestExtractBibcodes:
    def test_none_returns_empty(self) -> None:
        assert _extract_bibcodes_from_result(None) == ()

    def test_empty_string_returns_empty(self) -> None:
        assert _extract_bibcodes_from_result("") == ()

    def test_invalid_json_returns_empty(self) -> None:
        assert _extract_bibcodes_from_result("not-json{") == ()

    def test_non_dict_root_returns_empty(self) -> None:
        assert _extract_bibcodes_from_result("[1,2,3]") == ()

    def test_multi_paper_result(self) -> None:
        payload = json.dumps(
            {"papers": [{"bibcode": "2024A"}, {"bibcode": "2024B"}]}
        )
        assert _extract_bibcodes_from_result(payload) == ("2024A", "2024B")

    def test_single_paper_shape(self) -> None:
        payload = json.dumps({"bibcode": "2024Z"})
        assert _extract_bibcodes_from_result(payload) == ("2024Z",)

    def test_paper_without_bibcode_skipped(self) -> None:
        payload = json.dumps(
            {"papers": [{"bibcode": "2024A"}, {"title": "no bibcode"}, {"bibcode": "2024B"}]}
        )
        assert _extract_bibcodes_from_result(payload) == ("2024A", "2024B")

    def test_non_string_bibcode_skipped(self) -> None:
        payload = json.dumps({"papers": [{"bibcode": 12345}, {"bibcode": "2024A"}]})
        assert _extract_bibcodes_from_result(payload) == ("2024A",)

    def test_caps_at_max(self) -> None:
        papers = [{"bibcode": f"B{i:04d}"} for i in range(_MAX_TRACE_BIBCODES + 10)]
        payload = json.dumps({"papers": papers})
        result = _extract_bibcodes_from_result(payload)
        assert len(result) == _MAX_TRACE_BIBCODES
        assert result[0] == "B0000"

    def test_error_payload_returns_empty(self) -> None:
        payload = json.dumps({"error": "boom"})
        assert _extract_bibcodes_from_result(payload) == ()


# ---------------------------------------------------------------------------
# _emit_trace_event — direct helper tests
# ---------------------------------------------------------------------------


class TestEmitTraceEvent:
    def test_success_publishes_one_event(
        self, captured_events: list[TraceEvent]
    ) -> None:
        result_json = json.dumps({"papers": [{"bibcode": "2024A"}], "total": 1})
        _emit_trace_event("search", 12.5, {"query": "dark matter"}, result_json, True)

        assert len(captured_events) == 1
        ev = captured_events[0]
        assert ev.tool_name == "search"
        assert ev.latency_ms == 12.5
        assert ev.params == {"query": "dark matter"}
        assert ev.bibcodes == ("2024A",)
        assert ev.result_summary is None

    def test_failure_path_publishes_with_error_summary(
        self, captured_events: list[TraceEvent]
    ) -> None:
        result_json = json.dumps({"error": "bad arg"})
        _emit_trace_event("search", 3.0, {"query": "x"}, result_json, False)

        assert len(captured_events) == 1
        ev = captured_events[0]
        assert ev.tool_name == "search"
        assert ev.result_summary == "error: bad arg"
        assert ev.bibcodes == ()

    def test_missing_trace_stream_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mcp_server, "_trace_stream", None)
        # Must not raise; no events to assert on.
        _emit_trace_event("search", 1.0, {}, "{}", True)

    def test_publish_exception_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        broken = MagicMock()
        broken.TraceEvent = TraceEvent
        broken.publish.side_effect = RuntimeError("kaboom")
        monkeypatch.setattr(mcp_server, "_trace_stream", broken)
        # Must not raise.
        _emit_trace_event("search", 1.0, {"q": "x"}, "{}", True)
        broken.publish.assert_called_once()

    def test_none_params_handled(
        self, captured_events: list[TraceEvent]
    ) -> None:
        # `dict(params) if params else {}` should handle an empty dict
        # without issue; confirm empty dict produces empty params.
        _emit_trace_event("search", 1.0, {}, None, True)
        assert captured_events[0].params == {}


# ---------------------------------------------------------------------------
# call_tool end-to-end — publish happens after _log_query
# ---------------------------------------------------------------------------


def _invoke_call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Build a server, locate the call_tool handler, and invoke it.

    Uses the server's internal request_handlers registry so we drive the
    exact production code path.
    """
    from mcp.types import CallToolRequest, CallToolRequestParams

    server = mcp_server.create_server(_run_self_test=False)
    handler = server.request_handlers[CallToolRequest]
    req = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=tool_name, arguments=arguments),
    )
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(handler(req))
    finally:
        loop.close()


class TestCallToolEmission:
    def test_successful_tool_call_publishes_trace_event(
        self,
        monkeypatch: pytest.MonkeyPatch,
        captured_events: list[TraceEvent],
        mock_conn: MagicMock,
    ) -> None:
        # Stub _dispatch_tool to return a fixed JSON payload.
        monkeypatch.setattr(
            mcp_server,
            "_dispatch_tool",
            lambda conn, name, args: json.dumps(
                {"papers": [{"bibcode": "2024X"}], "total": 1}
            ),
        )
        # Avoid the self-test path.
        monkeypatch.setattr(mcp_server, "startup_self_test", lambda srv: None)

        _invoke_call_tool("search", {"query": "test"})

        assert len(captured_events) == 1
        ev = captured_events[0]
        assert ev.tool_name == "search"
        assert ev.bibcodes == ("2024X",)
        assert ev.latency_ms >= 0.0
        assert ev.result_summary is None

    def test_failed_tool_call_still_publishes_trace_event(
        self,
        monkeypatch: pytest.MonkeyPatch,
        captured_events: list[TraceEvent],
        mock_conn: MagicMock,
    ) -> None:
        def _raise(conn: Any, name: str, args: dict[str, Any]) -> str:
            raise RuntimeError("inner failure")

        monkeypatch.setattr(mcp_server, "_dispatch_tool", _raise)
        monkeypatch.setattr(mcp_server, "startup_self_test", lambda srv: None)

        # The MCP framework may catch + convert the raised exception into
        # an error response, but the emission hook runs in a `finally`
        # block that fires regardless. Invoke the handler and verify the
        # event landed, without asserting on whether the exception
        # propagates past the framework boundary.
        try:
            _invoke_call_tool("search", {"query": "boom"})
        except Exception:
            # Either outcome (caught inside MCP or re-raised) is valid;
            # emission is what we're testing.
            pass

        # The event must have been published regardless of where the
        # exception ultimately surfaces.
        assert len(captured_events) == 1
        ev = captured_events[0]
        assert ev.tool_name == "search"
        # result_summary surfaces the error for the failure path.
        assert ev.result_summary == "error: inner failure"
        assert ev.bibcodes == ()

    def test_trace_stream_import_failure_is_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_conn: MagicMock,
    ) -> None:
        """With _trace_stream = None, call_tool must still succeed and
        the existing _log_query must still be invoked."""
        monkeypatch.setattr(mcp_server, "_trace_stream", None)

        monkeypatch.setattr(
            mcp_server,
            "_dispatch_tool",
            lambda conn, name, args: json.dumps({"papers": [], "total": 0}),
        )

        log_calls: list[tuple[str, ...]] = []

        def _track_log(*a: Any, **k: Any) -> None:
            # (tool_name, params, latency_ms, success, error_msg)
            log_calls.append((a[1],))

        monkeypatch.setattr(mcp_server, "_log_query", _track_log)
        monkeypatch.setattr(mcp_server, "startup_self_test", lambda srv: None)

        # Must not raise.
        _invoke_call_tool("search", {"query": "noop"})

        # _log_query must still have run exactly once.
        assert log_calls == [("search",)]


# ---------------------------------------------------------------------------
# Overhead budget — 100 calls with hook must be <100 ms slower than without
# ---------------------------------------------------------------------------


class TestOverhead:
    def test_overhead_under_100ms_for_100_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Directly time the _emit_trace_event hook over 100 invocations
        vs the no-op path. The delta must stay under 100 ms total."""
        result_json = json.dumps({"papers": [{"bibcode": "2024A"}], "total": 1})
        args = {"query": "x"}

        # --- With hook active (real TraceEvent, no-op publish) ---
        stub = MagicMock()
        stub.TraceEvent = TraceEvent
        stub.publish.side_effect = lambda ev: None
        monkeypatch.setattr(mcp_server, "_trace_stream", stub)

        # Warmup (avoid first-call import effects)
        for _ in range(5):
            _emit_trace_event("search", 0.0, args, result_json, True)

        t0 = time.perf_counter()
        for _ in range(100):
            _emit_trace_event("search", 0.0, args, result_json, True)
        with_hook_ms = (time.perf_counter() - t0) * 1000

        # --- With hook disabled (trace_stream = None, fast return) ---
        monkeypatch.setattr(mcp_server, "_trace_stream", None)

        for _ in range(5):
            _emit_trace_event("search", 0.0, args, result_json, True)

        t0 = time.perf_counter()
        for _ in range(100):
            _emit_trace_event("search", 0.0, args, result_json, True)
        without_hook_ms = (time.perf_counter() - t0) * 1000

        overhead_ms = with_hook_ms - without_hook_ms
        assert overhead_ms < 100.0, (
            f"Overhead budget exceeded: {overhead_ms:.2f}ms "
            f"(with={with_hook_ms:.2f}ms, without={without_hook_ms:.2f}ms)"
        )
