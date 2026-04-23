"""Tests for POST /viz/api/demo/search.

Asserts that the demo endpoint dispatches through
:func:`scix.mcp_server.call_tool` (so the MCP instrumentation hook publishes
the TraceEvent stream) rather than synthesizing TraceEvents itself.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from scix.viz import api as viz_api
from scix.viz.server import app

client = TestClient(app)


@pytest.fixture
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Replace ``mcp_server.call_tool`` with a stub that records each call.

    The stub returns a fixed JSON payload so the demo endpoint can parse it
    into ``papers`` / ``bibcodes``. Each invocation is appended to the list
    returned to the test.
    """
    calls: list[tuple[str, dict[str, Any]]] = []

    def _fake_call_tool(name: str, arguments: dict[str, Any]) -> str:
        calls.append((name, dict(arguments)))
        if name == "search":
            return json.dumps(
                {
                    "papers": [
                        {"bibcode": "2024A", "title": "alpha", "rrf_score": 0.9},
                        {"bibcode": "2024B", "title": "beta", "rrf_score": 0.8},
                        {"bibcode": "2024C", "title": "gamma", "rrf_score": 0.7},
                        {"bibcode": "2024D", "title": "delta", "rrf_score": 0.6},
                        {"bibcode": "2024E", "title": "epsilon", "rrf_score": 0.5},
                    ],
                    "total": 5,
                    "timing_ms": {"lexical_ms": 12.0},
                }
            )
        if name == "get_paper":
            return json.dumps({"bibcode": arguments.get("bibcode"), "title": "x"})
        return "{}"

    monkeypatch.setattr(viz_api.mcp_server, "call_tool", _fake_call_tool)
    return calls


def test_demo_search_dispatches_through_mcp_call_tool(
    captured_calls: list[tuple[str, dict[str, Any]]],
) -> None:
    """One ``search`` call followed by three ``get_paper`` drill calls."""
    response = client.post(
        "/viz/api/demo/search", json={"query": "dark matter", "top_n": 5}
    )

    assert response.status_code == 200, response.text
    body = response.json()

    # Endpoint shape unchanged so the frontend keeps working.
    assert body["query"] == "dark matter"
    assert body["bibcodes"] == ["2024A", "2024B", "2024C", "2024D", "2024E"]
    assert body["drilled"] == ["2024A", "2024B", "2024C"]
    assert body["total"] == 5
    assert body["latency_ms"] == 12.0

    # Exactly four real tool dispatches: 1x search + 3x get_paper drill-ins.
    tool_names = [name for name, _ in captured_calls]
    assert tool_names == ["search", "get_paper", "get_paper", "get_paper"]

    # Search uses keyword mode for the fast lexical path.
    search_args = captured_calls[0][1]
    assert search_args["query"] == "dark matter"
    assert search_args["mode"] == "keyword"
    assert search_args["limit"] == 5

    # Each drill targets a top-3 bibcode in order.
    assert [args["bibcode"] for _, args in captured_calls[1:]] == [
        "2024A",
        "2024B",
        "2024C",
    ]


def test_demo_search_does_not_publish_trace_events_directly(
    captured_calls: list[tuple[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint must not call ``trace_stream.publish`` itself.

    Trace events should now be emitted only by the instrumentation hook
    inside ``mcp_server.call_tool``. We verify by patching ``publish`` on the
    ``trace_stream`` module and asserting it is never invoked from this
    endpoint's call path (which now goes through the stubbed ``call_tool``).
    """
    from scix.viz import trace_stream

    publish_spy = MagicMock()
    monkeypatch.setattr(trace_stream, "publish", publish_spy)

    response = client.post(
        "/viz/api/demo/search", json={"query": "exoplanet atmospheres", "top_n": 3}
    )
    assert response.status_code == 200, response.text

    publish_spy.assert_not_called()


def test_demo_search_surfaces_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``call_tool('search', ...)`` returns an error payload, return 502."""

    def _error_call_tool(name: str, arguments: dict[str, Any]) -> str:
        if name == "search":
            return json.dumps({"error": "search failed: backend unavailable"})
        return "{}"

    monkeypatch.setattr(viz_api.mcp_server, "call_tool", _error_call_tool)

    response = client.post(
        "/viz/api/demo/search", json={"query": "x" * 5, "top_n": 5}
    )
    assert response.status_code == 502
    assert "backend unavailable" in response.json()["detail"]


def test_demo_search_drill_tolerates_get_paper_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing ``get_paper`` drill call must not poison the response."""
    invocations: list[str] = []

    def _flaky_call_tool(name: str, arguments: dict[str, Any]) -> str:
        invocations.append(name)
        if name == "search":
            return json.dumps(
                {
                    "papers": [
                        {"bibcode": "2024A"},
                        {"bibcode": "2024B"},
                        {"bibcode": "2024C"},
                    ],
                    "total": 3,
                    "timing_ms": {"lexical_ms": 5.0},
                }
            )
        # Second drill blows up; first and third must still proceed.
        if arguments.get("bibcode") == "2024B":
            raise RuntimeError("simulated failure")
        return json.dumps({"bibcode": arguments.get("bibcode")})

    monkeypatch.setattr(viz_api.mcp_server, "call_tool", _flaky_call_tool)

    response = client.post(
        "/viz/api/demo/search", json={"query": "test query", "top_n": 5}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # Drill only records bibcodes whose call succeeded.
    assert body["drilled"] == ["2024A", "2024C"]
    # All four dispatches were attempted.
    assert invocations == ["search", "get_paper", "get_paper", "get_paper"]
