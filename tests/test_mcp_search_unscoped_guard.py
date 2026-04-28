"""Tests for the unscoped-broad-query guard in the MCP `search` tool.

Bead: scix_experiments-uerc

The `search` tool's description warns that unscoped queries run a full-text
scan over all 32M papers and may hit the statement timeout. These tests
verify the guard that intercepts unscoped + broad queries before they hit
Postgres and returns a structured ``unscoped_broad_query`` error with
actionable hints, plus a ``bypass_unscoped_guard`` escape hatch for tests
and power users.

All tests mock the DB connection and the underlying search functions so
no real DB or models are touched. The guard MUST fire before any
disambiguation, embedding, or DB call, so success of these tests proves
both the heuristic and the early-return ordering.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix import mcp_server
from scix.mcp_server import _dispatch_tool, _is_unscoped_broad_query
from scix.search import SearchResult

# ---------------------------------------------------------------------------
# Fixtures — disable disambiguation, force lexical-only path
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_disambiguator() -> Generator[None, None, None]:
    """Patch the disambiguator so the search path never short-circuits."""
    with patch("scix.mcp_server.disambiguate_query", return_value=[]):
        yield


@pytest.fixture(autouse=True)
def _disable_hnsw() -> Generator[None, None, None]:
    """Force the lexical-only path inside _handle_search — no embedding load."""
    with patch("scix.mcp_server._hnsw_index_exists", return_value=False):
        yield


def _hybrid_stub() -> MagicMock:
    """Return a MagicMock that mimics ``search.hybrid_search`` with a tiny result."""
    return MagicMock(
        return_value=SearchResult(
            papers=[{"bibcode": "2024ApJ...001A", "title": "stub"}],
            total=1,
            timing_ms={"total_ms": 0.0},
        )
    )


def _lexical_stub() -> MagicMock:
    """Return a MagicMock that mimics ``search.lexical_search`` with a tiny result."""
    return MagicMock(
        return_value=SearchResult(
            papers=[{"bibcode": "2024ApJ...001A", "title": "stub"}],
            total=1,
            timing_ms={"total_ms": 0.0},
        )
    )


# ---------------------------------------------------------------------------
# Heuristic unit tests — _is_unscoped_broad_query
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,filters,bypass,expected",
    [
        # Single token → narrow, never blocked.
        ("exoplanets", None, False, False),
        ("exoplanets", {}, False, False),
        # Two-token query under 30 chars → not blocked (heuristic floor at 3 tokens).
        ("quantum gravity", None, False, False),
        ("black holes", {}, False, False),
        # Three+ tokens → broad → blocked when unscoped.
        ("quantum gravity loop integration", None, False, True),
        ("supernova remnant magnetic field", {}, False, True),
        # Long query (>= 30 chars) even with 2 tokens → blocked.
        ("supernova-remnant-magnetic-field-evolution", None, False, True),
        # Scoped via filters.year_min → not blocked even if broad.
        (
            "quantum gravity loop integration",
            {"year_min": 2020},
            False,
            False,
        ),
        # Scoped via filters.arxiv_class → not blocked.
        (
            "quantum gravity loop integration",
            {"arxiv_class": "astro-ph"},
            False,
            False,
        ),
        # Scoped via filters.entity_types → not blocked.
        (
            "quantum gravity loop integration",
            {"entity_types": ["instrument"]},
            False,
            False,
        ),
        # Filters present but all-None → still unscoped (full corpus scan).
        (
            "quantum gravity loop integration",
            {"year_min": None, "year_max": None, "arxiv_class": None},
            False,
            True,
        ),
        # Filters with empty list → still unscoped.
        (
            "quantum gravity loop integration",
            {"entity_types": [], "entity_ids": []},
            False,
            True,
        ),
        # year_min=0 (or any non-positive) doesn't constrain anything → still unscoped.
        (
            "quantum gravity loop integration",
            {"year_min": 0},
            False,
            True,
        ),
        (
            "quantum gravity loop integration",
            {"year_max": 0},
            False,
            True,
        ),
        (
            "quantum gravity loop integration",
            {"year_min": -1, "year_max": 0},
            False,
            True,
        ),
        # Bypass set → never blocked even if unscoped + broad.
        ("quantum gravity loop integration", None, True, False),
        ("quantum gravity loop integration", {}, True, False),
        # Edge cases.
        ("", None, False, False),  # empty string → not blocked (schema enforces presence)
        ("   ", None, False, False),  # whitespace only → not blocked
    ],
)
def test_is_unscoped_broad_query_heuristic(
    query: str, filters: dict[str, Any] | None, bypass: bool, expected: bool
) -> None:
    """Direct unit test of the heuristic."""
    assert _is_unscoped_broad_query(query, filters, bypass=bypass) is expected


# ---------------------------------------------------------------------------
# End-to-end: guard fires through _dispatch_tool
# ---------------------------------------------------------------------------


def test_unscoped_broad_query_returns_structured_error() -> None:
    """AC1: Unscoped 3+ token query returns structured error, no DB call."""
    stub_hybrid = _hybrid_stub()
    stub_lexical = _lexical_stub()
    mock_conn = MagicMock()

    with (
        patch("scix.mcp_server.search.hybrid_search", stub_hybrid),
        patch("scix.mcp_server.search.lexical_search", stub_lexical),
    ):
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            {"query": "supernova remnant magnetic field evolution"},
        )

    # Neither search backend was invoked — guard fired before DB call.
    stub_hybrid.assert_not_called()
    stub_lexical.assert_not_called()

    data = json.loads(result_json)
    assert data["error_code"] == "unscoped_broad_query"
    assert isinstance(data["error"], str) and data["error"].strip()
    assert "hint" in data and data["hint"]
    assert data["query"] == "supernova remnant magnetic field evolution"
    assert "suggestions" in data
    # Telemetry tag for operators to track rate.
    assert data["unscoped_broad_blocked"] is True
    # Bypass instructions surfaced for the agent.
    assert "bypass" in data


def test_long_query_returns_structured_error() -> None:
    """AC1b: Long unscoped query (>=30 chars) blocked even at 2 tokens."""
    stub_hybrid = _hybrid_stub()
    mock_conn = MagicMock()

    long_two_token = "supernova-remnant-magnetic-field-evolution"
    assert len(long_two_token) >= 30

    with patch("scix.mcp_server.search.hybrid_search", stub_hybrid):
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            {"query": long_two_token},
        )

    stub_hybrid.assert_not_called()
    data = json.loads(result_json)
    assert data["error_code"] == "unscoped_broad_query"
    assert isinstance(data["error"], str) and data["error"].strip()


def test_scoped_query_runs_normal_flow() -> None:
    """AC2: Query with filters.year_min runs through hybrid_search normally."""
    stub_hybrid = _hybrid_stub()
    mock_conn = MagicMock()

    with patch("scix.mcp_server.search.hybrid_search", stub_hybrid):
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            {
                "query": "quantum gravity loop integration",
                "filters": {"year_min": 2020},
            },
        )

    stub_hybrid.assert_called_once()
    data = json.loads(result_json)
    assert "error" not in data
    assert data["total"] == 1


def test_scoped_via_arxiv_class_runs_normal_flow() -> None:
    """AC2b: filters.arxiv_class scope runs normal flow."""
    stub_hybrid = _hybrid_stub()
    mock_conn = MagicMock()

    with patch("scix.mcp_server.search.hybrid_search", stub_hybrid):
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            {
                "query": "quantum gravity loop integration",
                "filters": {"arxiv_class": "astro-ph"},
            },
        )

    stub_hybrid.assert_called_once()
    data = json.loads(result_json)
    assert "error" not in data


def test_narrow_unscoped_query_runs_normal_flow() -> None:
    """AC2c: Single-token unscoped query runs normal flow (not blocked)."""
    stub_hybrid = _hybrid_stub()
    mock_conn = MagicMock()

    with patch("scix.mcp_server.search.hybrid_search", stub_hybrid):
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            {"query": "exoplanets"},
        )

    stub_hybrid.assert_called_once()
    data = json.loads(result_json)
    assert "error" not in data


def test_two_token_unscoped_query_runs_normal_flow() -> None:
    """AC2d: Two-token unscoped query under 30 chars runs normal flow."""
    stub_hybrid = _hybrid_stub()
    mock_conn = MagicMock()

    with patch("scix.mcp_server.search.hybrid_search", stub_hybrid):
        _dispatch_tool(
            mock_conn,
            "search",
            {"query": "quantum gravity"},
        )

    stub_hybrid.assert_called_once()


def test_bypass_unscoped_guard_runs_normal_flow() -> None:
    """AC3: bypass_unscoped_guard=True skips the guard for unscoped broad queries."""
    stub_hybrid = _hybrid_stub()
    mock_conn = MagicMock()

    with patch("scix.mcp_server.search.hybrid_search", stub_hybrid):
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            {
                "query": "quantum gravity loop integration",
                "bypass_unscoped_guard": True,
            },
        )

    stub_hybrid.assert_called_once()
    data = json.loads(result_json)
    assert "error" not in data
    assert data["total"] == 1


def test_keyword_mode_blocks_unscoped_broad() -> None:
    """Guard fires for keyword mode too (any mode hits the corpus)."""
    stub_lexical = _lexical_stub()
    mock_conn = MagicMock()

    with patch("scix.mcp_server.search.lexical_search", stub_lexical):
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            {
                "query": "quantum gravity loop integration",
                "mode": "keyword",
            },
        )

    stub_lexical.assert_not_called()
    data = json.loads(result_json)
    assert data["error_code"] == "unscoped_broad_query"
    assert isinstance(data["error"], str) and data["error"].strip()


def test_semantic_mode_blocks_unscoped_broad() -> None:
    """Guard fires for semantic mode too."""
    mock_conn = MagicMock()

    # No need to patch search.vector_search — guard fires first.
    result_json = _dispatch_tool(
        mock_conn,
        "search",
        {
            "query": "quantum gravity loop integration",
            "mode": "semantic",
        },
    )

    data = json.loads(result_json)
    assert data["error_code"] == "unscoped_broad_query"
    assert isinstance(data["error"], str) and data["error"].strip()


def test_guard_fires_before_disambiguation() -> None:
    """Guard runs before disambiguation — disambiguator is not invoked when blocked."""
    mock_conn = MagicMock()

    with patch("scix.mcp_server.disambiguate_query") as mock_disamb:
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            {"query": "quantum gravity loop integration"},
        )

        # Disambiguator never called when the guard fires.
        mock_disamb.assert_not_called()

    data = json.loads(result_json)
    assert data["error_code"] == "unscoped_broad_query"
    assert isinstance(data["error"], str) and data["error"].strip()


# ---------------------------------------------------------------------------
# Schema + description assertions
# ---------------------------------------------------------------------------


def _get_search_tool_schema() -> dict[str, Any]:
    """Extract the live ``search`` tool's inputSchema from the MCP server."""
    import asyncio

    from mcp.types import ListToolsRequest

    from scix.mcp_server import create_server

    server = create_server(_run_self_test=False)
    handler = server.request_handlers[ListToolsRequest]
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(handler(ListToolsRequest(method="tools/list")))
    finally:
        loop.close()
    tools = result.root.tools if hasattr(result, "root") else result.tools
    search_tool = next(t for t in tools if t.name == "search")
    return {
        "schema": search_tool.inputSchema,
        "description": search_tool.description,
    }


def test_schema_has_bypass_unscoped_guard_field() -> None:
    """The search tool exposes bypass_unscoped_guard: bool = False."""
    try:
        info = _get_search_tool_schema()
    except (ImportError, AttributeError):
        pytest.skip("mcp SDK not installed or server API changed")

    props = info["schema"]["properties"]
    assert "bypass_unscoped_guard" in props, "search tool schema missing 'bypass_unscoped_guard'"
    prop = props["bypass_unscoped_guard"]
    assert prop["type"] == "boolean"
    assert prop["default"] is False
    assert prop.get("description"), "bypass_unscoped_guard must have a description"


def test_description_mentions_structured_error_behavior() -> None:
    """Description tells agents about the structured-error behavior."""
    try:
        info = _get_search_tool_schema()
    except (ImportError, AttributeError):
        pytest.skip("mcp SDK not installed or server API changed")

    desc = info["description"].lower()
    # Mentions the structured error so agents know what to expect.
    assert "unscoped_broad_query" in desc or "structured error" in desc
    # Mentions the bypass arg.
    assert "bypass_unscoped_guard" in desc


# ---------------------------------------------------------------------------
# Telemetry — _log_query surfaces the unscoped tag in error_msg
# ---------------------------------------------------------------------------


def test_log_query_surfaces_unscoped_broad_tag() -> None:
    """When result_json carries unscoped_broad_blocked=true, _log_query
    sets error_msg='unscoped_broad_query' so operators can SELECT count(*)
    FROM query_log WHERE error_msg='unscoped_broad_query' to track rate.
    """
    captured: dict[str, Any] = {}

    class _FakeCursor:
        def __enter__(self) -> "_FakeCursor":
            return self

        def __exit__(self, *_a: Any) -> None:
            pass

        def execute(self, _sql: str, params: tuple) -> None:
            # _log_query positional order:
            #   tool_name, params_json, latency_ms, success, error_msg,
            #   tool_name (dup), query, result_count, session_id, is_test
            captured["error_msg"] = params[4]
            captured["success"] = params[3]
            captured["result_count"] = params[7]

    class _FakeConn:
        def cursor(self) -> _FakeCursor:
            return _FakeCursor()

        def commit(self) -> None:
            pass

    payload = json.dumps(
        {
            "error": "Unscoped broad query rejected.",
            "error_code": "unscoped_broad_query",
            "hint": "...",
            "query": "x",
            "unscoped_broad_blocked": True,
        }
    )

    mcp_server._log_query(
        _FakeConn(),
        "search",
        {"query": "x"},
        12.3,
        True,
        None,
        result_json=payload,
    )

    assert captured["error_msg"] == "unscoped_broad_query"
    # Result count is 0 because the response carried an "error" key.
    assert captured["result_count"] == 0


def test_log_query_does_not_surface_tag_for_normal_results() -> None:
    """Normal search results don't get the unscoped_broad_query tag."""
    captured: dict[str, Any] = {}

    class _FakeCursor:
        def __enter__(self) -> "_FakeCursor":
            return self

        def __exit__(self, *_a: Any) -> None:
            pass

        def execute(self, _sql: str, params: tuple) -> None:
            captured["error_msg"] = params[4]

    class _FakeConn:
        def cursor(self) -> _FakeCursor:
            return _FakeCursor()

        def commit(self) -> None:
            pass

    payload = json.dumps({"papers": [{"bibcode": "2024ApJ...001A"}], "total": 1})

    mcp_server._log_query(
        _FakeConn(),
        "search",
        {"query": "x"},
        12.3,
        True,
        None,
        result_json=payload,
    )

    # error_msg stays None when no unscoped marker is present.
    assert captured["error_msg"] is None


# ---------------------------------------------------------------------------
# Telemetry convention pinning — bead scix_experiments-3qun
#
# These tests pin the documented query_log convention for structured-error
# responses so that any future change to _log_query / call_tool semantics
# breaks visibly (signaling a breaking change for operator dashboards).
#
# Convention (see docs/mcp_tool_audit_2026-04.md "Telemetry conventions for
# query_log"):
#   1. Exceptions raised in _dispatch_tool       -> success=False, error_msg=str(exc)
#   2. Structured errors WITH a lifted tag       -> success=True,  error_msg=<stable tag>
#                                                  (currently only "unscoped_broad_query")
#   3. Structured errors WITHOUT a lifted tag    -> success=True,  error_msg=NULL
#                                                  (e.g. missing_required_params,
#                                                   entity legacy-type rejection,
#                                                   invalid mode/action/method)
#
# Recommended operator dashboard query for blocked + failed requests:
#   SELECT * FROM query_log WHERE success = FALSE OR error_msg IS NOT NULL;
# This catches cases (1) + (2). Case (3) remains hidden until a follow-up
# bead extends the _log_query lift list with additional stable tags.
# ---------------------------------------------------------------------------


class _CaptureCursor:
    """Test double that captures the params tuple from a single execute()."""

    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    def __enter__(self) -> "_CaptureCursor":
        return self

    def __exit__(self, *_a: Any) -> None:
        pass

    def execute(self, _sql: str, params: tuple) -> None:
        # _log_query positional order:
        #   tool_name, params_json, latency_ms, success, error_msg,
        #   tool_name (dup), query, result_count, session_id, is_test
        self._captured["success"] = params[3]
        self._captured["error_msg"] = params[4]


class _CaptureConn:
    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    def cursor(self) -> _CaptureCursor:
        return _CaptureCursor(self._captured)

    def commit(self) -> None:
        pass


def test_telemetry_convention_lifted_structured_error_logs_success_true_and_tag() -> None:
    """Convention (case 2): lifted structured-error responses log success=True
    AND error_msg=<stable tag>. Pins the unscoped_broad_query case as the
    reference example. If this assertion ever flips to success=False, that's
    a breaking change for any dashboard built against the documented contract
    in docs/mcp_tool_audit_2026-04.md.
    """
    captured: dict[str, Any] = {}

    payload = json.dumps(
        {
            "error": "Unscoped broad query rejected.",
            "error_code": "unscoped_broad_query",
            "hint": "...",
            "query": "x",
            "unscoped_broad_blocked": True,
        }
    )

    mcp_server._log_query(
        _CaptureConn(captured),
        "search",
        {"query": "x"},
        12.3,
        True,  # call_tool sets success=True because no exception fired
        None,  # _log_query lifts the tag from result_json
        result_json=payload,
    )

    # Pinned convention: structured-error + lifted tag => (True, <tag>).
    assert captured["success"] is True
    assert captured["error_msg"] == "unscoped_broad_query"


def test_telemetry_convention_unlifted_structured_error_logs_success_true_and_null() -> None:
    """Convention (case 3): structured-error responses WITHOUT a lifted tag
    log success=True AND error_msg=NULL. Covers missing_required_params,
    entity legacy-type rejection, invalid-mode errors, etc.

    Documents a known dashboard blind spot: these blocked requests are NOT
    surfaced by `WHERE success=False OR error_msg IS NOT NULL`. The fix is
    a follow-up bead that extends _log_query's lift list with additional
    stable tags — NOT a silent semantics flip in _log_query.
    """
    captured: dict[str, Any] = {}

    # missing_required_params is one of the unlifted structured errors.
    payload = json.dumps(
        {
            "error": "bibcode is required when mode='graph'",
            "error_code": "missing_required_params",
            "mode": "graph",
            "required": ["bibcode"],
            "got": [],
        }
    )

    mcp_server._log_query(
        _CaptureConn(captured),
        "citation_traverse",
        {"mode": "graph"},
        4.5,
        True,  # call_tool sets success=True because no exception fired
        None,
    # No result_json kwarg: simulates the lift-skipped case for an unlifted
    # structured-error payload. Even if result_json were passed, the
    # current lift list (only unscoped_broad_blocked) would not match.
        result_json=payload,
    )

    # Pinned convention: structured-error WITHOUT lifted tag => (True, None).
    # If a future change adds an error_code-based lift, this test breaks
    # and the convention doc + operator dashboards must be updated together.
    assert captured["success"] is True
    assert captured["error_msg"] is None
