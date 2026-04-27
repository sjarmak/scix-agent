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
def _disable_disambiguator() -> None:
    """Patch the disambiguator so the search path never short-circuits."""
    with patch("scix.mcp_server.disambiguate_query", return_value=[]):
        yield


@pytest.fixture(autouse=True)
def _disable_hnsw() -> None:
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
    assert data["error"] == "unscoped_broad_query"
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
    assert data["error"] == "unscoped_broad_query"


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
    assert data["error"] == "unscoped_broad_query"


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
    assert data["error"] == "unscoped_broad_query"


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
    assert data["error"] == "unscoped_broad_query"


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
            "error": "unscoped_broad_query",
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
