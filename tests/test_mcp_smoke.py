"""Smoke tests for all 13 consolidated MCP tools.

These tests catch total breakage of any tool on every deploy. They:

1. Verify ``startup_self_test`` succeeds against a freshly-created server
   (13 tools, valid schemas).
2. Call each of the 13 consolidated tools via ``_dispatch_tool`` with a
   minimal golden-path input and assert the returned JSON is a valid
   non-error response (no exception raised, no top-level ``error`` key).

The dispatch layer talks to ``scix.search`` functions directly — we mock
those at the module boundary so the smoke tests do not require database
state. Real integration tests live elsewhere; this suite is about
catching routing breakage and schema drift.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix.mcp_server import (
    EXPECTED_TOOLS,
    _dispatch_tool,
    _session_state,
    startup_self_test,
)
from scix.search import SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_result() -> SearchResult:
    """Empty but structurally-valid SearchResult for mocking."""
    return SearchResult(papers=[], total=0, timing_ms={"query_ms": 0.1})


def _assert_non_error(result_json: str, tool: str) -> dict[str, Any]:
    """Assert that the JSON response has no top-level 'error' key."""
    data = json.loads(result_json)
    assert isinstance(
        data, (dict, list)
    ), f"{tool}: response is not a dict or list, got {type(data).__name__}"
    if isinstance(data, dict):
        assert (
            "error" not in data
        ), f"{tool}: response contains top-level 'error': {data.get('error')}"
    return data if isinstance(data, dict) else {"items": data}


@pytest.fixture(autouse=True)
def _reset_session() -> Any:
    """Clear implicit session state between tests."""
    _session_state.clear_working_set()
    _session_state.clear_focused()
    yield
    _session_state.clear_working_set()
    _session_state.clear_focused()


@pytest.fixture
def mock_conn() -> MagicMock:
    """A MagicMock standing in for a psycopg connection."""
    conn = MagicMock()
    # Cursor context manager for tools that hit the DB directly (entity search).
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    cursor.execute.return_value = None
    conn.cursor.return_value = cursor
    return conn


# ---------------------------------------------------------------------------
# AC1-3: startup_self_test
# ---------------------------------------------------------------------------


class TestStartupSelfTest:
    """Validates the server's self-test catches missing/broken tools."""

    def test_expected_tools_has_13_entries(self) -> None:
        assert len(EXPECTED_TOOLS) == 13
        assert len(set(EXPECTED_TOOLS)) == 13  # no duplicates

    def test_self_test_passes_on_fresh_server(self) -> None:
        """A freshly created server must pass the self-test."""
        try:
            import mcp.types  # noqa: F401
        except ImportError:
            pytest.skip("mcp SDK not installed")

        with patch("scix.mcp_server._init_model_impl"):
            status = startup_self_test()

        assert status["ok"] is True
        assert status["tool_count"] == 13
        assert status["errors"] == []
        assert sorted(EXPECTED_TOOLS) == status["tool_names"]

    def test_self_test_raises_on_wrong_tool_count(self) -> None:
        """If list_tools() returns the wrong count, self-test raises."""
        try:
            from mcp.types import ListToolsRequest, ListToolsResult, Tool
        except ImportError:
            pytest.skip("mcp SDK not installed")

        # Build a fake server where the list_tools handler returns 12 tools.
        fake_server = MagicMock()
        bad_tools = [
            Tool(
                name=f"tool_{i}",
                description="x",
                inputSchema={"type": "object", "properties": {}},
            )
            for i in range(12)
        ]

        async def bad_handler(_req: Any) -> Any:
            return ListToolsResult(tools=bad_tools)

        fake_server.request_handlers = {ListToolsRequest: bad_handler}

        with pytest.raises(RuntimeError, match="startup_self_test failed"):
            startup_self_test(server=fake_server)

    def test_self_test_raises_on_missing_expected_tool(self) -> None:
        """If an expected tool name is missing, self-test raises."""
        try:
            from mcp.types import ListToolsRequest, ListToolsResult, Tool
        except ImportError:
            pytest.skip("mcp SDK not installed")

        # 13 tools but one expected name replaced with a bogus one.
        swapped = list(EXPECTED_TOOLS)
        swapped[0] = "not_a_real_tool"
        bad_tools = [
            Tool(
                name=n,
                description="x",
                inputSchema={"type": "object", "properties": {}},
            )
            for n in swapped
        ]

        async def bad_handler(_req: Any) -> Any:
            return ListToolsResult(tools=bad_tools)

        fake_server = MagicMock()
        fake_server.request_handlers = {ListToolsRequest: bad_handler}

        with pytest.raises(RuntimeError, match="missing expected tools"):
            startup_self_test(server=fake_server)


# ---------------------------------------------------------------------------
# AC4-5: One smoke test per consolidated tool
# ---------------------------------------------------------------------------


class TestToolSmoke:
    """Golden-path smoke test for each of the 13 consolidated tools."""

    @patch("scix.mcp_server._log_query")
    @patch(
        "scix.mcp_server._hnsw_index_exists",
        return_value=True,
    )
    @patch(
        "scix.mcp_server.embed_batch",
        return_value=[[0.0] * 768],
    )
    @patch(
        "scix.mcp_server.load_model",
        return_value=(MagicMock(), MagicMock()),
    )
    @patch("scix.search.hybrid_search")
    def test_search(
        self,
        mock_hybrid: MagicMock,
        _mock_load: MagicMock,
        _mock_embed: MagicMock,
        _mock_guard: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_hybrid.return_value = _empty_result()
        out = _dispatch_tool(mock_conn, "search", {"query": "dark matter"})
        _assert_non_error(out, "search")

    @patch("scix.mcp_server._log_query")
    @patch(
        "scix.mcp_server._hnsw_index_exists",
        return_value=True,
    )
    @patch(
        "scix.mcp_server.embed_batch",
        return_value=[[0.0] * 768],
    )
    @patch(
        "scix.mcp_server.load_model",
        return_value=(MagicMock(), MagicMock()),
    )
    @patch("scix.search.hybrid_search")
    def test_search_with_entity_filters(
        self,
        mock_hybrid: MagicMock,
        _mock_load: MagicMock,
        _mock_embed: MagicMock,
        _mock_guard: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """xz4.1.27: MCP search with entity_types + entity_ids propagates to hybrid_search."""
        mock_hybrid.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "search",
            {
                "query": "papers about JWST instruments",
                "filters": {
                    "entity_types": ["instrument"],
                    "entity_ids": [27867],
                },
            },
        )
        _assert_non_error(out, "search")
        # Confirm the filter threaded through to hybrid_search.
        called_filters = mock_hybrid.call_args.kwargs["filters"]
        assert called_filters.entity_types == ("instrument",)
        assert called_filters.entity_ids == (27867,)

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.concept_search")
    def test_concept_search(
        self,
        mock_cs: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_cs.return_value = _empty_result()
        out = _dispatch_tool(mock_conn, "concept_search", {"query": "Galaxies"})
        _assert_non_error(out, "concept_search")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_paper")
    def test_get_paper(
        self,
        mock_gp: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_gp.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "get_paper",
            {"bibcode": "2024ApJ...962L..15J"},
        )
        _assert_non_error(out, "get_paper")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.read_paper_section")
    def test_read_paper(
        self,
        mock_rps: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_rps.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "read_paper",
            {"bibcode": "2024ApJ...962L..15J", "section": "introduction"},
        )
        _assert_non_error(out, "read_paper")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_citations")
    def test_citation_graph(
        self,
        mock_cit: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_cit.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_graph",
            {"bibcode": "2024ApJ...962L..15J", "direction": "forward"},
        )
        _assert_non_error(out, "citation_graph")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.co_citation_analysis")
    def test_citation_similarity(
        self,
        mock_cca: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_cca.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_similarity",
            {"bibcode": "2024ApJ...962L..15J", "method": "co_citation"},
        )
        _assert_non_error(out, "citation_similarity")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.citation_chain")
    def test_citation_chain(
        self,
        mock_cc: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_cc.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_chain",
            {
                "source_bibcode": "2024ApJ...962L..15J",
                "target_bibcode": "2023ApJ...900L...1A",
                "max_depth": 3,
            },
        )
        _assert_non_error(out, "citation_chain")

    @patch("scix.mcp_server._log_query")
    def test_entity(
        self,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        # entity action=search hits conn.cursor() directly; mock_conn's
        # cursor already returns an empty fetchall, so we get an empty-but-
        # valid response.
        out = _dispatch_tool(
            mock_conn,
            "entity",
            {"action": "search", "entity_type": "methods", "query": "MCMC"},
        )
        _assert_non_error(out, "entity")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_entity_context")
    def test_entity_context(
        self,
        mock_gec: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_gec.return_value = _empty_result()
        out = _dispatch_tool(mock_conn, "entity_context", {"entity_id": 1})
        _assert_non_error(out, "entity_context")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_paper_metrics")
    def test_graph_context(
        self,
        mock_gpm: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_gpm.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "graph_context",
            {"bibcode": "2024ApJ...962L..15J"},
        )
        _assert_non_error(out, "graph_context")

    @patch("scix.mcp_server._log_query")
    def test_find_gaps(
        self,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        # find_gaps queries via conn.cursor(); mock_conn returns an empty
        # fetchall, yielding {"papers": [], "total": 0, ...}.
        out = _dispatch_tool(
            mock_conn,
            "find_gaps",
            {"resolution": "coarse", "limit": 5},
        )
        _assert_non_error(out, "find_gaps")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.temporal_evolution")
    def test_temporal_evolution(
        self,
        mock_te: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_te.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "temporal_evolution",
            {"bibcode_or_query": "dark matter"},
        )
        _assert_non_error(out, "temporal_evolution")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.facet_counts")
    def test_facet_counts(
        self,
        mock_fc: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_fc.return_value = _empty_result()
        out = _dispatch_tool(mock_conn, "facet_counts", {"field": "year"})
        _assert_non_error(out, "facet_counts")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.facet_counts")
    def test_facet_counts_threads_entity_filters(
        self,
        mock_fc: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """xz4.1.27 HIGH follow-up: facet_counts must propagate entity filters, not drop them."""
        mock_fc.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "facet_counts",
            {
                "field": "year",
                "filters": {"entity_types": ["instrument"], "entity_ids": [27867]},
            },
        )
        _assert_non_error(out, "facet_counts")
        called_filters = mock_fc.call_args.kwargs["filters"]
        assert called_filters.entity_types == ("instrument",)
        assert called_filters.entity_ids == (27867,)

    @patch("scix.mcp_server._log_query")
    @patch(
        "scix.mcp_server._hnsw_index_exists",
        return_value=True,
    )
    @patch(
        "scix.mcp_server.embed_batch",
        return_value=[[0.0] * 768],
    )
    @patch(
        "scix.mcp_server.load_model",
        return_value=(MagicMock(), MagicMock()),
    )
    @patch("scix.search.hybrid_search")
    def test_search_returns_json_error_for_bad_entity_filter(
        self,
        mock_hybrid: MagicMock,
        _mock_load: MagicMock,
        _mock_embed: MagicMock,
        _mock_guard: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """xz4.1.27 HIGH follow-up: invalid filters must return JSON {"error": ...}, not raise."""
        out = _dispatch_tool(
            mock_conn,
            "search",
            {"query": "dark matter", "filters": {"entity_types": "instrument"}},
        )
        data = json.loads(out)
        assert "error" in data
        assert "entity_types" in data["error"]
        mock_hybrid.assert_not_called()

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.facet_counts")
    def test_facet_counts_returns_json_error_for_bad_entity_filter(
        self,
        mock_fc: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """xz4.1.27 HIGH follow-up: invalid filters on facet_counts also return JSON error."""
        out = _dispatch_tool(
            mock_conn,
            "facet_counts",
            {"field": "year", "filters": {"entity_ids": ["not-an-int"]}},
        )
        data = json.loads(out)
        assert "error" in data
        assert "entity_ids" in data["error"]
        mock_fc.assert_not_called()


# ---------------------------------------------------------------------------
# Meta: ensure every expected tool has a smoke test
# ---------------------------------------------------------------------------


def test_every_expected_tool_has_a_smoke_test() -> None:
    """Guard: if the expected tool list changes, this test reminds us."""
    smoke_test_methods = {
        name[len("test_") :] for name in dir(TestToolSmoke) if name.startswith("test_")
    }
    missing = set(EXPECTED_TOOLS) - smoke_test_methods
    assert not missing, f"Missing smoke tests for: {sorted(missing)}"
