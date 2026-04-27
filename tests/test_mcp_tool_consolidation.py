"""Tests for the 2026-04-25 MCP tool consolidation pass.

This pass:

1. Merged ``citation_graph`` and ``citation_chain`` into a single
   ``citation_traverse`` tool with a ``mode`` enum (``graph`` | ``chain``).
2. Retired ``find_similar_by_examples`` (Qdrant backend not in active use).
3. Kept ``claim_blame``, ``find_replications``, and ``section_retrieval``
   as full first-class tools (they were missed by the original audit).

Final active tool count is 15. The deprecated names ``citation_graph``
and ``citation_chain`` continue to work via the ``_DEPRECATED_ALIASES``
shim and arrive at the new handler with the appropriate ``mode``
injected.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix.mcp_server import (
    _DEPRECATED_ALIASES,
    _dispatch_consolidated,
    _dispatch_tool,
    _expected_tool_set,
    EXPECTED_TOOLS,
)
from scix.search import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_result() -> SearchResult:
    return SearchResult(papers=[], total=0, timing_ms={"query_ms": 0.1})


@pytest.fixture
def mock_conn() -> MagicMock:
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    cursor.execute.return_value = None
    conn.cursor.return_value = cursor
    return conn


# ---------------------------------------------------------------------------
# AC1: citation_traverse mode='graph' is equivalent to old citation_graph
# ---------------------------------------------------------------------------


class TestCitationTraverseGraphMode:
    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_citations")
    def test_graph_mode_calls_get_citations(
        self,
        mock_cit: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_cit.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_traverse",
            {
                "mode": "graph",
                "bibcode": "2024ApJ...962L..15J",
                "direction": "forward",
                "limit": 10,
            },
        )
        data = json.loads(out)
        assert "error" not in data
        mock_cit.assert_called_once_with(
            mock_conn, "2024ApJ...962L..15J", limit=10
        )

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_references")
    def test_graph_mode_backward_calls_get_references(
        self,
        mock_ref: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_ref.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_traverse",
            {
                "mode": "graph",
                "bibcode": "2024ApJ...962L..15J",
                "direction": "backward",
                "limit": 5,
            },
        )
        data = json.loads(out)
        assert "error" not in data
        mock_ref.assert_called_once_with(
            mock_conn, "2024ApJ...962L..15J", limit=5
        )

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_citations")
    def test_default_mode_is_graph(
        self,
        mock_cit: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """When ``mode`` is omitted, citation_traverse defaults to mode='graph'."""
        mock_cit.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_traverse",
            {"bibcode": "2024ApJ...962L..15J"},
        )
        data = json.loads(out)
        assert "error" not in data
        mock_cit.assert_called_once()

    def test_graph_mode_missing_bibcode_returns_error(
        self,
        mock_conn: MagicMock,
    ) -> None:
        out = _dispatch_consolidated(
            mock_conn,
            "citation_traverse",
            {"mode": "graph"},
        )
        data = json.loads(out)
        assert "error" in data
        assert "bibcode" in data["error"]


# ---------------------------------------------------------------------------
# AC2: citation_traverse mode='chain' is equivalent to old citation_chain
# ---------------------------------------------------------------------------


class TestCitationTraverseChainMode:
    @patch("scix.mcp_server._log_query")
    @patch("scix.search.citation_chain")
    def test_chain_mode_calls_citation_chain(
        self,
        mock_cc: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_cc.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_traverse",
            {
                "mode": "chain",
                "source_bibcode": "2024ApJ...962L..15J",
                "target_bibcode": "2023ApJ...900L...1A",
                "max_depth": 3,
            },
        )
        data = json.loads(out)
        assert "error" not in data
        mock_cc.assert_called_once_with(
            mock_conn,
            "2024ApJ...962L..15J",
            "2023ApJ...900L...1A",
            max_depth=3,
        )

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.citation_chain")
    def test_chain_mode_clamps_max_depth_to_5(
        self,
        mock_cc: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_cc.return_value = _empty_result()
        _dispatch_tool(
            mock_conn,
            "citation_traverse",
            {
                "mode": "chain",
                "source_bibcode": "A",
                "target_bibcode": "B",
                "max_depth": 99,
            },
        )
        # The handler clamps to [1, 5].
        assert mock_cc.call_args.kwargs["max_depth"] == 5

    def test_chain_mode_missing_endpoints_returns_error(
        self,
        mock_conn: MagicMock,
    ) -> None:
        out = _dispatch_consolidated(
            mock_conn,
            "citation_traverse",
            {"mode": "chain", "source_bibcode": "A"},  # missing target
        )
        data = json.loads(out)
        assert "error" in data
        assert "target_bibcode" in data["error"]


# ---------------------------------------------------------------------------
# AC3: invalid mode returns a structured error
# ---------------------------------------------------------------------------


class TestCitationTraverseInvalidMode:
    def test_invalid_mode_returns_error(self, mock_conn: MagicMock) -> None:
        out = _dispatch_consolidated(
            mock_conn,
            "citation_traverse",
            {"mode": "bogus", "bibcode": "X"},
        )
        data = json.loads(out)
        assert "error" in data
        assert "bogus" in data["error"]


# ---------------------------------------------------------------------------
# AC4: deprecated aliases shim correctly through to citation_traverse
# ---------------------------------------------------------------------------


class TestDeprecatedCitationAliases:
    def test_citation_graph_in_deprecated_aliases(self) -> None:
        assert _DEPRECATED_ALIASES["citation_graph"] == "citation_traverse"

    def test_citation_chain_in_deprecated_aliases(self) -> None:
        assert _DEPRECATED_ALIASES["citation_chain"] == "citation_traverse"

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_citations")
    def test_old_citation_graph_routes_to_traverse_graph_mode(
        self,
        mock_cit: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """Old citation_graph name dispatches to citation_traverse(mode='graph')."""
        mock_cit.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_graph",
            {"bibcode": "2024ApJ...962L..15J", "direction": "forward"},
        )
        data = json.loads(out)
        # Routed: the underlying handler ran.
        mock_cit.assert_called_once()
        # Deprecation envelope is attached.
        assert data["deprecated"] is True
        assert data["use_instead"] == "citation_traverse"
        assert data["original_tool"] == "citation_graph"

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.citation_chain")
    def test_old_citation_chain_routes_to_traverse_chain_mode(
        self,
        mock_cc: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """Old citation_chain name dispatches to citation_traverse(mode='chain')."""
        mock_cc.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_chain",
            {"source_bibcode": "A", "target_bibcode": "B", "max_depth": 2},
        )
        data = json.loads(out)
        mock_cc.assert_called_once_with(
            mock_conn, "A", "B", max_depth=2
        )
        assert data["deprecated"] is True
        assert data["use_instead"] == "citation_traverse"
        assert data["original_tool"] == "citation_chain"

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_citations")
    def test_old_get_citations_routes_to_traverse_graph_forward(
        self,
        mock_cit: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """get_citations -> citation_traverse(mode='graph', direction='forward')."""
        mock_cit.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "get_citations",
            {"bibcode": "X", "limit": 5},
        )
        data = json.loads(out)
        mock_cit.assert_called_once_with(mock_conn, "X", limit=5)
        assert data["deprecated"] is True
        assert data["use_instead"] == "citation_traverse"


# ---------------------------------------------------------------------------
# AC5: find_similar_by_examples retired — dispatch returns tool_removed
# ---------------------------------------------------------------------------


class TestFindSimilarByExamplesRetired:
    def test_dispatch_returns_tool_removed_error(
        self,
        mock_conn: MagicMock,
    ) -> None:
        out = _dispatch_consolidated(
            mock_conn,
            "find_similar_by_examples",
            {"positive_bibcodes": ["X"]},
        )
        data = json.loads(out)
        assert data["error"] == "tool_removed"
        assert data["removed_in"] == "2026-04-25"
        assert "find_similar_by_examples" in data["message"]

    def test_not_in_deprecated_aliases(self) -> None:
        """find_similar_by_examples is hard-removed, not renamed."""
        assert "find_similar_by_examples" not in _DEPRECATED_ALIASES

    def test_not_in_expected_tool_set(self) -> None:
        assert "find_similar_by_examples" not in _expected_tool_set()
        assert "find_similar_by_examples" not in EXPECTED_TOOLS


# ---------------------------------------------------------------------------
# AC6: list_tools count is exactly 20 and contains the expected names
# ---------------------------------------------------------------------------


class TestListToolsCount:
    def test_expected_tools_has_20_entries(self) -> None:
        # Subsequent PRDs grew the list past the original 15: section_retrieval
        # (section-embeddings-mcp-consolidation) + 2 paper_claims retrieval
        # tools (nanopub-claim-extraction) + cited_by_intent (structural
        # citation lookup) + synthesize_findings (bead cfh9). Final = 20.
        assert len(EXPECTED_TOOLS) == 20
        assert len(set(EXPECTED_TOOLS)) == 20  # no duplicates

    def test_expected_tools_contains_citation_traverse(self) -> None:
        assert "citation_traverse" in EXPECTED_TOOLS

    def test_expected_tools_does_not_contain_old_citation_tools(self) -> None:
        assert "citation_graph" not in EXPECTED_TOOLS
        assert "citation_chain" not in EXPECTED_TOOLS

    def test_expected_tools_contains_audit_missed_keeps(self) -> None:
        """The 3 tools the original audit missed: 2 keeps + 1 retirement."""
        assert "claim_blame" in EXPECTED_TOOLS
        assert "find_replications" in EXPECTED_TOOLS
        # find_similar_by_examples is the retirement.
        assert "find_similar_by_examples" not in EXPECTED_TOOLS

    def test_list_tools_returns_20_via_self_test(self) -> None:
        """Round-trip through startup_self_test: registers exactly 20 tools."""
        try:
            import mcp.types  # noqa: F401
        except ImportError:
            pytest.skip("mcp SDK not installed")

        from scix.mcp_server import startup_self_test

        with patch("scix.mcp_server._init_model_impl"):
            status = startup_self_test()
        assert status["ok"] is True
        assert status["tool_count"] == 20
        assert "citation_traverse" in status["tool_names"]
        assert "citation_graph" not in status["tool_names"]
        assert "citation_chain" not in status["tool_names"]
        assert "find_similar_by_examples" not in status["tool_names"]
        # PRD nanopub-claim-extraction registrations are present.
        assert "read_paper_claims" in status["tool_names"]
        assert "find_claims" in status["tool_names"]
