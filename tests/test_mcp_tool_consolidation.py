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
        mock_cit.assert_called_once_with(mock_conn, "2024ApJ...962L..15J", limit=10)

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
        mock_ref.assert_called_once_with(mock_conn, "2024ApJ...962L..15J", limit=5)

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
        mock_cc.assert_called_once_with(mock_conn, "A", "B", max_depth=2)
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
        """Round-trip through startup_self_test: registers exactly 20 tools.

        EXPECTED_TOOLS has 20 entries, but list_tools() drops the
        ``_HIDDEN_TOOLS`` set (default: section_retrieval, read_paper_claims,
        find_claims — backing data not yet populated). The visible tool count
        equals ``len(_expected_tool_set())``, which collapses to 17 with the
        defaults above. Test against that derived expectation rather than a
        hardcoded number so future hide/unhide changes don't churn this test.
        """
        try:
            import mcp.types  # noqa: F401
        except ImportError:
            pytest.skip("mcp SDK not installed")

        from scix.mcp_server import _expected_tool_set, startup_self_test

        expected_visible = _expected_tool_set()
        with patch("scix.mcp_server._init_model_impl"):
            status = startup_self_test()
        assert status["ok"] is True
        assert status["tool_count"] == len(expected_visible)
        assert "citation_traverse" in status["tool_names"]
        assert "citation_graph" not in status["tool_names"]
        assert "citation_chain" not in status["tool_names"]
        assert "find_similar_by_examples" not in status["tool_names"]
        # PRD bead cfh9: synthesize_findings is not env-hidden, so it must
        # appear in the visible self-test surface.
        assert "synthesize_findings" in status["tool_names"]


# ---------------------------------------------------------------------------
# Bead scix_experiments-4c5v: no LIVE tool description should reference the
# retired ``citation_graph`` tool name. The deprecation alias entry in
# ``_DEPRECATED_ALIASES`` and internal handler names (``_handle_citation_graph``)
# are infrastructure, not agent-visible surface, and are explicitly allowed.
# ---------------------------------------------------------------------------


def _list_tools_via_server() -> list[Any]:
    """Drive the registered ``list_tools()`` handler and return the Tool list.

    Mirrors the dispatch in ``startup_self_test`` so we exercise the same
    registration surface that agents see, not a hand-rolled snapshot.
    """
    import asyncio

    pytest.importorskip("mcp.types")
    from mcp.types import ListToolsRequest

    from scix.mcp_server import create_server

    with patch("scix.mcp_server._init_model_impl"):
        server = create_server(_run_self_test=False)

    handler = server.request_handlers[ListToolsRequest]
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(handler(ListToolsRequest(method="tools/list")))
    finally:
        loop.close()

    if hasattr(result, "root") and hasattr(result.root, "tools"):
        return list(result.root.tools)
    if hasattr(result, "tools"):
        return list(result.tools)
    raise RuntimeError(f"unexpected list_tools result shape: {result!r}")


class TestNoLiveCitationGraphDescriptionReferences:
    def test_no_tool_description_mentions_citation_graph(self) -> None:
        """Agent-visible descriptions must point at ``citation_traverse``.

        The 28→13 consolidation in commit 7fe258d renamed ``citation_graph``
        to ``citation_traverse`` (with ``mode='graph'``). Any live tool
        description that still mentions ``citation_graph`` is a phantom
        breadcrumb — agents that follow it will hit the deprecation shim
        instead of routing directly to the modern tool.
        """
        tools = _list_tools_via_server()
        offenders: list[tuple[str, str]] = []
        for tool in tools:
            description = getattr(tool, "description", "") or ""
            if "citation_graph" in description:
                offenders.append((tool.name, description))
        assert not offenders, (
            "Live tool descriptions reference the retired tool 'citation_graph'. "
            "Replace with 'citation_traverse' (or 'citation_traverse(mode=\"graph\")' "
            f"where the distinction matters). Offending tools: "
            f"{[name for name, _ in offenders]}"
        )


# ---------------------------------------------------------------------------
# Bead scix_experiments-unmm: every tool that accepts a ``resolution``
# parameter must agree on the default ('medium' — the production-quality
# partition per CLAUDE.md community_labels_pipeline memory).
# ---------------------------------------------------------------------------


_RESOLUTION_DEFAULT = "medium"


class TestResolutionDefaultUnified:
    def test_graph_context_resolution_default_is_medium(self) -> None:
        tools = {t.name: t for t in _list_tools_via_server()}
        schema = tools["graph_context"].inputSchema
        assert schema["properties"]["resolution"]["default"] == _RESOLUTION_DEFAULT

    def test_find_gaps_resolution_default_is_medium(self) -> None:
        tools = {t.name: t for t in _list_tools_via_server()}
        schema = tools["find_gaps"].inputSchema
        assert schema["properties"]["resolution"]["default"] == _RESOLUTION_DEFAULT

    def test_all_tools_with_resolution_share_same_default(self) -> None:
        """Cross-tool consistency check: every tool exposing ``resolution``
        must default to the same value so back-to-back agent calls don't
        silently mismatch on partition granularity."""
        tools = _list_tools_via_server()
        defaults: dict[str, str] = {}
        for tool in tools:
            schema = getattr(tool, "inputSchema", None) or {}
            props = schema.get("properties") or {}
            res = props.get("resolution")
            if not isinstance(res, dict) or "default" not in res:
                continue
            defaults[tool.name] = res["default"]
        assert defaults, "Expected at least one tool to expose 'resolution'"
        unique_defaults = set(defaults.values())
        assert unique_defaults == {_RESOLUTION_DEFAULT}, (
            f"Inconsistent 'resolution' defaults across tools: {defaults}. "
            f"All must default to {_RESOLUTION_DEFAULT!r}."
        )

    def test_handle_graph_context_defaults_resolution_to_medium(self) -> None:
        """The handler's runtime default must match the schema default.

        ``_handle_graph_context`` reads ``args.get('resolution', ...)`` to
        cover callers that bypass schema validation (e.g. internal in-process
        dispatch). It must default to the same value advertised in the schema.
        """
        from scix.search import SearchResult

        with patch("scix.search.get_paper_metrics") as mock_metrics, patch(
            "scix.search.explore_community"
        ) as mock_explore:
            mock_metrics.return_value = SearchResult(
                papers=[], total=0, timing_ms={"query_ms": 1.0}
            )
            mock_explore.return_value = SearchResult(
                papers=[], total=0, timing_ms={"query_ms": 1.0}
            )
            from scix.mcp_server import _handle_graph_context

            _handle_graph_context(
                MagicMock(),
                {"bibcode": "X", "include_community": True},
            )
            assert mock_explore.call_args.kwargs["resolution"] == _RESOLUTION_DEFAULT
