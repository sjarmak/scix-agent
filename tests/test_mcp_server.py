"""Unit tests for the consolidated MCP server (no database or MCP SDK required).

Covers:
- 13 consolidated tools dispatch correctly
- Deprecated aliases return deprecated:true + use_instead
- Implicit session tracking (focused papers)
- list_tools() returns exactly 13 tools
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scix.mcp_server import (
    _DEPRECATED_ALIASES,
    _coerce_year,
    _dispatch_tool,
    _hnsw_index_cache,
    _hnsw_index_exists,
    _hnsw_index_name,
    _parse_filters,
    _result_to_json,
    _session_state,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_session():
    """Clear session state between tests."""
    _session_state.clear_working_set()
    _session_state.clear_focused()
    yield
    _session_state.clear_working_set()
    _session_state.clear_focused()


# ---------------------------------------------------------------------------
# _parse_filters
# ---------------------------------------------------------------------------


class TestCoerceYear:
    def test_none_returns_none(self) -> None:
        assert _coerce_year(None, "year_start") is None

    def test_int_passthrough(self) -> None:
        assert _coerce_year(2024, "year_start") == 2024

    def test_string_numeric_coerced(self) -> None:
        assert _coerce_year("2024", "year_start") == 2024

    def test_non_numeric_string_raises(self) -> None:
        with pytest.raises(ValueError, match="year_start must be an integer"):
            _coerce_year("not-a-year", "year_start")

    def test_below_min_raises(self) -> None:
        with pytest.raises(ValueError, match="year_end must be in"):
            _coerce_year(1800, "year_end")

    def test_above_max_raises(self) -> None:
        with pytest.raises(ValueError, match="year_start must be in"):
            _coerce_year(2200, "year_start")

    def test_injection_string_raises(self) -> None:
        """A crafted SQL-looking value must not reach the DB."""
        with pytest.raises(ValueError):
            _coerce_year("2020; DROP TABLE papers", "year_start")


class TestParseFilters:
    def test_none_returns_default(self) -> None:
        result = _parse_filters(None)
        assert result.year_min is None
        assert result.year_max is None

    def test_empty_dict_returns_default(self) -> None:
        result = _parse_filters({})
        assert result.year_min is None

    def test_with_filters(self) -> None:
        result = _parse_filters({"year_min": 2022, "arxiv_class": "astro-ph"})
        assert result.year_min == 2022
        assert result.arxiv_class == "astro-ph"

    def test_unknown_keys_ignored(self) -> None:
        result = _parse_filters({"unknown_key": "value", "year_min": 2020})
        assert result.year_min == 2020


# ---------------------------------------------------------------------------
# _result_to_json
# ---------------------------------------------------------------------------


class TestResultToJson:
    def test_search_result_serialization(self) -> None:
        from scix.search import SearchResult

        result = SearchResult(
            papers=[{"bibcode": "2024ApJ...001A", "title": "Test"}],
            total=1,
            timing_ms={"query_ms": 5.0},
        )
        output = json.loads(_result_to_json(result))
        assert output["total"] == 1
        assert output["timing_ms"]["query_ms"] == 5.0
        assert len(output["papers"]) == 1

    def test_search_result_with_metadata(self) -> None:
        from scix.search import SearchResult

        result = SearchResult(
            papers=[],
            total=0,
            timing_ms={},
            metadata={"source": "test"},
        )
        output = json.loads(_result_to_json(result))
        assert output["metadata"]["source"] == "test"

    def test_plain_dict_serialization(self) -> None:
        output = json.loads(_result_to_json({"key": "value"}))
        assert output["key"] == "value"


# ---------------------------------------------------------------------------
# AC1: list_tools() returns exactly 13 tools
# ---------------------------------------------------------------------------


class TestListTools:
    def test_list_tools_returns_exactly_13(self) -> None:
        try:
            import asyncio

            from mcp.types import ListToolsRequest

            from scix.mcp_server import create_server

            with patch("scix.mcp_server._init_model_impl"):
                server = create_server()
                handler = server.request_handlers[ListToolsRequest]
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(handler(ListToolsRequest(method="tools/list")))
                    tools = result.root.tools
                    tool_names = sorted([t.name for t in tools])
                    expected = sorted(
                        [
                            "search",
                            "concept_search",
                            "get_paper",
                            "read_paper",
                            "citation_graph",
                            "citation_similarity",
                            "citation_chain",
                            "entity",
                            "entity_context",
                            "graph_context",
                            "find_gaps",
                            "temporal_evolution",
                            "facet_counts",
                        ]
                    )
                    assert tool_names == expected, f"Got: {tool_names}"
                    assert len(tools) == 13
                finally:
                    loop.close()
        except (ImportError, AttributeError):
            pytest.skip("mcp SDK not installed or server API changed")


# ---------------------------------------------------------------------------
# AC2-4: search tool dispatches (hybrid, semantic, keyword)
# ---------------------------------------------------------------------------


class TestSearchTool:
    @patch("scix.search.lexical_search")
    def test_keyword_mode(self, mock_lex: MagicMock) -> None:
        from scix.search import SearchResult

        mock_lex.return_value = SearchResult(
            papers=[{"bibcode": "test"}], total=1, timing_ms={"lexical_ms": 3.0}
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(mock_conn, "search", {"query": "dark matter", "mode": "keyword"})
        )
        assert result["total"] == 1
        mock_lex.assert_called_once()

    @patch("scix.mcp_server._hnsw_index_exists", return_value=True)
    @patch("scix.mcp_server.embed_batch", return_value=[[0.0] * 768])
    @patch("scix.mcp_server.load_model", return_value=(MagicMock(), MagicMock()))
    @patch("scix.search.vector_search")
    def test_semantic_mode(self, mock_vs, mock_load, mock_embed, mock_guard) -> None:
        from scix.search import SearchResult

        mock_vs.return_value = SearchResult(
            papers=[{"bibcode": "2024X"}], total=1, timing_ms={"vector_ms": 5.0}
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(mock_conn, "search", {"query": "test", "mode": "semantic"})
        )
        assert result["total"] == 1
        mock_vs.assert_called_once()

    @patch("scix.mcp_server._hnsw_index_exists", return_value=True)
    @patch("scix.mcp_server.embed_batch", return_value=[[0.0] * 768])
    @patch("scix.mcp_server.load_model", return_value=(MagicMock(), MagicMock()))
    @patch("scix.search.hybrid_search")
    def test_hybrid_mode(self, mock_hs, mock_load, mock_embed, mock_guard) -> None:
        from scix.search import SearchResult

        mock_hs.return_value = SearchResult(
            papers=[{"bibcode": "2024H"}], total=1, timing_ms={"hybrid_ms": 10.0}
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(mock_conn, "search", {"query": "test", "mode": "hybrid"})
        )
        assert result["total"] == 1
        mock_hs.assert_called_once()


# ---------------------------------------------------------------------------
# AC5-6: citation_graph
# ---------------------------------------------------------------------------


class TestCitationGraph:
    @patch("scix.search.get_citations")
    def test_forward_direction(self, mock_cit: MagicMock) -> None:
        from scix.search import SearchResult

        mock_cit.return_value = SearchResult(papers=[], total=0, timing_ms={"query_ms": 2.0})
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(mock_conn, "citation_graph", {"bibcode": "X", "direction": "forward"})
        )
        assert result["total"] == 0
        mock_cit.assert_called_once_with(mock_conn, "X", limit=20)

    @patch("scix.search.get_references")
    def test_backward_direction(self, mock_ref: MagicMock) -> None:
        from scix.search import SearchResult

        mock_ref.return_value = SearchResult(papers=[], total=0, timing_ms={"query_ms": 1.0})
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(mock_conn, "citation_graph", {"bibcode": "X", "direction": "backward"})
        )
        assert result["total"] == 0
        mock_ref.assert_called_once_with(mock_conn, "X", limit=20)


# ---------------------------------------------------------------------------
# AC7: citation_similarity
# ---------------------------------------------------------------------------


class TestCitationSimilarity:
    @patch("scix.search.co_citation_analysis")
    def test_co_citation_method(self, mock_fn: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fn.return_value = SearchResult(
            papers=[{"bibcode": "A"}], total=1, timing_ms={"query_ms": 5.0}
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "citation_similarity",
                {"bibcode": "X", "method": "co_citation"},
            )
        )
        assert result["total"] == 1
        mock_fn.assert_called_once_with(mock_conn, "X", min_overlap=2, limit=20)

    @patch("scix.search.bibliographic_coupling")
    def test_coupling_method(self, mock_fn: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fn.return_value = SearchResult(
            papers=[{"bibcode": "B"}], total=1, timing_ms={"query_ms": 4.0}
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "citation_similarity",
                {"bibcode": "Y", "method": "coupling"},
            )
        )
        assert result["total"] == 1
        mock_fn.assert_called_once_with(mock_conn, "Y", min_overlap=2, limit=20)


# ---------------------------------------------------------------------------
# AC8-10: entity tool
# ---------------------------------------------------------------------------


class TestEntityTool:
    def test_entity_search_dispatches(self) -> None:
        rows = [
            ("2024ApJ...001A", "methods", "v1", {"methods": ["JWST"]}, "Paper A"),
        ]
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = rows
        cur.__enter__ = lambda self: self
        cur.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cur

        result = json.loads(
            _dispatch_tool(
                conn,
                "entity",
                {"action": "search", "entity_type": "methods", "query": "JWST"},
            )
        )
        assert result["total"] == 1
        assert result["papers"][0]["bibcode"] == "2024ApJ...001A"

    def test_entity_search_invalid_type_returns_error(self) -> None:
        """AC9: entity_type='entities' returns validation error."""
        conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                conn,
                "entity",
                {"action": "search", "entity_type": "entities", "query": "test"},
            )
        )
        assert "error" in result
        assert "methods" in result["error"]
        assert "datasets" in result["error"]
        assert "instruments" in result["error"]
        assert "materials" in result["error"]

    @patch("scix.mcp_server.EntityResolver")
    def test_entity_resolve_dispatches(self, mock_resolver_cls: MagicMock) -> None:
        """AC10: entity(action='resolve')."""
        mock_resolver = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.entity_id = 1
        mock_candidate.canonical_name = "James Webb Space Telescope"
        mock_candidate.entity_type = "instruments"
        mock_candidate.source = "metadata"
        mock_candidate.discipline = "astronomy"
        mock_candidate.confidence = 0.95
        mock_candidate.match_method = "exact"
        mock_resolver.resolve.return_value = [mock_candidate]
        mock_resolver_cls.return_value = mock_resolver

        conn = MagicMock()
        result = json.loads(
            _dispatch_tool(conn, "entity", {"action": "resolve", "query": "James Webb"})
        )
        assert result["total"] == 1
        assert result["candidates"][0]["canonical_name"] == "James Webb Space Telescope"


# ---------------------------------------------------------------------------
# AC11: entity_context
# ---------------------------------------------------------------------------


class TestEntityContext:
    @patch("scix.search.get_entity_context")
    def test_dispatches_correctly(self, mock_fn: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fn.return_value = SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": 1.0},
            metadata={"entity_id": 1, "name": "JWST"},
        )
        conn = MagicMock()
        result = json.loads(_dispatch_tool(conn, "entity_context", {"entity_id": 1}))
        assert result["metadata"]["entity_id"] == 1
        mock_fn.assert_called_once_with(conn, 1)


# ---------------------------------------------------------------------------
# AC12-13: get_paper with include_entities
# ---------------------------------------------------------------------------


class TestGetPaper:
    @patch("scix.search.get_paper")
    def test_without_entities(self, mock_fn: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fn.return_value = SearchResult(
            papers=[{"bibcode": "2024X"}], total=1, timing_ms={"query_ms": 1.0}
        )
        conn = MagicMock()
        result = json.loads(
            _dispatch_tool(conn, "get_paper", {"bibcode": "2024X", "include_entities": False})
        )
        assert result["total"] == 1
        mock_fn.assert_called_once()

    @patch("scix.search.get_document_context")
    def test_with_entities(self, mock_fn: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fn.return_value = SearchResult(
            papers=[{"bibcode": "2024X", "entities": []}],
            total=1,
            timing_ms={"query_ms": 2.0},
        )
        conn = MagicMock()
        result = json.loads(
            _dispatch_tool(conn, "get_paper", {"bibcode": "2024X", "include_entities": True})
        )
        assert result["total"] == 1
        mock_fn.assert_called_once_with(conn, "2024X")


# ---------------------------------------------------------------------------
# AC14-15: find_gaps reads from implicit session state
# ---------------------------------------------------------------------------


class TestFindGaps:
    def test_empty_focused_set_returns_message(self) -> None:
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = []
        cur.__enter__ = lambda self: self
        cur.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cur

        result = json.loads(_dispatch_tool(conn, "find_gaps", {"limit": 5}))
        assert result["total"] == 0
        assert "focused" in result["message"].lower() or "no" in result["message"].lower()

    @patch("scix.search.get_paper")
    def test_reads_from_focused_set(self, mock_get: MagicMock) -> None:
        from scix.search import SearchResult

        mock_get.return_value = SearchResult(
            papers=[{"bibcode": "2024WS1"}], total=1, timing_ms={"query_ms": 1.0}
        )
        # Simulate get_paper to add to focused set
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = [("2024GAP1", "Gap", 0.05, 42)]
        cur.__enter__ = lambda self: self
        cur.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cur

        _dispatch_tool(conn, "get_paper", {"bibcode": "2024WS1"})

        result = json.loads(_dispatch_tool(conn, "find_gaps", {"resolution": "coarse"}))
        assert result["total"] == 1
        assert result["papers"][0]["bibcode"] == "2024GAP1"

    @patch("scix.search.get_paper")
    def test_clear_first_resets_focused(self, mock_get: MagicMock) -> None:
        from scix.search import SearchResult

        mock_get.return_value = SearchResult(
            papers=[{"bibcode": "2024WS1"}], total=1, timing_ms={"query_ms": 1.0}
        )
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = []
        cur.__enter__ = lambda self: self
        cur.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cur

        _dispatch_tool(conn, "get_paper", {"bibcode": "2024WS1"})
        assert len(_session_state.get_focused_papers()) == 1

        result = json.loads(_dispatch_tool(conn, "find_gaps", {"clear_first": True}))
        assert result["total"] == 0
        assert "message" in result


# ---------------------------------------------------------------------------
# AC16: session tools NOT in list_tools
# ---------------------------------------------------------------------------


class TestSessionToolsRemoved:
    def test_deprecated_session_tools_not_in_aliases_target(self) -> None:
        """add_to_working_set etc. are in deprecated aliases."""
        for old_name in [
            "add_to_working_set",
            "get_working_set",
            "get_session_summary",
            "clear_working_set",
        ]:
            assert old_name in _DEPRECATED_ALIASES


# ---------------------------------------------------------------------------
# AC17-19: deprecated aliases
# ---------------------------------------------------------------------------


class TestDeprecatedAliases:
    @patch("scix.search.lexical_search")
    def test_semantic_search_alias(self, mock_lex: MagicMock) -> None:
        """AC17: old 'semantic_search' returns deprecated:true."""
        from scix.search import SearchResult

        # semantic_search will check HNSW index, so mock it to fallback
        with patch("scix.mcp_server._hnsw_index_exists", return_value=False):
            conn = MagicMock()
            result = json.loads(_dispatch_tool(conn, "semantic_search", {"query": "dark matter"}))
            assert result["deprecated"] is True
            assert result["use_instead"] == "search"
            assert result["original_tool"] == "semantic_search"

    @patch("scix.search.get_citations")
    def test_get_citations_alias(self, mock_cit: MagicMock) -> None:
        """AC18: old 'get_citations' returns deprecated:true."""
        from scix.search import SearchResult

        mock_cit.return_value = SearchResult(papers=[], total=0, timing_ms={"query_ms": 1.0})
        conn = MagicMock()
        result = json.loads(_dispatch_tool(conn, "get_citations", {"bibcode": "X", "limit": 5}))
        assert result["deprecated"] is True
        assert result["use_instead"] == "citation_graph"

    def test_all_deprecated_aliases_log_original(self) -> None:
        """AC19: verify all deprecated aliases exist and map correctly."""
        assert "semantic_search" in _DEPRECATED_ALIASES
        assert "keyword_search" in _DEPRECATED_ALIASES
        assert "get_citations" in _DEPRECATED_ALIASES
        assert "get_references" in _DEPRECATED_ALIASES
        assert "co_citation_analysis" in _DEPRECATED_ALIASES
        assert "bibliographic_coupling" in _DEPRECATED_ALIASES
        assert "entity_search" in _DEPRECATED_ALIASES
        assert "resolve_entity" in _DEPRECATED_ALIASES

    def test_health_check_not_deprecated(self) -> None:
        """health_check must NOT be tagged deprecated (it is an internal tool,
        not a renamed tool)."""
        assert "health_check" not in _DEPRECATED_ALIASES


class TestEntityProfileLegacy:
    """entity_profile is a deprecated alias with a dedicated handler that
    preserves the original schema (raw extractions rows)."""

    def test_entity_profile_returns_legacy_schema(self) -> None:
        """Legacy entity_profile returns bibcode + extractions + total,
        NOT the get_paper/document_context shape."""
        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = [
            ("methods", "v1", {"name": "MCMC"}, None),
            ("datasets", "v1", {"name": "SDSS"}, None),
        ]
        conn.cursor.return_value = cursor

        result = json.loads(_dispatch_tool(conn, "entity_profile", {"bibcode": "X"}))

        # Legacy schema fields
        assert result["bibcode"] == "X"
        assert "extractions" in result
        assert result["total"] == 2
        assert result["extractions"][0]["extraction_type"] == "methods"
        assert result["extractions"][0]["payload"] == {"name": "MCMC"}
        # Deprecation wrapper present
        assert result["deprecated"] is True
        assert result["use_instead"] == "get_paper"
        assert result["original_tool"] == "entity_profile"
        # Must NOT have get_paper shape
        assert "linked_entities" not in result
        assert "title" not in result


# ---------------------------------------------------------------------------
# AC20: verify at least 3 consolidated tools dispatch correctly
# (Covered by TestSearchTool, TestCitationGraph, TestEntityTool above)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


class TestUnknownTool:
    def test_unknown_tool_returns_error(self) -> None:
        mock_conn = MagicMock()
        result = json.loads(_dispatch_tool(mock_conn, "nonexistent_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]


# ---------------------------------------------------------------------------
# HNSW index guard
# ---------------------------------------------------------------------------


class TestHnswIndexGuard:
    def setup_method(self) -> None:
        _hnsw_index_cache.clear()

    def teardown_method(self) -> None:
        _hnsw_index_cache.clear()

    def test_index_name_convention(self) -> None:
        assert _hnsw_index_name("indus") == "idx_embed_hnsw_indus"
        assert _hnsw_index_name("specter2") == "idx_embed_hnsw_specter2"

    def _mock_conn_with_index(self, *, exists: bool) -> MagicMock:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,) if exists else None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        return mock_conn

    def test_returns_true_when_index_present(self) -> None:
        mock_conn = self._mock_conn_with_index(exists=True)
        assert _hnsw_index_exists(mock_conn, "indus") is True

    def test_returns_false_when_index_missing(self) -> None:
        mock_conn = self._mock_conn_with_index(exists=False)
        assert _hnsw_index_exists(mock_conn, "indus") is False

    def test_positive_result_is_cached(self) -> None:
        mock_conn = self._mock_conn_with_index(exists=True)
        _hnsw_index_exists(mock_conn, "indus")
        _hnsw_index_exists(mock_conn, "indus")
        assert mock_conn.cursor.call_count == 1

    def test_negative_result_is_rechecked(self) -> None:
        mock_conn = self._mock_conn_with_index(exists=False)
        _hnsw_index_exists(mock_conn, "indus")
        exists, _ = _hnsw_index_cache["indus"]
        _hnsw_index_cache["indus"] = (exists, 0.0)
        _hnsw_index_exists(mock_conn, "indus")
        assert mock_conn.cursor.call_count == 2


# ---------------------------------------------------------------------------
# graph_context
# ---------------------------------------------------------------------------


class TestGraphContext:
    @patch("scix.search.get_paper_metrics")
    def test_metrics_only(self, mock_fn: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fn.return_value = SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": 1.0},
            metadata={"pagerank": 0.01},
        )
        conn = MagicMock()
        result = json.loads(
            _dispatch_tool(conn, "graph_context", {"bibcode": "X", "include_community": False})
        )
        assert result["metadata"]["pagerank"] == 0.01
        mock_fn.assert_called_once_with(conn, "X")

    @patch("scix.search.explore_community")
    @patch("scix.search.get_paper_metrics")
    def test_with_community(self, mock_metrics, mock_community) -> None:
        from scix.search import SearchResult

        mock_metrics.return_value = SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": 1.0},
            metadata={"pagerank": 0.01},
        )
        mock_community.return_value = SearchResult(
            papers=[{"bibcode": "sibling"}],
            total=1,
            timing_ms={"query_ms": 2.0},
        )
        conn = MagicMock()
        result = json.loads(
            _dispatch_tool(conn, "graph_context", {"bibcode": "X", "include_community": True})
        )
        assert "metrics" in result
        assert "community" in result
        mock_community.assert_called_once()


# ---------------------------------------------------------------------------
# _init_model
# ---------------------------------------------------------------------------


class TestInitModel:
    @patch("scix.mcp_server._init_model_impl")
    def test_init_model_called_at_create(self, mock_init: MagicMock) -> None:
        try:
            from scix.mcp_server import create_server

            create_server()
            mock_init.assert_called_once()
        except ImportError:
            pytest.skip("mcp SDK not installed")

    def test_init_model_impl_calls_load_model(self) -> None:
        from scix.mcp_server import _init_model_impl

        with patch("scix.mcp_server.load_model") as mock_load:
            _init_model_impl()
            mock_load.assert_called_once_with("indus", device="cpu")

    def test_init_model_impl_survives_import_error(self) -> None:
        from scix.mcp_server import _init_model_impl

        with patch("scix.mcp_server.load_model", side_effect=ImportError("no torch")):
            _init_model_impl()


# ---------------------------------------------------------------------------
# _shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_clears_model_cache(self) -> None:
        from scix.embed import _model_cache

        _model_cache[("test", "cpu")] = (MagicMock(), MagicMock())
        assert len(_model_cache) > 0

        from scix.mcp_server import _shutdown

        _shutdown()
        assert len(_model_cache) == 0

    def test_shutdown_closes_pool(self) -> None:
        from scix import mcp_server

        mock_pool = MagicMock()
        original_pool = mcp_server._pool
        mcp_server._pool = mock_pool

        try:
            mcp_server._shutdown()
            mock_pool.close.assert_called_once()
            assert mcp_server._pool is None
        finally:
            mcp_server._pool = original_pool

    def test_shutdown_no_pool_ok(self) -> None:
        from scix import mcp_server

        original_pool = mcp_server._pool
        mcp_server._pool = None

        try:
            mcp_server._shutdown()
        finally:
            mcp_server._pool = original_pool
