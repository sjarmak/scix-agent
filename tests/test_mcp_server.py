"""Unit tests for the MCP server (no database or MCP SDK required)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scix.mcp_server import (
    _dispatch_tool,
    _parse_filters,
    _result_to_json,
)

# ---------------------------------------------------------------------------
# _parse_filters
# ---------------------------------------------------------------------------


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
# _dispatch_tool — unit tests with mocked DB
# ---------------------------------------------------------------------------


class TestDispatchTool:
    def test_unknown_tool_returns_error(self) -> None:
        mock_conn = MagicMock()
        result = json.loads(_dispatch_tool(mock_conn, "nonexistent_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    @patch("scix.search.lexical_search")
    def test_keyword_search_dispatches(self, mock_lexical: MagicMock) -> None:
        from scix.search import SearchResult

        mock_lexical.return_value = SearchResult(
            papers=[{"bibcode": "test"}], total=1, timing_ms={"lexical_ms": 3.0}
        )
        mock_conn = MagicMock()
        result = json.loads(_dispatch_tool(mock_conn, "keyword_search", {"terms": "dark matter"}))
        assert result["total"] == 1
        mock_lexical.assert_called_once()

    @patch("scix.search.get_paper")
    def test_get_paper_dispatches(self, mock_get: MagicMock) -> None:
        from scix.search import SearchResult

        mock_get.return_value = SearchResult(
            papers=[{"bibcode": "2024ApJ...001A"}],
            total=1,
            timing_ms={"query_ms": 1.0},
        )
        mock_conn = MagicMock()
        result = json.loads(_dispatch_tool(mock_conn, "get_paper", {"bibcode": "2024ApJ...001A"}))
        assert result["total"] == 1

    @patch("scix.search.get_citations")
    def test_get_citations_dispatches(self, mock_cit: MagicMock) -> None:
        from scix.search import SearchResult

        mock_cit.return_value = SearchResult(papers=[], total=0, timing_ms={"query_ms": 2.0})
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(mock_conn, "get_citations", {"bibcode": "X", "limit": 5})
        )
        assert result["total"] == 0
        mock_cit.assert_called_once_with(mock_conn, "X", limit=5)

    @patch("scix.search.facet_counts")
    def test_facet_counts_dispatches(self, mock_facets: MagicMock) -> None:
        from scix.search import SearchResult

        mock_facets.return_value = SearchResult(papers=[], total=0, timing_ms={"query_ms": 1.0})
        mock_conn = MagicMock()
        _dispatch_tool(mock_conn, "facet_counts", {"field": "year"})
        mock_facets.assert_called_once()


# ---------------------------------------------------------------------------
# _dispatch_tool — graph tool dispatch tests
# ---------------------------------------------------------------------------


class TestDispatchGraphTools:
    @patch("scix.search.co_citation_analysis")
    def test_co_citation_dispatches(self, mock_fn: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fn.return_value = SearchResult(
            papers=[{"bibcode": "A", "overlap_count": 5}],
            total=1,
            timing_ms={"query_ms": 10.0},
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(mock_conn, "co_citation_analysis", {"bibcode": "X", "min_overlap": 3})
        )
        assert result["total"] == 1
        mock_fn.assert_called_once_with(mock_conn, "X", min_overlap=3, limit=20)

    @patch("scix.search.bibliographic_coupling")
    def test_bibliographic_coupling_dispatches(self, mock_fn: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fn.return_value = SearchResult(
            papers=[{"bibcode": "B", "shared_refs": 3}],
            total=1,
            timing_ms={"query_ms": 8.0},
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(mock_conn, "bibliographic_coupling", {"bibcode": "Y", "limit": 10})
        )
        assert result["total"] == 1
        mock_fn.assert_called_once_with(mock_conn, "Y", min_overlap=2, limit=10)

    @patch("scix.search.citation_chain")
    def test_citation_chain_dispatches(self, mock_fn: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fn.return_value = SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": 5.0},
            metadata={"path_length": -1, "path_bibcodes": []},
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "citation_chain",
                {"source_bibcode": "A", "target_bibcode": "B", "max_depth": 3},
            )
        )
        assert result["metadata"]["path_length"] == -1
        mock_fn.assert_called_once_with(mock_conn, "A", "B", max_depth=3)

    @patch("scix.search.citation_chain")
    def test_citation_chain_caps_depth_at_5(self, mock_fn: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fn.return_value = SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": 1.0},
            metadata={"path_length": -1, "path_bibcodes": []},
        )
        mock_conn = MagicMock()
        _dispatch_tool(
            mock_conn,
            "citation_chain",
            {"source_bibcode": "A", "target_bibcode": "B", "max_depth": 99},
        )
        mock_fn.assert_called_once_with(mock_conn, "A", "B", max_depth=5)

    @patch("scix.search.temporal_evolution")
    def test_temporal_evolution_dispatches(self, mock_fn: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fn.return_value = SearchResult(
            papers=[],
            total=2,
            timing_ms={"query_ms": 3.0},
            metadata={"mode": "citations", "yearly_counts": [{"year": 2023, "count": 10}]},
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "temporal_evolution",
                {"bibcode_or_query": "X", "year_start": 2020},
            )
        )
        assert result["metadata"]["mode"] == "citations"
        mock_fn.assert_called_once_with(mock_conn, "X", year_start=2020, year_end=None)


# ---------------------------------------------------------------------------
# _init_model
# ---------------------------------------------------------------------------


class TestInitModel:
    @patch("scix.mcp_server._init_model_impl")
    def test_init_model_called_at_create(self, mock_init: MagicMock) -> None:
        """create_server should call _init_model_impl for eager model load."""
        try:
            from scix.mcp_server import create_server

            create_server()
            mock_init.assert_called_once()
        except ImportError:
            pytest.skip("mcp SDK not installed")

    def test_init_model_impl_calls_load_model(self) -> None:
        """_init_model_impl should call load_model."""
        from scix.mcp_server import _init_model_impl

        with patch("scix.mcp_server.load_model") as mock_load:
            _init_model_impl()
            mock_load.assert_called_once_with("specter2", device="cpu")

    def test_init_model_impl_survives_import_error(self) -> None:
        """_init_model_impl should not crash if torch/transformers missing."""
        from scix.mcp_server import _init_model_impl

        with patch("scix.mcp_server.load_model", side_effect=ImportError("no torch")):
            # Should not raise
            _init_model_impl()


# ---------------------------------------------------------------------------
# health_check tool dispatch
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health_check_returns_status(self) -> None:
        """health_check tool should return structured status."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = json.loads(_dispatch_tool(mock_conn, "health_check", {}))
        assert "db" in result
        assert "model_cached" in result
        assert "pool" in result

    def test_health_check_db_failure(self) -> None:
        """health_check should report DB errors gracefully."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("connection refused")
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = json.loads(_dispatch_tool(mock_conn, "health_check", {}))
        assert result["db"] == "error"


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
        """_shutdown should not crash when no pool exists."""
        from scix import mcp_server

        original_pool = mcp_server._pool
        mcp_server._pool = None

        try:
            mcp_server._shutdown()  # Should not raise
        finally:
            mcp_server._pool = original_pool
