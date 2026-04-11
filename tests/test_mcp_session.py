"""Tests for MCP server session tracking and entity tools (consolidated API).

Covers:
- entity tool (search + resolve actions)
- Implicit session tracking (focused/seen papers)
- find_gaps with implicit state
- Deprecated session tool aliases
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scix import mcp_server

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_session_state():
    """Clear session state between tests."""
    mcp_server._session_state.clear_working_set()
    mcp_server._session_state.clear_focused()
    yield
    mcp_server._session_state.clear_working_set()
    mcp_server._session_state.clear_focused()


def _make_conn(rows: list[tuple] | None = None) -> MagicMock:
    """Create a mock psycopg connection returning *rows* from fetchall."""
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchall.return_value = rows or []
    cur.fetchone.return_value = (1,)
    cur.__enter__ = lambda self: self
    cur.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cur
    return conn


# ---------------------------------------------------------------------------
# entity(action=search)
# ---------------------------------------------------------------------------


class TestEntitySearch:
    def test_returns_papers_with_containment_query(self) -> None:
        rows = [
            (
                "2024ApJ...001A",
                "instruments",
                "v1",
                {"instruments": ["dark matter"]},
                "Dark Matter Paper",
            ),
        ]
        conn = _make_conn(rows)
        result_json = mcp_server._dispatch_tool(
            conn,
            "entity",
            {"action": "search", "entity_type": "instruments", "query": "dark matter", "limit": 10},
        )
        result = json.loads(result_json)
        assert result["total"] == 1
        assert result["papers"][0]["bibcode"] == "2024ApJ...001A"
        assert "in_working_set" in result["papers"][0]

    def test_sql_uses_containment_operator(self) -> None:
        conn = _make_conn([])
        mcp_server._dispatch_tool(
            conn,
            "entity",
            {"action": "search", "entity_type": "methods", "query": "MCMC"},
        )
        cur = conn.cursor.return_value
        call_args = cur.execute.call_args
        sql = call_args[0][0]
        assert "@>" in sql
        params = call_args[0][1]
        assert json.loads(params[0]) == {"methods": ["MCMC"]}

    def test_invalid_entity_type_returns_error(self) -> None:
        conn = _make_conn([])
        result_json = mcp_server._dispatch_tool(
            conn,
            "entity",
            {"action": "search", "entity_type": "entities", "query": "test"},
        )
        result = json.loads(result_json)
        assert "error" in result
        assert "methods" in result["error"]
        assert "datasets" in result["error"]


# ---------------------------------------------------------------------------
# Implicit session tracking
# ---------------------------------------------------------------------------


class TestImplicitTracking:
    @patch("scix.search.get_paper")
    def test_get_paper_adds_to_focused(self, mock_get: MagicMock) -> None:
        from scix.search import SearchResult

        mock_get.return_value = SearchResult(
            papers=[{"bibcode": "2024X"}], total=1, timing_ms={"query_ms": 1.0}
        )
        conn = _make_conn()
        mcp_server._dispatch_tool(conn, "get_paper", {"bibcode": "2024X"})
        assert "2024X" in mcp_server._session_state.get_focused_papers()

    @patch("scix.search.lexical_search")
    def test_search_tracks_seen_bibcodes(self, mock_lex: MagicMock) -> None:
        from scix.search import SearchResult

        mock_lex.return_value = SearchResult(
            papers=[{"bibcode": "2024A"}, {"bibcode": "2024B"}],
            total=2,
            timing_ms={"lexical_ms": 3.0},
        )
        conn = _make_conn()
        mcp_server._dispatch_tool(conn, "search", {"query": "test", "mode": "keyword"})
        data = mcp_server._session_state._get("_default")
        assert "2024A" in data.seen_papers
        assert "2024B" in data.seen_papers


# ---------------------------------------------------------------------------
# find_gaps with implicit state
# ---------------------------------------------------------------------------


class TestFindGapsImplicit:
    def test_empty_returns_message(self) -> None:
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "find_gaps", {"limit": 5})
        result = json.loads(result_json)
        assert result["total"] == 0
        assert "message" in result

    @patch("scix.search.get_paper")
    def test_uses_focused_papers(self, mock_get: MagicMock) -> None:
        from scix.search import SearchResult

        mock_get.return_value = SearchResult(
            papers=[{"bibcode": "2024WS1"}], total=1, timing_ms={"query_ms": 1.0}
        )
        conn = _make_conn()

        # Focus a paper via get_paper
        mcp_server._dispatch_tool(conn, "get_paper", {"bibcode": "2024WS1"})

        # Now find_gaps should use it
        rows = [("2024GAP1", "Gap Paper", 0.05, 42)]
        conn2 = _make_conn(rows)
        result_json = mcp_server._dispatch_tool(conn2, "find_gaps", {"resolution": "coarse"})
        result = json.loads(result_json)
        assert result["total"] == 1
        assert result["papers"][0]["bibcode"] == "2024GAP1"

    @patch("scix.search.get_paper")
    def test_clear_first_resets(self, mock_get: MagicMock) -> None:
        from scix.search import SearchResult

        mock_get.return_value = SearchResult(
            papers=[{"bibcode": "2024WS1"}], total=1, timing_ms={"query_ms": 1.0}
        )
        conn = _make_conn()
        mcp_server._dispatch_tool(conn, "get_paper", {"bibcode": "2024WS1"})

        result_json = mcp_server._dispatch_tool(conn, "find_gaps", {"clear_first": True})
        result = json.loads(result_json)
        assert result["total"] == 0

    def test_invalid_resolution_returns_error(self) -> None:
        mcp_server._session_state.track_focused("2024WS1")
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "find_gaps", {"resolution": "invalid"})
        result = json.loads(result_json)
        assert "error" in result


# ---------------------------------------------------------------------------
# Deprecated session tool aliases
# ---------------------------------------------------------------------------


class TestDeprecatedSessionTools:
    def test_add_to_working_set_still_works(self) -> None:
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(
            conn,
            "add_to_working_set",
            {"bibcodes": ["2024A", "2024B"], "source_tool": "test"},
        )
        result = json.loads(result_json)
        assert result["deprecated"] is True
        assert result["added"] == 2

    def test_get_working_set_still_works(self) -> None:
        mcp_server._session_state.add_to_working_set(bibcode="2024Z", source_tool="test")
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "get_working_set", {})
        result = json.loads(result_json)
        assert result["deprecated"] is True
        assert result["total"] == 1

    def test_clear_working_set_still_works(self) -> None:
        mcp_server._session_state.add_to_working_set(bibcode="2024C1", source_tool="test")
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "clear_working_set", {})
        result = json.loads(result_json)
        assert result["deprecated"] is True
        assert result["removed"] == 1


# ---------------------------------------------------------------------------
# in_working_set annotation
# ---------------------------------------------------------------------------


class TestInWorkingSetAnnotation:
    def test_annotation_on_search_result(self) -> None:
        from scix.search import SearchResult

        mcp_server._session_state.add_to_working_set(bibcode="2024IN", source_tool="test")
        result = SearchResult(
            papers=[
                {"bibcode": "2024IN", "title": "In WS"},
                {"bibcode": "2024OUT", "title": "Not in WS"},
            ],
            total=2,
            timing_ms={"query": 1.0},
        )
        output = json.loads(mcp_server._result_to_json(result))
        assert output["papers"][0]["in_working_set"] is True
        assert output["papers"][1]["in_working_set"] is False
