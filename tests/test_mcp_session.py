"""Tests for MCP server entity and session tools.

Covers all 7 new tools (entity_search, entity_profile, add_to_working_set,
get_working_set, get_session_summary, find_gaps, clear_working_set) and
the in_working_set annotation on paper-returning tools.
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
    yield
    mcp_server._session_state.clear_working_set()


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
# entity_search
# ---------------------------------------------------------------------------


class TestEntitySearch:
    def test_returns_papers_with_containment_query(self) -> None:
        rows = [
            (
                "2024ApJ...001A",
                "entities",
                "v1",
                {"entities": ["dark matter"]},
                "Dark Matter Paper",
            ),
        ]
        conn = _make_conn(rows)
        result_json = mcp_server._dispatch_tool(
            conn,
            "entity_search",
            {"entity_type": "instruments", "entity_name": "dark matter", "limit": 10},
        )
        result = json.loads(result_json)
        assert result["total"] == 1
        assert result["papers"][0]["bibcode"] == "2024ApJ...001A"
        assert "in_working_set" in result["papers"][0]

    def test_sql_uses_containment_operator(self) -> None:
        conn = _make_conn([])
        mcp_server._dispatch_tool(
            conn,
            "entity_search",
            {"entity_type": "methods", "entity_name": "MCMC"},
        )
        cur = conn.cursor.return_value
        call_args = cur.execute.call_args
        sql = call_args[0][0]
        assert "@>" in sql
        params = call_args[0][1]
        # Containment JSON should be {"methods": ["MCMC"]}
        assert json.loads(params[0]) == {"methods": ["MCMC"]}


# ---------------------------------------------------------------------------
# entity_profile
# ---------------------------------------------------------------------------


class TestEntityProfile:
    def test_returns_extractions_for_bibcode(self) -> None:
        rows = [
            ("entities", "v1", {"entities": ["star"]}, "2024-01-01T00:00:00"),
            ("keywords", "v1", {"keywords": ["stellar"]}, "2024-01-01T00:00:00"),
        ]
        conn = _make_conn(rows)
        result_json = mcp_server._dispatch_tool(
            conn, "entity_profile", {"bibcode": "2024ApJ...002B"}
        )
        result = json.loads(result_json)
        assert result["bibcode"] == "2024ApJ...002B"
        assert result["total"] == 2
        assert result["extractions"][0]["extraction_type"] == "entities"

    def test_empty_extractions(self) -> None:
        conn = _make_conn([])
        result_json = mcp_server._dispatch_tool(conn, "entity_profile", {"bibcode": "NONEXISTENT"})
        result = json.loads(result_json)
        assert result["total"] == 0
        assert result["extractions"] == []


# ---------------------------------------------------------------------------
# add_to_working_set
# ---------------------------------------------------------------------------


class TestAddToWorkingSet:
    def test_adds_multiple_bibcodes(self) -> None:
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(
            conn,
            "add_to_working_set",
            {
                "bibcodes": ["2024A", "2024B"],
                "source_tool": "keyword_search",
                "source_context": "dark energy",
            },
        )
        result = json.loads(result_json)
        assert result["added"] == 2
        assert len(result["entries"]) == 2
        assert result["entries"][0]["bibcode"] == "2024A"

    def test_delegates_to_session_state(self) -> None:
        conn = _make_conn()
        mcp_server._dispatch_tool(
            conn,
            "add_to_working_set",
            {"bibcodes": ["2024X"], "source_tool": "test"},
        )
        assert mcp_server._session_state.is_in_working_set("2024X")


# ---------------------------------------------------------------------------
# get_working_set
# ---------------------------------------------------------------------------


class TestGetWorkingSet:
    def test_empty_working_set(self) -> None:
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "get_working_set", {})
        result = json.loads(result_json)
        assert result["total"] == 0
        assert result["entries"] == []

    def test_returns_added_entries(self) -> None:
        mcp_server._session_state.add_to_working_set(bibcode="2024Z", source_tool="test")
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "get_working_set", {})
        result = json.loads(result_json)
        assert result["total"] == 1
        assert result["entries"][0]["bibcode"] == "2024Z"


# ---------------------------------------------------------------------------
# get_session_summary
# ---------------------------------------------------------------------------


class TestGetSessionSummary:
    def test_returns_summary(self) -> None:
        mcp_server._session_state.add_to_working_set(bibcode="2024S", source_tool="test")
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "get_session_summary", {})
        result = json.loads(result_json)
        assert result["working_set_size"] == 1
        assert result["seen_papers_count"] >= 1
        assert "session_id" in result


# ---------------------------------------------------------------------------
# find_gaps
# ---------------------------------------------------------------------------


class TestFindGaps:
    def test_empty_working_set_returns_message(self) -> None:
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "find_gaps", {"limit": 5})
        result = json.loads(result_json)
        assert result["total"] == 0
        assert "empty" in result["message"].lower()

    def test_returns_gap_papers(self) -> None:
        mcp_server._session_state.add_to_working_set(bibcode="2024WS1", source_tool="test")
        rows = [
            ("2024GAP1", "Gap Paper 1", 0.05, 42),
            ("2024GAP2", "Gap Paper 2", 0.03, 43),
        ]
        conn = _make_conn(rows)
        result_json = mcp_server._dispatch_tool(
            conn, "find_gaps", {"resolution": "coarse", "limit": 10}
        )
        result = json.loads(result_json)
        assert result["total"] == 2
        assert result["papers"][0]["bibcode"] == "2024GAP1"
        assert result["papers"][0]["community_id"] == 42
        assert "in_working_set" in result["papers"][0]
        assert result["resolution"] == "coarse"

    def test_invalid_resolution_returns_error(self) -> None:
        mcp_server._session_state.add_to_working_set(bibcode="2024WS1", source_tool="test")
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "find_gaps", {"resolution": "invalid"})
        result = json.loads(result_json)
        assert "error" in result


# ---------------------------------------------------------------------------
# clear_working_set
# ---------------------------------------------------------------------------


class TestClearWorkingSet:
    def test_clears_and_returns_count(self) -> None:
        mcp_server._session_state.add_to_working_set(bibcode="2024C1", source_tool="test")
        mcp_server._session_state.add_to_working_set(bibcode="2024C2", source_tool="test")
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "clear_working_set", {})
        result = json.loads(result_json)
        assert result["removed"] == 2
        assert mcp_server._session_state.get_working_set() == []


# ---------------------------------------------------------------------------
# in_working_set annotation
# ---------------------------------------------------------------------------


class TestInWorkingSetAnnotation:
    def test_annotation_on_search_result(self) -> None:
        """Verify _result_to_json annotates papers with in_working_set."""
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

    def test_entity_search_annotates_results(self) -> None:
        mcp_server._session_state.add_to_working_set(bibcode="2024ApJ...001A", source_tool="test")
        rows = [
            ("2024ApJ...001A", "entities", "v1", {}, "Paper A"),
            ("2024ApJ...002B", "entities", "v1", {}, "Paper B"),
        ]
        conn = _make_conn(rows)
        result_json = mcp_server._dispatch_tool(
            conn,
            "entity_search",
            {"entity_type": "instruments", "entity_name": "test"},
        )
        result = json.loads(result_json)
        assert result["papers"][0]["in_working_set"] is True
        assert result["papers"][1]["in_working_set"] is False
