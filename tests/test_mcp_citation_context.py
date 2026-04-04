"""Tests for get_citation_context in search.py and MCP dispatch."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scix.search import SearchResult, get_citation_context
from scix.mcp_server import _dispatch_tool

# ---------------------------------------------------------------------------
# search.get_citation_context — unit tests with mocked DB
# ---------------------------------------------------------------------------


class TestGetCitationContext:
    """Tests for search.get_citation_context()."""

    def _make_conn(self, rows: list[tuple]) -> MagicMock:
        """Create a mock connection that returns the given rows."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        return mock_conn

    def test_existing_pair_returns_context(self) -> None:
        """A pair with context should return the context_text, intent, and char_offset."""
        rows = [
            {
                "context_text": "As shown by Smith et al. (2020), dark matter...",
                "intent": "background",
                "char_offset": 1234,
            }
        ]
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        result = get_citation_context(mock_conn, "2024ApJ...001A", "2020ApJ...002B")

        assert isinstance(result, SearchResult)
        assert result.total == 1
        assert len(result.papers) == 1
        assert result.papers[0]["context_text"] == "As shown by Smith et al. (2020), dark matter..."
        assert result.papers[0]["intent"] == "background"
        assert result.papers[0]["char_offset"] == 1234
        assert "query_ms" in result.timing_ms

    def test_nonexistent_pair_returns_empty(self) -> None:
        """A pair without context should return empty result, not an error."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        result = get_citation_context(mock_conn, "2024ApJ...001A", "2020ApJ...999Z")

        assert isinstance(result, SearchResult)
        assert result.total == 0
        assert result.papers == []
        assert "query_ms" in result.timing_ms

    def test_multiple_contexts_for_same_pair(self) -> None:
        """A pair cited multiple times should return all context rows."""
        rows = [
            {
                "context_text": "First mention in introduction...",
                "intent": "background",
                "char_offset": 500,
            },
            {
                "context_text": "Second mention in methods...",
                "intent": "method",
                "char_offset": 3200,
            },
            {
                "context_text": "Third mention in discussion...",
                "intent": "result",
                "char_offset": 8900,
            },
        ]
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        result = get_citation_context(mock_conn, "2024ApJ...001A", "2020ApJ...002B")

        assert result.total == 3
        assert len(result.papers) == 3
        assert result.papers[0]["intent"] == "background"
        assert result.papers[1]["intent"] == "method"
        assert result.papers[2]["intent"] == "result"

    def test_null_intent_and_offset(self) -> None:
        """Context rows with null intent/char_offset are returned as-is."""
        rows = [
            {
                "context_text": "Some citation text...",
                "intent": None,
                "char_offset": None,
            }
        ]
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        result = get_citation_context(mock_conn, "2024ApJ...001A", "2020ApJ...002B")

        assert result.total == 1
        assert result.papers[0]["intent"] is None
        assert result.papers[0]["char_offset"] is None


# ---------------------------------------------------------------------------
# MCP dispatch — get_citation_context
# ---------------------------------------------------------------------------


class TestDispatchCitationContext:
    """Tests for MCP _dispatch_tool routing to get_citation_context."""

    @patch("scix.search.get_citation_context")
    def test_dispatch_routes_to_search(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = SearchResult(
            papers=[
                {
                    "context_text": "citation text",
                    "intent": "background",
                    "char_offset": 100,
                }
            ],
            total=1,
            timing_ms={"query_ms": 2.0},
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "get_citation_context",
                {
                    "source_bibcode": "2024ApJ...001A",
                    "target_bibcode": "2020ApJ...002B",
                },
            )
        )
        assert result["total"] == 1
        assert result["papers"][0]["context_text"] == "citation text"
        mock_fn.assert_called_once_with(mock_conn, "2024ApJ...001A", "2020ApJ...002B")

    @patch("scix.search.get_citation_context")
    def test_dispatch_empty_result(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = SearchResult(papers=[], total=0, timing_ms={"query_ms": 1.0})
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "get_citation_context",
                {
                    "source_bibcode": "2024ApJ...001A",
                    "target_bibcode": "2020ApJ...999Z",
                },
            )
        )
        assert result["total"] == 0
        assert result["papers"] == []
