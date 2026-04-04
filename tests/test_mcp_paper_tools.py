"""Tests for read_paper_section and search_within_paper MCP tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scix.mcp_server import _dispatch_tool
from scix.search import SearchResult, read_paper_section, search_within_paper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BODY = """Introduction
This paper studies dark matter halos in galaxy clusters.
We present new observations from the Hubble Space Telescope.

Methods
We used spectroscopic analysis of 500 galaxies.
The data was reduced using standard IRAF pipelines.

Results
We find a strong correlation between halo mass and cluster richness.
The best-fit relation has a slope of 1.3 +/- 0.2.

Conclusions
Our results confirm previous findings and extend them to higher redshifts.
"""

SAMPLE_ABSTRACT = "We study dark matter halos in galaxy clusters using HST observations."


def _mock_cursor_with_row(row):
    """Create a mock connection whose cursor returns a single row."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = row
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


# ---------------------------------------------------------------------------
# read_paper_section — unit tests
# ---------------------------------------------------------------------------


class TestReadPaperSection:
    def test_paper_with_body_full(self) -> None:
        """Reading full body returns the entire text with has_body=True."""
        row = {"body": SAMPLE_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is True
        assert paper["section_name"] == "full"
        assert paper["bibcode"] == "2024ApJ...001A"
        assert "dark matter" in paper["section_text"]
        assert paper["total_chars"] == len(SAMPLE_BODY)

    def test_paper_with_body_specific_section(self) -> None:
        """Reading a specific section returns only that section's text."""
        row = {"body": SAMPLE_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A", section="introduction")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is True
        assert paper["section_name"] == "introduction"
        assert "dark matter halos" in paper["section_text"]
        # Should NOT contain methods content
        assert "spectroscopic analysis" not in paper["section_text"]

    def test_paper_with_body_section_not_found(self) -> None:
        """Requesting a non-existent section returns empty with available sections."""
        row = {"body": SAMPLE_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A", section="acknowledgments")

        assert result.total == 0
        assert "available_sections" in result.metadata
        assert result.metadata["has_body"] is True

    def test_fallback_to_abstract(self) -> None:
        """Paper without body falls back to abstract with has_body=False."""
        row = {"body": None, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is False
        assert paper["section_name"] == "abstract"
        assert "dark matter" in paper["section_text"]
        assert result.metadata["has_body"] is False

    def test_fallback_to_abstract_empty_body(self) -> None:
        """Paper with empty string body falls back to abstract."""
        row = {"body": "", "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is False

    def test_pagination_char_offset(self) -> None:
        """Pagination with char_offset and limit works correctly."""
        row = {"body": SAMPLE_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A", char_offset=10, limit=50)

        assert result.total == 1
        paper = result.papers[0]
        assert paper["char_offset"] == 10
        assert len(paper["section_text"]) <= 50
        assert paper["total_chars"] == len(SAMPLE_BODY)

    def test_paper_not_found(self) -> None:
        """Non-existent bibcode returns empty result with error."""
        conn = _mock_cursor_with_row(None)

        result = read_paper_section(conn, "NONEXISTENT")

        assert result.total == 0
        assert "error" in result.metadata


# ---------------------------------------------------------------------------
# search_within_paper — unit tests
# ---------------------------------------------------------------------------


class TestSearchWithinPaper:
    def test_matching_query(self) -> None:
        """Search with matching terms returns headline with has_body=True."""
        row = {
            "bibcode": "2024ApJ...001A",
            "title": "Test Paper",
            "body": SAMPLE_BODY,
            "headline": "...studies <b>dark matter</b> halos in galaxy clusters...",
        }
        conn = _mock_cursor_with_row(row)

        result = search_within_paper(conn, "2024ApJ...001A", "dark matter")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is True
        assert "dark matter" in paper["headline"]
        assert paper["bibcode"] == "2024ApJ...001A"

    def test_no_body_text(self) -> None:
        """Paper without body returns empty with has_body=False metadata."""
        # First query (body search) returns None, second (existence check) returns paper
        mock_conn = MagicMock()

        mock_cursor_1 = MagicMock()
        mock_cursor_1.fetchone.return_value = None
        mock_cursor_1.__enter__ = MagicMock(return_value=mock_cursor_1)
        mock_cursor_1.__exit__ = MagicMock(return_value=False)

        mock_cursor_2 = MagicMock()
        mock_cursor_2.fetchone.return_value = {
            "bibcode": "2024ApJ...001A",
            "title": "Test Paper",
        }
        mock_cursor_2.__enter__ = MagicMock(return_value=mock_cursor_2)
        mock_cursor_2.__exit__ = MagicMock(return_value=False)

        mock_conn.cursor.side_effect = [mock_cursor_1, mock_cursor_2]

        result = search_within_paper(mock_conn, "2024ApJ...001A", "dark matter")

        assert result.total == 0
        assert result.metadata["has_body"] is False

    def test_paper_not_found(self) -> None:
        """Non-existent paper returns empty with error metadata."""
        mock_conn = MagicMock()

        mock_cursor_1 = MagicMock()
        mock_cursor_1.fetchone.return_value = None
        mock_cursor_1.__enter__ = MagicMock(return_value=mock_cursor_1)
        mock_cursor_1.__exit__ = MagicMock(return_value=False)

        mock_cursor_2 = MagicMock()
        mock_cursor_2.fetchone.return_value = None
        mock_cursor_2.__enter__ = MagicMock(return_value=mock_cursor_2)
        mock_cursor_2.__exit__ = MagicMock(return_value=False)

        mock_conn.cursor.side_effect = [mock_cursor_1, mock_cursor_2]

        result = search_within_paper(mock_conn, "NONEXISTENT", "dark matter")

        assert result.total == 0
        assert "error" in result.metadata


# ---------------------------------------------------------------------------
# MCP dispatch tests
# ---------------------------------------------------------------------------


class TestDispatchPaperTools:
    @patch("scix.search.read_paper_section")
    def test_read_paper_section_dispatches(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = SearchResult(
            papers=[
                {
                    "bibcode": "2024ApJ...001A",
                    "section_name": "full",
                    "section_text": "body text",
                    "has_body": True,
                    "char_offset": 0,
                    "total_chars": 100,
                }
            ],
            total=1,
            timing_ms={"query_ms": 2.0},
            metadata={"has_body": True},
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "read_paper_section",
                {
                    "bibcode": "2024ApJ...001A",
                    "section": "introduction",
                    "char_offset": 100,
                    "limit": 3000,
                },
            )
        )
        assert result["total"] == 1
        mock_fn.assert_called_once_with(
            mock_conn,
            "2024ApJ...001A",
            section="introduction",
            char_offset=100,
            limit=3000,
        )

    @patch("scix.search.read_paper_section")
    def test_read_paper_section_defaults(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = SearchResult(papers=[], total=0, timing_ms={"query_ms": 1.0})
        mock_conn = MagicMock()
        _dispatch_tool(
            mock_conn,
            "read_paper_section",
            {"bibcode": "2024ApJ...001A"},
        )
        mock_fn.assert_called_once_with(
            mock_conn,
            "2024ApJ...001A",
            section="full",
            char_offset=0,
            limit=5000,
        )

    @patch("scix.search.search_within_paper")
    def test_search_within_paper_dispatches(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = SearchResult(
            papers=[
                {
                    "bibcode": "2024ApJ...001A",
                    "headline": "<b>dark matter</b> halos",
                    "has_body": True,
                }
            ],
            total=1,
            timing_ms={"query_ms": 5.0},
            metadata={"has_body": True},
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "search_within_paper",
                {"bibcode": "2024ApJ...001A", "query": "dark matter"},
            )
        )
        assert result["total"] == 1
        assert "dark matter" in result["papers"][0]["headline"]
        mock_fn.assert_called_once_with(
            mock_conn,
            "2024ApJ...001A",
            "dark matter",
        )
