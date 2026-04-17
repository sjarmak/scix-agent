"""Tests for ADR-006 runtime guard in read_paper_section and search_within_paper.

Covers the scenario where papers.body could contain LaTeX-derived text
(from ar5iv or arxiv_local). The guard checks papers_fulltext.source for
the bibcode and, if the source is LaTeX-derived, applies snippet budget
enforcement before returning body text to the user.

Currently papers.body is ADS-only, so this is defense-in-depth. If ar5iv
text is ever promoted to papers.body, the guard prevents ADR-006 bypass.

No database, no network. Pure unit tests with mocked cursors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import psycopg

from scix.search import (
    _check_body_latex_provenance,
    read_paper_section,
    search_within_paper,
)
from scix.sources.licensing import DEFAULT_SNIPPET_BUDGET

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_BODY = (
    "Introduction\n"
    "This paper studies dark matter halos in galaxy clusters.\n"
    "We present new observations from the Hubble Space Telescope.\n"
    "\n"
    "Methods\n"
    "We used spectroscopic analysis of 500 galaxies.\n"
    "The data was reduced using standard IRAF pipelines.\n"
)

SAMPLE_ABSTRACT = "We study dark matter halos using HST observations."

LONG_BODY = "x" * (DEFAULT_SNIPPET_BUDGET + 500)


# ---------------------------------------------------------------------------
# Helper: multi-cursor mock
# ---------------------------------------------------------------------------


def _make_multi_cursor_conn(rows_per_cursor: list[dict | None]) -> MagicMock:
    """Create a mock connection that returns different rows for sequential cursor calls.

    Each entry in rows_per_cursor becomes the fetchone() return value for
    a successive cursor() call on the connection.
    """
    mock_conn = MagicMock()
    cursors = []
    for row in rows_per_cursor:
        cur = MagicMock()
        cur.fetchone.return_value = row
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cursors.append(cur)
    mock_conn.cursor.side_effect = cursors
    return mock_conn


def _make_single_cursor_conn(row: dict | None) -> MagicMock:
    """Create a mock connection whose cursor always returns the same row."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = row
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


# ---------------------------------------------------------------------------
# _check_body_latex_provenance — unit tests
# ---------------------------------------------------------------------------


class TestCheckBodyLatexProvenance:
    """Tests for the provenance-check helper."""

    def test_returns_none_when_no_fulltext_row(self) -> None:
        """No papers_fulltext row means body is not LaTeX-derived."""
        conn = _make_single_cursor_conn(None)
        result = _check_body_latex_provenance(conn, "2024ApJ...001A")
        assert result is None

    def test_returns_none_when_papers_fulltext_table_missing(self) -> None:
        """If papers_fulltext table is absent (migration 041 not yet applied),
        the guard returns None and rolls back the failed transaction so the
        caller's session stays usable."""
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.execute.side_effect = psycopg.errors.UndefinedTable(
            'relation "papers_fulltext" does not exist'
        )
        conn.cursor.return_value = cur

        result = _check_body_latex_provenance(conn, "2024ApJ...001A")

        assert result is None
        conn.rollback.assert_called_once()

    def test_returns_none_for_ads_body_source(self) -> None:
        """papers_fulltext with source='ads_body' is not LaTeX-derived."""
        conn = _make_single_cursor_conn({"source": "ads_body"})
        result = _check_body_latex_provenance(conn, "2024ApJ...001A")
        assert result is None

    def test_returns_source_for_ar5iv(self) -> None:
        """papers_fulltext with source='ar5iv' IS LaTeX-derived."""
        conn = _make_single_cursor_conn({"source": "ar5iv"})
        result = _check_body_latex_provenance(conn, "2024ApJ...001A")
        assert result == "ar5iv"

    def test_returns_source_for_arxiv_local(self) -> None:
        """papers_fulltext with source='arxiv_local' IS LaTeX-derived."""
        conn = _make_single_cursor_conn({"source": "arxiv_local"})
        result = _check_body_latex_provenance(conn, "2024ApJ...001A")
        assert result == "arxiv_local"

    def test_returns_none_for_s2orc_source(self) -> None:
        """papers_fulltext with source='s2orc' is not LaTeX-derived."""
        conn = _make_single_cursor_conn({"source": "s2orc"})
        result = _check_body_latex_provenance(conn, "2024ApJ...001A")
        assert result is None

    def test_returns_none_for_docling_source(self) -> None:
        """papers_fulltext with source='docling' is not LaTeX-derived."""
        conn = _make_single_cursor_conn({"source": "docling"})
        result = _check_body_latex_provenance(conn, "2024ApJ...001A")
        assert result is None

    def test_returns_none_for_abstract_source(self) -> None:
        """papers_fulltext with source='abstract' is not LaTeX-derived."""
        conn = _make_single_cursor_conn({"source": "abstract"})
        result = _check_body_latex_provenance(conn, "2024ApJ...001A")
        assert result is None


# ---------------------------------------------------------------------------
# read_paper_section — ADR-006 guard tests
# ---------------------------------------------------------------------------


class TestReadPaperSectionADR006Guard:
    """read_paper_section applies snippet budget when body is LaTeX-derived."""

    def test_ads_body_no_snippet_budget(self) -> None:
        """When papers_fulltext.source is 'ads_body', body text is returned fully."""
        papers_row = {"body": SAMPLE_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        fulltext_row = {"source": "ads_body"}
        conn = _make_multi_cursor_conn([papers_row, fulltext_row])

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        # Full body returned, not truncated
        assert paper["has_body"] is True
        assert len(paper["section_text"]) == len(SAMPLE_BODY)
        assert "adr006_guarded" not in result.metadata

    def test_no_fulltext_row_no_snippet_budget(self) -> None:
        """When there is no papers_fulltext row, body text is returned fully."""
        papers_row = {"body": SAMPLE_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        fulltext_row = None  # No fulltext entry
        conn = _make_multi_cursor_conn([papers_row, fulltext_row])

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is True
        assert len(paper["section_text"]) == len(SAMPLE_BODY)

    def test_ar5iv_source_applies_snippet_budget(self) -> None:
        """When papers_fulltext.source is 'ar5iv', body is capped to snippet budget."""
        papers_row = {"body": LONG_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        fulltext_row = {"source": "ar5iv"}
        identifier_row = {"identifier": ["2301.12345"]}
        conn = _make_multi_cursor_conn([papers_row, fulltext_row, identifier_row])

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is True
        # Snippet budget enforced
        assert len(paper["section_text"]) <= DEFAULT_SNIPPET_BUDGET
        assert paper["section_text"].endswith("...")
        assert paper.get("canonical_url") == "https://arxiv.org/abs/2301.12345"
        assert result.metadata.get("adr006_guarded") is True

    def test_arxiv_local_source_applies_snippet_budget(self) -> None:
        """When papers_fulltext.source is 'arxiv_local', body is capped."""
        papers_row = {"body": LONG_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        fulltext_row = {"source": "arxiv_local"}
        identifier_row = {"identifier": ["astro-ph/0301001"]}
        conn = _make_multi_cursor_conn([papers_row, fulltext_row, identifier_row])

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert len(paper["section_text"]) <= DEFAULT_SNIPPET_BUDGET
        assert paper.get("canonical_url") == "https://arxiv.org/abs/astro-ph/0301001"
        assert result.metadata.get("adr006_guarded") is True

    def test_latex_derived_short_body_not_truncated(self) -> None:
        """LaTeX-derived body under budget is returned in full but still guarded."""
        short_body = "Short LaTeX body text."
        papers_row = {"body": short_body, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        fulltext_row = {"source": "ar5iv"}
        identifier_row = {"identifier": ["2301.12345"]}
        conn = _make_multi_cursor_conn([papers_row, fulltext_row, identifier_row])

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["section_text"] == short_body  # Not truncated
        assert paper.get("canonical_url") == "https://arxiv.org/abs/2301.12345"
        assert result.metadata.get("adr006_guarded") is True

    def test_no_body_skips_guard(self) -> None:
        """Paper without body falls back to abstract — guard not needed."""
        papers_row = {"body": None, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        # Guard query should NOT be issued when there's no body
        conn = _make_single_cursor_conn(papers_row)

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is False
        assert paper["section_name"] == "abstract"

    def test_paper_not_found_skips_guard(self) -> None:
        """Non-existent paper returns error — guard not needed."""
        conn = _make_single_cursor_conn(None)

        result = read_paper_section(conn, "NONEXISTENT")

        assert result.total == 0
        assert "error" in result.metadata


# ---------------------------------------------------------------------------
# search_within_paper — ADR-006 guard tests
# ---------------------------------------------------------------------------


class TestSearchWithinPaperADR006Guard:
    """search_within_paper applies snippet budget when body is LaTeX-derived."""

    def test_ads_body_no_snippet_budget(self) -> None:
        """When papers_fulltext.source is 'ads_body', headline is returned unmodified."""
        search_row = {
            "bibcode": "2024ApJ...001A",
            "title": "Test Paper",
            "body": SAMPLE_BODY,
            "headline": "...studies <b>dark matter</b> halos in galaxy clusters...",
        }
        fulltext_row = {"source": "ads_body"}
        conn = _make_multi_cursor_conn([search_row, fulltext_row])

        result = search_within_paper(conn, "2024ApJ...001A", "dark matter")

        assert result.total == 1
        paper = result.papers[0]
        assert "dark matter" in paper["headline"]
        assert "adr006_guarded" not in result.metadata

    def test_no_fulltext_row_no_snippet_budget(self) -> None:
        """When there is no papers_fulltext row, headline is returned unmodified."""
        search_row = {
            "bibcode": "2024ApJ...001A",
            "title": "Test Paper",
            "body": SAMPLE_BODY,
            "headline": "...studies <b>dark matter</b> halos...",
        }
        fulltext_row = None
        conn = _make_multi_cursor_conn([search_row, fulltext_row])

        result = search_within_paper(conn, "2024ApJ...001A", "dark matter")

        assert result.total == 1
        paper = result.papers[0]
        assert "dark matter" in paper["headline"]

    def test_ar5iv_source_applies_snippet_budget_to_headline(self) -> None:
        """When papers_fulltext.source is 'ar5iv', headline is capped to snippet budget."""
        long_headline = "x" * (DEFAULT_SNIPPET_BUDGET + 200)
        search_row = {
            "bibcode": "2024ApJ...001A",
            "title": "Test Paper",
            "body": LONG_BODY,
            "headline": long_headline,
        }
        fulltext_row = {"source": "ar5iv"}
        identifier_row = {"identifier": ["2301.12345"]}
        conn = _make_multi_cursor_conn([search_row, fulltext_row, identifier_row])

        result = search_within_paper(conn, "2024ApJ...001A", "dark matter")

        assert result.total == 1
        paper = result.papers[0]
        assert len(paper["headline"]) <= DEFAULT_SNIPPET_BUDGET
        assert paper.get("canonical_url") == "https://arxiv.org/abs/2301.12345"
        assert result.metadata.get("adr006_guarded") is True

    def test_no_body_match_skips_guard(self) -> None:
        """No body match returns empty — guard not needed."""
        # First cursor: search returns None (no body match)
        # Second cursor: existence check returns paper
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
