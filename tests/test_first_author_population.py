"""Tests for ``first_author`` population in ``lit_review`` and ``read_paper`` responses.

Bead scix_experiments-nc1t — agents should never have to parse author names
from abstract text. ``papers.first_author`` is populated for the bulk of the
ADS corpus, so it must round-trip through both tool responses unmodified.

These are pure unit tests with mocked cursors: no DB, no network. The point
is to lock down the response shape so that any future SELECT or row-to-dict
mapping that drops ``first_author`` fails CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scix.search import (
    SearchResult,
    lit_review,
    read_paper_section,
    search_within_paper,
)

# ---------------------------------------------------------------------------
# Mock helpers (mirror tests/test_body_adr006_guard.py — same shape used
# elsewhere in the suite).
# ---------------------------------------------------------------------------


def _make_multi_cursor_conn(rows_per_cursor: list[dict | None]) -> MagicMock:
    """Build a mock connection whose successive cursor() calls return given rows."""
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
    """Build a mock connection whose cursor always returns the same row."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = row
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


# ---------------------------------------------------------------------------
# read_paper_section — first_author must appear on every return path
# ---------------------------------------------------------------------------


class TestReadPaperSectionFirstAuthor:
    """``read_paper_section`` returns ``first_author`` regardless of which
    body source it falls through to (full body, abstract fallback, or
    structured papers_fulltext.sections)."""

    def test_full_body_returns_first_author(self) -> None:
        """Default ``section='full'`` path: SELECT row carries first_author and
        it must surface in the response paper dict."""
        papers_row = {
            "body": "Introduction\nSome text.\n",
            "abstract": "Abstract text.",
            "title": "Test Paper",
            "first_author": "Breu, A. P.",
        }
        # Second cursor call is the ADR-006 provenance check; returning None
        # means "not LaTeX-derived" and short-circuits the guard.
        conn = _make_multi_cursor_conn([papers_row, None])

        result = read_paper_section(conn, "2003PhRvL..90a4302B")

        assert result.total == 1
        paper = result.papers[0]
        assert "first_author" in paper
        assert paper["first_author"] == "Breu, A. P."

    def test_abstract_fallback_returns_first_author(self) -> None:
        """When the paper has no body, the abstract-fallback dict must still
        carry first_author."""
        papers_row = {
            "body": None,
            "abstract": "Abstract only.",
            "title": "Test Paper",
            "first_author": "Doe, J.",
        }
        # No body → no provenance check; single cursor.
        conn = _make_single_cursor_conn(papers_row)

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert "first_author" in paper
        assert paper["first_author"] == "Doe, J."

    def test_paper_not_found_no_first_author(self) -> None:
        """Sanity: missing paper returns total=0 and metadata error; no paper
        dict to assert against, but the function must not raise."""
        conn = _make_single_cursor_conn(None)

        result = read_paper_section(conn, "NONEXISTENT")

        assert result.total == 0
        assert "error" in result.metadata

    def test_structured_papers_fulltext_returns_first_author(self) -> None:
        """When ``section`` is requested and ``papers_fulltext.sections`` has a
        match, the structured-section response must still carry first_author.
        The first SELECT (from papers) provides the value; the helper must
        propagate it down."""
        papers_row = {
            "body": "Introduction\nIntro text.\nMethods\nMethods text.\n",
            "abstract": "Abstract.",
            "title": "Test Paper",
            "first_author": "Smith, A.",
        }
        fulltext_row = {
            "sections": [
                {"heading": "Methods", "text": "Methods text body.", "level": 1, "offset": 0},
            ],
        }
        # 1st cursor: SELECT body, abstract, title, first_author FROM papers
        # 2nd cursor: SELECT sections FROM papers_fulltext (structured path)
        conn = _make_multi_cursor_conn([papers_row, fulltext_row])

        result = read_paper_section(conn, "2024ApJ...001A", section="methods")

        assert result.total == 1
        paper = result.papers[0]
        # Structured path was used (no ADR-006 source key set in metadata,
        # but the section_name comes from the heading).
        assert paper["section_name"] == "Methods"
        assert "first_author" in paper
        assert paper["first_author"] == "Smith, A."


# ---------------------------------------------------------------------------
# search_within_paper — first_author must appear on every return path
# ---------------------------------------------------------------------------


class TestSearchWithinPaperFirstAuthor:
    """``search_within_paper`` returns ``first_author`` on the body-search path."""

    def test_body_search_returns_first_author(self) -> None:
        """SELECT row from search_within_paper carries first_author and the
        response paper dict must include it."""
        search_row = {
            "bibcode": "2024ApJ...001A",
            "title": "Test Paper",
            "body": "We study dark matter halos in galaxy clusters.",
            "first_author": "Lee, K.",
            "headline": "We study <b>dark matter</b> halos in galaxy clusters.",
        }
        # Cursor sequence (mirrors tests/test_body_adr006_guard.py):
        #   1. main search (SELECT bibcode, title, first_author, body, headline)
        #   2. ts_rank section scoring (None → falls back to Python proxy)
        #   3. ADR-006 provenance check (None → not LaTeX-derived)
        conn = _make_multi_cursor_conn([search_row, None, None])

        result = search_within_paper(conn, "2024ApJ...001A", "dark matter")

        assert result.total == 1
        paper = result.papers[0]
        assert "first_author" in paper
        assert paper["first_author"] == "Lee, K."


# ---------------------------------------------------------------------------
# lit_review — seed papers must carry first_author through the pass-through
# ---------------------------------------------------------------------------


class TestLitReviewFirstAuthor:
    """``lit_review`` does not re-query author names; it relies on
    ``hybrid_search`` returning paper stubs with ``first_author``. This test
    locks down the pass-through behaviour: if ``lit_review`` ever drops the
    field while building its response, this fails."""

    @patch("scix.search.hybrid_search")
    def test_seed_papers_carry_first_author(self, mock_hybrid: MagicMock) -> None:
        """When hybrid_search returns paper stubs with first_author, lit_review's
        ``papers`` list must preserve the field."""
        mock_hybrid.return_value = SearchResult(
            papers=[
                {
                    "bibcode": "2003PhRvL..90a4302B",
                    "title": "Reversing the Brazil-Nut Effect",
                    "first_author": "Breu, A. P.",
                    "year": 2003,
                    "citation_count": 100,
                    "abstract_snippet": "...",
                }
            ],
            total=1,
            timing_ms={"query_ms": 1.0},
        )

        # Conn cursor calls in lit_review (after seed retrieval, with one
        # bibcode in the working set):
        #   1. communities aggregate
        #   2. top_venues
        #   3. year_distribution
        #   4. covered count (citation_contexts subquery)
        #   5. contexts_rows count
        #   6. abstracts pull (sample_abstracts > 0)
        # We don't care about their values — empty results are fine. Use a
        # default MagicMock conn whose cursors return empty fetchall/fetchone.
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = []
        cur.fetchone.return_value = (0,)
        conn.cursor.return_value = cur

        # Minimal call: top_seeds=1, expansion_seeds=0 to skip get_references /
        # get_citations side calls; sample_abstracts=0 to skip the abstract pull.
        result = lit_review(
            conn,
            "Brazil-nut effect",
            top_seeds=1,
            expansion_seeds=0,
            expand_per_seed=0,
            sample_abstracts=0,
        )

        assert result.total == 1
        seed = result.papers[0]
        assert "first_author" in seed
        assert seed["first_author"] == "Breu, A. P."
