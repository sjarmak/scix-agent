"""Tests for snippet budget enforcement wiring in ar5iv and papers_fulltext read paths.

Covers bead wqr.5.1 requirements:
- ar5iv.get_body_snippet() wraps ParsedFulltext body through enforce_snippet_budget
- ar5iv.get_body_snippet() always returns canonical_url
- LATEX_DERIVED_SOURCES correctly classifies source types
- apply_snippet_budget_if_needed() enforces budget for LaTeX sources only
- read_fulltext() in search.py applies snippet budget for LaTeX-derived sources
- read_fulltext() passes non-LaTeX sources through without truncation

No database required. Unit tests with mocks.
"""

from __future__ import annotations

import json

import pytest

from scix.sources.ar5iv import (
    ParsedFulltext,
    Section,
    get_body_snippet,
)
from scix.sources.licensing import DEFAULT_SNIPPET_BUDGET, SnippetPayload

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

LONG_BODY = "A" * 1000  # Well above default 500-char budget


def _make_parsed_fulltext(body_text: str = "Short body.") -> ParsedFulltext:
    """Build a minimal ParsedFulltext with a single section."""
    return ParsedFulltext(
        sections=[
            Section(heading="Introduction", level=1, text=body_text, offset=0),
        ],
        inline_cites=[],
        figures=[],
        tables=[],
        equations=[],
        parser_version="test-v1",
    )


# ---------------------------------------------------------------------------
# ar5iv.get_body_snippet
# ---------------------------------------------------------------------------


class TestGetBodySnippet:
    def test_returns_snippet_payload(self) -> None:
        parsed = _make_parsed_fulltext("Short body.")
        result = get_body_snippet(parsed, arxiv_id="2301.12345")
        assert isinstance(result, SnippetPayload)

    def test_canonical_url_present(self) -> None:
        parsed = _make_parsed_fulltext("Short body.")
        result = get_body_snippet(parsed, arxiv_id="2301.12345")
        assert result.canonical_url == "https://arxiv.org/abs/2301.12345"

    def test_short_body_not_truncated(self) -> None:
        parsed = _make_parsed_fulltext("Short body.")
        result = get_body_snippet(parsed, arxiv_id="2301.12345")
        assert result.truncated is False
        assert result.snippet == "Short body."

    def test_long_body_truncated_to_budget(self) -> None:
        parsed = _make_parsed_fulltext(LONG_BODY)
        result = get_body_snippet(parsed, arxiv_id="2301.12345")
        assert result.truncated is True
        assert len(result.snippet) <= DEFAULT_SNIPPET_BUDGET
        assert result.original_length == 1000

    def test_explicit_budget_override(self) -> None:
        parsed = _make_parsed_fulltext("A" * 200)
        result = get_body_snippet(parsed, arxiv_id="2301.12345", budget=50)
        assert result.budget == 50
        assert len(result.snippet) <= 50
        assert result.truncated is True

    def test_multi_section_body_concatenated(self) -> None:
        """Body text from multiple sections is joined."""
        parsed = ParsedFulltext(
            sections=[
                Section(heading="Intro", level=1, text="First section.", offset=0),
                Section(heading="Methods", level=1, text="Second section.", offset=15),
            ],
            inline_cites=[],
            figures=[],
            tables=[],
            equations=[],
            parser_version="test-v1",
        )
        result = get_body_snippet(parsed, arxiv_id="2301.12345")
        assert "First section." in result.snippet
        assert "Second section." in result.snippet

    def test_empty_sections_yields_empty_snippet(self) -> None:
        parsed = ParsedFulltext(
            sections=[],
            inline_cites=[],
            figures=[],
            tables=[],
            equations=[],
            parser_version="test-v1",
        )
        result = get_body_snippet(parsed, arxiv_id="2301.12345")
        assert result.snippet == ""
        assert result.truncated is False

    def test_old_style_arxiv_id(self) -> None:
        parsed = _make_parsed_fulltext("Body text.")
        result = get_body_snippet(parsed, arxiv_id="astro-ph/9901001")
        assert result.canonical_url == "https://arxiv.org/abs/astro-ph/9901001"


# ---------------------------------------------------------------------------
# LATEX_DERIVED_SOURCES constant
# ---------------------------------------------------------------------------


class TestLatexDerivedSources:
    def test_ar5iv_is_latex_derived(self) -> None:
        from scix.sources.ar5iv import LATEX_DERIVED_SOURCES

        assert "ar5iv" in LATEX_DERIVED_SOURCES

    def test_arxiv_local_is_latex_derived(self) -> None:
        from scix.sources.ar5iv import LATEX_DERIVED_SOURCES

        assert "arxiv_local" in LATEX_DERIVED_SOURCES

    def test_ads_body_not_latex_derived(self) -> None:
        from scix.sources.ar5iv import LATEX_DERIVED_SOURCES

        assert "ads_body" not in LATEX_DERIVED_SOURCES

    def test_s2orc_not_latex_derived(self) -> None:
        from scix.sources.ar5iv import LATEX_DERIVED_SOURCES

        assert "s2orc" not in LATEX_DERIVED_SOURCES


# ---------------------------------------------------------------------------
# search.apply_snippet_budget_if_needed
# ---------------------------------------------------------------------------


class TestApplySnippetBudgetIfNeeded:
    """Test the helper that conditionally applies snippet budget based on source."""

    def test_latex_source_gets_budget(self) -> None:
        from scix.search import apply_snippet_budget_if_needed

        result = apply_snippet_budget_if_needed(
            body_text=LONG_BODY,
            source="ar5iv",
            bibcode="2024arXiv240112345A",
            arxiv_id="2401.12345",
        )
        assert result["truncated"] is True
        assert len(result["snippet"]) <= DEFAULT_SNIPPET_BUDGET
        assert result["canonical_url"] == "https://arxiv.org/abs/2401.12345"

    def test_non_latex_source_passes_through(self) -> None:
        from scix.search import apply_snippet_budget_if_needed

        result = apply_snippet_budget_if_needed(
            body_text=LONG_BODY,
            source="ads_body",
            bibcode="2024ApJ...962L..15J",
            arxiv_id=None,
        )
        assert result["truncated"] is False
        assert result["snippet"] == LONG_BODY
        assert result.get("canonical_url") is None

    def test_arxiv_local_gets_budget(self) -> None:
        from scix.search import apply_snippet_budget_if_needed

        result = apply_snippet_budget_if_needed(
            body_text=LONG_BODY,
            source="arxiv_local",
            bibcode="2024arXiv240112345A",
            arxiv_id="2401.12345",
        )
        assert result["truncated"] is True
        assert result["canonical_url"] is not None

    def test_short_latex_text_not_truncated(self) -> None:
        from scix.search import apply_snippet_budget_if_needed

        result = apply_snippet_budget_if_needed(
            body_text="Short body.",
            source="ar5iv",
            bibcode="2024arXiv240112345A",
            arxiv_id="2401.12345",
        )
        assert result["truncated"] is False
        assert result["snippet"] == "Short body."
        # canonical_url MUST still be present for LaTeX-derived sources
        assert result["canonical_url"] == "https://arxiv.org/abs/2401.12345"

    def test_latex_source_without_arxiv_id_raises(self) -> None:
        from scix.search import apply_snippet_budget_if_needed

        with pytest.raises(ValueError, match="canonical_url"):
            apply_snippet_budget_if_needed(
                body_text="text",
                source="ar5iv",
                bibcode="2024ApJ...962L..15J",
                arxiv_id=None,
            )


# ---------------------------------------------------------------------------
# search.read_fulltext (reads papers_fulltext table)
# ---------------------------------------------------------------------------


class TestReadFulltext:
    """Test read_fulltext function in search.py.

    Uses a mock connection to avoid DB dependency.
    """

    def test_latex_source_body_is_budget_enforced(self) -> None:
        from unittest.mock import MagicMock, patch

        from scix.search import read_fulltext

        # Mock DB row: ar5iv source with long body
        sections = [{"heading": "Intro", "level": 1, "text": LONG_BODY, "offset": 0}]
        mock_row = {
            "bibcode": "2024arXiv240112345A",
            "source": "ar5iv",
            "sections": json.dumps(sections),
            "parser_version": "LaTeXML v0.8.8",
        }

        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = mock_row
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Mock identifier lookup to provide arxiv_id
        with patch("scix.search._get_arxiv_id_for_bibcode", return_value="2401.12345"):
            result = read_fulltext(conn, "2024arXiv240112345A")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["canonical_url"] == "https://arxiv.org/abs/2401.12345"
        assert paper["truncated"] is True
        assert len(paper["section_text"]) <= DEFAULT_SNIPPET_BUDGET

    def test_non_latex_source_body_not_truncated(self) -> None:
        from unittest.mock import MagicMock

        from scix.search import read_fulltext

        sections = [{"heading": "Intro", "level": 1, "text": LONG_BODY, "offset": 0}]
        mock_row = {
            "bibcode": "2024ApJ...962L..15J",
            "source": "ads_body",
            "sections": json.dumps(sections),
            "parser_version": "ads-v1",
        }

        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = mock_row
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = read_fulltext(conn, "2024ApJ...962L..15J")

        assert result.total == 1
        paper = result.papers[0]
        assert paper.get("canonical_url") is None
        assert paper.get("truncated") is None or paper["truncated"] is False

    def test_not_found_returns_empty(self) -> None:
        from unittest.mock import MagicMock

        from scix.search import read_fulltext

        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = read_fulltext(conn, "2024ApJ...000L..00X")

        assert result.total == 0
        assert result.papers == []
