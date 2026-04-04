"""Tests for citation context extraction pipeline."""

from __future__ import annotations

import pytest

from scix.citation_context import (
    CitationContext,
    CitationMarker,
    _parse_marker_numbers,
    extract_citation_contexts,
    process_paper,
    resolve_citation_markers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_REFERENCES = [
    "2020ApJ...900..100A",  # index 0 -> [1]
    "2021MNRAS.500..200B",  # index 1 -> [2]
    "2022A&A...650..300C",  # index 2 -> [3]
    "2023Natur.600..400D",  # index 3 -> [4]
    "2024Sci...380..500E",  # index 4 -> [5]
]


def _body_with_citation(before_words: int, after_words: int, marker: str = "[1]") -> str:
    """Build a body string with a known number of words before/after a marker."""
    before = " ".join(f"word{i}" for i in range(before_words))
    after = " ".join(f"after{i}" for i in range(after_words))
    return f"{before} {marker} {after}"


# ---------------------------------------------------------------------------
# _parse_marker_numbers
# ---------------------------------------------------------------------------


class TestParseMarkerNumbers:
    def test_single(self) -> None:
        assert _parse_marker_numbers("1") == (1,)

    def test_comma_separated(self) -> None:
        assert _parse_marker_numbers("1, 2, 3") == (1, 2, 3)

    def test_range(self) -> None:
        assert _parse_marker_numbers("1-3") == (1, 2, 3)

    def test_mixed_comma_and_range(self) -> None:
        assert _parse_marker_numbers("1, 3-5") == (1, 3, 4, 5)

    def test_empty_string(self) -> None:
        assert _parse_marker_numbers("") == ()

    def test_non_numeric(self) -> None:
        assert _parse_marker_numbers("abc") == ()


# ---------------------------------------------------------------------------
# extract_citation_contexts — single [1] marker
# ---------------------------------------------------------------------------


class TestExtractSingleMarker:
    def test_finds_single_marker(self) -> None:
        body = (
            "Some introductory text about stellar evolution [1] and more discussion follows here."
        )
        markers = extract_citation_contexts(body)
        assert len(markers) == 1
        assert markers[0].marker_text == "[1]"
        assert markers[0].marker_numbers == (1,)

    def test_context_contains_marker(self) -> None:
        body = (
            "Some introductory text about stellar evolution [1] and more discussion follows here."
        )
        markers = extract_citation_contexts(body)
        assert "[1]" in markers[0].context_text

    def test_char_offsets_correct(self) -> None:
        body = "Hello [1] world"
        markers = extract_citation_contexts(body)
        assert body[markers[0].char_start : markers[0].char_end] == "[1]"

    def test_empty_body(self) -> None:
        assert extract_citation_contexts("") == []

    def test_no_markers(self) -> None:
        body = "This text has no citation markers at all."
        assert extract_citation_contexts(body) == []


# ---------------------------------------------------------------------------
# extract_citation_contexts — multiple [1,2,3] markers
# ---------------------------------------------------------------------------


class TestExtractMultipleMarkers:
    def test_comma_separated_marker(self) -> None:
        body = "Previous work [1, 2, 3] established the framework."
        markers = extract_citation_contexts(body)
        assert len(markers) == 1
        assert markers[0].marker_numbers == (1, 2, 3)
        assert markers[0].marker_text == "[1, 2, 3]"

    def test_range_marker(self) -> None:
        body = "Several studies [1-3] have shown this effect."
        markers = extract_citation_contexts(body)
        assert len(markers) == 1
        assert markers[0].marker_numbers == (1, 2, 3)

    def test_multiple_separate_markers(self) -> None:
        body = "First point [1] and second point [2] in the text."
        markers = extract_citation_contexts(body)
        assert len(markers) == 2
        assert markers[0].marker_numbers == (1,)
        assert markers[1].marker_numbers == (2,)


# ---------------------------------------------------------------------------
# Author-year style — graceful skip
# ---------------------------------------------------------------------------


class TestAuthorYearSkip:
    def test_no_match_for_author_year(self) -> None:
        """Author-year citations like (Smith et al. 2020) should not match."""
        body = "As shown by Smith et al. (2020), the results are consistent."
        markers = extract_citation_contexts(body)
        assert markers == []

    def test_no_match_for_text_in_brackets(self) -> None:
        """Bracketed text like [see also] should not match."""
        body = "The results [see also discussion in Section 3] were surprising."
        markers = extract_citation_contexts(body)
        assert markers == []


# ---------------------------------------------------------------------------
# Edge cases: marker at start/end of text, N > len(references)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_marker_at_start(self) -> None:
        body = "[1] This paper begins with a citation."
        markers = extract_citation_contexts(body)
        assert len(markers) == 1
        assert markers[0].char_start == 0

    def test_marker_at_end(self) -> None:
        body = "This paper ends with a citation [1]"
        markers = extract_citation_contexts(body)
        assert len(markers) == 1
        assert markers[0].char_end == len(body)

    def test_context_window_size(self) -> None:
        """Context window should be approximately 250 words."""
        body = _body_with_citation(200, 200, "[1]")
        markers = extract_citation_contexts(body)
        assert len(markers) == 1
        context_words = markers[0].context_text.split()
        # ~125 before + marker + ~125 after = roughly 250
        assert 200 <= len(context_words) <= 260

    def test_short_text_includes_everything(self) -> None:
        """When body is shorter than window, include all text."""
        body = "Short text [1] here."
        markers = extract_citation_contexts(body)
        assert markers[0].context_text == body


# ---------------------------------------------------------------------------
# resolve_citation_markers
# ---------------------------------------------------------------------------


class TestResolveMarkers:
    def test_single_resolution(self) -> None:
        marker = CitationMarker(
            marker_text="[1]",
            marker_numbers=(1,),
            char_start=10,
            char_end=13,
            context_text="some context [1] text",
            context_start=0,
        )
        contexts = resolve_citation_markers([marker], SAMPLE_REFERENCES, "2025test..bibcode")
        assert len(contexts) == 1
        assert contexts[0].target_bibcode == "2020ApJ...900..100A"
        assert contexts[0].source_bibcode == "2025test..bibcode"

    def test_multi_resolution(self) -> None:
        marker = CitationMarker(
            marker_text="[1, 2, 3]",
            marker_numbers=(1, 2, 3),
            char_start=10,
            char_end=19,
            context_text="some context [1, 2, 3] text",
            context_start=0,
        )
        contexts = resolve_citation_markers([marker], SAMPLE_REFERENCES, "2025test..bibcode")
        assert len(contexts) == 3
        assert contexts[0].target_bibcode == "2020ApJ...900..100A"
        assert contexts[1].target_bibcode == "2021MNRAS.500..200B"
        assert contexts[2].target_bibcode == "2022A&A...650..300C"

    def test_out_of_bounds_skipped(self) -> None:
        """Marker [99] with only 5 references should produce no contexts."""
        marker = CitationMarker(
            marker_text="[99]",
            marker_numbers=(99,),
            char_start=0,
            char_end=4,
            context_text="[99] text",
            context_start=0,
        )
        contexts = resolve_citation_markers([marker], SAMPLE_REFERENCES, "2025test..bibcode")
        assert contexts == []

    def test_zero_marker_skipped(self) -> None:
        """[0] is out of bounds (1-indexed)."""
        marker = CitationMarker(
            marker_text="[0]",
            marker_numbers=(0,),
            char_start=0,
            char_end=3,
            context_text="[0] text",
            context_start=0,
        )
        contexts = resolve_citation_markers([marker], SAMPLE_REFERENCES, "src")
        assert contexts == []

    def test_empty_references(self) -> None:
        marker = CitationMarker(
            marker_text="[1]",
            marker_numbers=(1,),
            char_start=0,
            char_end=3,
            context_text="[1] text",
            context_start=0,
        )
        contexts = resolve_citation_markers([marker], [], "src")
        assert contexts == []

    def test_partial_resolution(self) -> None:
        """[1, 99] should resolve [1] and skip [99]."""
        marker = CitationMarker(
            marker_text="[1, 99]",
            marker_numbers=(1, 99),
            char_start=0,
            char_end=7,
            context_text="[1, 99] text",
            context_start=0,
        )
        contexts = resolve_citation_markers([marker], SAMPLE_REFERENCES, "src")
        assert len(contexts) == 1
        assert contexts[0].target_bibcode == "2020ApJ...900..100A"


# ---------------------------------------------------------------------------
# process_paper — integration
# ---------------------------------------------------------------------------


class TestProcessPaper:
    def test_full_pipeline(self) -> None:
        body = "We follow the method of [1] and extend results from [2, 3]."
        contexts = process_paper("2025src..bibcode", body, SAMPLE_REFERENCES)
        # [1] -> 1 context, [2, 3] -> 2 contexts = 3 total
        assert len(contexts) == 3
        bibcodes = [c.target_bibcode for c in contexts]
        assert "2020ApJ...900..100A" in bibcodes
        assert "2021MNRAS.500..200B" in bibcodes
        assert "2022A&A...650..300C" in bibcodes

    def test_empty_body(self) -> None:
        assert process_paper("bib", "", SAMPLE_REFERENCES) == []

    def test_empty_references(self) -> None:
        assert process_paper("bib", "Some text [1] here", []) == []

    def test_section_enrichment(self) -> None:
        """process_paper should annotate contexts with section names."""
        body = (
            "Introduction\n"
            "We build on prior work [1] in this field.\n"
            "Methods\n"
            "Following [2], we apply the standard approach.\n"
        )
        contexts = process_paper("2025src..bibcode", body, SAMPLE_REFERENCES)
        assert len(contexts) == 2
        # First citation is in Introduction section
        intro_ctx = [c for c in contexts if c.target_bibcode == "2020ApJ...900..100A"]
        assert len(intro_ctx) == 1
        assert intro_ctx[0].section_name == "introduction"
        # Second citation is in Methods section
        methods_ctx = [c for c in contexts if c.target_bibcode == "2021MNRAS.500..200B"]
        assert len(methods_ctx) == 1
        assert methods_ctx[0].section_name == "methods"

    def test_all_contexts_have_source_bibcode(self) -> None:
        body = "Text [1] and [2] here."
        contexts = process_paper("MY_BIB", body, SAMPLE_REFERENCES)
        for ctx in contexts:
            assert ctx.source_bibcode == "MY_BIB"
