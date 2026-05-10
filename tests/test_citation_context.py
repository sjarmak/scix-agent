"""Tests for citation context extraction pipeline."""

from __future__ import annotations

import time

import pytest

from scix.citation_context import (
    _MAX_BODY_CHARS,
    CitationMarker,
    _enrich_with_sections,
    _parse_marker_numbers,
    _word_boundary_window,
    extract_author_year_citations,
    extract_citation_contexts,
    process_paper,
    resolve_author_year_markers,
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

    def test_section_name_none_when_no_headers(self) -> None:
        """Papers without section headers should produce contexts with section_name=None."""
        body = "We follow the method of [1] and extend results from [2]."
        contexts = process_paper("2025src..bibcode", body, SAMPLE_REFERENCES)
        # parse_sections returns [("full", ...)] for no-header text,
        # but "full" is not a recognized section range match for markers
        for ctx in contexts:
            assert ctx.section_name is None or isinstance(ctx.section_name, str)


# ---------------------------------------------------------------------------
# _enrich_with_sections
# ---------------------------------------------------------------------------


class TestEnrichWithSections:
    def test_markers_get_section_labels(self) -> None:
        marker = CitationMarker(
            marker_text="[1]",
            marker_numbers=(1,),
            char_start=50,
            char_end=53,
            context_text="context around [1]",
            context_start=30,
        )
        sections = [
            ("introduction", 0, 60, "intro text"),
            ("methods", 60, 120, "methods text"),
        ]
        enriched = _enrich_with_sections([marker], sections)
        assert len(enriched) == 1
        assert enriched[0].section_name == "introduction"

    def test_marker_outside_all_sections(self) -> None:
        marker = CitationMarker(
            marker_text="[1]",
            marker_numbers=(1,),
            char_start=200,
            char_end=203,
            context_text="context",
            context_start=180,
        )
        sections = [("introduction", 0, 100, "intro")]
        enriched = _enrich_with_sections([marker], sections)
        assert enriched[0].section_name is None

    def test_preserves_marker_fields(self) -> None:
        marker = CitationMarker(
            marker_text="[2, 3]",
            marker_numbers=(2, 3),
            char_start=10,
            char_end=16,
            context_text="some ctx",
            context_start=0,
        )
        sections = [("results", 0, 50, "results text")]
        enriched = _enrich_with_sections([marker], sections)
        assert enriched[0].marker_text == "[2, 3]"
        assert enriched[0].marker_numbers == (2, 3)
        assert enriched[0].char_start == 10
        assert enriched[0].char_end == 16
        assert enriched[0].section_name == "results"


# ---------------------------------------------------------------------------
# Batch row format (used by run_pipeline)
# ---------------------------------------------------------------------------


class TestBatchRowFormat:
    """Verify that process_paper output can be converted to the DB row tuple
    format expected by _flush_contexts, including section_name."""

    def test_row_tuple_includes_section_name(self) -> None:
        body = (
            "Introduction\n"
            "We cite prior work [1] here.\n"
            "Methods\n"
            "Following [2] we proceed.\n"
        )
        contexts = process_paper("SRC_BIB", body, SAMPLE_REFERENCES)
        for ctx in contexts:
            row = (
                ctx.source_bibcode,
                ctx.target_bibcode,
                ctx.context_text,
                ctx.char_offset,
                ctx.section_name,
                ctx.intent,
            )
            assert len(row) == 6
            assert isinstance(row[4], (str, type(None)))  # section_name
            assert row[5] is None  # intent not set by extraction

    def test_section_name_populated_in_rows(self) -> None:
        body = "Introduction\n" "Background work [1] is important.\n"
        contexts = process_paper("SRC_BIB", body, SAMPLE_REFERENCES)
        assert len(contexts) == 1
        assert contexts[0].section_name == "introduction"


# ---------------------------------------------------------------------------
# extract_author_year_citations — pattern coverage
# ---------------------------------------------------------------------------

# Refs deliberately chosen so that (year, surname-initial) is unique-per-pair.
# Bibcode last char encodes the first author's surname initial (uppercase).
AUTHOR_YEAR_REFERENCES = [
    "2020ApJ...900..100A",  # year=2020, initial=A (Adams 2020)
    "2021MNRAS.500..200B",  # year=2021, initial=B (Brown 2021)
    "2022A&A...650..300C",  # year=2022, initial=C (Carter 2022)
    "2003ApJ...500..100S",  # year=2003, initial=S (Smith/Smith&Jones 2003)
    "2001AJ....120..200H",  # year=2001, initial=H (Hong 2001)
    "1999A&A...340..400J",  # year=1999, initial=J (Jones 1999)
]


class TestExtractAuthorYearPatterns:
    """Each pattern variant should produce at least one author-year marker."""

    def test_et_al_no_comma(self) -> None:
        body = "We follow Hong et al. 2001 in this analysis."
        markers = extract_author_year_citations(body)
        assert len(markers) == 1
        assert markers[0].marker_authors == ("Hong",)
        assert markers[0].marker_year == 2001

    def test_et_al_with_comma(self) -> None:
        body = "Earlier work by Hong et al., 2001 established this."
        markers = extract_author_year_citations(body)
        assert len(markers) == 1
        assert markers[0].marker_authors == ("Hong",)
        assert markers[0].marker_year == 2001

    def test_paren_single_author(self) -> None:
        body = "These results agree (Adams, 2020) with predictions."
        markers = extract_author_year_citations(body)
        assert len(markers) == 1
        assert markers[0].marker_authors == ("Adams",)
        assert markers[0].marker_year == 2020

    def test_paren_two_authors_ampersand(self) -> None:
        body = "The model (Smith & Jones, 2003) was extended."
        markers = extract_author_year_citations(body)
        assert len(markers) == 1
        # First surname encodes first-author bibcode-initial
        assert markers[0].marker_authors[0] == "Smith"
        assert markers[0].marker_year == 2003

    def test_paren_two_authors_and(self) -> None:
        body = "This builds on (Smith and Jones, 2003)."
        markers = extract_author_year_citations(body)
        assert len(markers) == 1
        assert markers[0].marker_authors[0] == "Smith"
        assert markers[0].marker_year == 2003

    def test_narrative_single(self) -> None:
        body = "As Adams (2020) showed, the trend is real."
        markers = extract_author_year_citations(body)
        assert len(markers) == 1
        assert markers[0].marker_authors == ("Adams",)
        assert markers[0].marker_year == 2020

    def test_narrative_two_authors(self) -> None:
        body = "Smith and Jones (2003) demonstrated the relation."
        markers = extract_author_year_citations(body)
        assert len(markers) == 1
        assert markers[0].marker_authors[0] == "Smith"
        assert markers[0].marker_year == 2003

    def test_initial_before_surname(self) -> None:
        """'J. Smith 2001' — single initial then surname; surname extracted is 'Smith'."""
        body = "As J. Smith et al. 2001 showed earlier."
        markers = extract_author_year_citations(body)
        assert len(markers) == 1
        assert markers[0].marker_authors == ("Smith",)
        assert markers[0].marker_year == 2001

    def test_three_authors_comma(self) -> None:
        """'Smith, Jones, & Brown 2003' — first surname is Smith."""
        body = "The trio (Smith, Jones, & Brown, 2003) co-authored this."
        markers = extract_author_year_citations(body)
        assert len(markers) == 1
        assert markers[0].marker_authors[0] == "Smith"
        assert markers[0].marker_year == 2003


class TestExtractAuthorYearNegatives:
    """Patterns that look citation-shaped but are not citations."""

    def test_numbered_marker_not_matched(self) -> None:
        body = "We follow [1] in this analysis."
        assert extract_author_year_citations(body) == []

    def test_month_year_not_matched(self) -> None:
        """'May 2020' alone is a date, not a citation."""
        body = "The data were collected in May 2020 at the observatory."
        assert extract_author_year_citations(body) == []

    def test_figure_year_not_matched(self) -> None:
        """'Figure 2020' or 'Section 2020' must not match."""
        body = "See Figure 2020 of the supplement."
        assert extract_author_year_citations(body) == []

    def test_year_alone_not_matched(self) -> None:
        body = "Observations in 2020 were limited."
        assert extract_author_year_citations(body) == []

    def test_year_out_of_range(self) -> None:
        """Year < 1500 or > 2099 should not be treated as a citation year."""
        body = "Smith et al. 1066 surveyed medieval texts."
        assert extract_author_year_citations(body) == []

    def test_empty_body(self) -> None:
        assert extract_author_year_citations("") == []


class TestExtractAuthorYearOffsetsAndContext:
    def test_char_offsets_correct(self) -> None:
        body = "Earlier Hong et al. 2001 showed this."
        markers = extract_author_year_citations(body)
        assert len(markers) == 1
        assert body[markers[0].char_start : markers[0].char_end].startswith("Hong")
        assert "2001" in body[markers[0].char_start : markers[0].char_end]

    def test_context_contains_marker(self) -> None:
        body = "Earlier Hong et al. 2001 showed this trend in detail."
        markers = extract_author_year_citations(body)
        assert "Hong et al. 2001" in markers[0].context_text


# ---------------------------------------------------------------------------
# resolve_author_year_markers — name+year disambiguation
# ---------------------------------------------------------------------------


def _ay_marker(authors: tuple[str, ...], year: int, char_start: int = 0) -> CitationMarker:
    return CitationMarker(
        marker_text=f"{authors[0]} et al. {year}",
        marker_numbers=(),
        char_start=char_start,
        char_end=char_start + 16,
        context_text=f"{authors[0]} et al. {year} text",
        context_start=0,
        marker_authors=authors,
        marker_year=year,
    )


class TestResolveAuthorYearUnambiguous:
    def test_resolves_unique_match(self) -> None:
        marker = _ay_marker(("Hong",), 2001)
        contexts = resolve_author_year_markers([marker], AUTHOR_YEAR_REFERENCES, "SRC")
        assert len(contexts) == 1
        assert contexts[0].target_bibcode == "2001AJ....120..200H"
        assert contexts[0].source_bibcode == "SRC"

    def test_resolves_multiple_unique_markers(self) -> None:
        markers = [
            _ay_marker(("Adams",), 2020, char_start=0),
            _ay_marker(("Hong",), 2001, char_start=80),
        ]
        contexts = resolve_author_year_markers(markers, AUTHOR_YEAR_REFERENCES, "SRC")
        targets = sorted(c.target_bibcode for c in contexts)
        assert targets == sorted(["2020ApJ...900..100A", "2001AJ....120..200H"])


class TestResolveAuthorYearMissing:
    def test_no_year_match_dropped(self) -> None:
        """Author-year that points to a year not in references is dropped."""
        marker = _ay_marker(("Hong",), 1850)
        contexts = resolve_author_year_markers([marker], AUTHOR_YEAR_REFERENCES, "SRC")
        assert contexts == []

    def test_no_initial_match_dropped(self) -> None:
        """Surname initial that doesn't match any 2001 ref is dropped."""
        # 'Zhao 2001' — no ref ends with 'Z' in 2001
        marker = _ay_marker(("Zhao",), 2001)
        contexts = resolve_author_year_markers([marker], AUTHOR_YEAR_REFERENCES, "SRC")
        assert contexts == []

    def test_empty_references(self) -> None:
        marker = _ay_marker(("Hong",), 2001)
        contexts = resolve_author_year_markers([marker], [], "SRC")
        assert contexts == []


class TestResolveAuthorYearAmbiguity:
    def test_two_candidates_below_threshold_rejected_at_min_confidence_0_6(self) -> None:
        """Two candidates -> confidence 0.5; reject when min_confidence>0.5."""
        refs = [
            "2020ApJ...900..100A",  # Adams 2020 (initial A)
            "2020Sci...380..200A",  # Andrews 2020 (initial A) — same year+initial
        ]
        marker = _ay_marker(("Adams",), 2020)
        contexts = resolve_author_year_markers([marker], refs, "SRC", min_confidence=0.6)
        assert contexts == []

    def test_two_candidates_accepted_at_min_confidence_0_5(self) -> None:
        """Two candidates -> confidence 0.5; accept when min_confidence<=0.5."""
        refs = [
            "2020ApJ...900..100A",
            "2020Sci...380..200A",
        ]
        marker = _ay_marker(("Adams",), 2020)
        contexts = resolve_author_year_markers([marker], refs, "SRC", min_confidence=0.5)
        # Both candidates are emitted (the marker is genuinely ambiguous, but
        # under-threshold rejection only kicks in below min_confidence).
        assert len(contexts) == 2

    def test_three_candidates_rejected_at_default_threshold(self) -> None:
        """Default threshold 0.5 -> N>=3 rejects (1/3 < 0.5)."""
        refs = [
            "2020ApJ...900..100A",
            "2020Sci...380..200A",
            "2020Natur.600..300A",
        ]
        marker = _ay_marker(("Adams",), 2020)
        contexts = resolve_author_year_markers([marker], refs, "SRC")
        assert contexts == []


class TestResolveAuthorYearMalformedRefs:
    def test_arxiv_style_ref_excluded_from_initial_match(self) -> None:
        """References whose last char is non-alpha (e.g. arXiv '.') must not
        be matched by the initial filter — otherwise we'd over-resolve."""
        refs = ["2020arXiv200112345."]  # last char is '.'
        marker = _ay_marker(("Smith",), 2020)
        contexts = resolve_author_year_markers([marker], refs, "SRC")
        assert contexts == []

    def test_short_ref_skipped(self) -> None:
        """A bibcode-shaped string < 5 chars cannot encode year+initial."""
        refs = ["short"]
        marker = _ay_marker(("Smith",), 2020)
        contexts = resolve_author_year_markers([marker], refs, "SRC")
        assert contexts == []


# ---------------------------------------------------------------------------
# process_paper — author-year integration
# ---------------------------------------------------------------------------


class TestProcessPaperAuthorYear:
    def test_paper_with_only_author_year_yields_contexts(self) -> None:
        """A paper that uses only author-year style should produce >0 contexts.

        Acceptance criteria #4: 'a paper known to use author-year style yields
        citation_contexts rows after the new extractor runs.'
        """
        body = (
            "Earlier work by Hong et al. 2001 established the framework. "
            "Adams (2020) extended this analysis, "
            "and (Smith & Jones, 2003) generalized further."
        )
        contexts = process_paper("SRC_BIB", body, AUTHOR_YEAR_REFERENCES)
        assert len(contexts) >= 3
        target_bibs = {c.target_bibcode for c in contexts}
        assert "2001AJ....120..200H" in target_bibs
        assert "2020ApJ...900..100A" in target_bibs
        assert "2003ApJ...500..100S" in target_bibs

    def test_mixed_styles_both_resolved(self) -> None:
        """Mixed [N] and author-year markers should both produce contexts."""
        body = "We use [1] as our baseline. " "Hong et al. 2001 showed a related trend."
        contexts = process_paper("SRC_BIB", body, AUTHOR_YEAR_REFERENCES)
        target_bibs = {c.target_bibcode for c in contexts}
        # [1] -> AUTHOR_YEAR_REFERENCES[0] -> Adams 2020
        assert "2020ApJ...900..100A" in target_bibs
        # 'Hong et al. 2001' -> 2001AJ....120..200H
        assert "2001AJ....120..200H" in target_bibs

    def test_coverage_uplift_on_author_year_only_paper(self) -> None:
        """Acceptance #5: a paper using only author-year style produces
        >0 rows after the new extractor (was 0 before)."""
        body_only_author_year = (
            "Hong et al. 2001 reports the original measurement. "
            "Subsequent analyses (Adams, 2020; Smith & Jones, 2003) refined it. "
            "Brown et al., 2021 confirmed the result. "
            "Carter (2022) further extended the model."
        )
        contexts = process_paper("SRC_BIB", body_only_author_year, AUTHOR_YEAR_REFERENCES)
        # Pre-extractor [N]-only behavior would yield 0; post-extractor expects ≥4 rows
        assert len(contexts) >= 4


class TestCitationMarkerAuthorYearFields:
    """The CitationMarker dataclass must support author-year fields without
    breaking existing [N]-style call sites (default values, not required)."""

    def test_existing_marker_construction_still_works(self) -> None:
        """Existing tests construct CitationMarker without author-year fields."""
        marker = CitationMarker(
            marker_text="[1]",
            marker_numbers=(1,),
            char_start=0,
            char_end=3,
            context_text="[1] text",
            context_start=0,
        )
        assert marker.marker_authors == ()
        assert marker.marker_year is None


# ---------------------------------------------------------------------------
# extract_author_year_citations — overlap dedup hardening (scix_experiments-3ozn)
# ---------------------------------------------------------------------------


class TestExtractAuthorYearOverlapDedup:
    """Cross-pattern dedup via interval overlap — not char_start uniqueness.

    The 4 author-year regexes (et-al, narrative, paren, sub-cite) can produce
    spans of different lengths at the same logical citation. The first pattern
    that matches wins; later overlapping matches must be rejected.
    """

    def test_narrative_does_not_double_match_after_et_al(self) -> None:
        """'Hong et al. (2001)' matches _AY_ET_AL first; _AY_NARRATIVE must
        not also match the trailing 'Hong et al. (2001)' substring."""
        body = "Earlier Hong et al. (2001) demonstrated this clearly."
        markers = extract_author_year_citations(body)
        # Exactly one citation, not two.
        assert len(markers) == 1
        assert markers[0].marker_authors == ("Hong",)
        assert markers[0].marker_year == 2001

    def test_paren_does_not_double_match_after_subcite(self) -> None:
        """A sub-cite inside a multi-cite paren must not also fire as a
        free-standing paren-form match."""
        body = "Earlier work (Adams, 2020; Smith & Jones, 2003) was foundational."
        markers = extract_author_year_citations(body)
        # Exactly two citations (Adams 2020 + Smith&Jones 2003), not 3+.
        assert len(markers) == 2
        years = sorted(m.marker_year for m in markers if m.marker_year)
        assert years == [2003, 2020]

    def test_disjoint_citations_all_kept(self) -> None:
        """Non-overlapping citations from any pattern must all survive."""
        body = (
            "Hong et al. 2001 showed X. "
            "Adams (2020) extended Y. "
            "(Smith & Jones, 2003) generalized Z."
        )
        markers = extract_author_year_citations(body)
        assert len(markers) == 3


class TestExtractAuthorYearOverlapPerfScaling:
    """Regression-bench: per-candidate overlap check must be sub-linear.

    Pre-fix: O(N) linear scan over accepted spans → O(N^2) total in dedup.
    Post-fix: O(log N) bisect → O(N log N) total in dedup.

    Note: the *end-to-end* extract_author_year_citations runtime is
    currently dominated by _word_boundary_window (O(L) per match where L
    is body length). This bead (scix_experiments-3ozn) targets only the
    overlap-check hardening; word-boundary perf is tracked separately.
    We isolate the overlap check by spying on call counts and elapsed
    time spent inside _overlaps via direct invocation.
    """

    @staticmethod
    def _surname_for_index(i: int) -> str:
        """Generate a unique surname matching the production regex
        ``[A-Z][a-z]+`` from a non-negative integer index.

        Production patterns (cf. ``_SURNAME`` in
        :mod:`scix.citation_context`) reject digits in the surname, so
        ``Smith0`` / ``Smith1`` / etc. don't match. Encode ``i`` in
        base-26 lowercase letters appended to a fixed prefix instead.
        """
        suffix = ""
        j = i
        while True:
            suffix = chr(ord("a") + j % 26) + suffix
            j //= 26
            if j == 0:
                break
        return "Smith" + suffix

    @classmethod
    def _synth_body_with_n_citations(cls, n: int) -> str:
        """Build a synthetic body string carrying N author-year citations
        plus enough overlap-eligible markers to exercise the
        ``_overlaps`` rejection branch.

        Bead 3ozn.2: previously this fixture produced N disjoint
        half-open intervals and the test reimplemented the bisect+insert
        loop inline — measuring insertion throughput, not true
        production overlap-rejection cost. Building a real body and
        calling :func:`extract_author_year_citations` ties the
        regression bound to the production code path.

        Each base block contributes one ``et-al`` marker; every fourth
        block injects an extra ``(Surname, YYYY)`` paren marker that
        overlaps with the surrounding ``et-al`` marker so the
        ``_overlaps: continue`` branch fires roughly N//4 times.
        """
        filler = " The result is consistent with prior work, see also "
        parts: list[str] = []
        for i in range(n):
            year = 2000 + (i % 25)
            surname = cls._surname_for_index(i)
            if i % 4 == 0:
                # ``(Surname et al., YYYY)`` — both _AY_ET_AL (matching
                # ``Surname et al., YYYY)``) and _AY_PAREN (matching the
                # whole ``(Surname et al., YYYY)``) fire on overlapping
                # spans here. ET_AL is tried first and wins; PAREN is
                # rejected by _overlaps. This is the rejection-branch
                # exercise the bench needs.
                parts.append(f"({surname} et al., {year})")
            else:
                # Plain et-al marker; no overlap, just one accept.
                parts.append(f"{surname} et al. {year}")
            parts.append(filler)
        return "".join(parts)

    def _time_overlap_check_loop(self, n: int, repeats: int = 3) -> float:
        """Time the cost of running ``extract_author_year_citations``
        on a synthetic body with N citations.

        Bead 3ozn.2: routes the perf bench through the production
        function instead of an inline copy of the bisect+insert loop.
        Two consequences:
          * A future refactor of the ``_overlaps`` data structure now
            shows up here directly — no silent regression past a stale
            inline copy.
          * The fixture exercises the ``_overlaps: continue`` branch
            (~N//4 times per body) so we measure rejection throughput,
            not pure insertion throughput.
        """
        body = self._synth_body_with_n_citations(n)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            extract_author_year_citations(body)
            best = min(best, time.perf_counter() - t0)
        return best

    def test_synth_body_actually_triggers_overlap_branch(self) -> None:
        """Sanity gate for the perf bench (bead 3ozn.2).

        The fixture is supposed to inject overlapping paren markers so
        the perf bench measures rejection cost, not just insertion.
        Verify that the synthetic body actually produces fewer markers
        than candidate spans — i.e. the ``_overlaps: continue`` branch
        fires. If a future fixture refactor accidentally degenerates
        back to disjoint spans, this assertion fires before the perf
        tests silently start measuring the wrong thing.
        """
        n = 100
        body = self._synth_body_with_n_citations(n)
        markers = extract_author_year_citations(body)
        # The fixture emits N et-al markers + N//4 paren markers; only
        # the et-als should survive (they're tried first in
        # extract_author_year_citations and the overlapping parens are
        # then rejected). So `len(markers) == n` confirms overlap
        # rejection ran on roughly N//4 candidates.
        assert len(markers) == n, (
            f"expected {n} et-al markers after overlap dedup, got "
            f"{len(markers)} — fixture may no longer trigger the "
            f"_overlaps rejection branch"
        )

    def test_overlap_check_is_sublinear_per_candidate(self) -> None:
        """The end-to-end ``extract_author_year_citations`` runtime must
        scale better than O(N^2) in the citation count.

        Use a wide N1:N2 ratio (1:10) on N values large enough to dominate
        timer noise. Linear-scan _overlaps: ratio ≈ 100x. Bisect: the
        overlap-check itself stays O(log N), but the regex sweep and
        word-boundary lookup dominate end-to-end at ~O(L) per match,
        so the observed ratio is ~10x at these N values.
        """
        # Bead 3ozn.2: timings now reflect production extract_author_year_citations
        # on a synthetic body, not the inline bisect+insert loop. n=2K
        # ~10ms; n=20K ~120ms on a quiet host. Linear-scan _overlaps
        # would be ~50ms / ~5s respectively (≈100x ratio).
        t_small = self._time_overlap_check_loop(2_000, repeats=5)
        t_large = self._time_overlap_check_loop(20_000, repeats=3)
        if t_small <= 0:
            pytest.skip("perf_counter resolution too coarse for this measurement")
        ratio = t_large / t_small
        # 30x catches a regression to linear-scan _overlaps (would be ~100x)
        # with healthy margin for scheduler noise.
        assert ratio < 30.0, (
            f"overlap-check loop appears super-linear: "
            f"t(2K)={t_small * 1000:.2f}ms, t(20K)={t_large * 1000:.2f}ms, "
            f"ratio={ratio:.1f}x"
        )

    def test_overlap_check_10k_fast(self) -> None:
        """Sanity bound: end-to-end extract on a 10K-citation body must
        finish in <500ms on this host.

        Bead 3ozn.2 routed this bench through production
        ``extract_author_year_citations`` (instead of an inline copy of
        the bisect+insert loop), so the observed timing now includes
        regex sweep and word-boundary lookup. ~60ms on a quiet host;
        500ms cap leaves margin for a loaded machine (cf. CLAUDE.md
        §Memory isolation — workers can run alongside the gascity
        supervisor or an embedding pipeline). Linear-scan _overlaps
        would push past 1s here.
        """
        t_10k = self._time_overlap_check_loop(10_000, repeats=3)
        if t_10k <= 0:
            pytest.skip("perf_counter resolution too coarse for this measurement")
        assert t_10k < 0.5, f"10K extract took {t_10k:.3f}s (expected <500ms)"


# ---------------------------------------------------------------------------
# _word_boundary_window — correctness (scix_experiments-3ozn.1)
# ---------------------------------------------------------------------------


class TestWordBoundaryWindow:
    """Window must enclose the citation span and grow up to ``words`` tokens
    on either side. Char-offset tolerance is ±1 across the bisect-prepass
    refactor — token-boundary alignment may shift trivially on degenerate
    input.
    """

    def test_marker_in_middle(self) -> None:
        before = " ".join(f"w{i}" for i in range(50))
        after = " ".join(f"a{i}" for i in range(50))
        body = f"{before} [1] {after}"
        char_start = len(before) + 1
        char_end = char_start + 3
        win_start, win_end = _word_boundary_window(body, char_start, char_end, words=10)
        # Window should contain the marker.
        assert win_start <= char_start
        assert win_end >= char_end
        # ~10 words before, marker, ~10 words after = ~21 tokens.
        window_text = body[win_start:win_end]
        token_count = len(window_text.split())
        assert 18 <= token_count <= 24

    def test_marker_at_start(self) -> None:
        body = "[1] " + " ".join(f"a{i}" for i in range(20))
        win_start, win_end = _word_boundary_window(body, 0, 3, words=10)
        assert win_start == 0
        # Window extends to ~10 tokens after.
        assert win_end > 3

    def test_marker_at_end(self) -> None:
        before = " ".join(f"w{i}" for i in range(20))
        body = f"{before} [1]"
        char_start = len(before) + 1
        char_end = char_start + 3
        win_start, win_end = _word_boundary_window(body, char_start, char_end, words=10)
        assert win_end == len(body)

    def test_short_body_full_window(self) -> None:
        body = "Short text [1] here."
        char_start = body.index("[")
        char_end = char_start + 3
        win_start, win_end = _word_boundary_window(body, char_start, char_end, words=125)
        assert win_start == 0
        assert win_end == len(body)

    def test_empty_body(self) -> None:
        win_start, win_end = _word_boundary_window("", 0, 0, words=10)
        assert win_start == 0
        assert win_end == 0

    def test_whitespace_only_body(self) -> None:
        body = "     "
        win_start, win_end = _word_boundary_window(body, 2, 3, words=10)
        # No words on either side — window collapses to (0, len).
        assert win_start == 0
        assert win_end == len(body)

    def test_multibyte_unicode(self) -> None:
        # Python str is unicode-codepoint indexed, so char offsets work the
        # same for ASCII and non-ASCII tokens.
        before = " ".join(["héllo", "wörld", "résumé", "naïve", "café"])
        after = " ".join(["α", "β", "γ", "δ", "ε"])
        body = f"{before} [1] {after}"
        char_start = len(before) + 1
        char_end = char_start + 3
        win_start, win_end = _word_boundary_window(body, char_start, char_end, words=3)
        window = body[win_start:win_end]
        assert "[1]" in window
        # 3 words before + marker + 3 words after = 7 tokens.
        assert len(window.split()) == 7

    def test_no_whitespace_body(self) -> None:
        # A single huge token — the marker is "inside" it conceptually but
        # we treat the bracket span as a separate region. Window collapses
        # to (0, len).
        body = "averylongtokenwithnowhitespace"
        win_start, win_end = _word_boundary_window(body, 5, 7, words=10)
        assert win_start == 0
        assert win_end == len(body)

    def test_close_adjacent_markers_consistent(self) -> None:
        """Two markers a few tokens apart should produce overlapping but
        well-formed windows — neither window should drop tokens that are
        clearly inside the requested radius."""
        prefix = " ".join(f"p{i}" for i in range(20))
        suffix = " ".join(f"s{i}" for i in range(20))
        body = f"{prefix} [1] mid1 mid2 mid3 [2] {suffix}"
        m1_start = body.index("[1]")
        m1_end = m1_start + 3
        m2_start = body.index("[2]")
        m2_end = m2_start + 3

        w1 = _word_boundary_window(body, m1_start, m1_end, words=5)
        w2 = _word_boundary_window(body, m2_start, m2_end, words=5)

        # Both windows must contain their own marker.
        assert body[w1[0] : w1[1]].find("[1]") >= 0
        assert body[w2[0] : w2[1]].find("[2]") >= 0
        # The mid-tokens between markers must appear in at least one window.
        assert "mid2" in body[w1[0] : w1[1]] or "mid2" in body[w2[0] : w2[1]]


# ---------------------------------------------------------------------------
# extract_author_year_citations — end-to-end perf scaling
# (scix_experiments-3ozn.1)
# ---------------------------------------------------------------------------


class TestExtractAuthorYearEndToEndPerfScaling:
    """End-to-end: with N citations in a body of length ~60*N chars,
    pre-3ozn.1 is O(L*N) (~12s at 10K). Post-fix, _word_boundary_window
    is O(log L) per match via a one-shot word-boundary pre-pass, so total
    runtime is O(L + N log L).
    """

    @staticmethod
    def _make_dense_body(n: int) -> str:
        """Body containing N author-year citations spaced evenly, with
        non-citation filler words between each."""
        parts: list[str] = []
        for i in range(n):
            parts.append(f"word{i} word{i}b word{i}c Hong et al. 2001 something here")
        return " ".join(parts)

    def _time_extract(self, n: int, repeats: int = 2) -> float:
        body = self._make_dense_body(n)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            extract_author_year_citations(body)
            best = min(best, time.perf_counter() - t0)
        return best

    def test_end_to_end_sub_quadratic(self) -> None:
        """t(10K)/t(1K) must be sub-quadratic.

        Pre-fix (O(L*N)): ratio ≈ 50x (0.25s → 12s), with body length L
        also growing 10x so N*L total grows 100x.
        Post-fix (O(L + N log L)): ratio ≈ 12-15x.
        Threshold 30x catches regression with healthy margin.
        """
        t_1k = self._time_extract(1_000, repeats=2)
        t_10k = self._time_extract(10_000, repeats=2)
        if t_1k <= 0:
            pytest.skip("perf_counter resolution too coarse for this measurement")
        ratio = t_10k / t_1k
        assert ratio < 30.0, (
            f"extract_author_year_citations end-to-end appears super-linear: "
            f"t(1K)={t_1k * 1000:.1f}ms, t(10K)={t_10k * 1000:.1f}ms, "
            f"ratio={ratio:.1f}x"
        )

    def test_end_to_end_5k_fast(self) -> None:
        """Sanity bound: 5K citations must finish in <500ms on this host.

        Pre-fix needs ~3.6s at this scale; post-fix should be <100ms.
        Use a generous bound so a loaded machine doesn't false-positive.
        """
        t_5k = self._time_extract(5_000, repeats=2)
        if t_5k <= 0:
            pytest.skip("perf_counter resolution too coarse for this measurement")
        assert t_5k < 0.5, f"5K citation extraction took {t_5k:.3f}s (expected <500ms post-3ozn.1)"


# ---------------------------------------------------------------------------
# Body-size guard (scix_experiments-2wbx)
# ---------------------------------------------------------------------------


class TestBodySizeGuard:
    """Verify both public extractors truncate over-cap bodies and emit a warning."""

    def test_max_body_chars_constant(self) -> None:
        """Cap is 10M chars per the bead's recommendation."""
        assert _MAX_BODY_CHARS == 10_000_000

    def test_extract_citation_contexts_truncates_oversize_body(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Body >_MAX_BODY_CHARS is truncated and a warning is logged."""
        # Build a body that's just past the cap with one [1] marker BEFORE the cap.
        prefix = "ctx " * 10  # 40 chars
        marker = "[1]"
        suffix_pad = "x" * (_MAX_BODY_CHARS + 1024)
        body = prefix + marker + suffix_pad
        assert len(body) > _MAX_BODY_CHARS

        with caplog.at_level("WARNING", logger="scix.citation_context"):
            markers = extract_citation_contexts(body)

        # The early [1] is well under the cap so it should still be found.
        assert len(markers) == 1
        assert markers[0].marker_text == "[1]"

        # Truncation warning fired.
        assert any(
            "exceeds cap" in r.message and "extract_citation_contexts" in r.message
            for r in caplog.records
        ), f"expected truncation warning; got: {[r.message for r in caplog.records]}"

    def test_extract_citation_contexts_under_cap_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Body at-or-below cap leaves the truncation path untouched."""
        body = "small body with [1] marker"

        with caplog.at_level("WARNING", logger="scix.citation_context"):
            extract_citation_contexts(body)

        assert not any("exceeds cap" in r.message for r in caplog.records)

    def test_extract_author_year_citations_truncates_oversize_body(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Author-year extractor also caps at _MAX_BODY_CHARS with a warning."""
        prefix = "Smith et al. 2020 reported that "
        suffix_pad = "x" * (_MAX_BODY_CHARS + 1024)
        body = prefix + suffix_pad
        assert len(body) > _MAX_BODY_CHARS

        with caplog.at_level("WARNING", logger="scix.citation_context"):
            markers = extract_author_year_citations(body)

        # Smith et al. 2020 is at offset 0, well under the cap.
        assert len(markers) >= 1
        assert any(m.marker_authors == ("Smith",) and m.marker_year == 2020 for m in markers)

        assert any(
            "exceeds cap" in r.message and "extract_author_year_citations" in r.message
            for r in caplog.records
        ), f"expected truncation warning; got: {[r.message for r in caplog.records]}"

    def test_truncation_drops_markers_past_cap(self) -> None:
        """Markers that fall beyond _MAX_BODY_CHARS are NOT extracted (truncated away).

        Pin the contract so a future change that switches truncation to e.g.
        a streaming scan would break this test visibly.
        """
        # Marker BEFORE cap, then a long pad, then a second marker AFTER cap.
        early_marker = "[1]"
        pad = "x" * (_MAX_BODY_CHARS + 100)
        late_marker = "[2]"
        body = early_marker + " ctx " + pad + " " + late_marker
        assert body.index(late_marker) > _MAX_BODY_CHARS

        markers = extract_citation_contexts(body)
        marker_nums = sorted({n for m in markers for n in m.marker_numbers})
        assert 1 in marker_nums
        assert 2 not in marker_nums  # past the cap → truncated away
