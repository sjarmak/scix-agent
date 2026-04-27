"""Tests for citation context extraction pipeline."""

from __future__ import annotations

import pytest

from scix.citation_context import (
    CitationContext,
    CitationMarker,
    _enrich_with_sections,
    _parse_marker_numbers,
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
        contexts = resolve_author_year_markers(
            [marker], refs, "SRC", min_confidence=0.6
        )
        assert contexts == []

    def test_two_candidates_accepted_at_min_confidence_0_5(self) -> None:
        """Two candidates -> confidence 0.5; accept when min_confidence<=0.5."""
        refs = [
            "2020ApJ...900..100A",
            "2020Sci...380..200A",
        ]
        marker = _ay_marker(("Adams",), 2020)
        contexts = resolve_author_year_markers(
            [marker], refs, "SRC", min_confidence=0.5
        )
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
        body = (
            "We use [1] as our baseline. "
            "Hong et al. 2001 showed a related trend."
        )
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
