"""Tests for section_parser — regex-based section splitter for astronomy papers."""

from __future__ import annotations

from scix.section_parser import parse_sections


class TestStandardIMRaD:
    """Standard IMRaD paper with numbered sections."""

    BODY = (
        "1. Introduction\n"
        "Stars form in molecular clouds.\n"
        "\n"
        "2. Methods\n"
        "We observed using the VLT.\n"
        "\n"
        "3. Results\n"
        "We found 42 new objects.\n"
        "\n"
        "4. Discussion\n"
        "Our findings are consistent with theory.\n"
        "\n"
        "5. Conclusions\n"
        "We conclude that stars are cool.\n"
    )

    def test_section_count(self) -> None:
        sections = parse_sections(self.BODY)
        assert len(sections) == 5

    def test_section_names(self) -> None:
        sections = parse_sections(self.BODY)
        names = [s[0] for s in sections]
        assert names == ["introduction", "methods", "results", "discussion", "conclusions"]

    def test_section_text_content(self) -> None:
        sections = parse_sections(self.BODY)
        # Introduction section text should contain the intro content
        assert "molecular clouds" in sections[0][3]
        # Results section text
        assert "42 new objects" in sections[2][3]

    def test_section_spans_cover_body(self) -> None:
        sections = parse_sections(self.BODY)
        # First section starts at 0
        assert sections[0][1] == 0
        # Last section ends at len(body)
        assert sections[-1][2] == len(self.BODY)

    def test_no_gaps_between_sections(self) -> None:
        sections = parse_sections(self.BODY)
        for i in range(len(sections) - 1):
            assert sections[i][2] == sections[i + 1][1]


class TestNoSections:
    """Paper without recognizable section headers."""

    BODY = (
        "This paper studies the evolution of galaxies. "
        "We find that mergers play a significant role. "
        "The implications are discussed in detail."
    )

    def test_returns_single_full_tuple(self) -> None:
        sections = parse_sections(self.BODY)
        assert len(sections) == 1

    def test_full_section_name(self) -> None:
        sections = parse_sections(self.BODY)
        assert sections[0][0] == "full"

    def test_full_section_spans(self) -> None:
        sections = parse_sections(self.BODY)
        assert sections[0][1] == 0
        assert sections[0][2] == len(self.BODY)

    def test_full_section_text(self) -> None:
        sections = parse_sections(self.BODY)
        assert sections[0][3] == self.BODY


class TestEmptyBody:
    """Edge case: empty string."""

    def test_empty_string(self) -> None:
        sections = parse_sections("")
        assert sections == [("full", 0, 0, "")]


class TestMixedCaseHeaders:
    """Headers in various cases: UPPER, Title, lower."""

    def test_uppercase_headers(self) -> None:
        body = "INTRODUCTION\nSome text.\n\nRESULTS\nMore text.\n"
        sections = parse_sections(body)
        names = [s[0] for s in sections]
        assert "introduction" in names
        assert "results" in names

    def test_title_case_headers(self) -> None:
        body = "Introduction\nSome text.\n\nResults\nMore text.\n"
        sections = parse_sections(body)
        names = [s[0] for s in sections]
        assert "introduction" in names
        assert "results" in names

    def test_lowercase_headers(self) -> None:
        body = "introduction\nSome text.\n\nresults\nMore text.\n"
        sections = parse_sections(body)
        names = [s[0] for s in sections]
        assert "introduction" in names
        assert "results" in names


class TestNumberedSections:
    """Various numbering formats."""

    def test_dot_numbered(self) -> None:
        body = "1. Introduction\nText here.\n\n2. Methods\nMore text.\n"
        sections = parse_sections(body)
        assert sections[0][0] == "introduction"
        assert sections[1][0] == "methods"

    def test_number_without_dot(self) -> None:
        body = "1 Introduction\nText here.\n\n2 Methods\nMore text.\n"
        sections = parse_sections(body)
        assert sections[0][0] == "introduction"
        assert sections[1][0] == "methods"

    def test_roman_numeral_numbered(self) -> None:
        body = "I. Introduction\nText.\n\nII. Results\nMore text.\n"
        sections = parse_sections(body)
        assert sections[0][0] == "introduction"
        assert sections[1][0] == "results"

    def test_subsection_numbered(self) -> None:
        body = "1. Introduction\nOpening.\n\n2.1 Methods\nProcedure.\n\n3. Results\nFindings.\n"
        sections = parse_sections(body)
        names = [s[0] for s in sections]
        assert "introduction" in names
        assert "methods" in names
        assert "results" in names


class TestSubsectionsAndPreamble:
    """Subsections and text before first header."""

    def test_preamble_captured(self) -> None:
        body = "Some preamble text here.\n\nIntroduction\nActual intro.\n"
        sections = parse_sections(body)
        assert sections[0][0] == "preamble"
        assert "preamble text" in sections[0][3]
        assert sections[1][0] == "introduction"

    def test_no_preamble_when_header_first(self) -> None:
        body = "Introduction\nSome text.\n"
        sections = parse_sections(body)
        assert sections[0][0] == "introduction"


class TestAliasNormalization:
    """Alias names normalize to canonical forms."""

    def test_methodology_to_methods(self) -> None:
        body = "Methodology\nWe did stuff.\n\nConclusion\nDone.\n"
        sections = parse_sections(body)
        assert sections[0][0] == "methods"
        assert sections[1][0] == "conclusions"

    def test_acknowledgements_spelling(self) -> None:
        body = "Introduction\nHello.\n\nAcknowledgements\nThanks.\n"
        sections = parse_sections(body)
        assert sections[1][0] == "acknowledgments"

    def test_summary_and_conclusions(self) -> None:
        body = "Introduction\nHello.\n\nSummary and Conclusions\nWe are done.\n"
        sections = parse_sections(body)
        assert sections[1][0] == "conclusions"


class TestObservationsSection:
    """Astronomy-specific 'Observations' section."""

    def test_observations_header(self) -> None:
        body = (
            "1. Introduction\nBackground.\n\n"
            "2. Observations\nWe pointed the telescope.\n\n"
            "3. Results\nWe saw things.\n"
        )
        sections = parse_sections(body)
        names = [s[0] for s in sections]
        assert names == ["introduction", "observations", "results"]


class TestTrailingPeriodOnHeader:
    """Some papers have trailing periods after section names."""

    def test_header_with_trailing_period(self) -> None:
        body = "Introduction.\nSome text.\n\nResults.\nMore text.\n"
        sections = parse_sections(body)
        names = [s[0] for s in sections]
        assert "introduction" in names
        assert "results" in names
