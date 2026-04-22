"""Unit tests for ``src/scix/sources/ads_body_parser.py``.

Pure-Python regex tests — no DB, no network, no fixtures required.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from scix.sources.ads_body_parser import (
    PARSER_VERSION,
    Section,
    compute_confidence,
    parse_ads_body,
)


# ---------------------------------------------------------------------------
# Constants / shape
# ---------------------------------------------------------------------------


def test_parser_version_constant() -> None:
    """PARSER_VERSION is the exact string required by the work unit spec."""
    assert PARSER_VERSION == "ads_body_inline_v2"


def test_section_is_frozen_dataclass_with_required_fields() -> None:
    """Section is a frozen dataclass with heading/level/text/offset."""
    field_names = {f.name for f in fields(Section)}
    assert field_names == {"heading", "level", "text", "offset"}

    s = Section(heading="Intro", level=1, text="body", offset=0)
    with pytest.raises(FrozenInstanceError):
        s.heading = "mutated"  # type: ignore[misc]


def test_return_shape_is_tuple_of_list_and_dict() -> None:
    """parse_ads_body returns (list[Section], dict) with the required keys."""
    # Two canonical markers so we clear the min-2-markers threshold.
    result = parse_ads_body(
        "INTRODUCTION opening remarks METHODS procedure text.",
        bibstem="MNRAS",
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    sections, meta = result
    assert isinstance(sections, list)
    assert all(isinstance(s, Section) for s in sections)
    assert isinstance(meta, dict)
    assert set(meta.keys()) == {
        "n_sections",
        "coverage_frac",
        "first_heading_offset",
        "bibstem_family",
        "patterns_tried",
    }


# ---------------------------------------------------------------------------
# Body-parsing scenarios
# ---------------------------------------------------------------------------


def test_empty_body_returns_zero_sections() -> None:
    """Empty body yields an empty list and zeroed metadata."""
    sections, meta = parse_ads_body("", bibstem="MNRAS")
    assert sections == []
    assert meta["n_sections"] == 0
    assert meta["coverage_frac"] == 0.0
    assert meta["first_heading_offset"] == -1
    assert meta["bibstem_family"] == "inline_v2"
    assert meta["patterns_tried"] == 1


def test_flat_single_line_body_with_zero_newlines() -> None:
    """Flat body with no newlines yields >=4 ordered canonical sections.

    This is the exact fixture from acceptance criterion 13.
    """
    body = (
        "Some text INTRODUCTION more text METHODS data RESULTS numbers "
        "DISCUSSION wrap-up REFERENCES bib."
    )
    assert "\n" not in body
    sections, meta = parse_ads_body(body, bibstem="ApJ")
    headings = [s.heading for s in sections]
    # Acceptance criterion 13: at least 4 sections in the order they appear
    # in the body. ``data`` is also a canonical marker (DATA) in this
    # vocabulary, so the fixture yields 6 headings in total.
    assert headings == [
        "INTRODUCTION",
        "METHODS",
        "DATA",
        "RESULTS",
        "DISCUSSION",
        "REFERENCES",
    ]
    assert len(headings) >= 4
    assert meta["n_sections"] == len(headings)
    assert meta["bibstem_family"] == "inline_v2"
    assert meta["patterns_tried"] == 1


def test_case_insensitive_input_canonical_uppercase_output() -> None:
    """Mixed-case keywords in the body emit canonical UPPERCASE headings."""
    body = "Intro prose Introduction more text methods Results wrap."
    sections, _ = parse_ads_body(body, bibstem=None)
    assert [s.heading for s in sections] == ["INTRODUCTION", "METHODS", "RESULTS"]


def test_numbered_prefix_variants_yield_bare_canonical_heading() -> None:
    """``1 INTRODUCTION`` / ``1. Introduction`` / ``1.1 Background`` strip prefix."""
    body = "1 INTRODUCTION alpha 1. Introduction beta 1.1 Background gamma."
    sections, _ = parse_ads_body(body, bibstem="MNRAS")
    headings = [s.heading for s in sections]
    # Three canonical hits: INTRODUCTION, INTRODUCTION, BACKGROUND.
    # Distinct count = 2 (INTRODUCTION, BACKGROUND) which clears the
    # min-2-markers threshold.
    assert headings == ["INTRODUCTION", "INTRODUCTION", "BACKGROUND"]
    # First section's offset points at the '1' of "1 INTRODUCTION", not at
    # the keyword 'I'.
    assert sections[0].offset == 0


def test_numbered_prefix_offsets_include_numeric_span() -> None:
    """The heading SPAN (offset) includes the numeric prefix but STRING does not."""
    # Leading preamble then "1. Introduction" at a known non-zero offset.
    preamble = "Preamble text. "
    body = preamble + "1. Introduction prose continues. METHODS details."
    sections, meta = parse_ads_body(body, bibstem="MNRAS")
    assert sections[0].heading == "INTRODUCTION"
    assert sections[0].offset == len(preamble)
    assert meta["first_heading_offset"] == len(preamble)


def test_methodology_matches_methodology_not_methods() -> None:
    """Word-boundary correctness: ``METHODOLOGY`` does not match as ``METHODS``."""
    body = "Start INTRODUCTION body METHODOLOGY pipeline RESULTS numbers."
    sections, _ = parse_ads_body(body, bibstem=None)
    headings = [s.heading for s in sections]
    assert "METHODOLOGY" in headings
    assert "METHODS" not in headings
    assert headings == ["INTRODUCTION", "METHODOLOGY", "RESULTS"]


def test_conclusions_beats_conclusion_longest_match_wins() -> None:
    """At the same start position, ``CONCLUSIONS`` wins over ``CONCLUSION``."""
    body = "open INTRODUCTION mid CONCLUSIONS end."
    sections, _ = parse_ads_body(body, bibstem=None)
    headings = [s.heading for s in sections]
    assert headings == ["INTRODUCTION", "CONCLUSIONS"]
    # Ensure we did NOT emit the singular form.
    assert "CONCLUSION" not in headings


def test_conclusion_singular_still_matches_when_alone() -> None:
    """The singular ``CONCLUSION`` still matches when not followed by 'S'."""
    body = "open INTRODUCTION middle CONCLUSION. Final remarks."
    sections, _ = parse_ads_body(body, bibstem=None)
    assert [s.heading for s in sections] == ["INTRODUCTION", "CONCLUSION"]


def test_min_two_markers_threshold_returns_empty_on_single_marker() -> None:
    """A body with only one distinct canonical marker yields ``([], meta_zeroed)``."""
    body = "1 Introduction and then just a single heading and prose."
    sections, meta = parse_ads_body(body, bibstem="MNRAS")
    assert sections == []
    assert meta["n_sections"] == 0
    assert meta["coverage_frac"] == 0.0
    assert meta["first_heading_offset"] == -1
    assert meta["bibstem_family"] == "inline_v2"


def test_min_two_markers_threshold_counts_distinct_not_occurrences() -> None:
    """Two occurrences of the SAME marker still fail the distinct-count gate."""
    body = "INTRODUCTION first part INTRODUCTION second part more prose."
    sections, meta = parse_ads_body(body, bibstem=None)
    assert sections == []
    assert meta["n_sections"] == 0


def test_appendix_letters_emitted_as_distinct_sections() -> None:
    """``APPENDIX A`` and ``APPENDIX B`` are two distinct canonical headings."""
    body = "some prose APPENDIX A alpha content APPENDIX B beta content."
    sections, meta = parse_ads_body(body, bibstem=None)
    headings = [s.heading for s in sections]
    assert headings == ["APPENDIX A", "APPENDIX B"]
    assert meta["n_sections"] == 2


def test_appendix_alone_matches_when_no_letter_follows() -> None:
    """Plain ``APPENDIX`` (no trailing letter) matches the one-token form."""
    body = "open INTRODUCTION body content APPENDIX followed by prose."
    sections, _ = parse_ads_body(body, bibstem=None)
    headings = [s.heading for s in sections]
    assert "APPENDIX" in headings
    assert "INTRODUCTION" in headings


def test_bibstem_parameter_is_ignored() -> None:
    """Different bibstems produce identical output — bibstem is a no-op."""
    body = "INTRODUCTION alpha METHODS beta RESULTS gamma."
    sections_mnras, meta_mnras = parse_ads_body(body, bibstem="MNRAS")
    sections_apj, meta_apj = parse_ads_body(body, bibstem="ApJ")
    sections_none, meta_none = parse_ads_body(body, bibstem=None)
    sections_unknown, meta_unknown = parse_ads_body(body, bibstem="Nature")

    headings_mnras = [s.heading for s in sections_mnras]
    assert headings_mnras == [s.heading for s in sections_apj]
    assert headings_mnras == [s.heading for s in sections_none]
    assert headings_mnras == [s.heading for s in sections_unknown]

    # All metadata dicts report the same inline_v2 family tag.
    for m in (meta_mnras, meta_apj, meta_none, meta_unknown):
        assert m["bibstem_family"] == "inline_v2"
        assert m["patterns_tried"] == 1


def test_section_text_is_body_slice_between_headings() -> None:
    """Each Section's text is the body slice from end-of-heading to next-heading."""
    body = "INTRODUCTION alpha beta METHODS gamma delta."
    sections, _ = parse_ads_body(body, bibstem=None)
    assert len(sections) == 2
    # First section's text runs from the end of "INTRODUCTION" to the start
    # of "METHODS".
    first_text = sections[0].text
    assert first_text.startswith(" alpha beta ")
    assert first_text.endswith(" ")  # trailing space before "METHODS"
    # Second section's text runs from the end of "METHODS" to EOF.
    assert sections[1].text == " gamma delta."
    # All sections are level 1.
    assert all(s.level == 1 for s in sections)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_section_offsets_are_strictly_increasing_for_distinct_headings() -> None:
    """Offsets are monotone strictly increasing for distinct well-formed headings."""
    body = (
        "preamble INTRODUCTION intro text METHODS methods text "
        "RESULTS results text DISCUSSION discussion text."
    )
    sections, _ = parse_ads_body(body, bibstem="MNRAS")
    offsets = [s.offset for s in sections]
    assert all(o >= 0 for o in offsets)
    assert offsets == sorted(offsets)
    assert all(a < b for a, b in zip(offsets, offsets[1:], strict=False))


def test_first_heading_offset_matches_first_section() -> None:
    """metadata.first_heading_offset == sections[0].offset when non-empty."""
    # Put the first heading at a known non-zero offset.
    preamble = "Preamble text with no heading. "
    body = preamble + "INTRODUCTION intro body. METHODS methods body."
    sections, meta = parse_ads_body(body, bibstem="MNRAS")
    assert sections, "expected at least one section"
    assert meta["first_heading_offset"] == sections[0].offset
    assert meta["first_heading_offset"] == len(preamble)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_compute_confidence_monotone_in_n_sections() -> None:
    """With coverage/offset fixed, confidence is non-decreasing in n_sections."""
    lo = compute_confidence(n_sections=1, coverage_frac=0.5, first_heading_offset=100)
    mid = compute_confidence(n_sections=3, coverage_frac=0.5, first_heading_offset=100)
    hi = compute_confidence(n_sections=6, coverage_frac=0.5, first_heading_offset=100)
    assert lo < mid < hi
    # Saturates at the cap (6 headings), so going above doesn't move the score.
    sat = compute_confidence(
        n_sections=20, coverage_frac=0.5, first_heading_offset=100
    )
    assert sat == pytest.approx(hi)


def test_compute_confidence_monotone_in_coverage() -> None:
    """With n_sections/offset fixed, confidence is non-decreasing in coverage."""
    lo = compute_confidence(n_sections=3, coverage_frac=0.2, first_heading_offset=100)
    mid = compute_confidence(n_sections=3, coverage_frac=0.6, first_heading_offset=100)
    hi = compute_confidence(n_sections=3, coverage_frac=1.0, first_heading_offset=100)
    assert lo < mid < hi


def test_compute_confidence_bounded_in_unit_interval() -> None:
    """Confidence stays in [0, 1] for extreme inputs."""
    assert compute_confidence(0, 0.0, -1) == pytest.approx(0.0)
    assert compute_confidence(100, 1.0, 0) == pytest.approx(1.0)
    # Weird/out-of-range inputs still clamp.
    assert 0.0 <= compute_confidence(5, -0.5, 10_000) <= 1.0
    assert 0.0 <= compute_confidence(5, 2.0, -1) <= 1.0
