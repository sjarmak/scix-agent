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
    assert PARSER_VERSION == "ads_body_regex@v1"


def test_section_is_frozen_dataclass_with_required_fields() -> None:
    """Section is a frozen dataclass with heading/level/text/offset."""
    field_names = {f.name for f in fields(Section)}
    assert field_names == {"heading", "level", "text", "offset"}

    s = Section(heading="Intro", level=1, text="body", offset=0)
    with pytest.raises(FrozenInstanceError):
        s.heading = "mutated"  # type: ignore[misc]


def test_return_shape_is_tuple_of_list_and_dict() -> None:
    """parse_ads_body returns (list[Section], dict) with the required keys."""
    result = parse_ads_body("1 Introduction\nHello.\n", bibstem="MNRAS")
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
    assert meta["bibstem_family"] == "mnras"


def test_single_section_mnras_body() -> None:
    """One MNRAS-style heading produces exactly one Section."""
    body = "1 Introduction\nThis paper studies quasars.\n"
    sections, meta = parse_ads_body(body, bibstem="MNRAS")
    assert meta["n_sections"] == 1
    assert len(sections) == 1
    assert sections[0].heading == "Introduction"
    assert sections[0].level == 1
    assert "quasars" in sections[0].text
    assert sections[0].offset == 0


def test_mnras_numbered_headings() -> None:
    """MNRAS pattern splits bare ``N  Heading`` headings into sections."""
    body = (
        "1 Introduction\n"
        "We introduce the topic.\n"
        "2 Methods\n"
        "We apply standard techniques.\n"
        "3 Results\n"
        "Numbers follow.\n"
    )
    sections, meta = parse_ads_body(body, bibstem="MNRAS")
    headings = [s.heading for s in sections]
    assert headings == ["Introduction", "Methods", "Results"]
    assert meta["n_sections"] == 3
    assert meta["bibstem_family"] == "mnras"
    assert meta["patterns_tried"] == 1


def test_apj_numbered_with_descenders() -> None:
    """ApJ pattern matches ``N. Word-case`` headings with lowercase descenders."""
    body = (
        "1. Introduction\n"
        "Intro prose with descenders like g, j, p, q, y.\n"
        "2. Observations\n"
        "Some data description.\n"
        "3. Discussion\n"
        "Wrap-up.\n"
    )
    sections, meta = parse_ads_body(body, bibstem="ApJ")
    assert [s.heading for s in sections] == [
        "Introduction",
        "Observations",
        "Discussion",
    ]
    assert meta["bibstem_family"] == "apj"
    # MNRAS pattern would reject "1. Introduction" because the digit needs to
    # be followed by whitespace, not a dot; confirm the apj family is used.
    assert meta["n_sections"] == 3


def test_physrev_roman_numeral_headings() -> None:
    """PhysRev pattern recognizes ``I.`` / ``II.`` roman-numeral headings."""
    body = (
        "I. INTRODUCTION\n"
        "Physics intro text.\n"
        "II. THEORY\n"
        "Theoretical framework.\n"
        "III. RESULTS\n"
        "Numerical results.\n"
    )
    sections, meta = parse_ads_body(body, bibstem="PhRvD")
    assert [s.heading for s in sections] == ["INTRODUCTION", "THEORY", "RESULTS"]
    assert meta["bibstem_family"] == "physrev"


def test_bare_heading_fallback_allcaps() -> None:
    """Fallback family matches bare ALL-CAPS headings when bibstem is None."""
    body = (
        "INTRODUCTION\n"
        "Opening remarks.\n"
        "METHODS\n"
        "Procedure.\n"
        "CONCLUSIONS\n"
        "Wrap-up.\n"
    )
    sections, meta = parse_ads_body(body, bibstem=None)
    assert [s.heading for s in sections] == ["INTRODUCTION", "METHODS", "CONCLUSIONS"]
    assert meta["bibstem_family"] == "fallback"
    # Fallback tries the whole bank.
    assert meta["patterns_tried"] >= 2


def test_unknown_bibstem_uses_fallback() -> None:
    """A bibstem not in the family map routes to the fallback family."""
    body = "INTRODUCTION\nSome text.\nMETHODS\nMore text.\n"
    sections, meta = parse_ads_body(body, bibstem="Nature")
    assert meta["bibstem_family"] == "fallback"
    assert meta["n_sections"] == 2
    assert [s.heading for s in sections] == ["INTRODUCTION", "METHODS"]


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_section_offsets_are_non_decreasing() -> None:
    """Offsets returned by parse_ads_body are monotone non-decreasing and >= 0."""
    body = (
        "1 Introduction\n"
        "Intro text.\n"
        "2 Methods\n"
        "Methods text.\n"
        "3 Results\n"
        "Results text.\n"
        "4 Discussion\n"
        "Discussion text.\n"
    )
    sections, _ = parse_ads_body(body, bibstem="MNRAS")
    offsets = [s.offset for s in sections]
    assert all(o >= 0 for o in offsets)
    assert offsets == sorted(offsets)
    # Strictly increasing for this well-formed input (all headings distinct).
    assert all(a < b for a, b in zip(offsets, offsets[1:], strict=False))


def test_first_heading_offset_matches_first_section() -> None:
    """metadata.first_heading_offset == sections[0].offset when non-empty."""
    # Put the first heading at a known non-zero offset to exercise the field.
    preamble = "Preamble text with no heading.\n\n"
    body = preamble + "1 Introduction\nIntroductory body.\n"
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
