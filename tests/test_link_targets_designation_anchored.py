"""Unit tests for tier-3 designation-anchored target linker (xz4.9).

Pure logic tests — no DB required. Each test exercises one of:
* designation regex classification
* surface partitioning into name vs designation rows
* per-paper co-presence rule (named entity needs both, designation-only
  passes alone)
* confidence scoring
* min-length filter on name surfaces
"""

from __future__ import annotations

import sys
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import link_targets_designation_anchored as anchored
from link_targets_designation_anchored import (
    AnchoredLink,
    FetchedSurfaces,
    build_entity_rows,
    is_designation_shape,
    link_paper,
    _name_min_length_filter,
)
from scix.aho_corasick import build_automaton


# ---------------------------------------------------------------------------
# Designation regex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "surface",
    [
        "(2160)",
        "(2160) Spitzer",
        "(690713) 2014 KZ113",
        "2014 KZ113",
        "1999 AP10",
        "2003 EH1",
        "1P/Halley",
        "45P/Honda",
        "C/2017 K2",
        "C/1995 O1",
        "D/1993 F2",
        "P/2010 A2",
        "I/2017 U1",
    ],
)
def test_designation_shape_true(surface: str) -> None:
    assert is_designation_shape(surface)


@pytest.mark.parametrize(
    "surface",
    [
        "Spitzer",
        "Apollo",
        "Ceres",
        "Hayabusa",
        "Hale-Bopp",
        "Comet Hale-Bopp",  # 'Comet' prefix is not a designation
        "Spitzer Space Telescope",
        "the",
        "May",
        "2014",  # year alone — too generic
        "(twothousand)",  # parens but non-numeric
        "ABC123",  # not a year-leading designation
    ],
)
def test_designation_shape_false(surface: str) -> None:
    assert not is_designation_shape(surface)


# ---------------------------------------------------------------------------
# Name min-length filter
# ---------------------------------------------------------------------------


def test_min_length_filter_keeps_long_single_tokens() -> None:
    assert _name_min_length_filter("Spitzer", 4)
    assert _name_min_length_filter("Apollo", 4)


def test_min_length_filter_drops_short_single_tokens() -> None:
    assert not _name_min_length_filter("Io", 4)
    assert not _name_min_length_filter("Ra", 4)


def test_min_length_filter_keeps_multi_token_short_strings() -> None:
    # "X-1" is short but multi-token — kept.
    assert _name_min_length_filter("X-1", 4)


def test_min_length_filter_drops_empty() -> None:
    assert not _name_min_length_filter("", 4)
    assert not _name_min_length_filter("   ", 4)


# ---------------------------------------------------------------------------
# Surface partitioning
# ---------------------------------------------------------------------------


def _named_entity(eid: int, name: str) -> FetchedSurfaces:
    """Helper: a named asteroid (canonical is a name, has paren alias)."""
    return FetchedSurfaces(
        entity_id=eid,
        canonical_name=name,
        name_surfaces=(name,),
        designation_surfaces=(f"({eid}) {name}",),
        has_name_anchor=True,
    )


def _provisional_entity(eid: int, designation: str) -> FetchedSurfaces:
    """Helper: a provisional designation (canonical is a designation)."""
    return FetchedSurfaces(
        entity_id=eid,
        canonical_name=designation,
        name_surfaces=(),
        designation_surfaces=(designation, f"({eid}) {designation}"),
        has_name_anchor=False,
    )


def test_build_entity_rows_splits_named() -> None:
    ent = _named_entity(2160, "Spitzer")
    name_rows, desig_rows = build_entity_rows([ent])
    assert {r.surface for r in name_rows} == {"Spitzer"}
    assert {r.surface for r in desig_rows} == {"(2160) Spitzer"}


def test_build_entity_rows_splits_provisional() -> None:
    ent = _provisional_entity(690713, "2014 KZ113")
    name_rows, desig_rows = build_entity_rows([ent])
    assert name_rows == []
    assert {r.surface for r in desig_rows} == {"2014 KZ113", "(690713) 2014 KZ113"}


def test_build_entity_rows_drops_short_names() -> None:
    ent = FetchedSurfaces(
        entity_id=85,
        canonical_name="Io",
        name_surfaces=("Io",),
        designation_surfaces=("(85) Io",),
        has_name_anchor=True,
    )
    name_rows, desig_rows = build_entity_rows([ent], name_min_len=4)
    # 'Io' is too short for the name automaton — dropped.
    assert name_rows == []
    # Designation alias still in the designation automaton.
    assert {r.surface for r in desig_rows} == {"(85) Io"}


def test_build_entity_rows_marks_alias_correctly() -> None:
    ent = _named_entity(1862, "Apollo")
    name_rows, desig_rows = build_entity_rows([ent])
    canonical_rows = [r for r in name_rows if r.surface == "Apollo"]
    assert canonical_rows and canonical_rows[0].is_alias is False
    alias_rows = [r for r in desig_rows if r.surface == "(1862) Apollo"]
    assert alias_rows and alias_rows[0].is_alias is True


# ---------------------------------------------------------------------------
# link_paper — co-presence rule
# ---------------------------------------------------------------------------


def _build_pair(
    surfaces: list[FetchedSurfaces],
):
    """Compile (name_automaton, designation_automaton, entity_index) for tests."""
    name_rows, desig_rows = build_entity_rows(surfaces)
    name_a = build_automaton(name_rows) if name_rows else None
    desig_a = build_automaton(desig_rows)
    index = {s.entity_id: s for s in surfaces}
    return name_a, desig_a, index


def test_named_entity_with_both_hits_emits_link() -> None:
    spitzer = _named_entity(2160, "Spitzer")
    name_a, desig_a, idx = _build_pair([spitzer])
    title = "Photometry of asteroid (2160) Spitzer"
    abstract = "We observed the asteroid Spitzer over three nights."
    out = link_paper("BIB1", title, abstract, name_a, desig_a, idx)
    assert len(out) == 1
    link = out[0]
    assert link.entity_id == 2160
    assert link.name_copresent is True
    assert link.matched_name == "Spitzer"
    assert "Spitzer" in link.matched_designation


def test_named_entity_name_only_no_designation_emits_nothing() -> None:
    """The Spitzer Space Telescope namesake collision must not fire."""
    spitzer = _named_entity(2160, "Spitzer")
    name_a, desig_a, idx = _build_pair([spitzer])
    title = "Spitzer Space Telescope mid-infrared survey"
    abstract = "We use Spitzer to observe protoplanetary disks."
    out = link_paper("BIB2", title, abstract, name_a, desig_a, idx)
    assert out == []


def test_named_entity_designation_only_emits_nothing() -> None:
    """Designation hit alone for a *named* entity does NOT fire — both required."""
    apollo = _named_entity(1862, "Apollo")
    name_a, desig_a, idx = _build_pair([apollo])
    # Paper that mentions only the parenthetical designation, e.g. an
    # orbit table footnote — no name in body.
    title = "Orbital elements table"
    abstract = "The object (1862) appears in row 14."
    out = link_paper("BIB3", title, abstract, name_a, desig_a, idx)
    # Designation alias is "(1862) Apollo" — does NOT match "(1862)" alone
    # since AC is whole-string. So no designation hit either, hence no link.
    assert out == []


def test_designation_only_entity_passes_with_designation_hit() -> None:
    """Provisional designation: canonical IS a designation, no name needed."""
    prov = _provisional_entity(690713, "2014 KZ113")
    name_a, desig_a, idx = _build_pair([prov])
    title = "Spectroscopy of 2014 KZ113"
    abstract = "We measured the spectrum of trans-Neptunian object 2014 KZ113."
    out = link_paper("BIB4", title, abstract, name_a, desig_a, idx)
    assert len(out) == 1
    link = out[0]
    assert link.entity_id == 690713
    assert link.name_copresent is False
    assert link.matched_name is None


def test_field_seen_recorded() -> None:
    spitzer = _named_entity(2160, "Spitzer")
    name_a, desig_a, idx = _build_pair([spitzer])
    title = "(2160) Spitzer photometry"
    abstract = "We observed Spitzer at multiple epochs."
    out = link_paper("BIB5", title, abstract, name_a, desig_a, idx)
    assert len(out) == 1
    # 'Spitzer' is in both fields; (2160) Spitzer is in title only.
    assert "title" in out[0].field_seen
    assert "abstract" in out[0].field_seen


def test_repeat_count_records_multiple_designation_hits() -> None:
    spitzer = _named_entity(2160, "Spitzer")
    name_a, desig_a, idx = _build_pair([spitzer])
    title = "(2160) Spitzer"
    abstract = "(2160) Spitzer was observed; (2160) Spitzer rotation period."
    out = link_paper("BIB6", title, abstract, name_a, desig_a, idx)
    assert len(out) == 1
    assert out[0].designation_repeat >= 2


def test_empty_text_yields_nothing() -> None:
    spitzer = _named_entity(2160, "Spitzer")
    name_a, desig_a, idx = _build_pair([spitzer])
    assert link_paper("BIB7", "", "", name_a, desig_a, idx) == []


def test_no_designation_in_corpus_means_no_links() -> None:
    """If the paper has neither name nor designation hits, return empty."""
    spitzer = _named_entity(2160, "Spitzer")
    name_a, desig_a, idx = _build_pair([spitzer])
    out = link_paper(
        "BIB8",
        "Galactic dynamics",
        "We model dark matter halo profiles.",
        name_a,
        desig_a,
        idx,
    )
    assert out == []


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def test_confidence_named_with_copresence_scores_higher_than_designation_only() -> None:
    spitzer = _named_entity(2160, "Spitzer")
    prov = _provisional_entity(690713, "2014 KZ113")
    name_a, desig_a, idx = _build_pair([spitzer, prov])

    # Co-presence link
    out_named = link_paper(
        "BIB1",
        "(2160) Spitzer photometry",
        "We observed Spitzer.",
        name_a,
        desig_a,
        idx,
    )
    # Designation-only link
    out_prov = link_paper(
        "BIB2",
        "Spectroscopy of 2014 KZ113",
        "Trans-Neptunian object.",
        name_a,
        desig_a,
        idx,
    )
    assert out_named and out_prov
    # Named with co-presence gets the bonus.
    assert out_named[0].confidence > out_prov[0].confidence


def test_confidence_within_bounds() -> None:
    spitzer = _named_entity(2160, "Spitzer")
    name_a, desig_a, idx = _build_pair([spitzer])
    out = link_paper(
        "BIB1",
        "(2160) Spitzer is observed; (2160) Spitzer rotation; Spitzer photometry",
        "Spitzer rotation period.",
        name_a,
        desig_a,
        idx,
    )
    assert out
    assert (
        anchored.CONFIDENCE_MIN
        <= out[0].confidence
        <= anchored.CONFIDENCE_MAX
    )


# ---------------------------------------------------------------------------
# Multiple-entity isolation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Designation case filter — date+preposition phrase rejection
# ---------------------------------------------------------------------------


def test_year_letter_lowercase_phrase_rejected() -> None:
    """``2020 by`` (date + preposition) must NOT match designation '2020 BY'."""
    # SsODNet entity 1416982 has canonical '2020 BY' (the asteroid).
    ent = _provisional_entity(1416982, "2020 BY")
    name_a, desig_a, idx = _build_pair([ent])
    title = "Meteoroid ablation observations"
    abstract = (
        "18 mm-sized Orionid meteoroids were captured in 2019 and 2020 "
        "by the Canadian Automated Observatory's mirror tracking system."
    )
    out = link_paper("BIB1", title, abstract, name_a, desig_a, idx)
    assert out == []


def test_year_letter_uppercase_match_kept() -> None:
    """``2020 BY`` (real designation, all upper) must match."""
    ent = _provisional_entity(1416982, "2020 BY")
    name_a, desig_a, idx = _build_pair([ent])
    title = "Photometric study of asteroid 2020 BY"
    abstract = "Spectroscopy of 2020 BY shows S-type spectral features."
    out = link_paper("BIB1", title, abstract, name_a, desig_a, idx)
    assert len(out) == 1
    assert out[0].entity_id == 1416982


def test_comet_letter_lowercase_rejected() -> None:
    """``C/2017 k2`` (lowercase letters after C/) is not a real designation."""
    ent = _provisional_entity(99999, "C/2017 K2")
    name_a, desig_a, idx = _build_pair([ent])
    # Synthetic — no real corpus example, but the case filter must still fire.
    abstract = "We saw c/2017 k2 in mixed case."
    out = link_paper("BIB1", "Title", abstract, name_a, desig_a, idx)
    assert out == []


def test_paren_designation_not_subject_to_case_filter() -> None:
    """``(2160) Spitzer`` does not need uppercase-letter recheck — it has parens."""
    ent = _named_entity(2160, "Spitzer")
    name_a, desig_a, idx = _build_pair([ent])
    title = "(2160) Spitzer photometry"
    abstract = "Observations of Spitzer over multiple epochs."
    out = link_paper("BIB1", title, abstract, name_a, desig_a, idx)
    assert len(out) == 1
    assert out[0].entity_id == 2160


def test_period_comet_designation_not_subject_to_case_filter() -> None:
    """``45P/Honda`` always passes — letter portion is the comet name."""
    ent = FetchedSurfaces(
        entity_id=12345,
        canonical_name="Honda",
        name_surfaces=("Honda",),
        designation_surfaces=("45P/Honda",),
        has_name_anchor=True,
    )
    name_a, desig_a, idx = _build_pair([ent])
    title = "Comet 45P/Honda observations"
    abstract = "We observed Honda using ground-based telescopes."
    out = link_paper("BIB1", title, abstract, name_a, desig_a, idx)
    assert len(out) == 1


def test_multiple_entities_each_judged_independently() -> None:
    spitzer = _named_entity(2160, "Spitzer")
    apollo = _named_entity(1862, "Apollo")
    prov = _provisional_entity(690713, "2014 KZ113")
    name_a, desig_a, idx = _build_pair([spitzer, apollo, prov])

    # Paper mentions Spitzer with designation, Apollo only as a name (no
    # designation), and the trans-Neptunian designation. Expected: 2 links —
    # Spitzer (co-present) and 2014 KZ113 (designation-only). Apollo
    # rejected — no co-presence.
    title = "(2160) Spitzer and 2014 KZ113"
    abstract = (
        "We compare Spitzer and Apollo missions. Spectra of trans-"
        "Neptunian object 2014 KZ113 are reported."
    )
    out = link_paper("BIB1", title, abstract, name_a, desig_a, idx)
    eids = {link.entity_id for link in out}
    assert 2160 in eids
    assert 690713 in eids
    assert 1862 not in eids
