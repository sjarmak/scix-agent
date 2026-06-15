"""Unit tests for src/scix/sources/crossref_update_to.py — relation harvester.

Parser-level golden-file coverage for the Crossref ``message.relation``
correction harvester. Pins the relation-key/update_type mapping, the
date-parts decoding, and the target-DOI emission rule against a captured
``/works/{doi}`` payload.

No network, no DB: the parser operates on a decoded JSON dict only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scix.sources.crossref_update_to import (
    SOURCE_TAG,
    _extract_date,
    _map_crossref_type,
    _normalize_doi,
    parse_crossref_work,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "crossref_update_to_sample.json"


def _load_payload() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Golden-file: full parse of the captured /works payload
# ---------------------------------------------------------------------------

# Verified by hand against tests/fixtures/crossref_update_to_sample.json:
# 3 relation entries -> 3 events. Every emitted DOI is the *target* paper
# (message.DOI), normalized lower-case. update-to carries an explicit
# update_type ("correction"); is-corrected-by has no update_type so it maps
# from the relation key; the nested date-parts form decodes to YYYY-MM-DD.
_TARGET_DOI = "10.1103/physrevlett.112.241101"
_GOLDEN_EVENTS = [
    {"type": "correction", "source": "crossref", "doi": _TARGET_DOI, "date": "2014-09-19"},
    {"type": "correction", "source": "crossref", "doi": _TARGET_DOI, "date": "2015-03-09"},
    {"type": "retraction", "source": "crossref", "doi": _TARGET_DOI, "date": "2015-12-01"},
]


@pytest.mark.unit
def test_parse_golden_fixture_matches_expected() -> None:
    assert list(parse_crossref_work(_load_payload())) == _GOLDEN_EVENTS


@pytest.mark.unit
def test_all_events_target_paper_doi_and_source() -> None:
    events = list(parse_crossref_work(_load_payload()))
    assert all(e["doi"] == _TARGET_DOI for e in events)
    assert {e["source"] for e in events} == {SOURCE_TAG}


@pytest.mark.unit
def test_missing_message_or_relation_yields_nothing() -> None:
    assert list(parse_crossref_work({})) == []
    assert list(parse_crossref_work({"message": None})) == []
    assert list(parse_crossref_work({"message": {"DOI": "10.1/x"}})) == []
    assert list(parse_crossref_work({"message": {"relation": None}})) == []


# ---------------------------------------------------------------------------
# Helper: label -> event-type mapping (drift-tolerant fallback)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("correction", "correction"),
        ("retraction", "retraction"),
        ("erratum", "erratum"),
        ("Expression of Concern", "expression_of_concern"),  # spaces -> underscores
        ("expression-of-concern", "expression_of_concern"),
        ("is-corrected-by", "correction"),
        ("is-retracted-by", "retraction"),
        ("withdrawal", "retraction"),
        ("removal", "retraction"),
        (None, "correction"),  # default fallback
        ("totally-unknown-label", "correction"),  # unknown -> default
    ],
)
def test_map_crossref_type(label: str | None, expected: str) -> None:
    assert _map_crossref_type(label) == expected


# ---------------------------------------------------------------------------
# Helper: DOI normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://doi.org/10.1/AB", "10.1/ab"),
        ("doi:10.1/AB", "10.1/ab"),
        ("10.1/AB", "10.1/ab"),
        (None, None),
        (123, None),  # non-string
        ("", None),
    ],
)
def test_normalize_doi(raw: object, expected: str | None) -> None:
    assert _normalize_doi(raw) == expected


# ---------------------------------------------------------------------------
# Helper: date extraction (string and date-parts forms)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_date_string_form() -> None:
    assert _extract_date({"date": "2014-09-19T00:00:00Z"}) == "2014-09-19"


@pytest.mark.unit
def test_extract_date_parts_form() -> None:
    assert _extract_date({"date": {"date-parts": [[2015, 3, 9]]}}) == "2015-03-09"
    # Partial date-parts default month/day to 1.
    assert _extract_date({"updated": {"date-parts": [[2020]]}}) == "2020-01-01"


@pytest.mark.unit
def test_extract_date_missing_returns_none() -> None:
    assert _extract_date({}) is None
    assert _extract_date({"date": {"date-parts": []}}) is None
