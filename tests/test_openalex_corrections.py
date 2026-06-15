"""Unit tests for src/scix/sources/openalex_corrections.py — retraction feed.

Parser-level golden-file coverage for the OpenAlex ``is_retracted`` works
harvester. Pins the parse contract (DOI prefix stripping, date-field
precedence, skip rules) and cursor pagination extraction against a captured
API page payload.

No network, no DB: the parser operates on a decoded JSON dict only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scix.sources.openalex_corrections import (
    EVENT_TYPE,
    SOURCE_TAG,
    _build_url,
    _next_cursor,
    _pick_date,
    _strip_doi_prefix,
    parse_openalex_works,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "openalex_corrections_sample.json"


def _load_payload() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Golden-file: full parse of the captured works-page payload
# ---------------------------------------------------------------------------

# Verified by hand against tests/fixtures/openalex_corrections_sample.json:
# 5 results -> 3 events. The control row (is_retracted=false) and the
# null-DOI row are both dropped. _pick_date precedence is
# (retracted_date, updated_date, publication_date), so row 2 (no
# retracted_date) falls through to updated_date trimmed to YYYY-MM-DD.
_GOLDEN_EVENTS = [
    {
        "type": "retraction",
        "source": "openalex",
        "doi": "10.1126/science.1234567",
        "date": "2018-06-12",
    },
    {
        "type": "retraction",
        "source": "openalex",
        "doi": "10.1093/mnras/stx1234",
        "date": "2020-02-14",
    },
    {
        "type": "retraction",
        "source": "openalex",
        "doi": "10.1051/0004-6361/202142001",
        "date": "2022-01-09",
    },
]


@pytest.mark.unit
def test_parse_golden_fixture_matches_expected() -> None:
    assert list(parse_openalex_works(_load_payload())) == _GOLDEN_EVENTS


@pytest.mark.unit
def test_non_retracted_and_null_doi_rows_skipped() -> None:
    events = list(parse_openalex_works(_load_payload()))
    assert len(events) == 3
    # Control row title "Not a retraction" and "Missing DOI" must not survive.
    assert all(e["doi"] for e in events)
    assert {e["type"] for e in events} == {EVENT_TYPE}
    assert {e["source"] for e in events} == {SOURCE_TAG}


@pytest.mark.unit
def test_missing_results_key_yields_nothing() -> None:
    assert list(parse_openalex_works({})) == []
    assert list(parse_openalex_works({"results": None})) == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_next_cursor_null_in_fixture() -> None:
    # The captured page has meta.next_cursor = null -> no further pages.
    assert _next_cursor(_load_payload()) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"meta": {"next_cursor": "I4Lc="}}, "I4Lc="),
        ({"meta": {"next_cursor": None}}, None),
        ({"meta": {}}, None),
        ({}, None),
    ],
)
def test_next_cursor_variants(meta: dict, expected: str | None) -> None:
    assert _next_cursor(meta) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://doi.org/10.1/AB", "10.1/ab"),
        ("http://doi.org/10.1/AB", "10.1/ab"),
        ("doi:10.1/AB", "10.1/ab"),
        ("10.1/AB", "10.1/ab"),
        (None, None),
        ("", None),
    ],
)
def test_strip_doi_prefix(raw: str | None, expected: str | None) -> None:
    assert _strip_doi_prefix(raw) == expected


@pytest.mark.unit
def test_pick_date_precedence() -> None:
    # retracted_date wins over updated_date and publication_date.
    work = {
        "retracted_date": "2018-06-12",
        "updated_date": "2019-01-01T00:00:00",
        "publication_date": "2017-03-21",
    }
    assert _pick_date(work) == "2018-06-12"
    # Falls through to updated_date when retracted_date absent (trimmed).
    assert _pick_date({"updated_date": "2020-02-14T00:00:00"}) == "2020-02-14"
    # Falls through to publication_date last.
    assert _pick_date({"publication_date": "2021-09-15"}) == "2021-09-15"
    # No usable date field.
    assert _pick_date({}) is None


@pytest.mark.unit
def test_build_url_filters_for_retracted_works() -> None:
    url = _build_url("https://api.openalex.org/works", cursor="*", per_page=50)
    assert url.startswith("https://api.openalex.org/works?")
    assert "filter=is_retracted%3Atrue" in url
    assert "per_page=50" in url
    assert "cursor=%2A" in url
