"""Unit tests for src/scix/sources/retraction_watch.py — CSV correction feed.

Parser-level golden-file coverage for the Retraction Watch CSV harvester. The
upstream CSV schema has drifted before (see ``harvester_api_issues``), and a
silent parse break would cascade into missing retraction events, so these
tests pin the parse contract against a captured fixture payload.

No network, no DB: the parser operates on raw CSV text only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scix.sources.retraction_watch import (
    EVENT_TYPE,
    SOURCE_TAG,
    _normalize_date,
    _normalize_doi,
    parse_retraction_watch_csv,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "retraction_watch_sample.csv"


# ---------------------------------------------------------------------------
# Golden-file: full parse of the captured CSV payload
# ---------------------------------------------------------------------------

# Verified by hand against tests/fixtures/retraction_watch_sample.csv:
# 7 data rows, row 1006 dropped (empty DOI + date), DOIs lower-cased, dates
# already ISO. Every event is type=retraction source=retraction_watch.
_GOLDEN_EVENTS = [
    {
        "type": "retraction",
        "source": "retraction_watch",
        "doi": "10.1103/physrevlett.112.241101",
        "date": "2015-03-30",
    },
    {
        "type": "retraction",
        "source": "retraction_watch",
        "doi": "10.1038/nature12373",
        "date": "2014-07-15",
    },
    {
        "type": "retraction",
        "source": "retraction_watch",
        "doi": "10.1086/183954",
        "date": "1985-12-01",
    },
    {
        "type": "retraction",
        "source": "retraction_watch",
        "doi": "10.1051/0004-6361/201424563",
        "date": "2015-08-12",
    },
    {
        "type": "retraction",
        "source": "retraction_watch",
        "doi": "10.1088/0004-637x/799/1/77",
        "date": "2015-04-10",
    },
    {
        "type": "retraction",
        "source": "retraction_watch",
        "doi": "10.1038/s41586-019-1666-5",
        "date": "2020-01-22",
    },
]


@pytest.mark.unit
def test_parse_golden_fixture_matches_expected() -> None:
    text = _FIXTURE.read_text(encoding="utf-8")
    events = list(parse_retraction_watch_csv(text))
    assert events == _GOLDEN_EVENTS


@pytest.mark.unit
def test_row_missing_doi_or_date_is_skipped() -> None:
    text = _FIXTURE.read_text(encoding="utf-8")
    events = list(parse_retraction_watch_csv(text))
    # The fixture's RecordID 1006 has an empty DOI and date and must not appear.
    assert len(events) == 6
    assert all(e["doi"] and e["date"] for e in events)


@pytest.mark.unit
def test_all_events_use_canonical_type_and_source() -> None:
    text = _FIXTURE.read_text(encoding="utf-8")
    events = list(parse_retraction_watch_csv(text))
    assert {e["type"] for e in events} == {EVENT_TYPE}
    assert {e["source"] for e in events} == {SOURCE_TAG}


@pytest.mark.unit
def test_empty_payload_yields_nothing() -> None:
    # Header-only CSV: no data rows.
    header = "RecordID,OriginalPaperDOI,RetractionDOI,RetractionDate,Reason\n"
    assert list(parse_retraction_watch_csv(header)) == []
    assert list(parse_retraction_watch_csv("")) == []


# ---------------------------------------------------------------------------
# Helper: DOI normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://doi.org/10.1/ABC", "10.1/abc"),
        ("http://doi.org/10.1/ABC", "10.1/abc"),
        ("doi:10.1/ABC", "10.1/abc"),
        ("  10.1/ABC  ", "10.1/abc"),
        ("", None),
        ("   ", None),
    ],
)
def test_normalize_doi(raw: str, expected: str | None) -> None:
    assert _normalize_doi(raw) == expected


# ---------------------------------------------------------------------------
# Helper: date normalization across observed formats
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2015-03-30", "2015-03-30"),
        ("03/30/2015", "2015-03-30"),  # %m/%d/%Y
        ("2015/03/30", "2015-03-30"),  # %Y/%m/%d
        ("not-a-date", None),
        ("", None),
        ("   ", None),
    ],
)
def test_normalize_date(raw: str, expected: str | None) -> None:
    assert _normalize_date(raw) == expected
