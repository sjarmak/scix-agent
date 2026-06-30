"""Unit tests for src/scix/sources/journal_errata_rss.py — RSS errata feed.

Parser-level golden-file coverage for the publisher-RSS correction harvester.
Pins title classification, DOI extraction (across dc:identifier / guid / link
/ title fallback), and RFC-822 pubDate normalization against a captured ApJ
feed payload.

No network, no DB: the parser operates on raw RSS XML text only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scix.sources.journal_errata_rss import (
    SOURCE_TAG,
    _classify_title,
    _normalize_pubdate,
    parse_rss_feed,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "journal_errata_rss_sample.xml"


# ---------------------------------------------------------------------------
# Golden-file: full parse of the captured ApJ feed payload
# ---------------------------------------------------------------------------

# Verified by hand against tests/fixtures/journal_errata_rss_sample.xml:
# 7 items -> 6 events. The non-correction item ("A new model for galactic
# dynamos") is dropped. DOIs are pulled from the link/guid and lower-cased;
# RFC-822 pubDates collapse to YYYY-MM-DD. source is always journal_rss.
_GOLDEN_EVENTS = [
    {
        "type": "erratum",
        "source": "journal_rss",
        "doi": "10.3847/1538-4357/abcd0001",
        "date": "2023-08-14",
    },
    {
        "type": "correction",
        "source": "journal_rss",
        "doi": "10.3847/1538-4357/abcd0002",
        "date": "2022-11-02",
    },
    {
        "type": "retraction",
        "source": "journal_rss",
        "doi": "10.3847/1538-4357/abcd0003",
        "date": "2022-07-15",
    },
    {
        "type": "expression_of_concern",
        "source": "journal_rss",
        "doi": "10.3847/1538-4357/abcd0004",
        "date": "2023-03-21",
    },
    {
        "type": "recalibration_supersession",
        "source": "journal_rss",
        "doi": "10.3847/1538-4357/abcd0005",
        "date": "2023-10-05",
    },
    {
        "type": "erratum",
        "source": "journal_rss",
        "doi": "10.3847/1538-4357/abcd0007",
        "date": "2024-01-06",
    },
]


@pytest.mark.unit
def test_parse_golden_fixture_matches_expected() -> None:
    text = _FIXTURE.read_text(encoding="utf-8")
    assert list(parse_rss_feed(text)) == _GOLDEN_EVENTS


@pytest.mark.unit
def test_non_correction_item_is_dropped() -> None:
    text = _FIXTURE.read_text(encoding="utf-8")
    events = list(parse_rss_feed(text))
    assert len(events) == 6
    assert {e["source"] for e in events} == {SOURCE_TAG}


@pytest.mark.unit
def test_classifies_all_five_event_types() -> None:
    text = _FIXTURE.read_text(encoding="utf-8")
    events = list(parse_rss_feed(text))
    assert {e["type"] for e in events} == {
        "erratum",
        "correction",
        "retraction",
        "expression_of_concern",
        "recalibration_supersession",
    }


@pytest.mark.unit
def test_custom_source_label_propagates() -> None:
    text = _FIXTURE.read_text(encoding="utf-8")
    events = list(parse_rss_feed(text, source_label="apj_recent"))
    assert events
    assert {e["source"] for e in events} == {"apj_recent"}


@pytest.mark.unit
def test_malformed_xml_yields_nothing() -> None:
    assert list(parse_rss_feed("<rss><channel><item>")) == []
    assert list(parse_rss_feed("")) == []


# ---------------------------------------------------------------------------
# Helper: title classification
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Retraction Notice: foo", "retraction"),
        ("Withdrawal of paper X", "retraction"),
        ("Expression of Concern: bar", "expression_of_concern"),
        ("Erratum: baz", "erratum"),
        ("Corrigendum: qux", "erratum"),
        ("Correction to: quux", "correction"),
        ("Recalibration of the distance scale", "recalibration_supersession"),
        ("Superseded by new analysis", "recalibration_supersession"),
        ("A normal research article", None),
        (None, None),
        ("", None),
    ],
)
def test_classify_title(title: str | None, expected: str | None) -> None:
    assert _classify_title(title) == expected


# ---------------------------------------------------------------------------
# Helper: pubDate normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Mon, 14 Aug 2023 00:00:00 +0000", "2023-08-14"),  # RFC-822 with tz
        ("Mon, 14 Aug 2023 00:00:00 GMT", "2023-08-14"),  # RFC-822 named zone
        ("2023-08-14T09:30:00Z", "2023-08-14"),  # ISO-8601 Z
        ("2023-08-14", "2023-08-14"),  # bare ISO date
        ("garbage", None),
        (None, None),
        ("", None),
    ],
)
def test_normalize_pubdate(raw: str | None, expected: str | None) -> None:
    assert _normalize_pubdate(raw) == expected
