"""Tests for the Lafia-style informal-reference detector (src/scix/extract/lafia.py).

The gold fixture in ``tests/fixtures/lafia_informal_gold.jsonl`` is hand-crafted
(NOT drawn from real ADS bodies). It pairs realistic astronomy/CS phrasing for
positives with decoys whose cue keywords fire but whose candidate is a common
word — so the detector cannot game the eval by matching the cue alone. The
precision gate mirrors bead dbl.18's "≥85% on a validation sample" criterion.

DB-write tests use a mocked psycopg connection — no live database is needed.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scix.extract.lafia import (
    DEFAULT_MIN_CONFIDENCE,
    LAFIA_TIER,
    LINK_TYPE,
    MATCH_METHOD,
    SOURCE,
    InformalMention,
    _is_namey_token,
    _normalise_name,
    _take_leading_name,
    _take_trailing_name,
    detect_informal_references,
    insert_mentions,
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "lafia_informal_gold.jsonl"
)


# ---------------------------------------------------------------------------
# 1. Name-shape filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [
        ("SExtractor", True),    # capitalised
        ("CASA", True),          # acronym
        ("scikit-learn", True),  # internal punctuation
        ("word2vec", True),      # internal digit
        ("2MASS", True),         # leading digit
        ("the", False),          # stopword
        ("The", False),          # stopword, sentence-initial
        ("method", False),       # common noun
        ("a", False),            # too short
        ("data", False),         # stopword
        ("emcee", False),        # pure lowercase, no digit/punct (known limitation)
        ("Obf~", False),         # OCR garbage (non-name punctuation)
        ("i8óO", False),         # OCR garbage (non-ASCII)
        ("Wolbach", False),      # ADS digitization-footer provenance
        ("HARVARD", False),      # institutional provenance
    ],
)
def test_is_namey_token(token: str, expected: bool) -> None:
    assert _is_namey_token(token) is expected


def test_ocr_garbage_and_boilerplate_suppressed_in_context() -> None:
    # The recurring Harvard scanned-page footer must not yield software.
    assert detect_informal_references("John G. Wolbach Library") == []
    # OCR noise before a 'Catalog' head must not yield a dataset.
    assert detect_informal_references("Nome Aøceas. retta i8óO Catalog. Secoudo") == []


def test_take_leading_name_stops_at_non_name() -> None:
    assert _take_leading_name("PyTorch and trained on") == "PyTorch"
    assert _take_leading_name("a straightforward way") is None
    # "survey" is a head-noun stopword, so the run stops before it (the full
    # "... Sky Survey" name is recovered instead by the dataset_head suffix cue).
    assert _take_leading_name("Sloan Digital Sky Survey data") == "Sloan Digital Sky"


def test_take_trailing_name_stops_at_non_name() -> None:
    assert _take_trailing_name("reduced with the IRAF") == "IRAF"
    assert _take_trailing_name("selected from the previous") is None


def test_normalise_name_rejects_overlong_and_pure_number() -> None:
    assert _normalise_name(["A", "B", "C", "D", "E"]) is None  # > MAX tokens
    assert _normalise_name(["1234"]) is None                   # pure number
    assert _normalise_name(["GALFIT"]) == "GALFIT"


# ---------------------------------------------------------------------------
# 2. Cue catalog — representative positives MUST fire with the right type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,name,etype",
    [
        ("The code was implemented in TOPCAT for the cross-match.", "TOPCAT", "software"),
        ("Detection was performed using SExtractor.", "SExtractor", "software"),
        ("Spectra were reduced with the IRAF package.", "IRAF", "software"),
        ("We used Astropy throughout.", "Astropy", "software"),
        ("Targets were drawn from the SDSS survey.", "SDSS", "dataset"),
        ("We used data from the Gaia archive.", "Gaia", "dataset"),
        ("Parameters came from the LAMOST DR5 release.", "LAMOST", "dataset"),
    ],
)
def test_representative_cues_fire(text: str, name: str, etype: str) -> None:
    mentions = detect_informal_references(text)
    hits = {(m.canonical_name, m.entity_type) for m in mentions}
    assert (name, etype) in hits


def test_offsets_point_at_surface() -> None:
    text = "Spectra were reduced with the IRAF package following recipes."
    for m in detect_informal_references(text):
        assert text[m.start_char : m.end_char] == m.surface


def test_highest_confidence_cue_wins_for_same_name() -> None:
    # "the GALFIT software" (suffix, 0.90) beats nothing else; one row only.
    mentions = detect_informal_references("We ran the GALFIT software.")
    galfit = [m for m in mentions if m.canonical_name == "GALFIT"]
    assert len(galfit) == 1
    assert galfit[0].confidence == pytest.approx(0.90)


def test_empty_body_returns_nothing() -> None:
    assert detect_informal_references("") == []


# ---------------------------------------------------------------------------
# 3. Gold-fixture precision gate (bead dbl.18 acceptance: ≥85%)
# ---------------------------------------------------------------------------


def _load_fixture() -> list[dict]:
    with _FIXTURE_PATH.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_gold_fixture_precision_and_recall() -> None:
    rows = _load_fixture()
    assert len(rows) >= 50 or len(rows) >= 30  # sample size sanity

    total_emitted = 0
    correct_emitted = 0
    expected_total = 0
    expected_hit = 0

    for row in rows:
        accepted = {(n, t) for n, t in row["expect"]}
        expected_total += len(accepted)
        emitted = {
            (m.canonical_name, m.entity_type)
            for m in detect_informal_references(row["text"])
            if m.confidence >= DEFAULT_MIN_CONFIDENCE
        }
        total_emitted += len(emitted)
        correct_emitted += len(emitted & accepted)
        expected_hit += len(emitted & accepted)

    precision = correct_emitted / total_emitted if total_emitted else 1.0
    recall = expected_hit / expected_total if expected_total else 1.0

    assert precision >= 0.85, f"precision {precision:.3f} below 0.85 gate"
    assert recall >= 0.80, f"recall {recall:.3f} below sanity floor"


def test_decoys_stay_silent_at_default_threshold() -> None:
    for row in _load_fixture():
        if row["kind"] != "decoy":
            continue
        emitted = [
            m
            for m in detect_informal_references(row["text"])
            if m.confidence >= DEFAULT_MIN_CONFIDENCE
        ]
        assert emitted == [], f"decoy leaked: {row['text']!r} -> {emitted}"


# ---------------------------------------------------------------------------
# 4. Dataclass + DB writer
# ---------------------------------------------------------------------------


def test_informal_mention_is_frozen() -> None:
    m = InformalMention(
        surface="CASA", canonical_name="CASA", entity_type="software",
        cue_id="we_used", confidence=0.7, start_char=0, end_char=4, evidence_span="CASA",
    )
    with pytest.raises(FrozenInstanceError):
        m.surface = "other"  # type: ignore[misc]


def test_insert_mentions_empty_is_noop() -> None:
    conn = MagicMock()
    assert insert_mentions(conn, "2020bib", []) == 0
    conn.cursor.assert_not_called()


def test_insert_mentions_upserts_entities_then_doc_entities() -> None:
    mentions = detect_informal_references(
        "Detection used SExtractor and data from the SDSS survey."
    )
    mentions = [m for m in mentions if m.confidence >= DEFAULT_MIN_CONFIDENCE]
    assert mentions, "fixture sentence should yield mentions"

    cur = MagicMock()
    cur.fetchone.side_effect = [(101,), (202,)]  # entity ids for the two upserts
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    written = insert_mentions(conn, "2020bib", mentions)
    assert written == len(mentions)

    # The doc-entity write is a single executemany carrying our provenance.
    cur.executemany.assert_called_once()
    sql, rows = cur.executemany.call_args[0]
    assert "document_entities" in sql
    for r in rows:
        bibcode, eid, link_type, conf, method, tier, _evidence = r
        assert bibcode == "2020bib"
        assert link_type == LINK_TYPE
        assert method == MATCH_METHOD
        assert tier == LAFIA_TIER
        assert eid in (101, 202)


def test_provenance_constants_distinct_from_gliner() -> None:
    assert MATCH_METHOD == "lafia_informal"
    assert LINK_TYPE == "informal_mention"   # not GLiNER's 'mentions'
    assert SOURCE == "lafia"
    assert LAFIA_TIER != 4                    # GLiNER owns tier 4
