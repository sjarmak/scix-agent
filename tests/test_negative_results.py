"""Tests for the M3 negative-results detector (src/scix/negative_results.py).

The 100-span gold fixture in ``tests/fixtures/negative_results_gold_100.jsonl``
is hand-crafted by the agent, NOT drawn from real ADS papers. It uses common
astronomy phrasing for both positives (40) and decoy negatives (60) so the
detector cannot game the eval with single-keyword matching.

DB-write tests use a mocked psycopg connection — no live database is needed.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scix.negative_results import (
    EVIDENCE_SPAN_CHARS,
    EXTRACTION_TYPE,
    EXTRACTION_VERSION,
    SOURCE,
    NegativeResultSpan,
    _build_payload,
    detect_negative_results,
    insert_extractions,
)


_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "negative_results_gold_100.jsonl"
)


# ---------------------------------------------------------------------------
# 1. Pattern catalog: high-tier examples MUST fire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "snippet,expected_pattern",
    [
        ("we find no significant excess in the data", "no_significant"),
        ("the search yielded a null result", "null_result"),
        ("ALMA failed to detect the line", "failed_to_detect"),
        ("we do not detect the predicted feature", "do_not_detect"),
        ("there is no evidence for any companion", "no_evidence"),
        ("our analysis is a non-detection at 95 GHz", "non_detection"),
        ("the candidate companion is not detected", "not_detected"),
        ("we rule out the magnetar scenario", "ruled_out"),
        ("models are rejected at 4 sigma", "rejected_sigma"),
        ("the original claim has been refuted", "refuted"),
        ("the discovery paper was retracted", "retracted"),
    ],
)
def test_high_tier_patterns_fire_in_results_section(
    snippet: str, expected_pattern: str
) -> None:
    """A 'results' section containing the snippet should produce a tier=3 span
    keyed to ``expected_pattern``."""
    body = "Results\n" + snippet
    spans = detect_negative_results(body)
    assert spans, f"expected a span for snippet={snippet!r}"
    pattern_ids = {s.pattern_id for s in spans}
    assert expected_pattern in pattern_ids, (
        f"expected pattern_id={expected_pattern!r} in {pattern_ids!r} "
        f"for snippet={snippet!r}"
    )
    # All emitted spans here should be tier 3 (high).
    assert all(s.confidence_tier == 3 for s in spans), spans


# ---------------------------------------------------------------------------
# 2. Section guard
# ---------------------------------------------------------------------------


def test_section_guard_blocks_introduction_match() -> None:
    """A high-tier match in an 'introduction' section must NOT be emitted."""
    body = (
        "Introduction\n"
        "Earlier work failed to detect the line, motivating our new search.\n"
        "Results\n"
        "We measure the line at 5-sigma significance.\n"
    )
    spans = detect_negative_results(body)
    assert spans == []


def test_section_guard_blocks_methods_match() -> None:
    body = (
        "Methods\n"
        "We searched for cases where the pipeline did not detect a known calibrator.\n"
        "Results\n"
        "All sources are recovered with > 5-sigma confidence.\n"
    )
    spans = detect_negative_results(body)
    assert spans == []


def test_section_guard_allows_discussion_match() -> None:
    body = (
        "Methods\n"
        "We use standard photometric techniques.\n"
        "Discussion\n"
        "The data are consistent with no signal at the predicted frequency.\n"
    )
    spans = detect_negative_results(body)
    assert spans, "discussion-section match should fire"
    assert all(s.section == "discussion" for s in spans)


def test_full_text_no_headers_only_high_tier_emitted() -> None:
    """When parse_sections returns the catch-all 'full' bucket, tier=2/1
    matches must be suppressed; only tier=3 fires."""
    medium_only = "the data are consistent with no signal at the 2-sigma level"
    high_in_full = "we find no significant detection in the cleaned spectrum"

    spans_medium = detect_negative_results(medium_only)
    assert spans_medium == [], "medium-tier match in headerless body must NOT fire"

    spans_high = detect_negative_results(high_in_full)
    assert spans_high, "high-tier match in headerless body MUST fire"
    assert all(s.confidence_tier == 3 for s in spans_high)


# ---------------------------------------------------------------------------
# 3. Output shape invariants
# ---------------------------------------------------------------------------


def test_evidence_span_is_exactly_250_chars() -> None:
    """Acceptance criterion (5): evidence_span field is exactly 250 chars."""
    body = "Results\n" + ("padding " * 80) + "we find no significant excess. " + (
        "trailing " * 80
    )
    spans = detect_negative_results(body)
    assert spans
    for s in spans:
        assert len(s.evidence_span) == EVIDENCE_SPAN_CHARS, (
            f"len(evidence_span)={len(s.evidence_span)} expected={EVIDENCE_SPAN_CHARS}"
        )


def test_evidence_span_padded_when_body_shorter_than_window() -> None:
    body = "Results\nwe find no significant signal."
    spans = detect_negative_results(body)
    assert spans
    for s in spans:
        assert len(s.evidence_span) == EVIDENCE_SPAN_CHARS


def test_negative_result_dataclass_is_frozen() -> None:
    body = "Results\nwe do not detect the line"
    spans = detect_negative_results(body)
    assert spans
    span = spans[0]
    with pytest.raises(FrozenInstanceError):
        span.section = "mutated"  # type: ignore[misc]


def test_offsets_point_into_original_body() -> None:
    body = "Results\nwe find no significant excess in the integrated spectrum."
    spans = detect_negative_results(body)
    assert spans
    for s in spans:
        assert body[s.start_char:s.end_char].lower() == s.match_text.lower()


def test_overlapping_matches_dedup_to_highest_tier() -> None:
    """When two patterns hit overlapping windows, the higher-tier wins."""
    body = "Results\nwe find no significant excess at the 95% confidence level."
    spans = detect_negative_results(body)
    pattern_ids = {s.pattern_id for s in spans}
    # 'no_significant' (tier 3) must be present.
    assert "no_significant" in pattern_ids


def test_empty_body_returns_empty() -> None:
    assert detect_negative_results("") == []


# ---------------------------------------------------------------------------
# 4. DB writer (mocked psycopg)
# ---------------------------------------------------------------------------


def test_db_insert_targets_staging_extractions_with_correct_columns() -> None:
    """Acceptance criterion (6): writes target staging.extractions with
    extraction_type='negative_result'. Verified via a mocked psycopg cursor."""
    body = "Results\nwe find no significant detection of the line."
    spans = detect_negative_results(body)
    assert spans

    fake_cur = MagicMock()
    fake_conn = MagicMock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cur

    rows = insert_extractions(fake_conn, "2025ApJ...123..456X", spans)

    assert rows == 1
    assert fake_cur.execute.call_count == 1
    sql_text, params = fake_cur.execute.call_args.args
    assert "INSERT INTO staging.extractions" in sql_text
    assert "extraction_type" in sql_text
    assert "extraction_version" in sql_text
    assert "confidence_tier" in sql_text
    assert "source" in sql_text
    # Positional params: (bibcode, ext_type, ext_version, payload, source, tier)
    bibcode, ext_type, ext_version, payload, source, tier = params
    assert bibcode == "2025ApJ...123..456X"
    assert ext_type == EXTRACTION_TYPE == "negative_result"
    assert ext_version == EXTRACTION_VERSION
    assert source == SOURCE
    assert isinstance(tier, int)
    assert 1 <= tier <= 3
    # Jsonb wraps a dict containing the spans we computed.
    inner = payload.obj  # psycopg Jsonb stashes the dict on .obj
    assert inner["n_spans"] == len(spans)
    assert inner["extractor"] == SOURCE
    assert "spans" in inner and len(inner["spans"]) == len(spans)


def test_db_insert_with_no_spans_still_writes_row_with_null_tier() -> None:
    body = "Results\nWe report a 5-sigma detection of the molecular line."
    spans = detect_negative_results(body)
    assert spans == []  # decoy positive, must not match

    fake_cur = MagicMock()
    fake_conn = MagicMock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cur

    rows = insert_extractions(fake_conn, "2025ApJ...000..001X", spans)

    assert rows == 1
    _sql, params = fake_cur.execute.call_args.args
    bibcode, ext_type, _v, _payload, source, tier = params
    assert bibcode == "2025ApJ...000..001X"
    assert ext_type == "negative_result"
    assert source == SOURCE
    assert tier is None  # no spans -> no max tier


def test_build_payload_counts_tiers() -> None:
    spans = [
        NegativeResultSpan(
            section="results", pattern_id="no_significant",
            confidence_tier=3, confidence_label="high",
            match_text="no significant", start_char=0, end_char=14,
            evidence_span=" " * EVIDENCE_SPAN_CHARS,
        ),
        NegativeResultSpan(
            section="discussion", pattern_id="cannot_rule_out",
            confidence_tier=2, confidence_label="medium",
            match_text="cannot rule out", start_char=20, end_char=35,
            evidence_span=" " * EVIDENCE_SPAN_CHARS,
        ),
        NegativeResultSpan(
            section="conclusions", pattern_id="if_real",
            confidence_tier=1, confidence_label="low",
            match_text="signal, if real", start_char=40, end_char=55,
            evidence_span=" " * EVIDENCE_SPAN_CHARS,
        ),
    ]
    payload = _build_payload(spans)
    assert payload["n_spans"] == 3
    assert payload["tier_counts"] == {"high": 1, "medium": 1, "low": 1}
    assert payload["extractor"] == SOURCE


# ---------------------------------------------------------------------------
# 5. Precision/recall on the 100-span gold fixture
# ---------------------------------------------------------------------------


def _load_gold_fixture() -> list[dict]:
    items: list[dict] = []
    with _FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            # Filter out any record that is not a labeled span (defensive).
            if "text" not in obj or "label" not in obj:
                continue
            items.append(obj)
    return items


def test_gold_fixture_has_exactly_100_labeled_spans() -> None:
    items = _load_gold_fixture()
    assert len(items) == 100, f"expected 100 fixture rows, got {len(items)}"
    labels = {item["label"] for item in items}
    assert labels <= {True, False}


def _predict(text: str) -> bool:
    """Wrap the candidate text in a 'Results' section so the section guard
    permits all tiers to fire — mirrors how the detector will see body
    sections at runtime."""
    body = "Results\n" + text
    return bool(detect_negative_results(body))


def test_precision_recall_on_gold_fixture() -> None:
    """Acceptance criterion (4): precision >= 0.70 AND recall >= 0.60 on the
    100-span gold fixture."""
    items = _load_gold_fixture()
    tp = fp = fn = tn = 0
    misses_pos: list[str] = []
    false_alarms: list[str] = []
    for item in items:
        truth = bool(item["label"])
        pred = _predict(item["text"])
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
            misses_pos.append(item["text"])
        elif (not truth) and pred:
            fp += 1
            false_alarms.append(item["text"])
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    # Surface diagnostics on failure for fast iteration.
    diag = (
        f"tp={tp} fp={fp} fn={fn} tn={tn} "
        f"precision={precision:.3f} recall={recall:.3f}\n"
        f"first 5 missed positives: {misses_pos[:5]}\n"
        f"first 5 false alarms: {false_alarms[:5]}"
    )
    assert precision >= 0.70, f"precision below 0.70: {diag}"
    assert recall >= 0.60, f"recall below 0.60: {diag}"
