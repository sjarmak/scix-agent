"""Tests for the manual extraction quality gate script.

Unit tests use mocked database connections to verify sampling, JSONL dump,
metric aggregation, and gate-decision report formatting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from manual_extraction_eval import (  # noqa: E402
    EXTRACTION_TYPES,
    GateDecision,
    PaperSample,
    PerTypeMetrics,
    aggregate_metrics,
    decide_gate,
    dump_samples_jsonl,
    fetch_samples,
    format_gate_report,
    score_paper,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_conn(rows: list[tuple]) -> MagicMock:
    """Build a mock psycopg connection that returns rows on fetchall."""
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


# ---------------------------------------------------------------------------
# fetch_samples
# ---------------------------------------------------------------------------


def test_fetch_samples_returns_paper_samples_grouped_by_type() -> None:
    """fetch_samples should return PaperSample objects with extracted entities by type."""
    rows = [
        (
            "2024Test..1",
            "An abstract about VASP and silicon.",
            {
                "materials": ["silicon"],
                "methods": ["VASP", "DFT"],
                "instruments": [],
                "datasets": [],
            },
        ),
        (
            "2024Test..2",
            "Hubble images of a galaxy.",
            {
                "materials": [],
                "methods": [],
                "instruments": ["Hubble Space Telescope"],
                "datasets": ["MAST"],
            },
        ),
    ]
    conn = _mock_conn(rows)

    samples = fetch_samples(conn, sample_size=2, seed=7)

    assert len(samples) == 2
    assert isinstance(samples[0], PaperSample)
    assert samples[0].bibcode == "2024Test..1"
    assert samples[0].extracted["materials"] == ("silicon",)
    assert samples[0].extracted["methods"] == ("VASP", "DFT")
    assert samples[1].extracted["instruments"] == ("Hubble Space Telescope",)
    assert samples[1].extracted["datasets"] == ("MAST",)


def test_fetch_samples_handles_missing_types() -> None:
    """When the SQL row omits some extraction types, defaults should be empty tuples."""
    rows = [("2024Test..3", "abstract", {"materials": ["x"]})]
    conn = _mock_conn(rows)

    samples = fetch_samples(conn, sample_size=1, seed=1)

    assert samples[0].extracted["methods"] == ()
    assert samples[0].extracted["instruments"] == ()
    assert samples[0].extracted["datasets"] == ()
    assert samples[0].extracted["materials"] == ("x",)


# ---------------------------------------------------------------------------
# dump_samples_jsonl
# ---------------------------------------------------------------------------


def test_dump_samples_jsonl_writes_one_record_per_paper(tmp_path: Path) -> None:
    """dump_samples_jsonl writes one JSONL record per PaperSample."""
    samples = [
        PaperSample(
            bibcode="2024Test..1",
            abstract="abstract one",
            extracted={
                "materials": ("silicon",),
                "methods": ("DFT",),
                "instruments": (),
                "datasets": (),
            },
        ),
        PaperSample(
            bibcode="2024Test..2",
            abstract="abstract two",
            extracted={
                "materials": (),
                "methods": (),
                "instruments": ("Hubble",),
                "datasets": ("MAST",),
            },
        ),
    ]
    out = tmp_path / "samples.jsonl"

    dump_samples_jsonl(samples, out)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["bibcode"] == "2024Test..1"
    assert first["abstract"] == "abstract one"
    assert first["extracted"]["materials"] == ["silicon"]
    assert first["extracted"]["methods"] == ["DFT"]
    assert first["extracted"]["instruments"] == []


# ---------------------------------------------------------------------------
# score_paper
# ---------------------------------------------------------------------------


def test_score_paper_computes_tp_fp_fn_per_type() -> None:
    """score_paper compares extracted vs gold and reports tp/fp/fn per type."""
    extracted = {
        "materials": ("silicon", "aluminum"),
        "methods": ("DFT",),
        "instruments": (),
        "datasets": (),
    }
    gold = {
        "materials": ("silicon",),  # silicon TP, aluminum FP
        "methods": ("DFT", "VASP"),  # DFT TP, VASP FN
        "instruments": (),
        "datasets": ("MAST",),  # MAST FN
    }

    scored = score_paper(extracted, gold)

    assert scored["materials"] == (1, 1, 0)  # tp, fp, fn
    assert scored["methods"] == (1, 0, 1)
    assert scored["instruments"] == (0, 0, 0)
    assert scored["datasets"] == (0, 0, 1)


def test_score_paper_is_case_insensitive_and_strips_whitespace() -> None:
    """Matching tolerates simple casing/whitespace differences."""
    extracted = {
        "materials": ("Silicon ",),
        "methods": (),
        "instruments": (),
        "datasets": (),
    }
    gold = {
        "materials": ("silicon",),
        "methods": (),
        "instruments": (),
        "datasets": (),
    }

    scored = score_paper(extracted, gold)
    assert scored["materials"] == (1, 0, 0)


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------


def test_aggregate_metrics_returns_per_type_and_overall() -> None:
    """aggregate_metrics produces micro precision/recall/F1 per type and overall."""
    per_paper = [
        {
            "materials": (1, 1, 0),
            "methods": (1, 0, 1),
            "instruments": (0, 0, 0),
            "datasets": (0, 0, 1),
        },
        {
            "materials": (2, 0, 0),
            "methods": (0, 0, 0),
            "instruments": (1, 1, 0),
            "datasets": (0, 0, 0),
        },
    ]

    metrics = aggregate_metrics(per_paper)

    materials = metrics["materials"]
    assert isinstance(materials, PerTypeMetrics)
    assert materials.tp == 3
    assert materials.fp == 1
    assert materials.fn == 0
    assert materials.precision == pytest.approx(3 / 4)
    assert materials.recall == pytest.approx(1.0)
    assert materials.f1 == pytest.approx(2 * (3 / 4) * 1.0 / ((3 / 4) + 1.0))

    overall = metrics["overall"]
    assert overall.tp == 5  # 1+1+0+0 + 2+0+1+0
    assert overall.fp == 2  # 1+0+0+0 + 0+0+1+0
    assert overall.fn == 2  # 0+1+0+1 + 0+0+0+0


def test_aggregate_metrics_handles_zero_division() -> None:
    """When tp+fp or tp+fn is zero, precision/recall default to 0."""
    per_paper = [
        {
            "materials": (0, 0, 0),
            "methods": (0, 0, 0),
            "instruments": (0, 0, 0),
            "datasets": (0, 0, 0),
        }
    ]
    metrics = aggregate_metrics(per_paper)
    assert metrics["overall"].precision == 0.0
    assert metrics["overall"].recall == 0.0
    assert metrics["overall"].f1 == 0.0


# ---------------------------------------------------------------------------
# decide_gate
# ---------------------------------------------------------------------------


def test_decide_gate_proceed_when_overall_f1_above_threshold() -> None:
    metrics = {
        "overall": PerTypeMetrics(tp=80, fp=10, fn=10),
    }
    decision = decide_gate(metrics, threshold=0.6)
    assert decision == GateDecision.PROCEED


def test_decide_gate_redesign_when_overall_f1_below_threshold() -> None:
    metrics = {
        "overall": PerTypeMetrics(tp=10, fp=50, fn=40),
    }
    decision = decide_gate(metrics, threshold=0.6)
    assert decision == GateDecision.REDESIGN


# ---------------------------------------------------------------------------
# format_gate_report
# ---------------------------------------------------------------------------


def test_format_gate_report_includes_decision_and_metrics() -> None:
    metrics = {
        "materials": PerTypeMetrics(tp=3, fp=1, fn=0),
        "methods": PerTypeMetrics(tp=1, fp=0, fn=1),
        "instruments": PerTypeMetrics(tp=1, fp=1, fn=0),
        "datasets": PerTypeMetrics(tp=0, fp=0, fn=1),
        "overall": PerTypeMetrics(tp=5, fp=2, fn=2),
    }
    report = format_gate_report(
        papers_evaluated=2,
        metrics=metrics,
        decision=GateDecision.REDESIGN,
        threshold=0.6,
        notes="Annotator: Claude. Sample: 2 papers.",
    )

    assert "# Extraction Quality Gate" in report
    assert "REDESIGN" in report
    assert "0.60" in report or "0.6" in report
    assert "materials" in report
    assert "Precision" in report
    assert "Recall" in report
    assert "F1" in report
    assert "Annotator: Claude" in report
