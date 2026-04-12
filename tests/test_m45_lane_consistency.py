"""Tests for PRD §M4.5 lane-consistency + gate (u12)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scix import resolve_entities as re_mod
from scix.eval.lane_delta import (
    GATE_THRESHOLD,
    BibcodeDivergence,
    LaneEntitySets,
    adjusted_jaccard,
    aggregate_divergence,
    compute_lane_delta_set,
    divergence,
    gate_p90,
    jaccard,
    per_bibcode_divergence,
)


@pytest.fixture(autouse=True)
def _reset_resolver_mocks():
    re_mod._reset_mocks()
    yield
    re_mod._reset_mocks()


# ---------------------------------------------------------------------------
# Basic set math
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_sets_yield_one(self):
        assert jaccard(frozenset({1, 2, 3}), frozenset({1, 2, 3})) == 1.0

    def test_disjoint_sets_yield_zero(self):
        assert jaccard(frozenset({1, 2}), frozenset({3, 4})) == 0.0

    def test_both_empty_yield_one(self):
        # Agreement that neither lane has any entity.
        assert jaccard(frozenset(), frozenset()) == 1.0

    def test_half_overlap(self):
        assert jaccard(frozenset({1, 2}), frozenset({2, 3})) == pytest.approx(1.0 / 3.0)

    def test_divergence_is_complement(self):
        j = jaccard(frozenset({1, 2}), frozenset({2, 3}))
        assert divergence(j) == pytest.approx(1.0 - j)


class TestAdjustedJaccard:
    def test_empty_delta_matches_raw(self):
        a = frozenset({1, 2, 3})
        b = frozenset({2, 3, 4})
        assert adjusted_jaccard(a, b, frozenset()) == pytest.approx(jaccard(a, b))

    def test_delta_subtracts_from_both_sides(self):
        a = frozenset({1, 2, 3})
        b = frozenset({1, 2, 4})
        # Raw Jaccard: |{1,2}| / |{1,2,3,4}| = 2/4 = 0.5
        assert jaccard(a, b) == pytest.approx(0.5)
        # Remove 3 and 4 from consideration: both sides become {1,2},
        # Jaccard climbs to 1.0.
        delta = frozenset({3, 4})
        assert adjusted_jaccard(a, b, delta) == pytest.approx(1.0)


class TestLaneDeltaStub:
    def test_empty_at_u12(self):
        # u12 stub must return an empty set; u07 will replace later.
        assert compute_lane_delta_set("2024ApJ...1A") == frozenset()


# ---------------------------------------------------------------------------
# per_bibcode_divergence
# ---------------------------------------------------------------------------


class TestPerBibcodeDivergence:
    def test_full_agreement_is_zero_divergence(self):
        sets = LaneEntitySets(
            bibcode="2024M45..1",
            citation_chain=frozenset({1, 2}),
            hybrid_enrich=frozenset({1, 2}),
            static_canonical=frozenset({1, 2}),
        )
        record = per_bibcode_divergence(sets, lane_delta_set=frozenset())
        assert record.raw_jaccard_chain_hybrid == 1.0
        assert record.raw_jaccard_chain_static == 1.0
        assert record.raw_jaccard_hybrid_static == 1.0
        assert record.mean_raw_divergence == 0.0
        assert record.mean_adj_divergence == 0.0

    def test_full_disagreement_is_one_divergence(self):
        sets = LaneEntitySets(
            bibcode="2024M45..2",
            citation_chain=frozenset({1}),
            hybrid_enrich=frozenset({2}),
            static_canonical=frozenset({3}),
        )
        record = per_bibcode_divergence(sets, lane_delta_set=frozenset())
        assert record.raw_jaccard_chain_hybrid == 0.0
        assert record.raw_jaccard_chain_static == 0.0
        assert record.raw_jaccard_hybrid_static == 0.0
        assert record.mean_raw_divergence == 1.0


# ---------------------------------------------------------------------------
# Aggregate + p90 gate
# ---------------------------------------------------------------------------


def _record(bib: str, mean_j: float) -> BibcodeDivergence:
    return BibcodeDivergence(
        bibcode=bib,
        raw_jaccard_chain_hybrid=mean_j,
        raw_jaccard_chain_static=mean_j,
        raw_jaccard_hybrid_static=mean_j,
        adj_jaccard_chain_hybrid=mean_j,
        adj_jaccard_chain_static=mean_j,
        adj_jaccard_hybrid_static=mean_j,
        mean_raw_jaccard=mean_j,
        mean_adj_jaccard=mean_j,
    )


class TestGate:
    def test_gate_passes_when_all_low_divergence(self):
        records = [_record(f"b{i}", 0.99) for i in range(10)]
        p90, passed = gate_p90(records)
        assert p90 <= GATE_THRESHOLD
        assert passed is True

    def test_gate_fails_when_high_divergence_tail(self):
        # 8 records at Jaccard=1.0 and 2 at Jaccard=0.5 → 2/10 = 20%
        # have divergence 0.5, so p90 ≥ 0.5, well over the 0.05 gate.
        records = [_record(f"ok{i}", 1.0) for i in range(8)] + [
            _record(f"bad{i}", 0.5) for i in range(2)
        ]
        p90, passed = gate_p90(records)
        assert p90 > GATE_THRESHOLD
        assert passed is False

    def test_gate_matches_numpy_percentile(self):
        # Fixture with deterministic divergences — verify the gate
        # computes numpy.percentile(divergences, 90) exactly.
        records = [_record(f"b{i}", j) for i, j in enumerate([1.0, 1.0, 1.0, 2.0 / 3.0, 0.0])]
        expected_divergences = [0.0, 0.0, 0.0, 1.0 - 2.0 / 3.0, 1.0]
        expected_p90 = float(np.percentile(expected_divergences, 90))
        agg = aggregate_divergence(records)
        assert agg.p90_adj_divergence == pytest.approx(expected_p90)
        assert agg.n == 5
        assert agg.gate_passed is (expected_p90 <= GATE_THRESHOLD)


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


class TestEvalLaneConsistencyScript:
    def test_run_writes_all_three_artifact_files(self, tmp_path):
        import eval_lane_consistency

        consistency_path = tmp_path / "m45_consistency.md"
        delta_path = tmp_path / "m45_lane_delta.md"
        inputs = eval_lane_consistency.run(
            consistency_output=consistency_path,
            lane_delta_output=delta_path,
        )
        assert consistency_path.exists()
        assert delta_path.exists()
        # AC4 — lane delta artifact exists (header + one-row-per-entity
        # contract; u12 stub may produce zero rows, which is documented
        # in the artifact itself).
        delta_content = delta_path.read_text()
        assert "Lane Delta" in delta_content
        assert "bibcode" in delta_content
        # AC3 — consistency artifact has raw + adjusted + distribution +
        # per-lane-pair breakdown.
        consistency_content = consistency_path.read_text()
        assert "raw_chain_hybrid" in consistency_content
        assert "adj_chain_hybrid" in consistency_content
        assert "p90_adj_divergence" in consistency_content
        assert "Per-lane-pair" in consistency_content
        # Gate value is printed into the report.
        assert "Gate" in consistency_content

        agg = inputs.aggregate
        assert agg.n == 5
        # Fixture divergences: three 0.0, one ~1/3 on pair 4, one 1.0 on pair 5.
        # Mean for bib4 ≈ (2/3 etc); we only assert the aggregate shape.
        assert 0.0 <= agg.p90_adj_divergence <= 1.0
        assert isinstance(agg.gate_passed, bool)

    def test_run_fixture_has_known_p90_divergence(self, tmp_path):
        import eval_lane_consistency

        inputs = eval_lane_consistency.run(
            consistency_output=tmp_path / "c.md",
            lane_delta_output=tmp_path / "d.md",
        )
        # Recompute p90 over the fixture and confirm the aggregate
        # matches numpy.percentile — this is AC5 "test asserts
        # computation correctness on fixture with known divergence".
        divergences = [r.mean_adj_divergence for r in inputs.records]
        expected_p90 = float(np.percentile(divergences, 90))
        assert inputs.aggregate.p90_adj_divergence == pytest.approx(expected_p90)
        # Fixture is designed with a 1.0-divergence outlier so the
        # gate must fail — sanity-checks the pass/fail path.
        assert inputs.aggregate.gate_passed is False
