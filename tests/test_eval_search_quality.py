"""Tests for scripts/eval_search_quality.py ranking comparison helpers.

Unit tests cover Jaccard overlap, Kendall's tau, and data types.
No database required for unit tests.
"""

from __future__ import annotations

# Import from the script using sys.path manipulation matching the script's own pattern
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_search_quality import (
    EvalSummary,
    QueryComparison,
    QueryResult,
    jaccard_overlap,
    kendall_tau_on_shared,
)

# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


class TestQueryResult:
    def test_construction(self) -> None:
        qr = QueryResult(
            query="test query",
            backend="tsvector",
            bibcodes=("A", "B", "C"),
            scores=(0.9, 0.8, 0.7),
            latency_ms=5.0,
            result_count=3,
        )
        assert qr.backend == "tsvector"
        assert len(qr.bibcodes) == 3
        assert qr.result_count == 3

    def test_frozen(self) -> None:
        qr = QueryResult(
            query="q",
            backend="bm25",
            bibcodes=(),
            scores=(),
            latency_ms=0.0,
            result_count=0,
        )
        with pytest.raises(AttributeError):
            qr.backend = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Jaccard overlap
# ---------------------------------------------------------------------------


class TestJaccardOverlap:
    def test_identical_sets(self) -> None:
        assert jaccard_overlap({"A", "B", "C"}, {"A", "B", "C"}) == 1.0

    def test_disjoint_sets(self) -> None:
        assert jaccard_overlap({"A", "B"}, {"C", "D"}) == 0.0

    def test_partial_overlap(self) -> None:
        # {A, B, C} & {B, C, D} = {B, C}, union = {A, B, C, D}
        result = jaccard_overlap({"A", "B", "C"}, {"B", "C", "D"})
        assert result == pytest.approx(0.5)

    def test_empty_sets(self) -> None:
        # Both empty => defined as 1.0 (identical vacuously)
        assert jaccard_overlap(set(), set()) == 1.0

    def test_one_empty(self) -> None:
        assert jaccard_overlap({"A"}, set()) == 0.0

    def test_single_element_match(self) -> None:
        assert jaccard_overlap({"A"}, {"A"}) == 1.0

    def test_subset(self) -> None:
        # {A, B} & {A, B, C} = {A, B}, union = {A, B, C}
        result = jaccard_overlap({"A", "B"}, {"A", "B", "C"})
        assert result == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Kendall's tau on shared items
# ---------------------------------------------------------------------------


class TestKendallTau:
    def test_identical_ranking(self) -> None:
        ranking = ("A", "B", "C", "D")
        assert kendall_tau_on_shared(ranking, ranking) == 1.0

    def test_reversed_ranking(self) -> None:
        a = ("A", "B", "C", "D")
        b = ("D", "C", "B", "A")
        assert kendall_tau_on_shared(a, b) == -1.0

    def test_partial_overlap(self) -> None:
        # Only B and C are shared
        a = ("A", "B", "C")
        b = ("B", "C", "D")
        # B before C in both => concordant => tau = 1.0
        assert kendall_tau_on_shared(a, b) == 1.0

    def test_partial_overlap_reversed(self) -> None:
        a = ("A", "B", "C")
        b = ("C", "B", "D")
        # B before C in a, C before B in b => discordant => tau = -1.0
        assert kendall_tau_on_shared(a, b) == -1.0

    def test_no_shared_items(self) -> None:
        assert kendall_tau_on_shared(("A", "B"), ("C", "D")) == 0.0

    def test_single_shared_item(self) -> None:
        # Need at least 2 shared items for meaningful correlation
        assert kendall_tau_on_shared(("A", "B"), ("B", "C")) == 0.0

    def test_empty_rankings(self) -> None:
        assert kendall_tau_on_shared((), ()) == 0.0

    def test_four_items_partial_agreement(self) -> None:
        a = ("A", "B", "C", "D")
        b = ("A", "C", "B", "D")
        # Pairs: (A,B)=conc, (A,C)=conc, (A,D)=conc, (B,C)=disc, (B,D)=conc, (C,D)=conc
        # 5 concordant, 1 discordant => tau = (5-1)/6 = 2/3
        tau = kendall_tau_on_shared(a, b)
        assert tau == pytest.approx(2 / 3, abs=0.01)


# ---------------------------------------------------------------------------
# QueryComparison
# ---------------------------------------------------------------------------


class TestQueryComparison:
    def test_construction(self) -> None:
        tsv = QueryResult("q", "tsvector", ("A", "B"), (0.9, 0.8), 5.0, 2)
        bm25 = QueryResult("q", "bm25", ("B", "C"), (1.2, 0.9), 3.0, 2)
        comp = QueryComparison(
            query="q",
            tsvector=tsv,
            bm25=bm25,
            overlap_at_10=0.3333,
            overlap_at_20=0.3333,
            rank_correlation=0.5,
            tsvector_only=("A",),
            bm25_only=("C",),
        )
        assert comp.query == "q"
        assert comp.tsvector_only == ("A",)
        assert comp.bm25_only == ("C",)


# ---------------------------------------------------------------------------
# EvalSummary
# ---------------------------------------------------------------------------


class TestEvalSummary:
    def test_construction(self) -> None:
        s = EvalSummary(
            num_queries=10,
            tsvector_mean_ms=5.0,
            bm25_mean_ms=3.0,
            mean_overlap_at_10=0.65,
            mean_overlap_at_20=0.55,
            mean_rank_correlation=0.42,
        )
        assert s.num_queries == 10
        assert s.comparisons == []
