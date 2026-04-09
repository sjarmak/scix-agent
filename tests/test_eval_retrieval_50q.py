"""Tests for the 50-query retrieval evaluation metrics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_retrieval_50q import (
    QueryEval,
    aggregate_results,
    dcg_at_k,
    mrr,
    ndcg_at_k,
    paired_difference_test,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# DCG / nDCG
# ---------------------------------------------------------------------------


class TestDCG:
    def test_perfect_ranking(self) -> None:
        # All relevant at top
        assert dcg_at_k([1, 1, 1], 3) == pytest.approx(
            1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        )

    def test_no_relevant(self) -> None:
        assert dcg_at_k([0, 0, 0], 3) == 0.0

    def test_single_relevant_at_top(self) -> None:
        assert dcg_at_k([1, 0, 0], 3) == pytest.approx(1 / math.log2(2))

    def test_k_truncation(self) -> None:
        # Only considers first k elements
        assert dcg_at_k([1, 0, 1, 1, 1], 2) == pytest.approx(1 / math.log2(2))


class TestNDCG:
    def test_perfect_ndcg(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_zero_ndcg(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_ndcg(self) -> None:
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        val = ndcg_at_k(retrieved, relevant, 3)
        assert 0.0 < val < 1.0

    def test_empty_relevant(self) -> None:
        assert ndcg_at_k(["a", "b"], set(), 2) == 0.0

    def test_relevant_after_k(self) -> None:
        # Relevant doc at position 3, but k=2
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        assert ndcg_at_k(retrieved, relevant, 2) == 0.0


# ---------------------------------------------------------------------------
# Recall / Precision / MRR
# ---------------------------------------------------------------------------


class TestRecall:
    def test_perfect_recall(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_partial_recall(self) -> None:
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_zero_recall(self) -> None:
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, 3) == 0.0

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a", "b"], set(), 2) == 0.0


class TestPrecision:
    def test_perfect_precision(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b", "c"}, 2) == 1.0

    def test_half_precision(self) -> None:
        assert precision_at_k(["a", "x"], {"a", "b"}, 2) == 0.5

    def test_zero_k(self) -> None:
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestMRR:
    def test_first_position(self) -> None:
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self) -> None:
        assert mrr(["x", "a", "c"], {"a"}) == 0.5

    def test_no_relevant(self) -> None:
        assert mrr(["x", "y", "z"], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def _make_eval(
        self, method: str, ndcg: float, seed: str = "test"
    ) -> QueryEval:
        return QueryEval(
            seed_bibcode=seed,
            method=method,
            ndcg_10=ndcg,
            recall_10=0.5,
            recall_20=0.6,
            precision_10=0.3,
            mrr_val=0.8,
            relevant_count=10,
            retrieved_count=20,
            latency_ms=5.0,
        )

    def test_single_method(self) -> None:
        evals = [
            self._make_eval("specter2", 0.8, "s1"),
            self._make_eval("specter2", 0.6, "s2"),
        ]
        summaries = aggregate_results(evals)
        assert len(summaries) == 1
        assert summaries[0].method == "specter2"
        assert summaries[0].mean_ndcg_10 == pytest.approx(0.7)
        assert summaries[0].n_queries == 2

    def test_multiple_methods(self) -> None:
        evals = [
            self._make_eval("specter2", 0.8),
            self._make_eval("lexical", 0.4),
        ]
        summaries = aggregate_results(evals)
        assert len(summaries) == 2


# ---------------------------------------------------------------------------
# Significance test
# ---------------------------------------------------------------------------


class TestSignificance:
    def _make_eval(
        self, method: str, ndcg: float, seed: str
    ) -> QueryEval:
        return QueryEval(
            seed_bibcode=seed,
            method=method,
            ndcg_10=ndcg,
            recall_10=0.5,
            recall_20=0.6,
            precision_10=0.3,
            mrr_val=0.8,
            relevant_count=10,
            retrieved_count=20,
            latency_ms=5.0,
        )

    def test_insufficient_pairs(self) -> None:
        evals = [
            self._make_eval("a", 0.8, "s1"),
            self._make_eval("b", 0.6, "s1"),
        ]
        result = paired_difference_test(evals, "a", "b")
        assert result["n_pairs"] == 1
        assert result["p_value"] is None

    def test_positive_difference(self) -> None:
        evals = []
        for i in range(10):
            evals.append(self._make_eval("a", 0.8, f"s{i}"))
            evals.append(self._make_eval("b", 0.3, f"s{i}"))
        result = paired_difference_test(evals, "a", "b")
        assert result["n_pairs"] == 10
        assert result["mean_diff"] > 0
