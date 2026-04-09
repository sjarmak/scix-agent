"""Tests for IR evaluation metrics."""

from __future__ import annotations

import math

import pytest

from scix.ir_metrics import (
    EvalReport,
    RetrievalScore,
    aggregate_scores,
    compute_retrieval_score,
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# dcg_at_k
# ---------------------------------------------------------------------------


class TestDcgAtK:
    def test_empty(self) -> None:
        assert dcg_at_k([], 10) == 0.0

    def test_single_relevant(self) -> None:
        # rel=1 at rank 1: (2^1 - 1) / log2(2) = 1.0
        assert dcg_at_k([1.0], 1) == pytest.approx(1.0)

    def test_known_values(self) -> None:
        # Standard example: rels = [3, 2, 3, 0, 1, 2]
        rels = [3.0, 2.0, 3.0, 0.0, 1.0, 2.0]
        # DCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^3-1)/log2(4) + 0 + (2^1-1)/log2(6) + (2^2-1)/log2(7)
        expected = 7 / 1 + 3 / math.log2(3) + 7 / math.log2(4) + 0 + 1 / math.log2(6) + 3 / math.log2(7)
        assert dcg_at_k(rels, 6) == pytest.approx(expected, rel=1e-6)

    def test_k_truncation(self) -> None:
        rels = [1.0, 1.0, 1.0]
        dcg_2 = dcg_at_k(rels, 2)
        dcg_3 = dcg_at_k(rels, 3)
        assert dcg_2 < dcg_3


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        # Retrieved in ideal order
        retrieved = ["a", "b", "c"]
        relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
        assert ndcg_at_k(retrieved, relevance, k=3) == pytest.approx(1.0)

    def test_reversed_ranking(self) -> None:
        retrieved = ["c", "b", "a"]
        relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
        score = ndcg_at_k(retrieved, relevance, k=3)
        assert 0 < score < 1.0

    def test_no_relevant_docs(self) -> None:
        assert ndcg_at_k(["a", "b"], {}, k=2) == 0.0

    def test_empty_retrieved(self) -> None:
        assert ndcg_at_k([], {"a": 1.0}, k=10) == 0.0

    def test_partial_overlap(self) -> None:
        retrieved = ["x", "a", "y"]  # only "a" is relevant
        relevance = {"a": 2.0, "b": 1.0}
        score = ndcg_at_k(retrieved, relevance, k=3)
        assert 0 < score < 1.0

    def test_binary_relevance(self) -> None:
        # With binary relevance (0 or 2), perfect retrieval at rank 1
        retrieved = ["a", "b", "c"]
        relevance = {"a": 2.0}
        assert ndcg_at_k(retrieved, relevance, k=3) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_all_found(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_none_found(self) -> None:
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == pytest.approx(0.0)

    def test_partial(self) -> None:
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == pytest.approx(0.5)

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a"], set(), k=1) == 0.0

    def test_k_cutoff(self) -> None:
        # "b" is relevant but at rank 3 — not found at k=2
        assert recall_at_k(["x", "y", "b"], {"b"}, k=2) == pytest.approx(0.0)
        assert recall_at_k(["x", "y", "b"], {"b"}, k=3) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b", "c"}, k=2) == pytest.approx(1.0)

    def test_none_relevant(self) -> None:
        assert precision_at_k(["x", "y"], {"a"}, k=2) == pytest.approx(0.0)

    def test_half_relevant(self) -> None:
        assert precision_at_k(["a", "x"], {"a"}, k=2) == pytest.approx(0.5)

    def test_k_zero(self) -> None:
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_position(self) -> None:
        assert mean_reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)

    def test_second_position(self) -> None:
        assert mean_reciprocal_rank(["x", "a"], {"a"}) == pytest.approx(0.5)

    def test_not_found(self) -> None:
        assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_retrieved(self) -> None:
        assert mean_reciprocal_rank([], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# compute_retrieval_score
# ---------------------------------------------------------------------------


class TestComputeRetrievalScore:
    def test_basic(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        relevance = {"a": 2.0, "c": 2.0, "f": 2.0}  # 3 relevant, 2 found
        score = compute_retrieval_score("q1", "test_sys", retrieved, relevance, 5.0)

        assert score.query_id == "q1"
        assert score.system == "test_sys"
        assert score.latency_ms == 5.0
        assert score.num_relevant == 3
        assert score.mrr == pytest.approx(1.0)  # "a" is first and relevant
        assert score.recall_at_10 == pytest.approx(2 / 3, rel=1e-3)

    def test_no_relevant(self) -> None:
        score = compute_retrieval_score("q1", "sys", ["a", "b"], {}, 1.0)
        assert score.ndcg_at_10 == 0.0
        assert score.recall_at_10 == 0.0
        assert score.mrr == 0.0


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------


class TestAggregateScores:
    def test_empty(self) -> None:
        report = aggregate_scores("sys", [])
        assert report.num_queries == 0
        assert report.mean_ndcg_at_10 == 0.0

    def test_aggregation(self) -> None:
        s1 = RetrievalScore("q1", "sys", 0.8, 0.5, 0.7, 0.4, 1.0, 10.0, 20, 10)
        s2 = RetrievalScore("q2", "sys", 0.6, 0.3, 0.5, 0.2, 0.5, 20.0, 20, 5)
        report = aggregate_scores("sys", [s1, s2])

        assert report.num_queries == 2
        assert report.mean_ndcg_at_10 == pytest.approx(0.7)
        assert report.mean_recall_at_10 == pytest.approx(0.4)
        assert report.mean_mrr == pytest.approx(0.75)
        assert report.mean_latency_ms == pytest.approx(15.0)
        assert len(report.per_query) == 2
