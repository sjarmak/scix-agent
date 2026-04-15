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
    def _make_eval(self, method: str, ndcg: float, seed: str = "test") -> QueryEval:
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
    def _make_eval(self, method: str, ndcg: float, seed: str) -> QueryEval:
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


# ---------------------------------------------------------------------------
# Full-corpus mode helpers
# ---------------------------------------------------------------------------


class TestFullCorpusMethods:
    """Tests for the FULL_CORPUS_METHODS constant and method routing."""

    def test_full_corpus_methods_exist(self) -> None:
        from eval_retrieval_50q import FULL_CORPUS_METHODS

        assert "indus" in FULL_CORPUS_METHODS
        assert "lexical" in FULL_CORPUS_METHODS
        assert "hybrid_indus" in FULL_CORPUS_METHODS

    def test_full_corpus_methods_excludes_pilot_only(self) -> None:
        """Full-corpus mode should not include specter2/nomic (no full embeddings)."""
        from eval_retrieval_50q import FULL_CORPUS_METHODS

        assert "specter2" not in FULL_CORPUS_METHODS
        assert "nomic" not in FULL_CORPUS_METHODS
        assert "hybrid_specter2" not in FULL_CORPUS_METHODS


class TestMakeLexicalQuery:
    """Tests for _make_lexical_query robustness."""

    def test_basic_extraction(self) -> None:
        from eval_retrieval_50q import _make_lexical_query

        seed = {"title": "Dark matter halos in galaxy clusters"}
        query = _make_lexical_query(seed)
        assert "dark" in query
        assert "matter" in query
        assert "halos" in query

    def test_html_stripping(self) -> None:
        from eval_retrieval_50q import _make_lexical_query

        seed = {"title": "The <sub>13</sub>CO emission from molecular clouds"}
        query = _make_lexical_query(seed)
        # HTML tags should be stripped, not leaked into query
        assert "<sub>" not in query
        assert "emission" in query

    def test_max_terms_limit(self) -> None:
        from eval_retrieval_50q import _make_lexical_query

        seed = {
            "title": "Spectroscopic observations revealing chemical abundances "
            "metallicity gradients stellar populations evolutionary sequences"
        }
        query = _make_lexical_query(seed, max_terms=4)
        assert len(query.split()) <= 4

    def test_empty_title(self) -> None:
        from eval_retrieval_50q import _make_lexical_query

        assert _make_lexical_query({"title": ""}) == ""
        assert _make_lexical_query({}) == ""


class TestGenerateReport:
    """Tests for report generation with full-corpus metadata."""

    def test_report_contains_corpus_label(self) -> None:
        from eval_retrieval_50q import EvalSummary, generate_report

        summary = EvalSummary(
            method="indus",
            n_queries=50,
            mean_ndcg_10=0.45,
            mean_recall_10=0.30,
            mean_recall_20=0.50,
            mean_precision_10=0.38,
            mean_mrr=0.72,
            mean_latency_ms=120.0,
            std_ndcg_10=0.15,
        )
        seeds = [
            {
                "bibcode": "2024TEST",
                "title": "Test",
                "n_neighbors": 15,
                "year": 2024,
            }
        ]
        report = generate_report(
            [summary],
            [],
            seeds,
            [],
            corpus_label="32.4M full corpus",
        )
        assert "32.4M full corpus" in report

    def test_report_default_corpus_label(self) -> None:
        from eval_retrieval_50q import EvalSummary, generate_report

        summary = EvalSummary(
            method="indus",
            n_queries=1,
            mean_ndcg_10=0.45,
            mean_recall_10=0.30,
            mean_recall_20=0.50,
            mean_precision_10=0.38,
            mean_mrr=0.72,
            mean_latency_ms=120.0,
            std_ndcg_10=0.15,
        )
        seeds = [
            {
                "bibcode": "2024TEST",
                "title": "Test",
                "n_neighbors": 15,
                "year": 2024,
            }
        ]
        report = generate_report([summary], [], seeds, [])
        assert "10K stratified sample" in report


class TestSignificancePairs:
    """Tests for _significance_pairs method-pair filtering."""

    def test_full_corpus_pairs(self) -> None:
        from eval_retrieval_50q import FULL_CORPUS_METHODS, _significance_pairs

        pairs = _significance_pairs(FULL_CORPUS_METHODS)
        # Should include hybrid_indus vs indus, hybrid_indus vs lexical,
        # indus vs lexical
        assert ("hybrid_indus", "indus") in pairs
        assert ("hybrid_indus", "lexical") in pairs
        assert ("indus", "lexical") in pairs
        # Should NOT include specter2 pairs
        assert all("specter2" not in p[0] and "specter2" not in p[1] for p in pairs)

    def test_pilot_methods_pairs(self) -> None:
        from eval_retrieval_50q import METHODS, _significance_pairs

        pairs = _significance_pairs(METHODS)
        assert ("hybrid_indus", "indus") in pairs
        assert ("hybrid_specter2", "specter2") in pairs

    def test_single_method_no_pairs(self) -> None:
        from eval_retrieval_50q import _significance_pairs

        pairs = _significance_pairs(["indus"])
        assert pairs == []
