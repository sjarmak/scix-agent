"""Tests for the SCIX_LEXICAL_POOL recall-regression eval driver (bead a2fp).

Covers the pure aggregation / delta logic that decides the acceptance verdict:
    - aggregate() means over scored queries, errored queries excluded
    - paired_delta() pairs per query and excludes any pair where either side
      errored (a timed-out INF baseline must not look like a recall floor)
    - paired_delta() sign convention: positive drop == capped pool scored lower
    - _parse_pools() validation of CLI pool tokens
    - QueryResult.scored gating on error and empty gold

Pure-Python: no DB connection, no model load. The driver module imports only
stdlib at module scope, so importing it here is cheap and side-effect-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``scripts/`` importable so we can pull the driver as a module.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eval_lexical_recall_pool as drv  # noqa: E402


def _qr(
    query: str,
    *,
    bucket: str = "concept",
    ndcg: float | None,
    recall: float | None,
    error: str | None = None,
    latency_ms: float = 1.0,
    n_hits: int = 20,
) -> drv.QueryResult:
    return drv.QueryResult(
        query=query,
        bucket=bucket,
        ndcg=ndcg,
        recall=recall,
        latency_ms=latency_ms,
        error=error,
        n_hits=n_hits,
    )


# ---------------------------------------------------------------------------
# QueryResult.scored
# ---------------------------------------------------------------------------


class TestQueryResultScored:
    def test_clean_result_with_gold_is_scored(self) -> None:
        assert _qr("q", ndcg=0.5, recall=0.5).scored is True

    def test_errored_result_is_not_scored(self) -> None:
        assert _qr("q", ndcg=None, recall=None, error="timeout").scored is False

    def test_empty_gold_yields_none_metrics_not_scored(self) -> None:
        # ndcg/recall are None when gold is empty (the metric primitives
        # return None) — such a query must not contribute to an average.
        assert _qr("q", ndcg=None, recall=None).scored is False


# ---------------------------------------------------------------------------
# aggregate()
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_mean_over_scored_queries(self) -> None:
        results = [
            _qr("a", bucket="concept", ndcg=0.8, recall=1.0),
            _qr("b", bucket="concept", ndcg=0.4, recall=0.5),
        ]
        agg = drv.aggregate(results)
        assert agg["overall"]["ndcg_at_10"] == pytest.approx(0.6)
        assert agg["overall"]["recall_at_20"] == pytest.approx(0.75)
        assert agg["overall"]["n_scored"] == 2
        assert agg["n_errored"] == 0

    def test_errored_query_excluded_from_mean(self) -> None:
        results = [
            _qr("a", ndcg=0.8, recall=1.0),
            _qr("b", ndcg=None, recall=None, error="QueryCanceled: timeout"),
        ]
        agg = drv.aggregate(results)
        # Mean is over the single scored query, not 0.8/2.
        assert agg["overall"]["ndcg_at_10"] == pytest.approx(0.8)
        assert agg["overall"]["n_scored"] == 1
        assert agg["n_errored"] == 1
        assert agg["errored_queries"][0]["query"] == "b"

    def test_empty_gold_query_excluded_from_mean(self) -> None:
        results = [
            _qr("a", ndcg=0.6, recall=0.6),
            _qr("nogold", ndcg=None, recall=None),  # empty gold -> None metrics
        ]
        agg = drv.aggregate(results)
        assert agg["overall"]["ndcg_at_10"] == pytest.approx(0.6)
        assert agg["overall"]["n_scored"] == 1
        # An empty-gold query is not an error.
        assert agg["n_errored"] == 0

    def test_all_errored_yields_none_mean(self) -> None:
        results = [_qr("a", ndcg=None, recall=None, error="boom")]
        agg = drv.aggregate(results)
        assert agg["overall"]["ndcg_at_10"] is None
        assert agg["overall"]["n_scored"] == 0

    def test_by_bucket_partitions_results(self) -> None:
        results = [
            _qr("a", bucket="concept", ndcg=1.0, recall=1.0),
            _qr("b", bucket="method", ndcg=0.0, recall=0.0),
        ]
        agg = drv.aggregate(results)
        assert agg["by_bucket"]["concept"]["ndcg_at_10"] == pytest.approx(1.0)
        assert agg["by_bucket"]["method"]["ndcg_at_10"] == pytest.approx(0.0)
        assert agg["by_bucket"]["title_matchable"]["n_scored"] == 0

    def test_latency_diagnostics(self) -> None:
        results = [
            _qr("a", ndcg=0.5, recall=0.5, latency_ms=10.0),
            _qr("b", ndcg=0.5, recall=0.5, latency_ms=30.0),
        ]
        agg = drv.aggregate(results)
        assert agg["latency_ms"]["max"] == pytest.approx(30.0)
        assert agg["latency_ms"]["mean"] == pytest.approx(20.0)
        assert agg["latency_ms"]["total"] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# paired_delta()
# ---------------------------------------------------------------------------


class TestPairedDelta:
    def test_positive_drop_when_pool_scores_lower(self) -> None:
        pool = [_qr("a", ndcg=0.6, recall=0.5)]
        base = [_qr("a", ndcg=0.8, recall=0.7)]
        d = drv.paired_delta(pool, base)
        assert d["n_paired"] == 1
        # base 0.8 - pool 0.6 = 0.2 -> +20.00 pp
        assert d["ndcg_drop_pp"] == pytest.approx(20.0)
        assert d["recall_drop_pp"] == pytest.approx(20.0)

    def test_negative_drop_when_pool_scores_higher(self) -> None:
        pool = [_qr("a", ndcg=0.9, recall=0.9)]
        base = [_qr("a", ndcg=0.8, recall=0.8)]
        d = drv.paired_delta(pool, base)
        assert d["ndcg_drop_pp"] == pytest.approx(-10.0)

    def test_pair_excluded_when_baseline_errored(self) -> None:
        # INF (baseline) timed out on this query — the pair must be dropped,
        # not scored with the pool side against a zero baseline.
        pool = [_qr("a", ndcg=0.6, recall=0.6)]
        base = [_qr("a", ndcg=None, recall=None, error="timeout")]
        d = drv.paired_delta(pool, base)
        assert d["n_paired"] == 0
        assert d["ndcg_drop_pp"] is None

    def test_pair_excluded_when_pool_errored(self) -> None:
        pool = [_qr("a", ndcg=None, recall=None, error="timeout")]
        base = [_qr("a", ndcg=0.8, recall=0.8)]
        d = drv.paired_delta(pool, base)
        assert d["n_paired"] == 0

    def test_only_overlapping_queries_are_paired(self) -> None:
        pool = [
            _qr("a", ndcg=0.6, recall=0.6),
            _qr("b", ndcg=0.4, recall=0.4),
        ]
        base = [_qr("a", ndcg=0.8, recall=0.8)]  # 'b' absent from baseline
        d = drv.paired_delta(pool, base)
        assert d["n_paired"] == 1
        assert d["ndcg_drop_pp"] == pytest.approx(20.0)

    def test_by_bucket_drop_is_partitioned(self) -> None:
        pool = [
            _qr("a", bucket="concept", ndcg=0.5, recall=0.5),
            _qr("b", bucket="method", ndcg=0.9, recall=0.9),
        ]
        base = [
            _qr("a", bucket="concept", ndcg=0.9, recall=0.9),
            _qr("b", bucket="method", ndcg=0.9, recall=0.9),
        ]
        d = drv.paired_delta(pool, base)
        assert d["by_bucket"]["concept"]["ndcg_drop_pp"] == pytest.approx(40.0)
        assert d["by_bucket"]["concept"]["n_paired"] == 1
        assert d["by_bucket"]["method"]["ndcg_drop_pp"] == pytest.approx(0.0)
        assert d["by_bucket"]["title_matchable"]["ndcg_drop_pp"] is None


# ---------------------------------------------------------------------------
# _parse_pools()
# ---------------------------------------------------------------------------


class TestParsePools:
    def test_integers_and_inf(self) -> None:
        assert drv._parse_pools("1000,5000,INF") == ["1000", "5000", "INF"]

    def test_dedupes_preserving_order(self) -> None:
        assert drv._parse_pools("5000,5000,INF") == ["5000", "INF"]

    def test_unbounded_keywords_accepted(self) -> None:
        assert drv._parse_pools("all,none,inf") == ["all", "none", "inf"]

    def test_rejects_zero_and_negative(self) -> None:
        with pytest.raises(Exception):
            drv._parse_pools("0")
        with pytest.raises(Exception):
            drv._parse_pools("-100")

    def test_rejects_non_integer(self) -> None:
        with pytest.raises(Exception):
            drv._parse_pools("lots")

    def test_rejects_empty(self) -> None:
        with pytest.raises(Exception):
            drv._parse_pools("  ,  ")


# ---------------------------------------------------------------------------
# Constant invariants
# ---------------------------------------------------------------------------


class TestPoolCap:
    def test_integer_label(self) -> None:
        assert drv._pool_cap("5000") == 5000

    def test_unbounded_labels_resolve_to_none(self) -> None:
        assert drv._pool_cap("INF") is None
        assert drv._pool_cap("all") is None
        assert drv._pool_cap("none") is None


def test_default_pool_label_matches_search_module() -> None:
    """The eval's notion of the 'default' pool must equal the live default."""
    from scix.search import _LEXICAL_POOL_DEFAULT

    assert drv.DEFAULT_POOL_LABEL == str(_LEXICAL_POOL_DEFAULT)


def test_retrieve_limit_covers_both_cutoffs() -> None:
    """top-N retrieved must be >= the larger of the nDCG/Recall cutoffs."""
    assert drv.RETRIEVE_LIMIT >= max(drv.NDCG_K, drv.RECALL_K)
