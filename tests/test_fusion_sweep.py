"""Unit tests for the pure fusion primitives in ``scripts/fusion_sweep.py``.

These exercise only the DB-free, model-free fusion math, so they run in CI
without Postgres, Qdrant, or torch — mirroring ``test_eval_retrieval_50q.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    p = str(_REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_retrieval_50q import BUCKETS  # noqa: E402
from fusion_sweep import (  # noqa: E402
    Lane,
    build_sweep,
    fuse_bm25_only,
    fuse_dense_only,
    fuse_dense_prior,
    fuse_naive_rrf,
    fuse_rank_cutoff_rrf,
    fuse_weighted_sum,
    min_max_normalize,
    render_markdown,
)

DENSE: Lane = [("A", 0.90), ("B", 0.50), ("C", 0.10)]
BM25: Lane = [("C", 3.0), ("D", 2.0), ("A", 1.0)]


# --- min_max_normalize ------------------------------------------------------


def test_min_max_normalize_maps_endpoints() -> None:
    norm = min_max_normalize(DENSE)
    assert norm["A"] == pytest.approx(1.0)
    assert norm["C"] == pytest.approx(0.0)
    assert 0.0 < norm["B"] < 1.0


def test_min_max_normalize_constant_lane_is_one() -> None:
    # No spread → every member is fully relevant within the lane.
    assert min_max_normalize([("X", 5.0), ("Y", 5.0)]) == {"X": 1.0, "Y": 1.0}


def test_min_max_normalize_empty() -> None:
    assert min_max_normalize([]) == {}


# --- reference passthroughs -------------------------------------------------


def test_dense_only_preserves_dense_order() -> None:
    assert fuse_dense_only(DENSE, BM25) == ["A", "B", "C"]


def test_bm25_only_preserves_bm25_order() -> None:
    assert fuse_bm25_only(DENSE, BM25) == ["C", "D", "A"]


# --- weighted sum -----------------------------------------------------------


def test_weighted_sum_full_dense_weight_recovers_dense_order() -> None:
    # w_dense=1.0 → BM25-exclusive doc D (norm_dense 0) sinks to the bottom.
    out = fuse_weighted_sum(DENSE, BM25, w_dense=1.0)
    assert out[:3] == ["A", "B", "C"]
    assert out[-1] == "D"


def test_weighted_sum_full_bm25_weight_recovers_bm25_order() -> None:
    out = fuse_weighted_sum(DENSE, BM25, w_dense=0.0)
    assert out[0] == "C"  # BM25 top
    assert out.index("D") < out.index("B")  # D (bm25) outranks B (dense-only)


def test_weighted_sum_is_a_permutation_of_the_union() -> None:
    out = fuse_weighted_sum(DENSE, BM25, w_dense=0.7)
    assert sorted(out) == ["A", "B", "C", "D"]


# --- rank-cutoff RRF --------------------------------------------------------


def test_rank_cutoff_zero_is_dense_only() -> None:
    # cutoff=0 → BM25 contributes nothing; result is the dense ranking.
    assert fuse_rank_cutoff_rrf(DENSE, BM25, cutoff=0) == ["A", "B", "C"]


def test_rank_cutoff_limits_bm25_tail() -> None:
    # cutoff=1 → only BM25's top doc (C) can be boosted; D (bm25 rank 2) is
    # excluded from BM25's contribution, so it can only appear via dense (it
    # never does) → D absent from the fused list.
    out = fuse_rank_cutoff_rrf(DENSE, BM25, cutoff=1)
    assert "D" not in out
    assert set(out) == {"A", "B", "C"}


# --- dense prior ------------------------------------------------------------


def test_dense_prior_keeps_dense_as_anchor() -> None:
    # Small lambda → every dense doc outranks the BM25-exclusive doc D.
    out = fuse_dense_prior(DENSE, BM25, lam=0.05)
    assert out[:3] == ["A", "B", "C"]
    assert out[-1] == "D"


def test_dense_prior_larger_lambda_can_promote_shared_docs() -> None:
    # A is in both lanes; a bigger lambda only strengthens it, never demotes
    # it below a dense-only doc with lower dense score.
    out = fuse_dense_prior(DENSE, BM25, lam=0.2)
    assert out[0] == "A"


# --- naive RRF / sweep grid -------------------------------------------------


def test_naive_rrf_rewards_cross_lane_agreement() -> None:
    # A and C appear in both lanes; they should outrank single-lane docs.
    out = fuse_naive_rrf(DENSE, BM25, k=60)
    assert set(out[:2]) == {"A", "C"}


def test_build_sweep_has_all_strategies_and_is_deterministic() -> None:
    names = [c.name for c in build_sweep()]
    assert "dense_only" in names
    assert "bm25_only" in names
    assert any(n.startswith("naive_rrf(") for n in names)
    assert any(n.startswith("weighted_sum(") for n in names)
    assert any(n.startswith("rank_cutoff_rrf(") for n in names)
    assert any(n.startswith("dense_prior(") for n in names)
    # Late-binding closure guard: each swept config must capture its own param.
    assert len(names) == len(set(names))
    assert names == [c.name for c in build_sweep()]


# --- report rendering -------------------------------------------------------


def _block(ndcg: float) -> dict:
    """A minimal results block: flat nDCG across overall + every bucket."""
    overall = {"ndcg_at_10": ndcg, "mrr_at_10": ndcg, "recall_at_50": ndcg}
    return {"overall": dict(overall), "by_bucket": {b: dict(overall) for b in BUCKETS}}


def test_render_flags_premise_inversion_when_bm25_beats_dense() -> None:
    # dense < bm25 → the "naive RRF hurts dense" regime does not hold; the
    # report must say so instead of crowning a hybrid "winner".
    results = {
        "dense_only": _block(0.04),
        "bm25_only": _block(0.08),
        "weighted_sum(w_dense=0.5)": _block(0.07),
    }
    md = render_markdown(results, queries_path="g.jsonl", n_queries=50, k=10)
    assert "Premise does NOT reproduce" in md
    assert "Winning fusion" not in md


def test_render_declares_winner_only_when_dense_dominates() -> None:
    results = {
        "dense_only": _block(0.50),
        "bm25_only": _block(0.20),
        "weighted_sum(w_dense=0.5)": _block(0.60),
    }
    md = render_markdown(results, queries_path="g.jsonl", n_queries=50, k=10)
    assert "Winning fusion: `weighted_sum(w_dense=0.5)`" in md
    assert "Premise does NOT reproduce" not in md


def test_render_uses_one_consistent_headline_config() -> None:
    # The per-bucket table header must name the SAME config the verdict crowns,
    # never a second different "winner" (the rows[0]-vs-verdict mismatch bug).
    results = {
        "dense_only": _block(0.50),
        "bm25_only": _block(0.20),
        "weighted_sum(w_dense=0.5)": _block(0.60),
    }
    md = render_markdown(results, queries_path="g.jsonl", n_queries=50, k=10)
    header = next(ln for ln in md.splitlines() if ln.startswith("## Per-bucket"))
    assert "weighted_sum(w_dense=0.5)" in header
