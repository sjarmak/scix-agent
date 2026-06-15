"""Unit tests for the UMBRELA rerank-judge harness (bead 9na0).

Covers the pure logic — ranking construction, dedup-union triple collection,
score indexing, judge-graded summarization, and the disagreement analysis —
without touching the DB, Qdrant, the reranker checkpoints, or ``claude -p``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# The module imports heavy transitive deps (scix.search) lazily inside main(),
# so importing it at module top level is cheap and safe.
_spec = importlib.util.spec_from_file_location(
    "eval_rerank_umbrela_judge",
    _REPO_ROOT / "scripts" / "eval_rerank_umbrela_judge.py",
)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
# Register before exec: dataclass annotation resolution (under
# ``from __future__ import annotations``) looks the module up in sys.modules.
sys.modules["eval_rerank_umbrela_judge"] = mod
_spec.loader.exec_module(mod)

from scix.eval.persona_judge import JudgeScore, JudgeTriple  # noqa: E402


def _seed_record(bib: str, title: str = "T", abstract: str = "A"):
    return mod.SeedRecord(bibcode=bib, title=title, abstract=abstract, embedding=[0.0])


def _pool(bibs: list[str]) -> list[dict]:
    return [{"bibcode": b, "title": f"title {b}", "abstract": f"abstract {b}"} for b in bibs]


class _ReverseReranker:
    """Deterministic stand-in reranker: returns the pool in reversed order."""

    def __call__(self, query_text, candidates):
        return list(reversed(candidates))


class _PromoteReranker:
    """Reranker that moves a named bibcode to the front, rest unchanged."""

    def __init__(self, promote: str) -> None:
        self.promote = promote

    def __call__(self, query_text, candidates):
        head = [c for c in candidates if c["bibcode"] == self.promote]
        tail = [c for c in candidates if c["bibcode"] != self.promote]
        return head + tail


# ---------------------------------------------------------------------------
# build_seed_rankings
# ---------------------------------------------------------------------------


def test_build_seed_rankings_drops_seed_and_truncates_to_k():
    seed = _seed_record("SEED")
    pool = _pool(["SEED", "P1", "P2", "P3", "P4"])
    rerankers = {"hybrid_indus": None, "minilm": None, "indus_ranker": None}

    sr = mod.build_seed_rankings(seed, pool, {"P1"}, rerankers, k=2)

    # Seed itself is excluded from every ranking.
    for config in mod.JUDGED_CONFIGS:
        assert "SEED" not in sr.rankings[config]
        assert len(sr.rankings[config]) == 2
    # Baseline keeps pool order (minus the seed).
    assert sr.rankings["hybrid_indus"] == ["P1", "P2"]


def test_build_seed_rankings_union_is_deduped():
    seed = _seed_record("SEED")
    pool = _pool(["SEED", "P1", "P2", "P3", "P4"])
    # baseline -> P1,P2 ; reverse -> P4,P3 ; promote P2 -> P2,P1
    rerankers = {
        "hybrid_indus": None,
        "minilm": _ReverseReranker(),
        "indus_ranker": _PromoteReranker("P2"),
    }
    sr = mod.build_seed_rankings(seed, pool, set(), rerankers, k=2)

    # Union of {P1,P2} ∪ {P4,P3} ∪ {P2,P1} = {P1,P2,P3,P4}; one snippet each.
    assert set(sr.snippets) == {"P1", "P2", "P3", "P4"}
    assert len(sr.snippets) == 4
    assert "Title: title P1" in sr.snippets["P1"]


def test_collect_triples_namespaces_by_seed():
    seed_a = mod.SeedRankings(
        seed_bibcode="A",
        query_text="qa",
        rankings={"hybrid_indus": ["P1"], "minilm": ["P1"], "indus_ranker": ["P1"]},
        snippets={"P1": "snip"},
        relevant=set(),
    )
    seed_b = mod.SeedRankings(
        seed_bibcode="B",
        query_text="qb",
        rankings={"hybrid_indus": ["P1"], "minilm": ["P1"], "indus_ranker": ["P1"]},
        snippets={"P1": "snip"},
        relevant=set(),
    )
    triples = mod.collect_triples([seed_a, seed_b])
    bibs = {t.bibcode for t in triples}
    # Same paper P1 under two seeds is two distinct, namespaced triples.
    assert bibs == {"A|P1", "B|P1"}


# ---------------------------------------------------------------------------
# index_scores
# ---------------------------------------------------------------------------


def test_index_scores_maps_back_and_drops_failures():
    scores = [
        JudgeScore(score=3, reason="", triple=JudgeTriple("q", "SEED|P1", "s")),
        JudgeScore(score=1, reason="", triple=JudgeTriple("q", "SEED|P2", "s")),
        JudgeScore(score=-1, reason="err", triple=JudgeTriple("q", "SEED|P3", "s")),
    ]
    idx = mod.index_scores(scores)
    assert idx == {"SEED": {"P1": 3, "P2": 1}}  # P3 (failed) dropped


# ---------------------------------------------------------------------------
# summarize_config — judge-graded nDCG
# ---------------------------------------------------------------------------


def test_summarize_config_perfect_vs_inverted_order():
    # One seed, judged union P1=3 P2=2 P3=0. minilm orders best-first, indus
    # orders worst-first → minilm judge-nDCG must exceed indus.
    sr = mod.SeedRankings(
        seed_bibcode="S",
        query_text="q",
        rankings={
            "hybrid_indus": ["P1", "P2", "P3"],
            "minilm": ["P1", "P2", "P3"],
            "indus_ranker": ["P3", "P2", "P1"],
        },
        snippets={"P1": "", "P2": "", "P3": ""},
        relevant={"P1"},
    )
    judge_by_seed = {"S": {"P1": 3, "P2": 2, "P3": 0}}

    minilm = mod.summarize_config("minilm", [sr], judge_by_seed, k=3)
    indus = mod.summarize_config("indus_ranker", [sr], judge_by_seed, k=3)

    assert minilm.judge_ndcg_10 == 1.0  # ideal order
    assert indus.judge_ndcg_10 < minilm.judge_ndcg_10
    # mean top-K judge score is order-independent → identical for both.
    assert (
        minilm.mean_judge_score_top10
        == indus.mean_judge_score_top10
        == pytest.approx(5 / 3, abs=1e-4)
    )
    assert minilm.judged_relevant_at_10 == 2.0  # P1,P2 are >=2


def test_summarize_config_empty_returns_zeroed():
    s = mod.summarize_config("minilm", [], {}, k=10)
    assert s.n_queries == 0
    assert s.judge_ndcg_10 == 0.0
    assert s.score_hist_top10 == {0: 0, 1: 0, 2: 0, 3: 0}


# ---------------------------------------------------------------------------
# analyze_disagreement — the deliverable
# ---------------------------------------------------------------------------


def test_analyze_disagreement_detects_indus_promoting_low_scores():
    # indus uniquely promotes P_BAD (score 0); minilm uniquely promotes
    # P_GOOD (score 3). The delta must be strongly negative.
    sr = mod.SeedRankings(
        seed_bibcode="S",
        query_text="q",
        rankings={
            "hybrid_indus": ["P_SHARED", "P_GOOD"],
            "minilm": ["P_SHARED", "P_GOOD"],
            "indus_ranker": ["P_SHARED", "P_BAD"],
        },
        snippets={},
        relevant=set(),
    )
    judge_by_seed = {"S": {"P_SHARED": 2, "P_GOOD": 3, "P_BAD": 0}}

    out = mod.analyze_disagreement([sr], judge_by_seed, k=2)
    assert out["indus_only"]["mean_judge_score"] == 0.0  # only P_BAD
    assert out["minilm_only"]["mean_judge_score"] == 3.0  # only P_GOOD
    assert out["delta_mean_judge_score_indus_minus_minilm"] == -3.0
    assert out["n_seeds_with_divergence"] == 1


# ---------------------------------------------------------------------------
# resume: load_prior_scores + partition_reusable
# ---------------------------------------------------------------------------


def test_load_prior_scores_skips_malformed_and_scoreless(tmp_path):
    p = tmp_path / "scores.jsonl"
    p.write_text(
        '{"bibcode": "S|P1", "score": 3, "needs_human_review": false}\n'
        "\n"  # blank line skipped
        '{"bibcode": "S|P2", "score": -1, "needs_human_review": true}\n'
        '{"needs_human_review": true}\n'  # no bibcode/score skipped
    )
    prior = mod.load_prior_scores(p)
    assert prior == {"S|P1": (3, False), "S|P2": (-1, True)}


def test_partition_reusable_rejudges_failures_and_missing():
    triples = [
        JudgeTriple("q", "S|GOOD", "s"),  # has a good prior score -> reuse
        JudgeTriple("q", "S|FAILED", "s"),  # prior score < 0 -> re-judge
        JudgeTriple("q", "S|NEW", "s"),  # absent from prior -> re-judge
    ]
    prior = {"S|GOOD": (2, False), "S|FAILED": (-1, True)}

    reused, pending = mod.partition_reusable(triples, prior)

    assert [s.triple.bibcode for s in reused] == ["S|GOOD"]
    assert reused[0].score == 2
    assert {t.bibcode for t in pending} == {"S|FAILED", "S|NEW"}


def test_analyze_disagreement_no_divergence_is_zero():
    sr = mod.SeedRankings(
        seed_bibcode="S",
        query_text="q",
        rankings={
            "hybrid_indus": ["P1", "P2"],
            "minilm": ["P1", "P2"],
            "indus_ranker": ["P1", "P2"],
        },
        snippets={},
        relevant=set(),
    )
    out = mod.analyze_disagreement([sr], {"S": {"P1": 3, "P2": 2}}, k=2)
    assert out["n_seeds_with_divergence"] == 0
    assert out["indus_only"]["n_papers"] == 0
    assert out["minilm_only"]["n_papers"] == 0
    assert out["delta_mean_judge_score_indus_minus_minilm"] == 0.0
