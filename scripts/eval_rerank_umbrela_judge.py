#!/usr/bin/env python3
"""UMBRELA LLM-judge second opinion on the 50q rerank A/B (bead 9na0).

Follow-up to bead 4skc (``scripts/eval_rerank_local_ab.py``). That run scored
rerankers against **binary citation relevance** and reached a decisive NO-GO:
the domain-tuned ``indus_ranker`` (``nasa-impact/nasa-smd-ibm-ranker``)
*regressed* nDCG@10 by -0.0400 and was the worst of four configs, while the
generic ``minilm`` (``cross-encoder/ms-marco-MiniLM-L-12-v2``) was the nominal
winner. Citation edges are a noisy binary proxy, so this script asks a second,
independent question with a **graded** relevance signal:

    When a human-calibrated relevance judge (UMBRELA, 0-3) scores the top-10
    each reranker actually returns, does ``indus_ranker`` still lose to
    ``minilm``? And if so, *what* is it promoting that the judge rates low?

## What it does

For each of the 50 seeds (loaded from ``results/retrieval_eval_50q.json``):

1. Build ONE shared INDUS-hybrid candidate pool via
   ``scix.search.hybrid_search`` (Qdrant dense + title/abstract BM25 + body
   BM25, RRF) — identical to the 4skc harness, so the comparison is apples to
   apples.
2. Produce three top-K rankings from that single pool:
   ``hybrid_indus`` (RRF order, no rerank), ``minilm``, ``indus_ranker``.
3. Judge the **deduplicated union** of those three top-K lists with the
   ``umbrela_judge`` OAuth subagent (``claude -p`` — NOT the paid Anthropic
   API, per ``feedback_claude_judge_via_oauth``). One judge call per distinct
   (seed, paper) pair; a paper appearing in two configs' top-K is judged once
   and its score is reused in both rankings.
4. Score each config with **judge-graded nDCG@K** (relevance grade = the
   UMBRELA 0-3 score) over the per-seed judged union, alongside the
   **citation-nDCG@K** for the same lists (binary GT from ``citation_edges``)
   so the two relevance notions can be compared directly.

The deliverable is a finding, not a gate: the disagreement (or agreement)
between citation-nDCG and judge-nDCG is itself the result.

## Judged-pool nDCG semantics (read before interpreting numbers)

Judge scores are gathered only for the **pooled top-K union** per seed, not the
full 50-candidate pool. The ideal-DCG denominator in
``scix.ir_metrics.ndcg_at_k`` is therefore computed over the judged union: this
is a *reranking-quality* nDCG measuring how well each config orders the
pooled-relevant set within its own top-K. It is the correct comparison here
because all three configs draw their top-K from one shared pool. It is NOT
comparable to the citation-nDCG absolute values in the 4skc memo, which use the
full citation neighbour set as the ideal.

## Memory envelope

Loads the INDUS encoder + minilm + indus_ranker checkpoints and runs 50
``hybrid_search`` calls — the same peak as ``eval_rerank_local_ab.py``. Wrap in
``scix-batch`` per CLAUDE.md. The ``claude -p`` judge subprocesses are separate
processes with their own memory.

    QDRANT_URL=http://localhost:6633 scix-batch \\
        .venv/bin/python scripts/eval_rerank_umbrela_judge.py --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Reuse the 4skc harness wholesale — same retrieval, same rerank wiring, same
# seed/GT loaders — so this run differs from it ONLY in the relevance signal.
from eval_rerank_local_ab import (  # noqa: E402
    INDUS_RANKER_MODEL_NAME,
    MINILM_MODEL_NAME,
    RRF_K,
    TOP_N_FROM_HYBRID,
    SeedRecord,
    _detect_device,
    _run_hybrid_for_seed,
    build_query_text,
    fetch_ground_truth,
    fetch_seed_record,
    load_seed_bibcodes,
)

from scix.eval.persona_judge import (  # noqa: E402
    DEFAULT_PERSONA,
    DEFAULT_PROMPT_VERSION,
    ClaudeSubprocessDispatcher,
    Dispatcher,
    JudgeScore,
    JudgeTriple,
    PersonaJudge,
    StubDispatcher,
    build_snippet,
)
from scix.ir_metrics import ndcg_at_k  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_rerank_umbrela_judge")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GOLD_PATH = Path("results/retrieval_eval_50q.json")
DEFAULT_OUT_JSON = Path("results/retrieval_eval_50q_rerank_umbrela_judge.json")
DEFAULT_OUT_MD = Path("results/retrieval_eval_50q_rerank_umbrela_judge.md")

# Judge the baseline plus the two rerankers named in the bead. bge_large is
# out of scope for the second opinion (4skc already found it non-significant).
JUDGED_CONFIGS = ("hybrid_indus", "minilm", "indus_ranker")

K_JUDGE = 10  # top-K of each ranking that gets judged + scored
RELEVANT_THRESHOLD = 2  # UMBRELA score >= 2 collapses to "relevant" (matches NASA-IR)
JUDGE_SNIPPET_BUDGET = 1500  # body chars passed to judge; abstract is included verbatim


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedRankings:
    """The three top-K rankings for one seed, plus the judged union."""

    seed_bibcode: str
    query_text: str
    rankings: dict[str, list[str]]  # config -> ordered bibcodes (top-K)
    snippets: dict[str, str]  # bibcode -> judge snippet (union of all top-K)
    relevant: set[str]  # citation ground truth for this seed


@dataclass(frozen=True)
class ConfigJudgeSummary:
    """Aggregated judge metrics across seeds for one config."""

    config: str
    n_queries: int
    judge_ndcg_10: float
    citation_ndcg_10: float
    mean_judge_score_top10: float
    judged_relevant_at_10: float  # mean count of top-10 papers with score >= 2
    score_hist_top10: dict[int, int]  # pooled 0/1/2/3 counts across all seeds
    per_query: tuple[dict[str, Any], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Ranking construction
# ---------------------------------------------------------------------------


def rank_pool(
    pool: list[dict[str, Any]],
    *,
    query_text: str,
    reranker: Any | None,
) -> list[dict[str, Any]]:
    """Return the candidate dicts ordered by a config.

    ``reranker is None`` keeps the RRF (pool) order — the ``hybrid_indus``
    baseline. Otherwise the reranker callable scores and reorders the pool.
    """
    if reranker is None:
        return list(pool)
    return list(reranker(query_text, pool))


def build_seed_rankings(
    seed: SeedRecord,
    pool: list[dict[str, Any]],
    relevant: set[str],
    rerankers: dict[str, Any],
    *,
    k: int,
) -> SeedRankings:
    """Build the three top-K rankings and the dedup judge-snippet union.

    The seed itself is dropped from every ranking (mirrors the 4skc eval
    semantics). Snippets for the judge are built once per distinct bibcode
    from the candidate's title + full abstract.
    """
    query_text = build_query_text(seed.title, seed.abstract)
    candidates = [p for p in pool if p.get("bibcode") and p.get("bibcode") != seed.bibcode]

    by_bib = {p["bibcode"]: p for p in candidates}

    rankings: dict[str, list[str]] = {}
    for config in JUDGED_CONFIGS:
        ordered = rank_pool(candidates, query_text=query_text, reranker=rerankers[config])
        rankings[config] = [p["bibcode"] for p in ordered[:k] if p.get("bibcode")]

    union: set[str] = set()
    for bibs in rankings.values():
        union.update(bibs)

    snippets: dict[str, str] = {}
    for bib in union:
        cand = by_bib.get(bib, {})
        snippets[bib] = build_snippet(
            title=cand.get("title") or bib,
            abstract=cand.get("abstract") or cand.get("abstract_snippet") or None,
            body=None,
            body_char_budget=JUDGE_SNIPPET_BUDGET,
        )

    return SeedRankings(
        seed_bibcode=seed.bibcode,
        query_text=query_text,
        rankings=rankings,
        snippets=snippets,
        relevant=relevant,
    )


def collect_triples(seed_rankings: list[SeedRankings]) -> list[JudgeTriple]:
    """Flatten every (seed, pooled-union paper) pair into judge triples.

    Bibcodes are namespaced by seed in the triple's ``bibcode`` field so the
    same paper judged for two different seeds maps back unambiguously.
    """
    triples: list[JudgeTriple] = []
    for sr in seed_rankings:
        for bib, snippet in sr.snippets.items():
            triples.append(
                JudgeTriple(
                    query=sr.query_text,
                    bibcode=f"{sr.seed_bibcode}|{bib}",
                    snippet=snippet,
                )
            )
    return triples


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def index_scores(scores: list[JudgeScore]) -> dict[str, dict[str, int]]:
    """Map judge scores back to ``{seed_bibcode: {paper_bibcode: score}}``.

    Failed triples (ERROR_SENTINEL, score < 0) are dropped: a paper with no
    usable judge score contributes 0 gain, identical to an unjudged paper.
    """
    out: dict[str, dict[str, int]] = {}
    for s in scores:
        if s.triple is None or s.score < 0:
            continue
        seed_bib, _, paper_bib = s.triple.bibcode.partition("|")
        out.setdefault(seed_bib, {})[paper_bib] = s.score
    return out


def load_prior_scores(path: Path) -> dict[str, tuple[int, bool | None]]:
    """Load ``{namespaced_bibcode: (score, needs_human_review)}`` from a scores JSONL.

    Used to resume a quota-interrupted judge run. Malformed or scoreless lines
    are skipped; the caller decides which scores are reusable.
    """
    prior: dict[str, tuple[int, bool | None]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bib = rec.get("bibcode")
            if bib is None or "score" not in rec:
                continue
            prior[bib] = (int(rec["score"]), rec.get("needs_human_review"))
    return prior


def partition_reusable(
    triples: list[JudgeTriple],
    prior: dict[str, tuple[int, bool | None]],
) -> tuple[list[JudgeScore], list[JudgeTriple]]:
    """Split triples into reusable prior scores and triples still needing a judge call.

    A prior entry is reused only when its score is non-negative (a real 0-3
    grade); failed sentinels (score < 0) and missing bibcodes fall through to
    ``pending`` so they are re-judged.
    """
    reused: list[JudgeScore] = []
    pending: list[JudgeTriple] = []
    for t in triples:
        hit = prior.get(t.bibcode)
        if hit is not None and hit[0] >= 0:
            reused.append(
                JudgeScore(score=hit[0], reason="reused", triple=t, needs_human_review=hit[1])
            )
        else:
            pending.append(t)
    return reused, pending


def _histogram(values: Iterable[int]) -> dict[int, int]:
    hist = {0: 0, 1: 0, 2: 0, 3: 0}
    for v in values:
        if 0 <= v <= 3:
            hist[v] += 1
    return hist


def summarize_config(
    config: str,
    seed_rankings: list[SeedRankings],
    judge_by_seed: dict[str, dict[str, int]],
    *,
    k: int,
) -> ConfigJudgeSummary:
    """Aggregate judge-graded + citation nDCG for one config across seeds.

    Per seed the relevance map for judge-nDCG is the *judged union* of that
    seed's top-K papers (graded 0-3); the ideal DCG is taken over that union,
    so this measures ordering quality of the pooled-relevant set within the
    config's own top-K (see module docstring).
    """
    per_query: list[dict[str, Any]] = []
    pooled_top10_scores: list[int] = []

    for sr in seed_rankings:
        ranking = sr.rankings[config]
        if not ranking:
            continue
        judged = judge_by_seed.get(sr.seed_bibcode, {})

        judge_rel_map = {bib: float(score) for bib, score in judged.items()}
        citation_rel_map = {bib: 1.0 for bib in sr.relevant}

        top10_scores = [judged.get(bib, 0) for bib in ranking[:k]]
        pooled_top10_scores.extend(top10_scores)

        per_query.append(
            {
                "seed_bibcode": sr.seed_bibcode,
                "judge_ndcg_10": round(ndcg_at_k(ranking, judge_rel_map, k=k), 4),
                "citation_ndcg_10": round(ndcg_at_k(ranking, citation_rel_map, k=k), 4),
                "mean_judge_score_top10": round(
                    sum(top10_scores) / max(len(top10_scores), 1), 4
                ),
                "judged_relevant_at_10": sum(1 for s in top10_scores if s >= RELEVANT_THRESHOLD),
                "n_judged_union": len(judged),
            }
        )

    n = len(per_query)
    if n == 0:
        return ConfigJudgeSummary(
            config=config,
            n_queries=0,
            judge_ndcg_10=0.0,
            citation_ndcg_10=0.0,
            mean_judge_score_top10=0.0,
            judged_relevant_at_10=0.0,
            score_hist_top10=_histogram([]),
            per_query=(),
        )
    return ConfigJudgeSummary(
        config=config,
        n_queries=n,
        judge_ndcg_10=round(sum(q["judge_ndcg_10"] for q in per_query) / n, 4),
        citation_ndcg_10=round(sum(q["citation_ndcg_10"] for q in per_query) / n, 4),
        mean_judge_score_top10=round(
            sum(q["mean_judge_score_top10"] for q in per_query) / n, 4
        ),
        judged_relevant_at_10=round(
            sum(q["judged_relevant_at_10"] for q in per_query) / n, 4
        ),
        score_hist_top10=_histogram(pooled_top10_scores),
        per_query=tuple(per_query),
    )


# ---------------------------------------------------------------------------
# Disagreement analysis — the actual deliverable
# ---------------------------------------------------------------------------


def analyze_disagreement(
    seed_rankings: list[SeedRankings],
    judge_by_seed: dict[str, dict[str, int]],
    *,
    k: int,
) -> dict[str, Any]:
    """Why does ``indus_ranker`` underperform ``minilm`` on judged relevance?

    For each seed, partition the top-K bibcodes into:
      - ``indus_only``: in indus_ranker top-K but not minilm top-K
      - ``minilm_only``: in minilm top-K but not indus_ranker top-K
    and compare the mean judge score of each disjoint set. If indus_ranker is
    promoting judge-low papers in place of judge-high ones, ``indus_only``
    papers score materially below ``minilm_only`` papers — that is the
    mechanism behind the citation-nDCG regression.
    """
    indus_only_scores: list[int] = []
    minilm_only_scores: list[int] = []
    n_seeds_with_divergence = 0

    for sr in seed_rankings:
        judged = judge_by_seed.get(sr.seed_bibcode, {})
        indus = set(sr.rankings.get("indus_ranker", [])[:k])
        minilm = set(sr.rankings.get("minilm", [])[:k])
        if not indus or not minilm:
            continue

        indus_only = indus - minilm
        minilm_only = minilm - indus
        if indus_only or minilm_only:
            n_seeds_with_divergence += 1
        indus_only_scores.extend(judged.get(b, 0) for b in indus_only)
        minilm_only_scores.extend(judged.get(b, 0) for b in minilm_only)

    def _mean(xs: list[int]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "n_seeds_with_divergence": n_seeds_with_divergence,
        "indus_only": {
            "n_papers": len(indus_only_scores),
            "mean_judge_score": _mean(indus_only_scores),
            "score_hist": _histogram(indus_only_scores),
            "frac_relevant": round(
                sum(1 for s in indus_only_scores if s >= RELEVANT_THRESHOLD)
                / max(len(indus_only_scores), 1),
                4,
            ),
        },
        "minilm_only": {
            "n_papers": len(minilm_only_scores),
            "mean_judge_score": _mean(minilm_only_scores),
            "score_hist": _histogram(minilm_only_scores),
            "frac_relevant": round(
                sum(1 for s in minilm_only_scores if s >= RELEVANT_THRESHOLD)
                / max(len(minilm_only_scores), 1),
                4,
            ),
        },
        "delta_mean_judge_score_indus_minus_minilm": round(
            _mean(indus_only_scores) - _mean(minilm_only_scores), 4
        ),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _summary_to_json(s: ConfigJudgeSummary) -> dict[str, Any]:
    return {
        "n_queries": s.n_queries,
        "judge_nDCG@10": s.judge_ndcg_10,
        "citation_nDCG@10": s.citation_ndcg_10,
        "mean_judge_score_top10": s.mean_judge_score_top10,
        "judged_relevant@10": s.judged_relevant_at_10,
        "score_hist_top10": s.score_hist_top10,
        "per_query": list(s.per_query),
    }


def write_outputs(
    out_json: Path,
    out_md: Path,
    *,
    summaries: dict[str, ConfigJudgeSummary],
    disagreement: dict[str, Any],
    n_seeds_used: int,
    n_triples: int,
    n_failed: int,
    gold_path: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gold_path": str(gold_path),
        "n_seeds_used": n_seeds_used,
        "n_judge_calls": n_triples,
        "n_judge_failed": n_failed,
        "configs": {name: _summary_to_json(s) for name, s in summaries.items()},
        "disagreement_analysis": disagreement,
        "provenance": provenance,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Wrote JSON report: %s", out_json)

    base = summaries["hybrid_indus"]
    minilm = summaries["minilm"]
    indus = summaries["indus_ranker"]

    lines: list[str] = []
    lines.append("# 50-Query Rerank — UMBRELA LLM-Judge Second Opinion")
    lines.append("")
    lines.append(
        "> **Provenance**: in-house authored. Graded relevance from the "
        "`umbrela_judge` OAuth subagent (verbatim Castorini UMBRELA rubric, "
        "0-3), dispatched via `claude -p` — no paid API. Seeds and the shared "
        f"INDUS-hybrid pool match `scripts/eval_rerank_local_ab.py` (bead 4skc). "
        "Self-reported engineering signal, not an external benchmark."
    )
    lines.append("")
    lines.append(f"**Generated**: {payload['generated_at']}")
    lines.append(f"**Seeds used**: {n_seeds_used}")
    lines.append(f"**Judge calls**: {n_triples} (failed: {n_failed})")
    lines.append(
        f"**Pool**: shared INDUS-hybrid (Qdrant dense + title/abstract BM25 + "
        f"body BM25, RRF k={RRF_K}, top-{TOP_N_FROM_HYBRID}); judged the dedup "
        f"union of each config's top-{K_JUDGE}."
    )
    lines.append("")

    lines.append("## Judge-graded vs citation relevance")
    lines.append("")
    lines.append(
        "| Config | judge nDCG@10 | citation nDCG@10 | mean judge score (top-10) "
        "| judged-relevant@10 (≥2) |"
    )
    lines.append("|--------|---------------|------------------|---------------------------|-------------------------|")
    for name in JUDGED_CONFIGS:
        s = summaries[name]
        lines.append(
            f"| `{name}` | {s.judge_ndcg_10:.4f} | {s.citation_ndcg_10:.4f} | "
            f"{s.mean_judge_score_top10:.4f} | {s.judged_relevant_at_10:.2f} |"
        )
    lines.append("")
    lines.append(
        "> **Judged-pool nDCG**: the ideal DCG is over the per-seed *judged "
        "union* (every paper in any config's top-10), so these values measure "
        "ordering quality of the pooled-relevant set and are NOT comparable to "
        "the citation-nDCG absolutes in the 4skc memo."
    )
    lines.append("")

    lines.append("## Score distribution in the judged top-10 (pooled over seeds)")
    lines.append("")
    lines.append("| Config | score 0 | score 1 | score 2 | score 3 |")
    lines.append("|--------|---------|---------|---------|---------|")
    for name in JUDGED_CONFIGS:
        h = summaries[name].score_hist_top10
        lines.append(f"| `{name}` | {h[0]} | {h[1]} | {h[2]} | {h[3]} |")
    lines.append("")

    lines.append("## Why does `indus_ranker` underperform `minilm`? (disagreement analysis)")
    lines.append("")
    io = disagreement["indus_only"]
    mo = disagreement["minilm_only"]
    lines.append(
        f"Across {disagreement['n_seeds_with_divergence']} seeds where the two "
        "top-10s diverge, comparing the papers each ranker promotes that the "
        "other does not:"
    )
    lines.append("")
    lines.append("| Disjoint set | n papers | mean judge score | frac relevant (≥2) | hist 0/1/2/3 |")
    lines.append("|--------------|----------|------------------|--------------------|--------------|")
    lines.append(
        f"| `indus_ranker` only | {io['n_papers']} | {io['mean_judge_score']:.4f} | "
        f"{io['frac_relevant']:.4f} | "
        f"{io['score_hist'][0]}/{io['score_hist'][1]}/{io['score_hist'][2]}/{io['score_hist'][3]} |"
    )
    lines.append(
        f"| `minilm` only | {mo['n_papers']} | {mo['mean_judge_score']:.4f} | "
        f"{mo['frac_relevant']:.4f} | "
        f"{mo['score_hist'][0]}/{mo['score_hist'][1]}/{mo['score_hist'][2]}/{mo['score_hist'][3]} |"
    )
    lines.append("")
    delta = disagreement["delta_mean_judge_score_indus_minus_minilm"]
    lines.append(
        f"**Δ mean judge score (indus-only − minilm-only) = {delta:+.4f}.** "
        + (
            "Negative confirms the citation-nDCG finding under a graded human-"
            "calibrated signal: the papers `indus_ranker` uniquely promotes into "
            "the top-10 are judged *less* relevant than the ones `minilm` "
            "promotes — the domain-tuned ranker is mis-ordering paper-to-paper "
            "relevance, not merely disagreeing with noisy citation edges."
            if delta < 0
            else "Non-negative: the judge does NOT corroborate the citation-nDCG "
            "regression — the two relevance notions disagree, which itself "
            "qualifies the 4skc NO-GO (the regression may be a citation-proxy "
            "artifact). See per-query rows in the JSON."
        )
    )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    judge_says_indus_worse = indus.judge_ndcg_10 < minilm.judge_ndcg_10
    lines.append(
        f"- Judge nDCG@10: `minilm` {minilm.judge_ndcg_10:.4f} vs `indus_ranker` "
        f"{indus.judge_ndcg_10:.4f} vs baseline `hybrid_indus` {base.judge_ndcg_10:.4f}."
    )
    lines.append(
        "- "
        + (
            "The UMBRELA judge **corroborates** 4skc: the domain-tuned reranker "
            "underperforms the generic one on graded relevance too."
            if judge_says_indus_worse
            else "The UMBRELA judge **does not corroborate** 4skc on this axis: "
            "indus_ranker is not worse than minilm by judge-nDCG. The citation "
            "regression may be partly a binary-proxy artifact."
        )
    )
    lines.append(
        "- The model card for `nasa-impact/nasa-smd-ibm-ranker` describes "
        "MS-MARCO-style short-question→passage fine-tuning. The seed-as-query "
        "task here (full paper title+abstract → related-paper ranking) is a "
        "different relevance notion; the disagreement table above quantifies "
        "how that mismatch surfaces."
    )
    lines.append("")

    lines.append("## Provenance details")
    lines.append("")
    for kk, vv in provenance.items():
        lines.append(f"- **{kk}**: `{vv}`")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    logger.info("Wrote Markdown report: %s", out_md)
    return payload


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval_rerank_umbrela_judge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    p.add_argument("--max-seeds", type=int, default=50)
    p.add_argument("--k", type=int, default=K_JUDGE, help="Top-K judged per config.")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--triples-out",
        type=Path,
        default=None,
        help="Optional path to dump the judge triples JSONL (audit trail).",
    )
    p.add_argument(
        "--scores-out",
        type=Path,
        default=None,
        help="Optional path to dump raw judge scores JSONL (audit trail).",
    )
    p.add_argument(
        "--reuse-scores",
        type=Path,
        default=None,
        help="Resume a quota-interrupted run: reuse score>=0 entries from a "
        "prior scores JSONL and only re-judge the failed/missing triples. "
        "Safe to point at the same path as --scores-out.",
    )
    p.add_argument(
        "--stub",
        action="store_true",
        help="Use StubDispatcher (no Claude calls) — wiring check only.",
    )
    p.add_argument("--claude-binary", default="claude")
    return p.parse_args(list(argv) if argv is not None else None)


def _provenance(device: str, k: int) -> dict[str, Any]:
    return {
        "host_python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "minilm_model": MINILM_MODEL_NAME,
        "indus_ranker_model": INDUS_RANKER_MODEL_NAME,
        "judge_persona": DEFAULT_PERSONA,
        "judge_prompt_version": DEFAULT_PROMPT_VERSION,
        "rrf_k": RRF_K,
        "top_n_from_hybrid": TOP_N_FROM_HYBRID,
        "k_judged": k,
        "relevant_threshold": RELEVANT_THRESHOLD,
        "qdrant_url": os.environ.get("QDRANT_URL", ""),
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    device = _detect_device()

    if not os.environ.get("QDRANT_URL"):
        logger.error(
            "QDRANT_URL not set — the INDUS dense lane is served from Qdrant "
            "(ADR-013). Re-run with QDRANT_URL=http://localhost:6633."
        )
        return 10

    seed_bibcodes, gold_path_used = load_seed_bibcodes(args.gold)
    seed_bibcodes = seed_bibcodes[: args.max_seeds]

    from scix.db import get_connection
    from scix.search import CrossEncoderReranker

    conn = get_connection()
    try:
        seeds: list[SeedRecord] = []
        ground_truth: dict[str, set[str]] = {}
        for bib in seed_bibcodes:
            rec = fetch_seed_record(conn, bib)
            if rec is None:
                continue
            gt = fetch_ground_truth(conn, bib)
            if not gt:
                continue
            seeds.append(rec)
            ground_truth[bib] = gt
        logger.info("Resolved %d/%d usable seeds", len(seeds), len(seed_bibcodes))
        if not seeds:
            logger.error("no usable seeds — aborting")
            return 6

        rerankers: dict[str, Any] = {
            "hybrid_indus": None,
            "minilm": CrossEncoderReranker(model_name=MINILM_MODEL_NAME),
            "indus_ranker": CrossEncoderReranker(model_name=INDUS_RANKER_MODEL_NAME),
        }
        for label, rr in rerankers.items():
            if rr is None:
                continue
            try:
                rr("warmup query", [{"bibcode": "_warm", "title": "warmup", "abstract_snippet": ""}])
            except Exception as exc:  # noqa: BLE001
                logger.warning("warmup failed for %s: %s", label, exc)

        seed_rankings: list[SeedRankings] = []
        for seed in seeds:
            try:
                pool = _run_hybrid_for_seed(conn, seed, top_n=TOP_N_FROM_HYBRID)
            except Exception as exc:  # noqa: BLE001
                logger.warning("pool build failed for %s: %s", seed.bibcode, exc)
                conn.rollback()
                continue
            if not pool:
                continue
            seed_rankings.append(
                build_seed_rankings(
                    seed, pool, ground_truth[seed.bibcode], rerankers, k=args.k
                )
            )
        logger.info("Built rankings for %d seeds", len(seed_rankings))
    finally:
        conn.close()

    triples = collect_triples(seed_rankings)
    logger.info("Collected %d judge triples (dedup union of top-%d)", len(triples), args.k)

    if args.triples_out:
        args.triples_out.parent.mkdir(parents=True, exist_ok=True)
        with args.triples_out.open("w", encoding="utf-8") as f:
            for t in triples:
                f.write(
                    json.dumps(
                        {"query": t.query, "bibcode": t.bibcode, "paper_snippet": t.snippet},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    reused: list[JudgeScore] = []
    triples_to_judge = triples
    if args.reuse_scores and args.reuse_scores.exists():
        prior = load_prior_scores(args.reuse_scores)
        reused, triples_to_judge = partition_reusable(triples, prior)
        logger.info(
            "Resume: reusing %d prior scores; re-judging %d of %d triples",
            len(reused),
            len(triples_to_judge),
            len(triples),
        )

    dispatcher: Dispatcher
    if args.stub:
        dispatcher = StubDispatcher(fixed_score=2, reason="stub")
    else:
        dispatcher = ClaudeSubprocessDispatcher(claude_binary=args.claude_binary)

    judge = PersonaJudge(
        dispatcher=dispatcher,
        max_concurrency=args.concurrency,
        max_retries=args.max_retries,
    )
    t0 = time.perf_counter()
    scores = reused + asyncio.run(judge.run(triples_to_judge))
    logger.info(
        "Judged %d triples in %.1fs (%d reused)",
        len(triples_to_judge),
        time.perf_counter() - t0,
        len(reused),
    )

    n_failed = sum(1 for s in scores if s.score < 0)
    if n_failed:
        logger.warning("%d/%d judge triples failed", n_failed, len(scores))

    if args.scores_out:
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        with args.scores_out.open("w", encoding="utf-8") as f:
            for s in scores:
                f.write(
                    json.dumps(
                        {
                            "bibcode": s.triple.bibcode if s.triple else None,
                            "score": s.score,
                            "needs_human_review": s.needs_human_review,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    judge_by_seed = index_scores(scores)
    summaries = {
        name: summarize_config(name, seed_rankings, judge_by_seed, k=args.k)
        for name in JUDGED_CONFIGS
    }
    disagreement = analyze_disagreement(seed_rankings, judge_by_seed, k=args.k)

    write_outputs(
        args.out_json,
        args.out_md,
        summaries=summaries,
        disagreement=disagreement,
        n_seeds_used=len(seed_rankings),
        n_triples=len(triples),
        n_failed=n_failed,
        gold_path=gold_path_used,
        provenance=_provenance(device, args.k),
    )

    for name in JUDGED_CONFIGS:
        s = summaries[name]
        logger.info(
            "%s: judge_nDCG@10=%.4f citation_nDCG@10=%.4f mean_judge=%.3f",
            name,
            s.judge_ndcg_10,
            s.citation_ndcg_10,
            s.mean_judge_score_top10,
        )
    logger.info(
        "disagreement Δ(indus-only − minilm-only) mean judge score = %+.4f",
        disagreement["delta_mean_judge_score_indus_minus_minilm"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
