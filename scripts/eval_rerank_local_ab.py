#!/usr/bin/env python3
"""A/B eval of cross-encoder reranking on the 50-query INDUS hybrid baseline.

Compares four configurations against ``results/retrieval_eval_50q.json``:

    1. ``hybrid_indus``  — INDUS + BM25 via RRF, no reranker (baseline).
    2. ``minilm``        — same hybrid, reranked by
       ``cross-encoder/ms-marco-MiniLM-L-12-v2``.
    3. ``bge_large``     — same hybrid, reranked by ``BAAI/bge-reranker-large``
       (HF revision pinned at :data:`BGE_RERANKER_LARGE_SHA`).
    4. ``indus_ranker``  — same hybrid, reranked by the INDUS domain-tuned
       cross-encoder ``nasa-impact/nasa-smd-ibm-ranker`` (bead 4skc).

Reports nDCG@10, Recall@10, Recall@20, MRR, P@10 and p50/p95 of the rerank
latency per config. Statistical significance is assessed with paired
Wilcoxon signed-rank tests on per-query nDCG@10 deltas; we run three pairwise
tests against the baseline so the Bonferroni-corrected significance
threshold is α/3 ≈ 0.0167.

POOL-COMPARABILITY DECISION (bead 4skc M3): the April run (commit 06a6cc3)
built the dense lane from the legacy ``paper_embeddings.embedding`` pgvector
column. That table was DROPPED in the Qdrant migration (ADR-013), so pool
comparability with April is impossible. This script re-baselines on the
CURRENT production stack: the candidate pool is built with
``scix.search.hybrid_search`` (INDUS dense served from Qdrant
``scix_indus_v2_papers_s1`` + title/abstract BM25 + body BM25, RRF). All four
configs share ONE cached pool per seed, so the reranker A/B is airtight
WITHIN this run; only cross-run comparison of absolute nDCG to April is lost.
Requires ``QDRANT_URL`` to be set (else hybrid_search has no dense lane).

Outputs (committed):

    * ``results/retrieval_eval_50q_rerank_indus.json``
    * ``results/retrieval_eval_50q_rerank_indus.md``

Memory note: reranker checkpoints + the INDUS encoder + 50 hybrid_search calls
peak under the scix-batch envelope; wrap in ``scix-batch`` per CLAUDE.md:

    QDRANT_URL=http://localhost:6633 scix-batch \\
        .venv/bin/python scripts/eval_rerank_local_ab.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Repo bootstrap so this script runs from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_rerank_local_ab")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HuggingFace commit SHA pin for BAAI/bge-reranker-large.
# Mirrors the value introduced in commit 8339ee2 on main; pinning makes
# rerank scores reproducible across runs even if HEAD on the model repo
# moves.
BGE_RERANKER_LARGE_SHA: str = "55611d7bca2a7133960a6d3b71e083071bbfc312"
BGE_RERANKER_REPO: str = "BAAI/bge-reranker-large"
BGE_RERANKER_LOCAL_DIR: str = "models/bge-reranker-large"

MINILM_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# INDUS domain-tuned cross-encoder (bead 4skc). HF revision pinned and local
# snapshot resolved by scix.search.CrossEncoderReranker; this script only needs
# the canonical model name.
INDUS_RANKER_MODEL_NAME: str = "nasa-impact/nasa-smd-ibm-ranker"

DEFAULT_GOLD_PATH = Path("results/retrieval_eval_50q.json")
PARENT_REPO_FALLBACK = Path("/home/ds/projects/scix_experiments/results/retrieval_eval_50q.json")
DEFAULT_OUT_JSON = Path("results/retrieval_eval_50q_rerank_indus.json")
DEFAULT_OUT_MD = Path("results/retrieval_eval_50q_rerank_indus.md")

CONFIGS = ("hybrid_indus", "minilm", "bge_large", "indus_ranker")

# Hybrid retrieval knobs — mirror eval_retrieval_50q.py defaults.
TOP_N_FROM_HYBRID = 50  # how many candidates feed the reranker
RRF_K = 60
K_METRIC = 10  # primary metric cutoff (nDCG@K, P@K, R@K)
K_RECALL_WIDE = 20

ALPHA = 0.05
N_PAIRWISE_TESTS = 3  # minilm, bge_large, indus_ranker — each vs baseline


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedRecord:
    """Seed paper used as a query."""

    bibcode: str
    title: str
    abstract: str
    embedding: list[float]


@dataclass(frozen=True)
class QueryResult:
    """Per-query metrics for one (config, seed) pair."""

    seed_bibcode: str
    config: str
    ndcg_10: float
    recall_10: float
    recall_20: float
    p_10: float
    mrr: float
    rerank_latency_ms: float
    n_relevant: int
    n_retrieved: int


@dataclass(frozen=True)
class ConfigSummary:
    """Aggregated metrics across all seeds for one config."""

    config: str
    n_queries: int
    ndcg_10: float
    recall_10: float
    recall_20: float
    p_10: float
    mrr: float
    p50_rerank_latency_ms: float
    p95_rerank_latency_ms: float
    per_query: tuple[QueryResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class WilcoxonResult:
    """Outcome of one paired Wilcoxon signed-rank test on nDCG@10."""

    comparison: str
    n_pairs: int
    statistic: float | None
    p_value: float | None
    mean_diff: float
    significant: bool


# ---------------------------------------------------------------------------
# Failure-mode helper (AC5)
# ---------------------------------------------------------------------------


def write_failure_report(
    out_json: Path,
    out_md: Path,
    reason: str,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist a documented failure rather than a fake report.

    Called when the eval cannot complete on this hardware (model download
    failure, DB unreachable, no usable seeds, etc.). Both .md and .json
    are written so the build artifact is a stable, reviewable record.
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "reason": reason,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configs": {},
        "stats": {},
    }
    if extra:
        payload.update(extra)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    md = (
        "# 50-query rerank A/B eval — FAILED\n\n"
        "> **Provenance**: in-house authored; eval did not complete on this run.\n"
        "> No fake numbers were committed (per AC5).\n\n"
        f"**Status**: failed\n"
        f"**Reason**: {reason}\n"
        f"**Generated**: {payload['generated_at']}\n"
    )
    if extra:
        md += "\n## Diagnostics\n\n"
        for k, v in extra.items():
            md += f"- **{k}**: `{v}`\n"
    out_md.write_text(md)
    logger.error("Wrote failure reports: %s, %s", out_md, out_json)


# ---------------------------------------------------------------------------
# Gold-set loading
# ---------------------------------------------------------------------------


def load_seed_bibcodes(gold_path: Path) -> tuple[list[str], Path]:
    """Read seed bibcodes from the 50-query eval JSON.

    Returns the deduplicated, ordered list of seed bibcodes plus the
    actual on-disk path that was loaded (helps the report record where
    the gold set came from).
    """
    actual = gold_path
    if not actual.exists() and PARENT_REPO_FALLBACK.exists():
        logger.info(
            "Gold path %s not found in this worktree; falling back to %s",
            gold_path,
            PARENT_REPO_FALLBACK,
        )
        actual = PARENT_REPO_FALLBACK
    if not actual.exists():
        raise FileNotFoundError(
            f"Gold-set file not found: {gold_path} (also tried {PARENT_REPO_FALLBACK})"
        )

    data = json.loads(actual.read_text())
    per_query = data.get("per_query", [])
    if not per_query:
        raise ValueError(f"Gold file {actual} has no 'per_query' entries")

    seen: set[str] = set()
    bibcodes: list[str] = []
    for entry in per_query:
        bib = entry.get("seed_bibcode")
        if not bib or bib in seen:
            continue
        seen.add(bib)
        bibcodes.append(bib)

    if not bibcodes:
        raise ValueError(f"No seed_bibcode values extracted from {actual}")

    logger.info("Loaded %d unique seed bibcodes from %s", len(bibcodes), actual)
    return bibcodes, actual


# ---------------------------------------------------------------------------
# BGE pre-download
# ---------------------------------------------------------------------------


def ensure_bge_weights(local_dir: Path) -> None:
    """Populate ``models/bge-reranker-large/`` from HF Hub at the pinned SHA.

    Pre-downloading avoids race conditions where the timer would include
    one-off network IO on the first call. Raises on any failure so the
    caller can write a documented FAILED report (AC5).
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    # Cheap presence check: a populated snapshot has a config.json at the
    # root of local_dir. Skip re-download if already present.
    if (local_dir / "config.json").exists():
        logger.info("bge-reranker-large already cached at %s — skipping download", local_dir)
        return

    from huggingface_hub import snapshot_download  # local import; may not be installed

    logger.info(
        "Downloading %s @ %s to %s ...",
        BGE_RERANKER_REPO,
        BGE_RERANKER_LARGE_SHA,
        local_dir,
    )
    snapshot_download(
        repo_id=BGE_RERANKER_REPO,
        revision=BGE_RERANKER_LARGE_SHA,
        local_dir=str(local_dir),
    )
    logger.info("bge-reranker-large download complete")


# ---------------------------------------------------------------------------
# DB-backed seed/embedding/ground-truth helpers
# ---------------------------------------------------------------------------


_WS_RE = re.compile(r"\s+")


def build_query_text(title: str | None, abstract: str | None, *, max_words: int = 50) -> str:
    """Title + first ``max_words`` words of abstract — mirrors the prior eval."""
    title_clean = _WS_RE.sub(" ", (title or "")).strip()
    abstract_clean = _WS_RE.sub(" ", (abstract or "")).strip()
    if abstract_clean:
        abstract_clean = " ".join(abstract_clean.split()[:max_words])
    if title_clean and abstract_clean:
        return f"{title_clean}. {abstract_clean}"
    return title_clean or abstract_clean


# Lazily-loaded INDUS encoder state. The legacy ``paper_embeddings`` table
# (which stored precomputed seed vectors) was dropped in the Qdrant migration,
# so seed query vectors are now computed on the fly with the same model and
# mean-pooling convention used to embed the corpus (see eval_retrieval_50q.py).
_indus_encoder: dict[str, Any] = {"loaded": False}


def _indus_encode(query: str) -> list[float]:
    """Encode a query string with the local INDUS model (mean pooling)."""
    if not _indus_encoder["loaded"]:
        from scix.embed import embed_batch, load_model

        model, tokenizer = load_model("indus", device="auto")
        _indus_encoder.update(
            loaded=True, model=model, tokenizer=tokenizer, embed_batch=embed_batch
        )
        logger.info("Loaded INDUS encoder for seed query embedding")
    vectors = _indus_encoder["embed_batch"](
        _indus_encoder["model"], _indus_encoder["tokenizer"], [query], pooling="mean"
    )
    if not vectors:
        raise RuntimeError("INDUS embed_batch returned no vectors")
    return vectors[0]


def fetch_seed_record(conn: Any, bibcode: str) -> SeedRecord | None:
    """Load (title, abstract) for one seed and encode its INDUS query vector.

    Reads metadata from ``papers`` only — the legacy ``paper_embeddings`` table
    no longer exists. The query vector is recomputed with the INDUS encoder over
    ``build_query_text(title, abstract)`` so it lives in the same space as the
    Qdrant dense collection.
    """
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT bibcode, title, abstract FROM papers WHERE bibcode = %s",
            [bibcode],
        )
        row = cur.fetchone()

    if row is None or not (row.get("title") or row.get("abstract")):
        return None

    title = row.get("title") or ""
    abstract = row.get("abstract") or ""
    embedding = _indus_encode(build_query_text(title, abstract))

    return SeedRecord(
        bibcode=row["bibcode"],
        title=title,
        abstract=abstract,
        embedding=embedding,
    )


def fetch_ground_truth(conn: Any, bibcode: str, *, max_relevant: int = 500) -> set[str]:
    """Citation-based ground truth: cited-by ∪ cites, capped per direction."""
    with conn.cursor() as cur:
        cur.execute(
            """
            (SELECT target_bibcode FROM citation_edges
              WHERE source_bibcode = %s
              LIMIT %s)
            UNION
            (SELECT source_bibcode FROM citation_edges
              WHERE target_bibcode = %s
              LIMIT %s)
            """,
            [bibcode, max_relevant, bibcode, max_relevant],
        )
        return {row[0] for row in cur.fetchall() if row[0] != bibcode}


# ---------------------------------------------------------------------------
# IR metrics — reuse canonical implementations from scix.ir_metrics
# ---------------------------------------------------------------------------


def _import_metric_helpers() -> tuple[Any, Any, Any, Any]:
    from scix.ir_metrics import (  # type: ignore[import-not-found]
        mean_reciprocal_rank,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )

    return ndcg_at_k, recall_at_k, precision_at_k, mean_reciprocal_rank


# ---------------------------------------------------------------------------
# Per-config eval loop
# ---------------------------------------------------------------------------


def _fetch_full_abstracts(conn: Any, bibcodes: list[str]) -> dict[str, str]:
    """Fetch full abstracts for the pooled candidates in one round-trip.

    ``hybrid_search`` returns ``PaperStub`` dicts whose ``abstract_snippet`` is
    truncated to 150 chars — too short for a cross-encoder. We fetch the full
    abstract and trim to 1000 chars (well inside the 512-token window) so every
    reranker sees the same rich input the April run used.
    """
    if not bibcodes:
        return {}
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT bibcode, abstract FROM papers WHERE bibcode = ANY(%s)",
            [bibcodes],
        )
        return {r["bibcode"]: (r.get("abstract") or "") for r in cur.fetchall()}


def _run_hybrid_for_seed(
    conn: Any,
    seed: SeedRecord,
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """One production hybrid retrieval (Qdrant dense + BM25 + body BM25, RRF).

    Uses :func:`scix.search.hybrid_search` — the same path the authoritative
    50q retrieval harness (``eval_retrieval_50q.py``) uses — so the candidate
    pool reflects the current production stack. The dense lane is served from
    Qdrant (requires ``QDRANT_URL``). Each candidate is enriched with its full
    abstract (trimmed to 1000 chars) for the reranker stage.
    """
    from scix.search import hybrid_search  # local import — heavy module

    query_text = build_query_text(seed.title, seed.abstract)
    result = hybrid_search(
        conn,
        query_text=query_text,
        query_embedding=seed.embedding,
        model_name="indus",
        top_n=top_n,
        rrf_k=RRF_K,
        include_body=True,
    )
    fused = [p for p in result.papers if p.get("bibcode")]

    abstracts = _fetch_full_abstracts(conn, [p["bibcode"] for p in fused])
    enriched: list[dict[str, Any]] = []
    for p in fused:
        abstract = abstracts.get(p["bibcode"], "")
        enriched.append(
            {
                "bibcode": p["bibcode"],
                "title": p.get("title"),
                "abstract": abstract,
                "abstract_snippet": abstract[:1000],
                "score": float(p.get("score") or 0.0),
            }
        )
    return enriched


def run_config(
    conn: Any,
    seeds: list[SeedRecord],
    ground_truth: dict[str, set[str]],
    cached_pools: dict[str, list[dict[str, Any]]],
    *,
    config: str,
    reranker: Any | None,
) -> ConfigSummary:
    """Re-rank a per-seed candidate pool and compute retrieval metrics.

    The candidate pool comes from ``cached_pools`` (computed once per
    seed and shared across configs) so retrieval cost is paid once and
    only the rerank step is timed.
    """
    ndcg_at_k, recall_at_k, precision_at_k, mean_reciprocal_rank = _import_metric_helpers()

    per_query: list[QueryResult] = []
    rerank_latencies: list[float] = []

    for seed in seeds:
        relevant = ground_truth.get(seed.bibcode, set())
        if not relevant:
            continue

        pool = cached_pools.get(seed.bibcode)
        if not pool:
            continue

        # Drop the seed itself before scoring (mirrors prior eval semantics).
        candidates = [p for p in pool if p.get("bibcode") != seed.bibcode]
        if not candidates:
            continue

        query_text = build_query_text(seed.title, seed.abstract)

        rerank_ms = 0.0
        if reranker is not None:
            t_rerank = time.perf_counter()
            try:
                ranked = reranker(query_text, candidates)
            except Exception as exc:  # noqa: BLE001
                logger.warning("rerank failed for %s (%s): %s", seed.bibcode, config, exc)
                continue
            rerank_ms = (time.perf_counter() - t_rerank) * 1000.0
            rerank_latencies.append(rerank_ms)
        else:
            ranked = candidates

        retrieved_bibs = [p["bibcode"] for p in ranked if p.get("bibcode")]
        if not retrieved_bibs:
            continue

        relevance_map = {bib: 1.0 for bib in relevant}
        per_query.append(
            QueryResult(
                seed_bibcode=seed.bibcode,
                config=config,
                ndcg_10=round(ndcg_at_k(retrieved_bibs, relevance_map, k=K_METRIC), 4),
                recall_10=round(recall_at_k(retrieved_bibs, relevant, k=K_METRIC), 4),
                recall_20=round(recall_at_k(retrieved_bibs, relevant, k=K_RECALL_WIDE), 4),
                p_10=round(precision_at_k(retrieved_bibs, relevant, k=K_METRIC), 4),
                mrr=round(mean_reciprocal_rank(retrieved_bibs, relevant), 4),
                rerank_latency_ms=round(rerank_ms, 2),
                n_relevant=len(relevant),
                n_retrieved=len(retrieved_bibs),
            )
        )

    return _summarize(config, per_query, rerank_latencies)


def _percentile(values: list[float], p: float) -> float:
    """Inclusive linear-interpolation percentile, no numpy dependency."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (p / 100.0) * (len(s) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(s[lower])
    frac = rank - lower
    return float(s[lower] + (s[upper] - s[lower]) * frac)


def _summarize(
    config: str,
    per_query: list[QueryResult],
    rerank_latencies: list[float],
) -> ConfigSummary:
    n = len(per_query)
    if n == 0:
        return ConfigSummary(
            config=config,
            n_queries=0,
            ndcg_10=0.0,
            recall_10=0.0,
            recall_20=0.0,
            p_10=0.0,
            mrr=0.0,
            p50_rerank_latency_ms=0.0,
            p95_rerank_latency_ms=0.0,
            per_query=(),
        )
    return ConfigSummary(
        config=config,
        n_queries=n,
        ndcg_10=round(sum(q.ndcg_10 for q in per_query) / n, 4),
        recall_10=round(sum(q.recall_10 for q in per_query) / n, 4),
        recall_20=round(sum(q.recall_20 for q in per_query) / n, 4),
        p_10=round(sum(q.p_10 for q in per_query) / n, 4),
        mrr=round(sum(q.mrr for q in per_query) / n, 4),
        p50_rerank_latency_ms=round(_percentile(rerank_latencies, 50), 2),
        p95_rerank_latency_ms=round(_percentile(rerank_latencies, 95), 2),
        per_query=tuple(per_query),
    )


# ---------------------------------------------------------------------------
# Wilcoxon + Bonferroni
# ---------------------------------------------------------------------------


def wilcoxon_pairwise(
    baseline: ConfigSummary,
    candidate: ConfigSummary,
    *,
    bonferroni_threshold: float,
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank on per-query nDCG@10 deltas (candidate − baseline)."""
    base_by_seed = {q.seed_bibcode: q.ndcg_10 for q in baseline.per_query}
    diffs: list[float] = []
    for q in candidate.per_query:
        if q.seed_bibcode in base_by_seed:
            diffs.append(q.ndcg_10 - base_by_seed[q.seed_bibcode])

    comparison = f"{candidate.config} vs {baseline.config}"
    if len(diffs) < 5:
        return WilcoxonResult(
            comparison=comparison,
            n_pairs=len(diffs),
            statistic=None,
            p_value=None,
            mean_diff=round(sum(diffs) / max(len(diffs), 1), 4) if diffs else 0.0,
            significant=False,
        )

    # If every diff is zero, scipy returns NaN — treat as not-significant.
    if all(abs(d) < 1e-12 for d in diffs):
        return WilcoxonResult(
            comparison=comparison,
            n_pairs=len(diffs),
            statistic=0.0,
            p_value=1.0,
            mean_diff=0.0,
            significant=False,
        )

    from scipy.stats import wilcoxon  # local import — heavy dep

    stat, p_value = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
    mean_diff = sum(diffs) / len(diffs)
    p_float = float(p_value)
    return WilcoxonResult(
        comparison=comparison,
        n_pairs=len(diffs),
        statistic=float(stat),
        p_value=round(p_float, 6),
        mean_diff=round(mean_diff, 4),
        significant=(p_float < bonferroni_threshold) and (mean_diff > 0),
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _config_to_json(summary: ConfigSummary) -> dict[str, Any]:
    return {
        "n_queries": summary.n_queries,
        "nDCG@10": summary.ndcg_10,
        "Recall@10": summary.recall_10,
        "Recall@20": summary.recall_20,
        "MRR": summary.mrr,
        "P@10": summary.p_10,
        "p50_rerank_latency_ms": summary.p50_rerank_latency_ms,
        "p95_rerank_latency_ms": summary.p95_rerank_latency_ms,
        "per_query": [asdict(q) for q in summary.per_query],
    }


def _wilcoxon_to_json(w: WilcoxonResult) -> dict[str, Any]:
    return {
        "comparison": w.comparison,
        "n_pairs": w.n_pairs,
        "statistic": w.statistic,
        "p_value": w.p_value,
        "mean_diff": w.mean_diff,
        "significant": w.significant,
    }


def write_outputs(
    out_json: Path,
    out_md: Path,
    *,
    summaries: dict[str, ConfigSummary],
    wilcoxon_results: list[WilcoxonResult],
    bonferroni_threshold: float,
    n_seeds_used: int,
    gold_path: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Persist machine-readable JSON + human-readable Markdown."""
    baseline = summaries["hybrid_indus"]
    # Winner: highest nDCG@10. Ties broken by lower p95 rerank latency
    # (mechanical tiebreaker, not a quality judgment — see ZFC patterns rule).
    winner_summary = max(
        summaries.values(),
        key=lambda s: (s.ndcg_10, -s.p95_rerank_latency_ms),
    )
    winner = {
        "config": winner_summary.config,
        "ndcg_10": winner_summary.ndcg_10,
        "delta_vs_baseline": round(winner_summary.ndcg_10 - baseline.ndcg_10, 4),
        "p95_rerank_latency_ms": winner_summary.p95_rerank_latency_ms,
    }

    payload = {
        "schema_version": 1,
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gold_path": str(gold_path),
        "n_seeds_used": n_seeds_used,
        "configs": {name: _config_to_json(s) for name, s in summaries.items()},
        "stats": {
            "alpha": ALPHA,
            "n_tests": N_PAIRWISE_TESTS,
            "bonferroni_threshold": bonferroni_threshold,
            "tests": [_wilcoxon_to_json(w) for w in wilcoxon_results],
        },
        "winner": winner,
        "provenance": provenance,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Wrote JSON report: %s", out_json)

    md_lines: list[str] = []
    md_lines.append("# 50-Query Rerank A/B Eval — INDUS hybrid + cross-encoders")
    md_lines.append("")
    md_lines.append(
        "> **Provenance**: in-house authored. Seed bibcodes are loaded from "
        f"`{gold_path}`; ground truth is re-derived from the live "
        "`citation_edges` table at run time. Metrics are self-reported and "
        "should be interpreted as an engineering signal, not an external benchmark."
    )
    md_lines.append("")
    md_lines.append(f"**Generated**: {payload['generated_at']}")
    md_lines.append(f"**Queries usable**: {n_seeds_used}")
    md_lines.append(
        f"**Hybrid stack**: INDUS dense (Qdrant) + title/abstract BM25 + body "
        f"BM25, RRF k={RRF_K}, top-{TOP_N_FROM_HYBRID} candidates fed to reranker."
    )
    md_lines.append("")

    md_lines.append("## Configs")
    md_lines.append("")
    md_lines.append(
        "| Config | nDCG@10 | Recall@10 | Recall@20 | MRR | P@10 | p50 rerank ms | p95 rerank ms |"
    )
    md_lines.append(
        "|--------|---------|-----------|-----------|-----|------|---------------|---------------|"
    )
    for name in CONFIGS:
        s = summaries[name]
        md_lines.append(
            f"| `{name}` | {s.ndcg_10:.4f} | {s.recall_10:.4f} | "
            f"{s.recall_20:.4f} | {s.mrr:.4f} | {s.p_10:.4f} | "
            f"{s.p50_rerank_latency_ms:.2f} | {s.p95_rerank_latency_ms:.2f} |"
        )
    md_lines.append("")

    md_lines.append("## Statistical significance")
    md_lines.append("")
    md_lines.append(
        f"{N_PAIRWISE_TESTS} pairwise paired Wilcoxon signed-rank tests on per-query "
        f"nDCG@10 deltas. "
        f"Bonferroni-corrected significance threshold: α={ALPHA} / {N_PAIRWISE_TESTS} = "
        f"{bonferroni_threshold:.4f}."
    )
    md_lines.append("")
    md_lines.append("| Comparison | n | mean Δ nDCG@10 | Wilcoxon stat | p-value | significant |")
    md_lines.append("|------------|---|----------------|---------------|---------|-------------|")
    for w in wilcoxon_results:
        p_str = f"{w.p_value:.6f}" if w.p_value is not None else "n/a"
        s_str = f"{w.statistic:.2f}" if w.statistic is not None else "n/a"
        md_lines.append(
            f"| {w.comparison} | {w.n_pairs} | {w.mean_diff:+.4f} | {s_str} | "
            f"{p_str} | {'yes' if w.significant else 'no'} |"
        )
    md_lines.append("")

    md_lines.append("## Winner")
    md_lines.append("")
    md_lines.append(
        f"**Winner**: `{winner['config']}` — nDCG@10 {winner['ndcg_10']:.4f} "
        f"({winner['delta_vs_baseline']:+.4f} vs `hybrid_indus` baseline), "
        f"p95 rerank latency {winner['p95_rerank_latency_ms']:.2f} ms."
    )
    md_lines.append("")

    md_lines.append("## Methodology")
    md_lines.append("")
    md_lines.append(
        f"- For each seed bibcode (loaded from `{gold_path}`), build a single "
        f"candidate pool of top-{TOP_N_FROM_HYBRID} via "
        f"`scix.search.hybrid_search` (the production path): INDUS dense lane "
        f"served from Qdrant (`scix_indus_v2_papers_s1`) + title/abstract BM25 "
        f"+ body BM25, fused with RRF (k={RRF_K}). The seed query vector is "
        f"recomputed with the INDUS encoder (mean pooling). The pool is reused "
        f"across all four configs so retrieval cost is paid once and only the "
        f"rerank stage is timed."
    )
    md_lines.append(
        "- The reranker (where present) scores all candidates; baseline "
        "returns the RRF order untouched. Candidates are enriched with their "
        "full abstract (trimmed to 1000 chars) before reranking — "
        "`hybrid_search` returns only a 150-char snippet, too short for a "
        "cross-encoder."
    )
    md_lines.append(
        f"- Metrics computed over the truncated ranking via `scix.ir_metrics`. "
        f"Recall@{K_METRIC}/{K_RECALL_WIDE}, P@{K_METRIC}, MRR, "
        f"nDCG@{K_METRIC} are reported."
    )
    md_lines.append(
        "- Rerank latency is measured around the reranker callable only "
        "(weights are pre-warmed before the bench loop so the first "
        "scored query does not include weight materialization). Baseline "
        "p50/p95 are zero because no rerank runs."
    )
    md_lines.append(
        "- Ground truth is binary citation relevance: papers that cite or are "
        "cited by the seed (capped at 500 per direction). Pulled live from "
        "`citation_edges`."
    )
    md_lines.append(
        "- **Pool-comparability note (bead 4skc M3):** the April run (06a6cc3) "
        "built the dense lane from the legacy `paper_embeddings.embedding` "
        "pgvector column, which was DROPPED in the ADR-013 Qdrant migration. "
        "Pool comparability with April is therefore impossible; this is a "
        "re-baseline on the current production stack. Absolute nDCG is NOT "
        "comparable to the April memo, but the four-way reranker A/B is "
        "airtight because every config shares one pool per seed."
    )
    md_lines.append("")

    md_lines.append("## Provenance details")
    md_lines.append("")
    for k, v in provenance.items():
        md_lines.append(f"- **{k}**: `{v}`")
    md_lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines))
    logger.info("Wrote Markdown report: %s", out_md)

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval_rerank_local_ab",
        description=(
            "A/B eval of MiniLM and BAAI/bge-reranker-large rerankers against the "
            "INDUS-hybrid baseline on the 50-query gold set. Reports nDCG@10, "
            "Recall@{10,20}, MRR, P@10, p50/p95 rerank latency, and Wilcoxon "
            "signed-rank significance with Bonferroni correction."
        ),
        epilog=(
            "Memory: peak RSS ~3-4 GB (BGE checkpoint + cuda buffers). Wrap with "
            "`scix-batch` if running on the shared host alongside the gascity "
            "supervisor — see CLAUDE.md."
        ),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help=(
            "Path to the 50-query eval JSON. Default: results/retrieval_eval_50q.json. "
            "If missing in the current worktree, the parent repo path is tried."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Where to write the machine-readable report (default: %(default)s).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_OUT_MD,
        help="Where to write the markdown report (default: %(default)s).",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=50,
        help="Cap on number of usable seeds (default: 50).",
    )
    parser.add_argument(
        "--bge-local-dir",
        type=Path,
        default=Path(BGE_RERANKER_LOCAL_DIR),
        help="Local cache directory for bge-reranker-large weights.",
    )
    parser.add_argument(
        "--skip-bge-download",
        action="store_true",
        help="Assume BGE weights are already populated; skip snapshot_download.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _provenance(device: str, bge_local_dir: Path) -> dict[str, Any]:
    return {
        "host_python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "bge_revision": BGE_RERANKER_LARGE_SHA,
        "bge_local_dir": str(bge_local_dir),
        "minilm_model": MINILM_MODEL_NAME,
        "indus_ranker_model": INDUS_RANKER_MODEL_NAME,
        "rrf_k": RRF_K,
        "top_n_from_hybrid": TOP_N_FROM_HYBRID,
        "k_metric": K_METRIC,
        "pool_source": (
            "hybrid_search (Qdrant dense scix_indus_v2_papers_s1 + title/abstract "
            "BM25 + body BM25, RRF) — re-baseline on production stack; "
            "paper_embeddings pgvector column dropped in ADR-013 Qdrant migration, "
            "so April pool comparability is not possible"
        ),
        "qdrant_url": os.environ.get("QDRANT_URL", ""),
    }


def _detect_device() -> str:
    try:
        import torch  # type: ignore[import-not-found]

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    device = _detect_device()
    provenance = _provenance(device, args.bge_local_dir)

    # The dense lane is served from Qdrant; without QDRANT_URL set,
    # hybrid_search silently degrades to lexical-only and the pool is no
    # longer representative of the production stack. Fail loudly instead.
    if not os.environ.get("QDRANT_URL"):
        write_failure_report(
            args.out_json,
            args.out_md,
            reason=(
                "QDRANT_URL not set — the INDUS dense lane is served from Qdrant "
                "(scix_indus_v2_papers_s1) since ADR-013. Re-run with "
                "QDRANT_URL=http://localhost:6633."
            ),
            extra={"phase": "preflight"},
        )
        return 10

    # Phase A: pre-download bge-reranker-large weights so the rerank-time
    # latency excludes one-off network IO.
    if not args.skip_bge_download:
        try:
            ensure_bge_weights(args.bge_local_dir)
        except Exception as exc:  # noqa: BLE001
            write_failure_report(
                args.out_json,
                args.out_md,
                reason=f"bge-reranker-large pre-download failed: {exc!r}",
                extra={"phase": "snapshot_download", "device": device},
            )
            return 2

    # Phase B: load seed bibcodes from the gold set.
    try:
        seed_bibcodes, gold_path_used = load_seed_bibcodes(args.gold)
    except (FileNotFoundError, ValueError) as exc:
        write_failure_report(
            args.out_json,
            args.out_md,
            reason=f"gold-set load failed: {exc!r}",
            extra={"phase": "load_gold", "gold_path": str(args.gold)},
        )
        return 3
    seed_bibcodes = seed_bibcodes[: args.max_seeds]

    # Phase C: open DB connection and resolve seed records + ground truth.
    try:
        from scix.db import get_connection  # local import keeps script importable
    except Exception as exc:  # noqa: BLE001
        write_failure_report(
            args.out_json,
            args.out_md,
            reason=f"scix import failed: {exc!r}",
            extra={"phase": "import_scix"},
        )
        return 4

    try:
        conn = get_connection()
    except Exception as exc:  # noqa: BLE001
        write_failure_report(
            args.out_json,
            args.out_md,
            reason=f"DB connection failed: {exc!r}",
            extra={"phase": "db_connect"},
        )
        return 5

    seeds: list[SeedRecord] = []
    ground_truth: dict[str, set[str]] = {}
    skipped_no_embedding: list[str] = []
    skipped_no_neighbors: list[str] = []

    for bib in seed_bibcodes:
        rec = fetch_seed_record(conn, bib)
        if rec is None:
            skipped_no_embedding.append(bib)
            continue
        gt = fetch_ground_truth(conn, bib)
        if not gt:
            skipped_no_neighbors.append(bib)
            continue
        seeds.append(rec)
        ground_truth[bib] = gt

    logger.info(
        "Resolved %d/%d seeds (skipped %d no-embedding, %d no-neighbors)",
        len(seeds),
        len(seed_bibcodes),
        len(skipped_no_embedding),
        len(skipped_no_neighbors),
    )

    if not seeds:
        conn.close()
        write_failure_report(
            args.out_json,
            args.out_md,
            reason="no usable seeds (all missing INDUS embedding or citation neighbors)",
            extra={
                "phase": "resolve_seeds",
                "n_loaded": len(seed_bibcodes),
                "n_no_embedding": len(skipped_no_embedding),
                "n_no_neighbors": len(skipped_no_neighbors),
            },
        )
        return 6

    # Phase D: run the three configs.
    try:
        from scix.search import CrossEncoderReranker  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        conn.close()
        write_failure_report(
            args.out_json,
            args.out_md,
            reason=f"CrossEncoderReranker import failed: {exc!r}",
            extra={"phase": "import_reranker"},
        )
        return 7

    # Phase D.1: build the per-seed candidate pool ONCE so retrieval
    # cost is shared across configs and we only time the rerank stage.
    logger.info("Pre-computing INDUS-hybrid candidate pools for %d seeds ...", len(seeds))
    cached_pools: dict[str, list[dict[str, Any]]] = {}
    pool_failures = 0
    for seed in seeds:
        try:
            pool = _run_hybrid_for_seed(conn, seed, top_n=TOP_N_FROM_HYBRID)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hybrid pool build failed for %s: %s", seed.bibcode, exc)
            pool = []
            pool_failures += 1
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
        cached_pools[seed.bibcode] = pool

    usable_pools = sum(1 for p in cached_pools.values() if p)
    logger.info(
        "Built %d/%d candidate pools (failures=%d)",
        usable_pools,
        len(seeds),
        pool_failures,
    )
    if usable_pools == 0:
        conn.close()
        write_failure_report(
            args.out_json,
            args.out_md,
            reason=(
                "INDUS-hybrid candidate pool empty for every seed — likely DB "
                "schema mismatch (e.g. embedding column missing) or text "
                "search config not installed"
            ),
            extra={
                "phase": "build_pools",
                "n_seeds": len(seeds),
                "pool_failures": pool_failures,
            },
        )
        return 9

    summaries: dict[str, ConfigSummary] = {}
    rerankers: dict[str, Any] = {
        "hybrid_indus": None,
        "minilm": CrossEncoderReranker(model_name=MINILM_MODEL_NAME),
        "bge_large": _build_bge_reranker(args.bge_local_dir, CrossEncoderReranker),
        "indus_ranker": CrossEncoderReranker(model_name=INDUS_RANKER_MODEL_NAME),
    }

    # Warm-load each reranker once with a tiny dummy pair so the first
    # measured rerank() doesn't include weight materialization time.
    # Without this the p50/p95 latency for the first config is dominated
    # by ~3s of one-off model loading rather than steady-state scoring.
    for label, rr in rerankers.items():
        if rr is None:
            continue
        logger.info("Warming reranker %s ...", label)
        try:
            rr("warmup query", [{"bibcode": "_warm", "title": "warmup", "abstract_snippet": ""}])
        except Exception as exc:  # noqa: BLE001
            logger.warning("warmup failed for %s: %s", label, exc)

    for config in CONFIGS:
        logger.info("Running config %s ...", config)
        t0 = time.perf_counter()
        try:
            summary = run_config(
                conn,
                seeds,
                ground_truth,
                cached_pools,
                config=config,
                reranker=rerankers[config],
            )
        except Exception as exc:  # noqa: BLE001
            conn.close()
            write_failure_report(
                args.out_json,
                args.out_md,
                reason=f"config {config!r} crashed: {exc!r}",
                extra={"phase": f"run_{config}"},
            )
            return 8
        elapsed = time.perf_counter() - t0
        logger.info(
            "  %s: n=%d  nDCG@10=%.4f  R@10=%.4f  R@20=%.4f  MRR=%.4f  "
            "p50=%.1fms  p95=%.1fms  wall=%.1fs",
            config,
            summary.n_queries,
            summary.ndcg_10,
            summary.recall_10,
            summary.recall_20,
            summary.mrr,
            summary.p50_rerank_latency_ms,
            summary.p95_rerank_latency_ms,
            elapsed,
        )
        summaries[config] = summary

    conn.close()

    # Phase E: stats + outputs.
    bonferroni_threshold = ALPHA / N_PAIRWISE_TESTS
    baseline = summaries["hybrid_indus"]
    wilcoxon_results = [
        wilcoxon_pairwise(baseline, summaries["minilm"], bonferroni_threshold=bonferroni_threshold),
        wilcoxon_pairwise(
            baseline, summaries["bge_large"], bonferroni_threshold=bonferroni_threshold
        ),
        wilcoxon_pairwise(
            baseline, summaries["indus_ranker"], bonferroni_threshold=bonferroni_threshold
        ),
    ]

    payload = write_outputs(
        args.out_json,
        args.out_md,
        summaries=summaries,
        wilcoxon_results=wilcoxon_results,
        bonferroni_threshold=bonferroni_threshold,
        n_seeds_used=len(seeds),
        gold_path=gold_path_used,
        provenance=provenance,
    )

    logger.info("Winner: %s", payload["winner"])
    return 0


def _build_bge_reranker(local_dir: Path, reranker_cls: Any) -> Any:
    """Construct CrossEncoderReranker pointing at the local BGE snapshot.

    Older `CrossEncoderReranker` revisions (pre-commit 8339ee2) pass the
    model_name directly to sentence_transformers.CrossEncoder. When the
    weights are pre-downloaded under ``local_dir`` we point the loader
    there instead of the HF Hub identifier so inference is offline.
    Falls back to the canonical repo id if the local snapshot is empty
    (HF cache will still serve it without network).
    """
    if local_dir.exists() and (local_dir / "config.json").exists():
        return reranker_cls(model_name=str(local_dir))
    return reranker_cls(model_name=BGE_RERANKER_REPO)


if __name__ == "__main__":
    raise SystemExit(main())
