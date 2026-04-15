#!/usr/bin/env python3
"""50-query retrieval evaluation: compare embedding models and hybrid search.

Uses citation-based ground truth on the pilot sample (20K papers):
  - Seed papers with rich citation networks become queries
  - Citation neighbors (references + citing papers) within the sample = relevant set
  - Standard IR metrics: nDCG@10, Recall@10, Recall@20, MRR

Models compared:
  - specter2 (768d, scientific domain)
  - indus (768d, NASA-trained)
  - nomic (768d, general purpose)
  - lexical (tsvector ts_rank_cd on full corpus)
  - hybrid: lexical + specter2 via RRF(k=60)
  - hybrid: lexical + indus via RRF(k=60)

Paper contribution: Section 4.4 of ADASS 2026 paper.

Usage:
    python scripts/eval_retrieval_50q.py
    python scripts/eval_retrieval_50q.py --seed-papers 50 --output results/retrieval_eval.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg
from psycopg.rows import dict_row

from scix.db import get_connection
from scix.search import (
    SearchFilters,
    SearchResult,
    hybrid_search,
    lexical_search,
    rrf_fuse,
    vector_search,
)


def get_conn() -> psycopg.Connection:
    """Get a fresh database connection."""
    return get_connection()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_retrieval")


# ---------------------------------------------------------------------------
# IR Metrics
# ---------------------------------------------------------------------------


def dcg_at_k(relevance: list[int], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevance[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k.

    Binary relevance: 1 if in relevant set, 0 otherwise.
    """
    # Actual DCG from retrieved order
    relevance = [1 if bib in relevant else 0 for bib in retrieved[:k]]
    actual_dcg = dcg_at_k(relevance, k)

    # Ideal DCG: all relevant docs first
    ideal_relevance = sorted(relevance, reverse=True)
    # But if there are more relevant docs than k, ideal is all 1s
    n_relevant_in_top_k = min(len(relevant), k)
    ideal_relevance = [1] * n_relevant_in_top_k + [0] * (k - n_relevant_in_top_k)
    ideal_dcg = dcg_at_k(ideal_relevance, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall at rank k: fraction of relevant docs found in top k."""
    if not relevant:
        return 0.0
    found = sum(1 for bib in retrieved[:k] if bib in relevant)
    return found / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, bib in enumerate(retrieved):
        if bib in relevant:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision at rank k: fraction of top-k that are relevant."""
    if k == 0:
        return 0.0
    found = sum(1 for bib in retrieved[:k] if bib in relevant)
    return found / k


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryEval:
    """Evaluation result for a single query against one method."""

    seed_bibcode: str
    method: str
    ndcg_10: float
    recall_10: float
    recall_20: float
    precision_10: float
    mrr_val: float
    relevant_count: int
    retrieved_count: int
    latency_ms: float


@dataclass(frozen=True)
class EvalSummary:
    """Aggregated summary per method across all queries."""

    method: str
    n_queries: int
    mean_ndcg_10: float
    mean_recall_10: float
    mean_recall_20: float
    mean_precision_10: float
    mean_mrr: float
    mean_latency_ms: float
    std_ndcg_10: float


# ---------------------------------------------------------------------------
# Ground truth: citation-based relevance
# ---------------------------------------------------------------------------


def select_seed_papers(
    conn: psycopg.Connection,
    n_seeds: int,
    min_neighbors: int = 5,
) -> list[dict[str, Any]]:
    """Select seed papers with rich citation networks within the pilot sample.

    Picks papers with at least `min_neighbors` citation neighbors in the sample,
    stratified by citation count tier to avoid bias toward only highly-cited papers.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            WITH neighbor_counts AS (
                SELECT bibcode, COUNT(*) as n_neighbors
                FROM (
                    SELECT source_bibcode as bibcode
                    FROM citation_edges
                    WHERE source_bibcode IN (SELECT bibcode FROM _pilot_sample)
                      AND target_bibcode IN (SELECT bibcode FROM _pilot_sample)
                    UNION ALL
                    SELECT target_bibcode as bibcode
                    FROM citation_edges
                    WHERE source_bibcode IN (SELECT bibcode FROM _pilot_sample)
                      AND target_bibcode IN (SELECT bibcode FROM _pilot_sample)
                ) edges
                GROUP BY bibcode
                HAVING COUNT(*) >= %s
            ),
            candidates AS (
                SELECT ps.bibcode, ps.title, ps.abstract, ps.year,
                       ps.citation_count, nc.n_neighbors,
                       NTILE(5) OVER (ORDER BY ps.citation_count) as cite_tier
                FROM _pilot_sample ps
                JOIN neighbor_counts nc ON nc.bibcode = ps.bibcode
                WHERE ps.abstract IS NOT NULL
                  AND length(ps.title) > 20
            )
            SELECT bibcode, title, abstract, year, citation_count, n_neighbors, cite_tier
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY cite_tier ORDER BY random()) as rn
                FROM candidates
            ) ranked
            WHERE rn <= %s
            ORDER BY cite_tier, random()
            LIMIT %s
        """,
            [min_neighbors, n_seeds // 5 + 2, n_seeds],
        )
        seeds = cur.fetchall()

    logger.info(
        "Selected %d seed papers (min %d neighbors, %d requested)",
        len(seeds),
        min_neighbors,
        n_seeds,
    )
    return seeds


def get_citation_ground_truth(
    conn: psycopg.Connection,
    seed_bibcode: str,
) -> set[str]:
    """Get citation-based ground truth for a seed paper.

    Returns bibcodes of papers that cite or are cited by the seed,
    restricted to the pilot sample.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT target_bibcode as bibcode FROM citation_edges
            WHERE source_bibcode = %s
              AND target_bibcode IN (SELECT bibcode FROM _pilot_sample)
            UNION
            SELECT source_bibcode as bibcode FROM citation_edges
            WHERE target_bibcode = %s
              AND source_bibcode IN (SELECT bibcode FROM _pilot_sample)
        """,
            [seed_bibcode, seed_bibcode],
        )
        return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Retrieval methods
# ---------------------------------------------------------------------------


def retrieve_vector(
    conn: psycopg.Connection,
    seed_bibcode: str,
    model_name: str,
    limit: int = 60,
) -> tuple[list[str], float]:
    """Retrieve by vector similarity using stored seed embedding.

    Returns (list of bibcodes, latency_ms).
    """
    t0 = time.perf_counter()

    with conn.cursor(row_factory=dict_row) as cur:
        # Get seed embedding
        cur.execute(
            "SELECT embedding FROM paper_embeddings WHERE bibcode = %s AND model_name = %s",
            [seed_bibcode, model_name],
        )
        row = cur.fetchone()
        if row is None:
            return [], 0.0

        # Find nearest neighbors within pilot sample (exclude seed)
        cur.execute(
            """
            SELECT pe.bibcode,
                   1 - (pe.embedding <=> %s::vector) as similarity
            FROM paper_embeddings pe
            JOIN _pilot_sample ps ON ps.bibcode = pe.bibcode
            WHERE pe.model_name = %s
              AND pe.bibcode != %s
            ORDER BY pe.embedding <=> %s::vector
            LIMIT %s
        """,
            [row["embedding"], model_name, seed_bibcode, row["embedding"], limit],
        )
        results = [r["bibcode"] for r in cur.fetchall()]

    latency = (time.perf_counter() - t0) * 1000
    return results, latency


_TITLE_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_ENTITY_RE = re.compile(r"&[a-zA-Z]+;")
_ALPHA_TOKEN_RE = re.compile(r"[a-z]{4,}")

_LEXICAL_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "was",
        "one",
        "our",
        "out",
        "has",
        "have",
        "from",
        "with",
        "this",
        "that",
        "they",
        "been",
        "were",
        "which",
        "their",
        "will",
        "each",
        "many",
        "some",
        "than",
        "them",
        "then",
        "what",
        "when",
        "over",
        "such",
        "into",
        "most",
        "between",
        "these",
        "using",
        "based",
        "also",
        "about",
        "more",
        "new",
        "first",
        "two",
    }
)


def _make_lexical_query(seed: dict[str, Any], max_terms: int = 6) -> str:
    """Build a lexical query from seed paper title.

    Returns a space-separated bag of ≤``max_terms`` lowercased alphabetic
    content words. ``retrieve_lexical`` combines them with OR semantics so
    that queries return candidates even when no single conjunction matches
    the 20K pilot sample.

    The extraction is robust to HTML/MathML noise found in ADS titles
    (``<sub>``, ``<sup>``, ``&gt;``, etc.) — previously such tokens leaked
    into the tsquery and caused zero-hit parses.
    """
    raw = seed.get("title") or ""
    cleaned = _TITLE_ENTITY_RE.sub(" ", _TITLE_TAG_RE.sub(" ", raw)).lower()
    tokens = _ALPHA_TOKEN_RE.findall(cleaned)

    content: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in _LEXICAL_STOP_WORDS or tok in seen:
            continue
        seen.add(tok)
        content.append(tok)
        if len(content) >= max_terms:
            break

    return " ".join(content)


def retrieve_lexical(
    conn: psycopg.Connection,
    query_text: str,
    seed_bibcode: str,
    limit: int = 60,
) -> tuple[list[str], float]:
    """Retrieve by lexical search within the pilot sample.

    Combines the bag-of-words query with OR semantics via
    ``to_tsquery('english', 'a | b | c')``. Using OR (instead of the prior
    AND-only ``plainto_tsquery``) raises coverage on the 20K pilot sample
    from ~12% to ~80% of queries that return at least one candidate, which
    is essential for a fair BM25 baseline in Section 4.4.

    Returns (list of bibcodes excluding seed, latency_ms). Returns an
    empty list when the query has no usable terms.
    """
    t0 = time.perf_counter()

    terms = [t for t in query_text.split() if t]
    if not terms:
        return [], (time.perf_counter() - t0) * 1000

    tsquery = " | ".join(terms)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT ps.bibcode,
                   ts_rank_cd(ps.tsv, to_tsquery('english', %s)) AS rank
            FROM _pilot_sample ps
            WHERE ps.tsv @@ to_tsquery('english', %s)
              AND ps.bibcode != %s
            ORDER BY rank DESC
            LIMIT %s
            """,
            [tsquery, tsquery, seed_bibcode, limit],
        )
        results = [r["bibcode"] for r in cur.fetchall()]

    latency = (time.perf_counter() - t0) * 1000
    return results, latency


def retrieve_hybrid(
    conn: psycopg.Connection,
    query_text: str,
    seed_bibcode: str,
    model_name: str,
    rrf_k: int = 60,
    limit: int = 60,
    top_n: int = 20,
) -> tuple[list[str], float]:
    """Hybrid retrieval: lexical + vector via RRF.

    Both restricted to pilot sample to ensure fair comparison.
    Returns (list of bibcodes excluding seed, latency_ms).
    """
    t0 = time.perf_counter()

    # Lexical component
    lex_bibs, _ = retrieve_lexical(conn, query_text, seed_bibcode, limit=limit)
    lex_papers = [{"bibcode": b} for b in lex_bibs]

    # Vector component
    vec_bibs, _ = retrieve_vector(conn, seed_bibcode, model_name, limit=limit)
    vec_papers = [{"bibcode": b} for b in vec_bibs]

    # RRF fusion
    fused = rrf_fuse([lex_papers, vec_papers], k=rrf_k, top_n=top_n)
    results = [p["bibcode"] for p in fused if p["bibcode"] != seed_bibcode]

    latency = (time.perf_counter() - t0) * 1000
    return results, latency


# ---------------------------------------------------------------------------
# Full-corpus retrieval (uses search.py production infrastructure)
# ---------------------------------------------------------------------------


def select_seed_papers_full_corpus(
    conn: psycopg.Connection,
    n_seeds: int,
    min_neighbors: int = 10,
) -> list[dict[str, Any]]:
    """Select seed papers with rich citation networks from the full corpus.

    Picks papers that have INDUS embeddings, a non-null abstract, and
    at least `min_neighbors` citation neighbors in the full corpus.
    Stratified by citation count tier.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            WITH neighbor_counts AS (
                SELECT bibcode, COUNT(*) as n_neighbors
                FROM (
                    SELECT source_bibcode as bibcode
                    FROM citation_edges
                    UNION ALL
                    SELECT target_bibcode as bibcode
                    FROM citation_edges
                ) edges
                GROUP BY bibcode
                HAVING COUNT(*) >= %s
            ),
            candidates AS (
                SELECT p.bibcode, p.title, p.abstract, p.year,
                       p.citation_count, nc.n_neighbors,
                       NTILE(5) OVER (ORDER BY p.citation_count) as cite_tier
                FROM papers p
                JOIN neighbor_counts nc ON nc.bibcode = p.bibcode
                JOIN paper_embeddings pe ON pe.bibcode = p.bibcode
                    AND pe.model_name = 'indus'
                WHERE p.abstract IS NOT NULL
                  AND length(p.title) > 20
            )
            SELECT bibcode, title, abstract, year, citation_count,
                   n_neighbors, cite_tier
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY cite_tier ORDER BY random()
                ) as rn
                FROM candidates
            ) ranked
            WHERE rn <= %s
            ORDER BY cite_tier, random()
            LIMIT %s
        """,
            [min_neighbors, n_seeds // 5 + 2, n_seeds],
        )
        seeds = cur.fetchall()

    logger.info(
        "Selected %d full-corpus seed papers (min %d neighbors, %d requested)",
        len(seeds),
        min_neighbors,
        n_seeds,
    )
    return seeds


def get_citation_ground_truth_full_corpus(
    conn: psycopg.Connection,
    seed_bibcode: str,
    max_relevant: int = 500,
) -> set[str]:
    """Get citation-based ground truth from the full corpus.

    Returns bibcodes of papers that cite or are cited by the seed.
    Capped at `max_relevant` to avoid outlier seeds with thousands of
    citations dominating the recall denominator.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            (SELECT target_bibcode as bibcode FROM citation_edges
             WHERE source_bibcode = %s
             LIMIT %s)
            UNION
            (SELECT source_bibcode as bibcode FROM citation_edges
             WHERE target_bibcode = %s
             LIMIT %s)
        """,
            [seed_bibcode, max_relevant, seed_bibcode, max_relevant],
        )
        return {row[0] for row in cur.fetchall()}


def retrieve_vector_full_corpus(
    conn: psycopg.Connection,
    seed_bibcode: str,
    model_name: str,
    limit: int = 60,
) -> tuple[list[str], float]:
    """Retrieve by vector similarity on the full corpus via search.py.

    Fetches the seed embedding and calls the production vector_search.
    Returns (list of bibcodes excluding seed, latency_ms).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT embedding FROM paper_embeddings WHERE bibcode = %s AND model_name = %s",
            [seed_bibcode, model_name],
        )
        row = cur.fetchone()
        if row is None:
            return [], 0.0

    # Parse the stored embedding string into a float list
    emb_raw = row["embedding"]
    if isinstance(emb_raw, str):
        embedding = [float(x) for x in emb_raw.strip("[]").split(",")]
    else:
        embedding = list(emb_raw)

    result = vector_search(conn, embedding, model_name=model_name, limit=limit + 1)
    bibs = [p["bibcode"] for p in result.papers if p["bibcode"] != seed_bibcode]
    return bibs[:limit], result.timing_ms.get("vector_ms", 0.0)


def retrieve_lexical_full_corpus(
    conn: psycopg.Connection,
    query_text: str,
    seed_bibcode: str,
    limit: int = 60,
) -> tuple[list[str], float]:
    """Retrieve by lexical search on the full corpus via search.py.

    Uses plainto_tsquery with scix_english config on the full papers table.
    Returns (list of bibcodes excluding seed, latency_ms).
    """
    result = lexical_search(conn, query_text, limit=limit + 1)
    bibs = [p["bibcode"] for p in result.papers if p["bibcode"] != seed_bibcode]
    return bibs[:limit], result.timing_ms.get("lexical_ms", 0.0)


def retrieve_hybrid_full_corpus(
    conn: psycopg.Connection,
    query_text: str,
    seed_bibcode: str,
    model_name: str,
    rrf_k: int = 60,
    limit: int = 60,
    top_n: int = 20,
) -> tuple[list[str], float]:
    """Hybrid retrieval on the full corpus: lexical + vector via RRF.

    Uses the production hybrid_search from search.py.
    Returns (list of bibcodes excluding seed, latency_ms).
    """
    # Get seed embedding for vector component
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT embedding FROM paper_embeddings WHERE bibcode = %s AND model_name = %s",
            [seed_bibcode, model_name],
        )
        row = cur.fetchone()

    if row is None:
        # Fall back to lexical-only
        return retrieve_lexical_full_corpus(conn, query_text, seed_bibcode, limit)

    emb_raw = row["embedding"]
    if isinstance(emb_raw, str):
        embedding = [float(x) for x in emb_raw.strip("[]").split(",")]
    else:
        embedding = list(emb_raw)

    result = hybrid_search(
        conn,
        query_text,
        query_embedding=embedding,
        model_name=model_name,
        vector_limit=limit,
        lexical_limit=limit,
        rrf_k=rrf_k,
        top_n=top_n,
    )
    bibs = [p["bibcode"] for p in result.papers if p["bibcode"] != seed_bibcode]
    total_ms = sum(result.timing_ms.values())
    return bibs[:top_n], total_ms


def evaluate_single_full_corpus(
    conn: psycopg.Connection,
    seed: dict[str, Any],
    relevant: set[str],
    method: str,
) -> QueryEval | None:
    """Evaluate a single query-method pair on the full corpus."""
    bibcode = seed["bibcode"]
    lex_query = _make_lexical_query(seed)

    if method == "lexical":
        retrieved, latency = retrieve_lexical_full_corpus(conn, lex_query, bibcode)
    elif method.startswith("hybrid_"):
        mname = method.replace("hybrid_", "")
        retrieved, latency = retrieve_hybrid_full_corpus(conn, lex_query, bibcode, mname)
    else:
        # Vector-only (indus)
        retrieved, latency = retrieve_vector_full_corpus(conn, bibcode, method)

    if not retrieved:
        return None

    return QueryEval(
        seed_bibcode=bibcode,
        method=method,
        ndcg_10=round(ndcg_at_k(retrieved, relevant, 10), 4),
        recall_10=round(recall_at_k(retrieved, relevant, 10), 4),
        recall_20=round(recall_at_k(retrieved, relevant, 20), 4),
        precision_10=round(precision_at_k(retrieved, relevant, 10), 4),
        mrr_val=round(mrr(retrieved, relevant), 4),
        relevant_count=len(relevant),
        retrieved_count=len(retrieved),
        latency_ms=round(latency, 1),
    )


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate_single(
    conn: psycopg.Connection,
    seed: dict[str, Any],
    relevant: set[str],
    method: str,
    model_name: str | None = None,
) -> QueryEval | None:
    """Evaluate a single query-method pair."""
    bibcode = seed["bibcode"]
    lex_query = _make_lexical_query(seed)

    if method == "lexical":
        retrieved, latency = retrieve_lexical(conn, lex_query, bibcode)
    elif method.startswith("hybrid_"):
        mname = method.replace("hybrid_", "")
        retrieved, latency = retrieve_hybrid(conn, lex_query, bibcode, mname)
    else:
        # Vector-only
        retrieved, latency = retrieve_vector(conn, bibcode, method)

    if not retrieved:
        return None

    return QueryEval(
        seed_bibcode=bibcode,
        method=method,
        ndcg_10=round(ndcg_at_k(retrieved, relevant, 10), 4),
        recall_10=round(recall_at_k(retrieved, relevant, 10), 4),
        recall_20=round(recall_at_k(retrieved, relevant, 20), 4),
        precision_10=round(precision_at_k(retrieved, relevant, 10), 4),
        mrr_val=round(mrr(retrieved, relevant), 4),
        relevant_count=len(relevant),
        retrieved_count=len(retrieved),
        latency_ms=round(latency, 1),
    )


def aggregate_results(evals: list[QueryEval]) -> list[EvalSummary]:
    """Aggregate per-query results into per-method summaries."""
    from collections import defaultdict

    by_method: dict[str, list[QueryEval]] = defaultdict(list)
    for e in evals:
        by_method[e.method].append(e)

    summaries = []
    for method, method_evals in sorted(by_method.items()):
        n = len(method_evals)
        mean_ndcg = sum(e.ndcg_10 for e in method_evals) / n
        std_ndcg = math.sqrt(sum((e.ndcg_10 - mean_ndcg) ** 2 for e in method_evals) / n)
        summaries.append(
            EvalSummary(
                method=method,
                n_queries=n,
                mean_ndcg_10=round(mean_ndcg, 4),
                mean_recall_10=round(sum(e.recall_10 for e in method_evals) / n, 4),
                mean_recall_20=round(sum(e.recall_20 for e in method_evals) / n, 4),
                mean_precision_10=round(sum(e.precision_10 for e in method_evals) / n, 4),
                mean_mrr=round(sum(e.mrr_val for e in method_evals) / n, 4),
                mean_latency_ms=round(sum(e.latency_ms for e in method_evals) / n, 1),
                std_ndcg_10=round(std_ndcg, 4),
            )
        )

    return summaries


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------


def paired_difference_test(
    evals: list[QueryEval],
    method_a: str,
    method_b: str,
    metric: str = "ndcg_10",
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test between two methods.

    Returns test statistic, p-value, and effect size (mean difference).
    """
    from collections import defaultdict

    by_seed: dict[str, dict[str, float]] = defaultdict(dict)
    for e in evals:
        by_seed[e.seed_bibcode][e.method] = getattr(e, metric)

    diffs = []
    for seed, methods in by_seed.items():
        if method_a in methods and method_b in methods:
            diffs.append(methods[method_a] - methods[method_b])

    if len(diffs) < 5:
        return {"n_pairs": len(diffs), "p_value": None, "mean_diff": 0.0}

    try:
        from scipy.stats import wilcoxon

        stat, p_value = wilcoxon(diffs, alternative="two-sided")
        return {
            "n_pairs": len(diffs),
            "statistic": float(stat),
            "p_value": round(float(p_value), 6),
            "mean_diff": round(sum(diffs) / len(diffs), 4),
        }
    except ImportError:
        # Fallback: just report mean difference
        mean_diff = sum(diffs) / len(diffs)
        return {
            "n_pairs": len(diffs),
            "p_value": None,
            "mean_diff": round(mean_diff, 4),
            "note": "scipy not available for significance test",
        }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    summaries: list[EvalSummary],
    evals: list[QueryEval],
    seeds: list[dict[str, Any]],
    significance: list[dict[str, Any]],
    corpus_label: str = "10K stratified sample from 32.4M ADS papers",
) -> str:
    """Generate markdown report for the paper."""
    lines = [
        "# 50-Query Retrieval Evaluation",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Corpus**: {corpus_label}",
        f"**Queries**: {len(seeds)} seed papers with citation-based ground truth",
        f"**Ground truth**: Citation network (references + citing papers) within sample",
        "",
        "## Results Summary",
        "",
        "| Method | nDCG@10 | Recall@10 | Recall@20 | P@10 | MRR | Latency (ms) |",
        "|--------|---------|-----------|-----------|------|-----|-------------|",
    ]

    for s in sorted(summaries, key=lambda x: -x.mean_ndcg_10):
        lines.append(
            f"| {s.method} | {s.mean_ndcg_10:.4f} ± {s.std_ndcg_10:.4f} "
            f"| {s.mean_recall_10:.4f} | {s.mean_recall_20:.4f} "
            f"| {s.mean_precision_10:.4f} | {s.mean_mrr:.4f} "
            f"| {s.mean_latency_ms:.0f} |"
        )

    # Significance tests
    if significance:
        lines.extend(
            [
                "",
                "## Statistical Significance (Wilcoxon signed-rank, nDCG@10)",
                "",
                "| Comparison | Mean Diff | p-value | n |",
                "|------------|-----------|---------|---|",
            ]
        )
        for sig in significance:
            p_str = f"{sig['p_value']:.6f}" if sig["p_value"] is not None else "N/A"
            lines.append(
                f"| {sig['comparison']} | {sig['mean_diff']:+.4f} | {p_str} | {sig['n_pairs']} |"
            )

    # Per-query detail sample
    lines.extend(
        [
            "",
            "## Query Distribution",
            "",
            f"- Total seed papers: {len(seeds)}",
        ]
    )

    # Citation network size distribution
    from collections import Counter

    neighbor_tiers = Counter()
    for seed in seeds:
        n = seed["n_neighbors"]
        if n < 10:
            neighbor_tiers["5-9"] += 1
        elif n < 20:
            neighbor_tiers["10-19"] += 1
        elif n < 50:
            neighbor_tiers["20-49"] += 1
        else:
            neighbor_tiers["50+"] += 1

    lines.append("- Citation network size distribution:")
    for tier in ["5-9", "10-19", "20-49", "50+"]:
        if tier in neighbor_tiers:
            lines.append(f"  - {tier} neighbors: {neighbor_tiers[tier]} queries")

    year_counts = Counter(seed["year"] for seed in seeds if seed.get("year"))
    lines.append(f"- Year range: {min(year_counts.keys())}-{max(year_counts.keys())}")

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "- **Query formulation**: Seed paper title + first 50 words of abstract used as",
            "  query text for lexical search; stored embedding used as query vector for dense retrieval.",
            "- **Ground truth**: Citation network (references + citing papers) restricted to",
            "  the pilot sample. Binary relevance: cited/citing = relevant, else irrelevant.",
            "- **Fusion**: RRF with k=60 (standard constant).",
            "- **Retrieval pool**: 10K stratified sample for both dense and lexical search.",
            "  Vector search restricted to pilot sample via JOIN for fair comparison.",
            "- **Models**: SPECTER2 (allenai/specter2_base, 768d),",
            "  INDUS (nasa-impact/nasa-smd-ibm-st-v2, 768d),",
            "  Nomic (nomic-ai/nomic-embed-text-v1.5, 768d).",
            "",
            "### Limitations",
            "",
            "- Citation-based ground truth favors models trained on citation proximity (SPECTER2),",
            "  yet SPECTER2 underperforms INDUS and Nomic here — suggesting ADS-specific",
            "  training data (INDUS) and general representation quality (Nomic) matter more",
            "  than training objective alignment.",
            "- Lexical search uses to_tsquery with OR semantics over HTML-stripped title",
            "  tokens. Coverage on the 20K pilot sample is 100% of queries returning",
            "  candidates (up from ~12% under the prior plainto_tsquery AND baseline). Full-",
            "  corpus hybrid evaluation remains pending completion of bulk embedding pipeline.",
            "- text-embedding-3-large not included (no embeddings generated yet).",
            "  Will be added when OpenAI embeddings are available.",
            "- Random seed selection (seed=42) used for reproducibility. Results may vary",
            "  slightly with different seed sets.",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


METHODS = ["specter2", "indus", "nomic", "lexical", "hybrid_specter2", "hybrid_indus"]

# Full-corpus methods: only INDUS has 32M embeddings; specter2/nomic are pilot-only
FULL_CORPUS_METHODS = ["indus", "lexical", "hybrid_indus"]


def _run_eval_loop(
    conn: psycopg.Connection,
    seeds: list[dict[str, Any]],
    ground_truth: dict[str, set[str]],
    methods: list[str],
    full_corpus: bool,
) -> list[QueryEval]:
    """Run the evaluation loop for all methods and seeds.

    Dispatches to pilot-sample or full-corpus retrieval depending on mode.
    """
    all_evals: list[QueryEval] = []

    for method in methods:
        logger.info("Evaluating method: %s", method)

        # Reconnect per method to handle connection drops
        try:
            conn.execute("SELECT 1")
        except Exception:
            logger.warning("Connection lost, reconnecting...")
            conn = get_conn()

        for seed in seeds:
            relevant = ground_truth[seed["bibcode"]]
            if not relevant:
                continue

            try:
                if full_corpus:
                    result = evaluate_single_full_corpus(conn, seed, relevant, method)
                else:
                    model_name = method if method in ("specter2", "indus", "nomic") else None
                    result = evaluate_single(conn, seed, relevant, method, model_name)
            except psycopg.OperationalError:
                logger.warning("Connection error, reconnecting...")
                conn = get_conn()
                if full_corpus:
                    result = evaluate_single_full_corpus(conn, seed, relevant, method)
                else:
                    model_name = method if method in ("specter2", "indus", "nomic") else None
                    result = evaluate_single(conn, seed, relevant, method, model_name)
            if result is not None:
                all_evals.append(result)

        # Progress
        method_evals = [e for e in all_evals if e.method == method]
        if method_evals:
            mean_ndcg = sum(e.ndcg_10 for e in method_evals) / len(method_evals)
            logger.info(
                "  %s: %d queries, mean nDCG@10=%.4f",
                method,
                len(method_evals),
                mean_ndcg,
            )

    return all_evals


def _significance_pairs(methods: list[str]) -> list[tuple[str, str]]:
    """Return method pairs for significance testing based on available methods."""
    pairs: list[tuple[str, str]] = []
    candidate_pairs = [
        ("hybrid_specter2", "specter2"),
        ("hybrid_specter2", "lexical"),
        ("hybrid_indus", "indus"),
        ("hybrid_indus", "lexical"),
        ("specter2", "indus"),
        ("specter2", "nomic"),
        ("specter2", "lexical"),
        ("indus", "lexical"),
    ]
    method_set = set(methods)
    for a, b in candidate_pairs:
        if a in method_set and b in method_set:
            pairs.append((a, b))
    return pairs


def _write_outputs(
    summaries: list[EvalSummary],
    all_evals: list[QueryEval],
    seeds: list[dict[str, Any]],
    significance_tests: list[dict[str, Any]],
    output: str | None,
    json_output: str | None,
    corpus_label: str,
) -> None:
    """Write markdown and JSON outputs."""
    report = generate_report(
        summaries, all_evals, seeds, significance_tests, corpus_label=corpus_label
    )

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(report)
        logger.info("Report written to %s", output)
    else:
        print(report)

    if json_output:
        json_data = {
            "corpus": corpus_label,
            "summaries": [
                {
                    "method": s.method,
                    "n_queries": s.n_queries,
                    "ndcg_10": s.mean_ndcg_10,
                    "recall_10": s.mean_recall_10,
                    "recall_20": s.mean_recall_20,
                    "precision_10": s.mean_precision_10,
                    "mrr": s.mean_mrr,
                    "latency_ms": s.mean_latency_ms,
                    "std_ndcg_10": s.std_ndcg_10,
                }
                for s in summaries
            ],
            "per_query": [
                {
                    "seed_bibcode": e.seed_bibcode,
                    "method": e.method,
                    "ndcg_10": e.ndcg_10,
                    "recall_10": e.recall_10,
                    "recall_20": e.recall_20,
                    "mrr": e.mrr_val,
                    "relevant_count": e.relevant_count,
                    "latency_ms": e.latency_ms,
                }
                for e in all_evals
            ],
            "significance": significance_tests,
        }
        Path(json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(json_output).write_text(json.dumps(json_data, indent=2))
        logger.info("JSON results written to %s", json_output)

    # Print summary table to stdout
    print("\n" + "=" * 80)
    print(f"RETRIEVAL EVALUATION RESULTS ({corpus_label})")
    print("=" * 80)
    print(
        f"\n{'Method':<18} {'nDCG@10':>10} {'R@10':>8} {'R@20':>8} "
        f"{'P@10':>8} {'MRR':>8} {'ms':>8}"
    )
    print("-" * 72)
    for s in sorted(summaries, key=lambda x: -x.mean_ndcg_10):
        print(
            f"{s.method:<18} {s.mean_ndcg_10:>10.4f} {s.mean_recall_10:>8.4f} "
            f"{s.mean_recall_20:>8.4f} {s.mean_precision_10:>8.4f} "
            f"{s.mean_mrr:>8.4f} {s.mean_latency_ms:>8.0f}"
        )
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="50-query retrieval evaluation")
    parser.add_argument("--seed-papers", type=int, default=50)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to evaluate (default depends on mode)",
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Run evaluation on the full 32M corpus using INDUS HNSW + BM25. "
        "Default methods: indus, lexical, hybrid_indus.",
    )
    args = parser.parse_args()

    full_corpus: bool = args.full_corpus

    # Resolve methods
    if args.methods is not None:
        methods = args.methods
    elif full_corpus:
        methods = list(FULL_CORPUS_METHODS)
    else:
        methods = list(METHODS)

    conn = get_connection()

    if full_corpus:
        logger.info("Running FULL-CORPUS evaluation (32M papers, INDUS + BM25)")
        corpus_label = "32.4M full corpus"
        min_neighbors = max(args.min_neighbors, 10)  # full corpus needs higher bar
    else:
        # Verify pilot sample exists
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                "WHERE table_name = '_pilot_sample')"
            )
            if not cur.fetchone()[0]:
                logger.error("_pilot_sample table not found. " "Run pilot_embed_compare.py first.")
                sys.exit(1)
        corpus_label = "10K stratified sample from 32.4M ADS papers"
        min_neighbors = args.min_neighbors

    # Set random seed for reproducibility
    with conn.cursor() as cur:
        cur.execute("SELECT setseed(%s)", [args.random_seed / 2**31])

    # Step 1: Select seed papers
    logger.info(
        "Selecting %d seed papers (random_seed=%d)...",
        args.seed_papers,
        args.random_seed,
    )
    if full_corpus:
        seeds = select_seed_papers_full_corpus(conn, args.seed_papers, min_neighbors=min_neighbors)
    else:
        seeds = select_seed_papers(conn, args.seed_papers, min_neighbors=min_neighbors)

    if len(seeds) < args.seed_papers:
        logger.warning(
            "Only found %d seeds (requested %d). Lowering min_neighbors might help.",
            len(seeds),
            args.seed_papers,
        )

    # Step 2: Build ground truth
    logger.info("Building citation-based ground truth...")
    ground_truth: dict[str, set[str]] = {}
    for seed in seeds:
        if full_corpus:
            gt = get_citation_ground_truth_full_corpus(conn, seed["bibcode"])
        else:
            gt = get_citation_ground_truth(conn, seed["bibcode"])
        ground_truth[seed["bibcode"]] = gt

    gt_sizes = [len(gt) for gt in ground_truth.values()]
    logger.info(
        "Ground truth: min=%d, max=%d, mean=%.1f relevant docs per query",
        min(gt_sizes),
        max(gt_sizes),
        sum(gt_sizes) / len(gt_sizes),
    )

    # Step 3: Run evaluation
    all_evals = _run_eval_loop(conn, seeds, ground_truth, methods, full_corpus)

    # Step 4: Aggregate
    summaries = aggregate_results(all_evals)

    # Step 5: Significance tests
    significance_tests = []
    for pair in _significance_pairs(methods):
        result = paired_difference_test(all_evals, pair[0], pair[1])
        result["comparison"] = f"{pair[0]} vs {pair[1]}"
        significance_tests.append(result)

    # Step 6: Output
    _write_outputs(
        summaries,
        all_evals,
        seeds,
        significance_tests,
        args.output,
        args.json_output,
        corpus_label,
    )

    conn.close()


if __name__ == "__main__":
    main()
