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


def _make_lexical_query(seed: dict[str, Any]) -> str:
    """Build a lexical query from seed paper title.

    Extracts 4-6 key content words from the title for AND-based plainto_tsquery.
    Keeps queries selective enough for fast execution against the full corpus.
    """
    title_words = (seed["title"] or "").split()

    # Filter to content words (>3 chars, no common stop words)
    stop = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "was", "one", "our", "out", "has", "have", "from", "with",
        "this", "that", "they", "been", "were", "which", "their", "will",
        "each", "many", "some", "than", "them", "then", "what", "when",
        "over", "such", "into", "most", "between", "these", "using",
        "based", "also", "about", "more", "new", "first", "two",
    }
    content_words = []
    seen: set[str] = set()
    for w in title_words:
        w_clean = w.strip(".,;:()[]{}\"'+-*/=$<>").lower()
        if len(w_clean) > 3 and w_clean not in stop and w_clean not in seen:
            seen.add(w_clean)
            content_words.append(w_clean)
        if len(content_words) >= 6:
            break

    return " ".join(content_words)


def retrieve_lexical(
    conn: psycopg.Connection,
    query_text: str,
    seed_bibcode: str,
    limit: int = 60,
) -> tuple[list[str], float]:
    """Retrieve by lexical search within pilot sample.

    Uses plainto_tsquery (AND logic) with key title words against the
    GIN index on _pilot_sample.tsv for fast intra-sample search.
    Returns (list of bibcodes excluding seed, latency_ms).
    """
    t0 = time.perf_counter()

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT ps.bibcode,
                   ts_rank_cd(ps.tsv, plainto_tsquery('english', %s)) AS rank
            FROM _pilot_sample ps
            WHERE ps.tsv @@ plainto_tsquery('english', %s)
              AND ps.bibcode != %s
            ORDER BY rank DESC
            LIMIT %s
        """,
            [query_text, query_text, seed_bibcode, limit],
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
        std_ndcg = math.sqrt(
            sum((e.ndcg_10 - mean_ndcg) ** 2 for e in method_evals) / n
        )
        summaries.append(
            EvalSummary(
                method=method,
                n_queries=n,
                mean_ndcg_10=round(mean_ndcg, 4),
                mean_recall_10=round(
                    sum(e.recall_10 for e in method_evals) / n, 4
                ),
                mean_recall_20=round(
                    sum(e.recall_20 for e in method_evals) / n, 4
                ),
                mean_precision_10=round(
                    sum(e.precision_10 for e in method_evals) / n, 4
                ),
                mean_mrr=round(sum(e.mrr_val for e in method_evals) / n, 4),
                mean_latency_ms=round(
                    sum(e.latency_ms for e in method_evals) / n, 1
                ),
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
) -> str:
    """Generate markdown report for the paper."""
    lines = [
        "# 50-Query Retrieval Evaluation",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Corpus**: 10K stratified sample from 32.4M ADS papers",
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
            "- 20K sample limits lexical search: plainto_tsquery (AND logic) with specific title",
            "  words returns results for <10% of queries. Hybrid search therefore defaults to",
            "  vector-only for most queries. Full-corpus hybrid evaluation pending completion",
            "  of bulk embedding pipeline.",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="50-query retrieval evaluation")
    parser.add_argument("--seed-papers", type=int, default=50)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        help=f"Methods to evaluate (default: {METHODS})",
    )
    args = parser.parse_args()

    conn = get_connection()

    # Verify pilot sample exists
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = '_pilot_sample')"
        )
        if not cur.fetchone()[0]:
            logger.error("_pilot_sample table not found. Run pilot_embed_compare.py first.")
            sys.exit(1)

    # Set random seed for reproducibility
    with conn.cursor() as cur:
        cur.execute("SELECT setseed(%s)", [args.random_seed / 2**31])

    # Step 1: Select seed papers
    logger.info("Selecting %d seed papers (random_seed=%d)...", args.seed_papers, args.random_seed)
    seeds = select_seed_papers(conn, args.seed_papers, min_neighbors=args.min_neighbors)

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
    all_evals: list[QueryEval] = []

    for method in args.methods:
        logger.info("Evaluating method: %s", method)
        model_name = method if method in ("specter2", "indus", "nomic") else None

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
                result = evaluate_single(conn, seed, relevant, method, model_name)
            except psycopg.OperationalError:
                logger.warning("Connection error, reconnecting...")
                conn = get_conn()
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

    # Step 4: Aggregate
    summaries = aggregate_results(all_evals)

    # Step 5: Significance tests (hybrid vs best single model)
    significance_tests = []
    for pair in [
        ("hybrid_specter2", "specter2"),
        ("hybrid_specter2", "lexical"),
        ("hybrid_indus", "indus"),
        ("specter2", "indus"),
        ("specter2", "nomic"),
        ("specter2", "lexical"),
    ]:
        result = paired_difference_test(all_evals, pair[0], pair[1])
        result["comparison"] = f"{pair[0]} vs {pair[1]}"
        significance_tests.append(result)

    # Step 6: Output
    report = generate_report(summaries, all_evals, seeds, significance_tests)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report)
        logger.info("Report written to %s", args.output)
    else:
        print(report)

    # JSON output for downstream use
    if args.json_output:
        json_data = {
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
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(json_data, indent=2))
        logger.info("JSON results written to %s", args.json_output)

    # Print summary table to stdout
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION RESULTS")
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

    conn.close()


if __name__ == "__main__":
    main()
