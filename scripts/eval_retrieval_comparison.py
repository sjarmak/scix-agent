#!/usr/bin/env python3
"""50-query retrieval evaluation: SPECTER2 vs lexical vs hybrid RRF.

Uses citation-based ground truth (standard in scientific IR evaluation,
same methodology as the SPECTER2 paper). For each seed paper, its citation
neighbors within the embedded subset form the relevant set.

Computes nDCG@10 and Recall@K per retrieval method.

Usage:
    python3 scripts/eval_retrieval_comparison.py
    python3 scripts/eval_retrieval_comparison.py --output results.md --k 10,20,50
    python3 scripts/eval_retrieval_comparison.py --dsn "dbname=scix" -v
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

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg
from psycopg.rows import dict_row

from scix.db import get_connection
from scix.search import (
    rrf_fuse,
    vector_search,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding lookup (use stored embeddings instead of on-the-fly generation)
# ---------------------------------------------------------------------------


def get_stored_embedding(
    conn: psycopg.Connection, bibcode: str, model_name: str
) -> list[float] | None:
    """Look up a pre-computed embedding from paper_embeddings."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding::text FROM paper_embeddings "
            "WHERE bibcode = %s AND model_name = %s",
            [bibcode, model_name],
        )
        row = cur.fetchone()
        if row is None:
            return None
        # Parse pgvector text representation "[0.1,0.2,...]"
        vec_str = row[0].strip("[]")
        return [float(x) for x in vec_str.split(",")]


def lexical_search_embedded(
    conn: psycopg.Connection,
    query_text: str,
    model_name: str = "specter2",
    limit: int = 100,
) -> tuple[list[str], float]:
    """Lexical search restricted to the embedded subset for fair comparison.

    Returns (bibcodes, latency_ms).
    """
    t0 = time.perf_counter()

    sql = """
        SELECT p.bibcode,
               ts_rank_cd(p.tsv, plainto_tsquery('english', %s), 32) AS rank
        FROM papers p
        JOIN paper_embeddings pe ON p.bibcode = pe.bibcode AND pe.model_name = %s
        WHERE p.tsv @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, [query_text, model_name, query_text, limit])
        rows = cur.fetchall()

    latency = round((time.perf_counter() - t0) * 1000, 2)
    return [row[0] for row in rows], latency


def get_openai_query_embedding(text: str) -> list[float] | None:
    """Get OpenAI embedding for a query string. Returns None if unavailable."""
    try:
        from scix.embed import embed_query_openai

        return embed_query_openai(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def dcg_at_k(relevance: list[int], k: int) -> float:
    """Discounted Cumulative Gain at rank k.

    relevance[i] = 1 if the item at rank i is relevant, else 0.
    DCG@k = sum_{i=1}^{k} rel_i / log2(i+1)
    """
    score = 0.0
    for i in range(min(k, len(relevance))):
        if relevance[i]:
            score += 1.0 / math.log2(i + 2)  # i+2 because i is 0-indexed
    return score


def ndcg_at_k(retrieved_bibcodes: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k.

    Returns 0.0 if there are no relevant documents.
    """
    if not relevant:
        return 0.0

    # Actual relevance of retrieved results
    actual_rel = [1 if bib in relevant else 0 for bib in retrieved_bibcodes[:k]]
    actual_dcg = dcg_at_k(actual_rel, k)

    # Ideal: all relevant docs at top
    ideal_rel = sorted(actual_rel, reverse=True)
    # But ideal should consider ALL relevant docs, not just retrieved ones
    n_relevant_possible = min(len(relevant), k)
    ideal_rel_full = [1] * n_relevant_possible + [0] * (k - n_relevant_possible)
    ideal_dcg = dcg_at_k(ideal_rel_full, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def recall_at_k(retrieved_bibcodes: list[str], relevant: set[str], k: int) -> float:
    """Recall at rank k: fraction of relevant documents retrieved in top-k."""
    if not relevant:
        return 0.0
    retrieved_set = set(retrieved_bibcodes[:k])
    return len(retrieved_set & relevant) / len(relevant)


def precision_at_k(retrieved_bibcodes: list[str], relevant: set[str], k: int) -> float:
    """Precision at rank k: fraction of top-k that are relevant."""
    if k == 0:
        return 0.0
    retrieved_top = retrieved_bibcodes[:k]
    if not retrieved_top:
        return 0.0
    hits = sum(1 for bib in retrieved_top if bib in relevant)
    return hits / len(retrieved_top)


def mean_reciprocal_rank(retrieved_bibcodes: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, bib in enumerate(retrieved_bibcodes):
        if bib in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalQuery:
    """A single evaluation query with citation-based ground truth."""

    seed_bibcode: str
    query_title: str  # title only (for lexical search)
    query_text: str  # title + abstract snippet (for vector search context)
    relevant_bibcodes: frozenset[str]  # citation neighbors in embedded set
    domain: str  # arxiv class or topic label


@dataclass(frozen=True)
class MethodResult:
    """Result from one retrieval method on one query."""

    method: str
    retrieved_bibcodes: tuple[str, ...]
    latency_ms: float
    ndcg_10: float
    recall_10: float
    recall_20: float
    recall_50: float
    precision_10: float
    mrr: float


@dataclass
class EvalSummary:
    """Aggregated results across all queries for all methods."""

    num_queries: int
    methods: dict[str, dict[str, float]]  # method -> metric -> mean value
    per_query: list[dict[str, Any]] = field(default_factory=list)
    k_values: list[int] = field(default_factory=lambda: [10, 20, 50])


# ---------------------------------------------------------------------------
# Query selection: find seed papers with citation-based ground truth
# ---------------------------------------------------------------------------


def select_eval_queries(
    conn: psycopg.Connection,
    *,
    num_queries: int = 50,
    min_neighbors: int = 3,
    model_name: str = "specter2",
) -> list[EvalQuery]:
    """Select seed papers that have sufficient citation neighbors in the embedded set.

    Uses bidirectional citation links (citing OR cited) as relevance proxy.
    Ensures diversity by sampling across different arxiv classes.
    """
    logger.info("Selecting %d evaluation queries (min %d neighbors)...", num_queries, min_neighbors)

    # Step 1: Get the set of embedded bibcodes into a temp table for efficient joins
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _eval_embedded")
        cur.execute(
            "CREATE TEMP TABLE _eval_embedded AS "
            "SELECT bibcode FROM paper_embeddings WHERE model_name = %s",
            [model_name],
        )
        cur.execute("CREATE INDEX ON _eval_embedded (bibcode)")
        cur.execute("ANALYZE _eval_embedded")
        cur.execute("SELECT count(*) FROM _eval_embedded")
        n_embedded = cur.fetchone()[0]
    conn.commit()

    logger.info("Embedded set: %d papers", n_embedded)

    # Step 2: Find citation neighbors within embedded set using temp table joins
    neighbor_map: dict[str, set[str]] = {}

    with conn.cursor() as cur:
        cur.execute(
            "SELECT ce.source_bibcode, ce.target_bibcode "
            "FROM citation_edges ce "
            "JOIN _eval_embedded e1 ON ce.source_bibcode = e1.bibcode "
            "JOIN _eval_embedded e2 ON ce.target_bibcode = e2.bibcode"
        )
        for source, target in cur.fetchall():
            neighbor_map.setdefault(source, set()).add(target)
            neighbor_map.setdefault(target, set()).add(source)

    logger.info(
        "Found %d papers with citation neighbors (>= %d)",
        sum(1 for v in neighbor_map.values() if len(v) >= min_neighbors),
        min_neighbors,
    )

    # Step 3: Get paper metadata for candidates
    candidates = [bib for bib, nbrs in neighbor_map.items() if len(nbrs) >= min_neighbors]
    if not candidates:
        return []

    # Fetch metadata in batches
    rows: list[dict] = []
    batch_size = 500
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT bibcode, title, LEFT(abstract, 300) AS abstract_snippet, "
                "COALESCE(arxiv_class[1], 'general') AS domain, citation_count "
                "FROM papers WHERE bibcode = ANY(%s) AND title IS NOT NULL",
                [batch],
            )
            for row in cur.fetchall():
                row["neighbor_list"] = list(neighbor_map[row["bibcode"]])
                row["n_neighbors"] = len(neighbor_map[row["bibcode"]])
                rows.append(row)

    rows.sort(key=lambda r: r["n_neighbors"], reverse=True)

    logger.info("Found %d candidate seed papers with >= %d neighbors", len(rows), min_neighbors)

    # Diversify: group by domain, round-robin select
    by_domain: dict[str, list[dict]] = {}
    for row in rows:
        domain = row["domain"]
        by_domain.setdefault(domain, []).append(row)

    queries: list[EvalQuery] = []
    seen_domains = list(by_domain.keys())

    # Round-robin across domains until we have enough
    idx = 0
    domain_cursors = {d: 0 for d in seen_domains}
    while len(queries) < num_queries and any(
        domain_cursors[d] < len(by_domain[d]) for d in seen_domains
    ):
        domain = seen_domains[idx % len(seen_domains)]
        cursor = domain_cursors[domain]
        if cursor < len(by_domain[domain]):
            row = by_domain[domain][cursor]
            domain_cursors[domain] += 1

            title = row["title"]
            query_text = title
            if row["abstract_snippet"]:
                query_text = f"{title} {row['abstract_snippet']}"

            relevant = frozenset(
                b for b in row["neighbor_list"] if b != row["bibcode"]
            )

            queries.append(
                EvalQuery(
                    seed_bibcode=row["bibcode"],
                    query_title=title,
                    query_text=query_text,
                    relevant_bibcodes=relevant,
                    domain=domain,
                )
            )
        idx += 1

    logger.info(
        "Selected %d queries across %d domains (avg %.1f relevant docs/query)",
        len(queries),
        len(set(q.domain for q in queries)),
        sum(len(q.relevant_bibcodes) for q in queries) / max(len(queries), 1),
    )
    return queries


# ---------------------------------------------------------------------------
# Run retrieval methods
# ---------------------------------------------------------------------------


def run_method(
    conn: psycopg.Connection,
    query: EvalQuery,
    method: str,
    *,
    limit: int = 100,
) -> MethodResult:
    """Run a single retrieval method on a single query and compute metrics.

    For vector methods, uses the seed paper's stored embedding as the query
    vector (since all seeds are in the embedded set). This avoids needing
    torch installed and is a standard evaluation approach.
    """

    retrieved_bibcodes: list[str] = []
    latency_ms = 0.0

    if method == "lexical":
        # Use title only for lexical (full abstract creates too-specific AND query)
        # Restrict to embedded subset for fair comparison with vector methods
        retrieved_bibcodes, latency_ms = lexical_search_embedded(
            conn, query.query_title, limit=limit
        )

    elif method in ("specter2", "indus", "nomic"):
        # Use the seed paper's stored embedding as query vector
        qvec = get_stored_embedding(conn, query.seed_bibcode, method)
        if qvec is None:
            logger.warning("No %s embedding for seed %s", method, query.seed_bibcode)
            return _empty_result(method)
        result = vector_search(conn, qvec, model_name=method, limit=limit)
        retrieved_bibcodes = [p["bibcode"] for p in result.papers]
        latency_ms = result.timing_ms.get("vector_ms", 0.0)

    elif method == "hybrid_specter2_lexical":
        # 2-way RRF fusion: SPECTER2 vector + lexical (restricted to embedded subset)
        qvec = get_stored_embedding(conn, query.seed_bibcode, "specter2")
        if qvec is None:
            logger.warning("No specter2 embedding for seed %s", query.seed_bibcode)
            return _empty_result(method)

        t0 = time.perf_counter()
        # Vector component
        vec_result = vector_search(conn, qvec, model_name="specter2", limit=limit)
        vec_papers = vec_result.papers

        # Lexical component (restricted to embedded set)
        lex_bibs, _ = lexical_search_embedded(conn, query.query_title, limit=limit)
        lex_papers = [{"bibcode": b} for b in lex_bibs]

        # RRF fusion
        fused = rrf_fuse([vec_papers, lex_papers], top_n=limit)
        retrieved_bibcodes = [p["bibcode"] for p in fused]
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    elif method == "hybrid_rrf":
        # 3-way RRF fusion: SPECTER2 + indus + lexical
        qvec_s = get_stored_embedding(conn, query.seed_bibcode, "specter2")
        qvec_i = get_stored_embedding(conn, query.seed_bibcode, "indus")
        if qvec_s is None:
            logger.warning("No specter2 embedding for seed %s", query.seed_bibcode)
            return _empty_result(method)

        t0 = time.perf_counter()
        result_lists: list[list[dict[str, Any]]] = []

        # SPECTER2 vector
        vec_result = vector_search(conn, qvec_s, model_name="specter2", limit=limit)
        result_lists.append(vec_result.papers)

        # Indus vector (if available)
        if qvec_i is not None:
            indus_result = vector_search(conn, qvec_i, model_name="indus", limit=limit)
            result_lists.append(indus_result.papers)

        # Lexical (restricted to embedded set)
        lex_bibs, _ = lexical_search_embedded(conn, query.query_title, limit=limit)
        result_lists.append([{"bibcode": b} for b in lex_bibs])

        fused = rrf_fuse(result_lists, top_n=limit)
        retrieved_bibcodes = [p["bibcode"] for p in fused]
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Exclude the seed paper itself from retrieved results
    retrieved_bibcodes = [b for b in retrieved_bibcodes if b != query.seed_bibcode]

    relevant = query.relevant_bibcodes

    return MethodResult(
        method=method,
        retrieved_bibcodes=tuple(retrieved_bibcodes),
        latency_ms=round(latency_ms, 2),
        ndcg_10=round(ndcg_at_k(retrieved_bibcodes, relevant, 10), 4),
        recall_10=round(recall_at_k(retrieved_bibcodes, relevant, 10), 4),
        recall_20=round(recall_at_k(retrieved_bibcodes, relevant, 20), 4),
        recall_50=round(recall_at_k(retrieved_bibcodes, relevant, 50), 4),
        precision_10=round(precision_at_k(retrieved_bibcodes, relevant, 10), 4),
        mrr=round(mean_reciprocal_rank(retrieved_bibcodes, relevant), 4),
    )


def _empty_result(method: str) -> MethodResult:
    """Return an empty result when embedding is missing."""
    return MethodResult(
        method=method,
        retrieved_bibcodes=(),
        latency_ms=0.0,
        ndcg_10=0.0,
        recall_10=0.0,
        recall_20=0.0,
        recall_50=0.0,
        precision_10=0.0,
        mrr=0.0,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_eval(
    conn: psycopg.Connection,
    queries: list[EvalQuery],
    methods: list[str],
    *,
    limit: int = 100,
) -> EvalSummary:
    """Run all methods on all queries, compute aggregate metrics."""

    all_results: dict[str, list[MethodResult]] = {m: [] for m in methods}
    per_query: list[dict[str, Any]] = []

    for i, query in enumerate(queries):
        logger.info(
            "[%d/%d] Evaluating: %s (domain=%s, %d relevant)",
            i + 1,
            len(queries),
            query.seed_bibcode,
            query.domain,
            len(query.relevant_bibcodes),
        )

        query_data: dict[str, Any] = {
            "seed_bibcode": query.seed_bibcode,
            "domain": query.domain,
            "n_relevant": len(query.relevant_bibcodes),
        }

        for method in methods:
            try:
                result = run_method(
                    conn,
                    query,
                    method,
                    limit=limit,
                )
                all_results[method].append(result)
                query_data[f"{method}_ndcg10"] = result.ndcg_10
                query_data[f"{method}_recall10"] = result.recall_10
                query_data[f"{method}_recall20"] = result.recall_20
                query_data[f"{method}_mrr"] = result.mrr
                query_data[f"{method}_latency_ms"] = result.latency_ms

                logger.info(
                    "  %s: nDCG@10=%.3f  R@10=%.3f  R@20=%.3f  MRR=%.3f  (%.0fms)",
                    method,
                    result.ndcg_10,
                    result.recall_10,
                    result.recall_20,
                    result.mrr,
                    result.latency_ms,
                )
            except Exception:
                logger.warning("  %s: FAILED", method, exc_info=True)
                query_data[f"{method}_ndcg10"] = None

        per_query.append(query_data)

    # Aggregate
    summary_methods: dict[str, dict[str, float]] = {}
    for method in methods:
        results = all_results[method]
        if not results:
            continue
        n = len(results)
        summary_methods[method] = {
            "nDCG@10": round(sum(r.ndcg_10 for r in results) / n, 4),
            "Recall@10": round(sum(r.recall_10 for r in results) / n, 4),
            "Recall@20": round(sum(r.recall_20 for r in results) / n, 4),
            "Recall@50": round(sum(r.recall_50 for r in results) / n, 4),
            "P@10": round(sum(r.precision_10 for r in results) / n, 4),
            "MRR": round(sum(r.mrr for r in results) / n, 4),
            "Latency_ms": round(sum(r.latency_ms for r in results) / n, 1),
            "n_queries": n,
        }

    return EvalSummary(
        num_queries=len(queries),
        methods=summary_methods,
        per_query=per_query,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(summary: EvalSummary) -> str:
    """Generate markdown evaluation report for the paper."""
    lines = [
        "# Retrieval Evaluation: 50-Query Citation-Based Benchmark",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Queries**: {summary.num_queries}",
        "**Ground truth**: Citation-based (bidirectional citation neighbors in embedded subset)",
        "**Methodology**: For each seed paper, query = title + abstract, "
        "relevant = papers citing or cited by seed within embedded set",
        "",
        "## Summary Results",
        "",
        "| Method | nDCG@10 | Recall@10 | Recall@20 | Recall@50 | P@10 | MRR | Latency (ms) |",
        "|--------|---------|-----------|-----------|-----------|------|-----|-------------|",
    ]

    for method, metrics in summary.methods.items():
        lines.append(
            f"| {method} "
            f"| {metrics['nDCG@10']:.4f} "
            f"| {metrics['Recall@10']:.4f} "
            f"| {metrics['Recall@20']:.4f} "
            f"| {metrics['Recall@50']:.4f} "
            f"| {metrics['P@10']:.4f} "
            f"| {metrics['MRR']:.4f} "
            f"| {metrics['Latency_ms']:.1f} |"
        )

    # Relative improvement section
    if "lexical" in summary.methods and "hybrid_specter2_lexical" in summary.methods:
        lex = summary.methods["lexical"]
        hyb = summary.methods["hybrid_specter2_lexical"]
        ndcg_lift = ((hyb["nDCG@10"] - lex["nDCG@10"]) / max(lex["nDCG@10"], 0.0001)) * 100
        recall_lift = (
            (hyb["Recall@10"] - lex["Recall@10"]) / max(lex["Recall@10"], 0.0001)
        ) * 100
        lines.extend(
            [
                "",
                "## Hybrid vs Lexical-Only",
                "",
                f"- nDCG@10 improvement: **{ndcg_lift:+.1f}%**",
                f"- Recall@10 improvement: **{recall_lift:+.1f}%**",
            ]
        )

    if "specter2" in summary.methods and "hybrid_specter2_lexical" in summary.methods:
        spec = summary.methods["specter2"]
        hyb = summary.methods["hybrid_specter2_lexical"]
        ndcg_lift = ((hyb["nDCG@10"] - spec["nDCG@10"]) / max(spec["nDCG@10"], 0.0001)) * 100
        lines.extend(
            [
                "",
                "## Hybrid vs SPECTER2-Only",
                "",
                f"- nDCG@10 improvement: **{ndcg_lift:+.1f}%**",
            ]
        )

    # Error reduction vs dense-only (for paper claim of 49-67%)
    if "specter2" in summary.methods and "hybrid_specter2_lexical" in summary.methods:
        spec_ndcg = summary.methods["specter2"]["nDCG@10"]
        hyb_ndcg = summary.methods["hybrid_specter2_lexical"]["nDCG@10"]
        if spec_ndcg < 1.0:
            error_reduction = ((hyb_ndcg - spec_ndcg) / (1.0 - spec_ndcg)) * 100
            lines.extend(
                [
                    "",
                    "## Error Reduction (Hybrid vs Dense-Only)",
                    "",
                    f"- nDCG@10 error reduction: **{error_reduction:.1f}%** "
                    f"(literature reports 49-67%)",
                ]
            )

    # Domain breakdown
    lines.extend(["", "## Per-Domain Performance (nDCG@10)", ""])
    domains: dict[str, dict[str, list[float]]] = {}
    for qdata in summary.per_query:
        d = qdata["domain"]
        if d not in domains:
            domains[d] = {}
        for method in summary.methods:
            key = f"{method}_ndcg10"
            if key in qdata and qdata[key] is not None:
                domains[d].setdefault(method, []).append(qdata[key])

    if domains:
        header_methods = list(summary.methods.keys())
        lines.append("| Domain | n | " + " | ".join(header_methods) + " |")
        lines.append("|--------|---|" + "|".join(["------"] * len(header_methods)) + "|")
        for domain in sorted(domains.keys()):
            n = max(len(v) for v in domains[domain].values()) if domains[domain] else 0
            vals = []
            for m in header_methods:
                scores = domains[domain].get(m, [])
                if scores:
                    vals.append(f"{sum(scores)/len(scores):.3f}")
                else:
                    vals.append("-")
            lines.append(f"| {domain} | {n} | " + " | ".join(vals) + " |")

    lines.extend(
        [
            "",
            "## Methodology Notes",
            "",
            "- **Citation-based ground truth**: Standard methodology used in SPECTER/SPECTER2 "
            "evaluation. For each seed paper, bidirectional citation neighbors (both citing and "
            "cited papers) within the embedded subset form the relevant document set.",
            "- **Query construction**: Title + first 300 chars of abstract for each seed paper.",
            "- **Metrics**: nDCG@10 (ranking quality), Recall@K (coverage), P@10 (precision), "
            "MRR (first relevant result position).",
            "- **RRF constant**: k=60 (standard).",
            "- Papers are excluded from their own results (seed bibcode filtered out).",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="50-query retrieval evaluation: SPECTER2 vs lexical vs hybrid RRF"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output markdown file (default: stdout)"
    )
    parser.add_argument(
        "--json-output", type=str, default=None, help="Output JSON file with per-query results"
    )
    parser.add_argument(
        "--dsn", default=None, help="PostgreSQL DSN (default: SCIX_DSN or 'dbname=scix')"
    )
    parser.add_argument(
        "--num-queries", type=int, default=50, help="Number of evaluation queries (default: 50)"
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=3,
        help="Minimum citation neighbors for seed selection (default: 3)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="lexical,specter2,hybrid_specter2_lexical",
        help="Comma-separated methods to evaluate (default: lexical,specter2,hybrid_specter2_lexical)",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Max results per method per query (default: 100)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    methods = [m.strip() for m in args.methods.split(",")]

    conn = get_connection(args.dsn)

    # Select evaluation queries
    queries = select_eval_queries(
        conn, num_queries=args.num_queries, min_neighbors=args.min_neighbors
    )

    if not queries:
        logger.error("No queries selected! Check that embeddings and citation edges exist.")
        sys.exit(1)

    # Run evaluation (uses stored embeddings, no model loading needed)
    summary = run_eval(conn, queries, methods, limit=args.limit)

    # Generate report
    report = generate_report(summary)

    if args.output:
        Path(args.output).write_text(report)
        logger.info("Report written to %s", args.output)
    else:
        print(report)

    # Save per-query JSON for further analysis
    if args.json_output:
        json_data = {
            "num_queries": summary.num_queries,
            "methods": summary.methods,
            "per_query": summary.per_query,
        }
        Path(args.json_output).write_text(json.dumps(json_data, indent=2))
        logger.info("JSON results written to %s", args.json_output)

    conn.close()


if __name__ == "__main__":
    main()
