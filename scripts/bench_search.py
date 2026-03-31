#!/usr/bin/env python3
"""Benchmark harness for the SciX search pipeline.

Measures latency (p50, p95, p99) for each search component:
  - Lexical search (tsvector/BM25-like)
  - Vector search (pgvector HNSW) -- if embeddings exist
  - Hybrid search (vector + lexical + RRF)
  - Hybrid + cross-encoder reranker

Generates a markdown report to stdout or a file.

Usage:
    python scripts/bench_search.py
    python scripts/bench_search.py --iterations 50 --output bench_report.md
    python scripts/bench_search.py --lexical-only  # skip vector if no embeddings
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg

from scix.db import get_connection
from scix.search import (
    SearchFilters,
    SearchResult,
    facet_counts,
    get_author_papers,
    get_citations,
    get_paper,
    get_references,
    hybrid_search,
    lexical_search,
    vector_search,
)

logger = logging.getLogger(__name__)

# Representative scientific queries for benchmarking
BENCHMARK_QUERIES = [
    "gravitational wave detection LIGO",
    "exoplanet atmospheres spectroscopy",
    "dark matter direct detection",
    "stellar evolution main sequence",
    "galaxy formation high redshift",
    "black hole accretion disk",
    "cosmic microwave background polarization",
    "supernova nucleosynthesis",
    "X-ray binary millisecond pulsar",
    "21cm cosmology reionization",
]


@dataclass
class BenchmarkRun:
    """Timing data from a single benchmark run."""

    query: str
    mode: str
    latency_ms: float
    result_count: int
    timing_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    """Aggregated statistics for a benchmark mode."""

    mode: str
    runs: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    avg_results: float


def _percentile(data: list[float], pct: float) -> float:
    """Calculate percentile using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (pct / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def summarize(mode: str, runs: list[BenchmarkRun]) -> BenchmarkSummary:
    """Compute summary statistics from a list of benchmark runs."""
    latencies = [r.latency_ms for r in runs]
    counts = [r.result_count for r in runs]
    return BenchmarkSummary(
        mode=mode,
        runs=len(runs),
        p50_ms=round(_percentile(latencies, 50), 2),
        p95_ms=round(_percentile(latencies, 95), 2),
        p99_ms=round(_percentile(latencies, 99), 2),
        mean_ms=round(statistics.mean(latencies), 2) if latencies else 0,
        min_ms=round(min(latencies), 2) if latencies else 0,
        max_ms=round(max(latencies), 2) if latencies else 0,
        avg_results=round(statistics.mean(counts), 1) if counts else 0,
    )


def _has_embeddings(conn: psycopg.Connection, model_name: str = "specter2") -> bool:
    """Check if any embeddings exist for the given model."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM paper_embeddings WHERE model_name = %s LIMIT 1)",
            (model_name,),
        )
        return cur.fetchone()[0]


def _get_sample_embedding(
    conn: psycopg.Connection, model_name: str = "specter2"
) -> list[float] | None:
    """Get a sample embedding vector to use as a query vector for benchmarking."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding FROM paper_embeddings WHERE model_name = %s LIMIT 1",
            (model_name,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        # pgvector returns a string like "[0.1,0.2,...]"
        vec = row[0]
        if isinstance(vec, str):
            return [float(x) for x in vec.strip("[]").split(",")]
        # psycopg with pgvector adapter might return list/array directly
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return list(vec)


def _get_sample_bibcode(conn: psycopg.Connection) -> str | None:
    """Get a bibcode with citation edges for graph benchmarks."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.bibcode FROM papers p
            JOIN citation_edges ce ON ce.source_bibcode = p.bibcode
            WHERE p.citation_count > 0
            ORDER BY p.citation_count DESC NULLS LAST
            LIMIT 1
        """)
        row = cur.fetchone()
        return row[0] if row else None


def _get_sample_author(conn: psycopg.Connection) -> str | None:
    """Get an author name for author search benchmarks."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT first_author FROM papers
            WHERE first_author IS NOT NULL
            ORDER BY citation_count DESC NULLS LAST
            LIMIT 1
        """)
        row = cur.fetchone()
        return row[0] if row else None


def bench_lexical(
    conn: psycopg.Connection,
    queries: list[str],
    iterations: int,
) -> list[BenchmarkRun]:
    """Benchmark lexical search."""
    runs: list[BenchmarkRun] = []
    for _ in range(iterations):
        for q in queries:
            t0 = time.perf_counter()
            result = lexical_search(conn, q, limit=20)
            latency = round((time.perf_counter() - t0) * 1000, 2)
            runs.append(BenchmarkRun(
                query=q,
                mode="lexical",
                latency_ms=latency,
                result_count=result.total,
                timing_breakdown=result.timing_ms,
            ))
    return runs


def bench_vector(
    conn: psycopg.Connection,
    query_embedding: list[float],
    queries: list[str],
    iterations: int,
) -> list[BenchmarkRun]:
    """Benchmark vector search using a fixed query embedding."""
    runs: list[BenchmarkRun] = []
    for _ in range(iterations):
        for q in queries:
            t0 = time.perf_counter()
            result = vector_search(conn, query_embedding, limit=20)
            latency = round((time.perf_counter() - t0) * 1000, 2)
            runs.append(BenchmarkRun(
                query=q,
                mode="vector",
                latency_ms=latency,
                result_count=result.total,
                timing_breakdown=result.timing_ms,
            ))
    return runs


def bench_hybrid(
    conn: psycopg.Connection,
    queries: list[str],
    query_embedding: list[float] | None,
    iterations: int,
    reranker: Any | None = None,
) -> list[BenchmarkRun]:
    """Benchmark hybrid search (with optional reranker)."""
    mode = "hybrid+rerank" if reranker else "hybrid"
    runs: list[BenchmarkRun] = []
    for _ in range(iterations):
        for q in queries:
            t0 = time.perf_counter()
            result = hybrid_search(
                conn, q, query_embedding, top_n=20, reranker=reranker
            )
            latency = round((time.perf_counter() - t0) * 1000, 2)
            runs.append(BenchmarkRun(
                query=q,
                mode=mode,
                latency_ms=latency,
                result_count=result.total,
                timing_breakdown=result.timing_ms,
            ))
    return runs


def bench_graph(
    conn: psycopg.Connection,
    bibcode: str,
    iterations: int,
) -> dict[str, list[BenchmarkRun]]:
    """Benchmark graph traversal operations."""
    results: dict[str, list[BenchmarkRun]] = {
        "get_paper": [],
        "get_citations": [],
        "get_references": [],
    }
    for _ in range(iterations):
        for func_name, func in [
            ("get_paper", lambda: get_paper(conn, bibcode)),
            ("get_citations", lambda: get_citations(conn, bibcode, limit=20)),
            ("get_references", lambda: get_references(conn, bibcode, limit=20)),
        ]:
            t0 = time.perf_counter()
            result = func()
            latency = round((time.perf_counter() - t0) * 1000, 2)
            results[func_name].append(BenchmarkRun(
                query=bibcode,
                mode=func_name,
                latency_ms=latency,
                result_count=result.total,
                timing_breakdown=result.timing_ms,
            ))
    return results


def bench_author(
    conn: psycopg.Connection,
    author: str,
    iterations: int,
) -> list[BenchmarkRun]:
    """Benchmark author search."""
    runs: list[BenchmarkRun] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = get_author_papers(conn, author, limit=50)
        latency = round((time.perf_counter() - t0) * 1000, 2)
        runs.append(BenchmarkRun(
            query=author,
            mode="author_search",
            latency_ms=latency,
            result_count=result.total,
            timing_breakdown=result.timing_ms,
        ))
    return runs


def bench_facets(
    conn: psycopg.Connection,
    iterations: int,
) -> list[BenchmarkRun]:
    """Benchmark facet counting."""
    runs: list[BenchmarkRun] = []
    for _ in range(iterations):
        for field in ["year", "doctype", "arxiv_class"]:
            t0 = time.perf_counter()
            result = facet_counts(conn, field, limit=50)
            latency = round((time.perf_counter() - t0) * 1000, 2)
            runs.append(BenchmarkRun(
                query=field,
                mode="facet_counts",
                latency_ms=latency,
                result_count=result.total,
                timing_breakdown=result.timing_ms,
            ))
    return runs


def generate_report(
    summaries: list[BenchmarkSummary],
    has_vectors: bool,
    paper_count: int,
    embedding_count: int,
) -> str:
    """Generate a markdown benchmark report."""
    lines = [
        "# SciX Search Benchmark Report",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Papers in corpus**: {paper_count:,}",
        f"**Embeddings**: {embedding_count:,} ({'available' if has_vectors else 'none'})",
        "",
        "## Latency Summary",
        "",
        "| Mode | Runs | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Min | Max | Avg Results |",
        "|------|------|----------|----------|----------|-----------|-----|-----|-------------|",
    ]

    for s in summaries:
        lines.append(
            f"| {s.mode} | {s.runs} | {s.p50_ms} | {s.p95_ms} | {s.p99_ms} "
            f"| {s.mean_ms} | {s.min_ms} | {s.max_ms} | {s.avg_results} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- All timings include PostgreSQL query execution and Python processing.",
        "- p50/p95/p99 are wall-clock latencies from the client perspective.",
        "- Vector search uses HNSW with ef_search=100 (default).",
        "- Hybrid search fetches top-60 from each source, fuses with RRF (k=60), returns top-20.",
        f"- {'Embeddings available: vector and hybrid modes benchmarked.' if has_vectors else 'No embeddings found: lexical-only mode benchmarked.'}",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SciX search pipeline")
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of iterations per query (default: 5)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for markdown report (default: stdout)",
    )
    parser.add_argument(
        "--dsn", default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--lexical-only", action="store_true",
        help="Skip vector/hybrid benchmarks even if embeddings exist",
    )
    parser.add_argument(
        "--include-reranker", action="store_true",
        help="Include cross-encoder reranker in hybrid benchmark",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    conn = get_connection(args.dsn)
    queries = BENCHMARK_QUERIES
    summaries: list[BenchmarkSummary] = []

    # Corpus stats
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM papers")
        paper_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM paper_embeddings")
        embedding_count = cur.fetchone()[0]

    has_vectors = not args.lexical_only and _has_embeddings(conn)
    query_embedding = _get_sample_embedding(conn) if has_vectors else None

    logger.info("Corpus: %d papers, %d embeddings", paper_count, embedding_count)
    logger.info("Vectors available: %s", has_vectors)
    logger.info("Benchmarking with %d queries x %d iterations", len(queries), args.iterations)

    # --- Lexical ---
    logger.info("Benchmarking lexical search...")
    lex_runs = bench_lexical(conn, queries, args.iterations)
    summaries.append(summarize("lexical", lex_runs))
    logger.info("  lexical p50=%.1fms p95=%.1fms", summaries[-1].p50_ms, summaries[-1].p95_ms)

    # --- Vector ---
    if has_vectors and query_embedding:
        logger.info("Benchmarking vector search...")
        vec_runs = bench_vector(conn, query_embedding, queries, args.iterations)
        summaries.append(summarize("vector", vec_runs))
        logger.info("  vector p50=%.1fms p95=%.1fms", summaries[-1].p50_ms, summaries[-1].p95_ms)

    # --- Hybrid ---
    logger.info("Benchmarking hybrid search...")
    hybrid_runs = bench_hybrid(conn, queries, query_embedding, args.iterations)
    summaries.append(summarize("hybrid", hybrid_runs))
    logger.info("  hybrid p50=%.1fms p95=%.1fms", summaries[-1].p50_ms, summaries[-1].p95_ms)

    # --- Hybrid + Reranker ---
    if args.include_reranker:
        try:
            from scix.search import CrossEncoderReranker

            reranker = CrossEncoderReranker()
            logger.info("Benchmarking hybrid + reranker...")
            rerank_runs = bench_hybrid(
                conn, queries, query_embedding, args.iterations, reranker=reranker
            )
            summaries.append(summarize("hybrid+rerank", rerank_runs))
            logger.info(
                "  hybrid+rerank p50=%.1fms p95=%.1fms",
                summaries[-1].p50_ms, summaries[-1].p95_ms,
            )
        except ImportError:
            logger.warning("Skipping reranker benchmark (sentence-transformers not installed)")

    # --- Graph operations ---
    sample_bibcode = _get_sample_bibcode(conn)
    if sample_bibcode:
        logger.info("Benchmarking graph operations (bibcode=%s)...", sample_bibcode)
        graph_results = bench_graph(conn, sample_bibcode, args.iterations)
        for mode, runs in graph_results.items():
            summaries.append(summarize(mode, runs))
            logger.info("  %s p50=%.1fms", mode, summaries[-1].p50_ms)

    # --- Author search ---
    sample_author = _get_sample_author(conn)
    if sample_author:
        logger.info("Benchmarking author search (author=%s)...", sample_author)
        author_runs = bench_author(conn, sample_author, args.iterations)
        summaries.append(summarize("author_search", author_runs))
        logger.info("  author p50=%.1fms", summaries[-1].p50_ms)

    # --- Facet counts ---
    logger.info("Benchmarking facet counts...")
    facet_runs = bench_facets(conn, args.iterations)
    summaries.append(summarize("facet_counts", facet_runs))
    logger.info("  facets p50=%.1fms", summaries[-1].p50_ms)

    # Generate report
    report = generate_report(summaries, has_vectors, paper_count, embedding_count)

    if args.output:
        Path(args.output).write_text(report)
        logger.info("Report written to %s", args.output)
    else:
        print(report)

    conn.close()


if __name__ == "__main__":
    main()
