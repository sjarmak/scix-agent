#!/usr/bin/env python3
"""Evaluate search quality: tsvector ts_rank_cd vs pg_search BM25.

Runs benchmark queries against both search backends (when available),
measures timing and ranking correlation, and outputs a comparison report.

pg_search (ParadeDB BM25) is optional — if the extension is not installed,
only tsvector results are reported.

Usage:
    python3 scripts/eval_search_quality.py
    python3 scripts/eval_search_quality.py --output eval_report.md
    python3 scripts/eval_search_quality.py --dsn "dbname=scix" --limit 20
"""

from __future__ import annotations

import argparse
import logging
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
    STUB_COLUMNS,
    SearchFilters,
    SearchResult,
    _elapsed_ms,
    lexical_search,
)

logger = logging.getLogger(__name__)

# Benchmark queries spanning diverse scientific domains
EVAL_QUERIES = [
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
    "neutrino oscillation mass hierarchy",
    "fast radio burst magnetar",
    "type Ia supernova cosmological distance",
    "primordial gravitational waves inflation",
    "active galactic nuclei jet",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryResult:
    """Result from a single query against one backend."""

    query: str
    backend: str
    bibcodes: tuple[str, ...]
    scores: tuple[float, ...]
    latency_ms: float
    result_count: int


@dataclass(frozen=True)
class QueryComparison:
    """Side-by-side comparison of two backends for the same query."""

    query: str
    tsvector: QueryResult
    bm25: QueryResult
    overlap_at_10: float  # Jaccard overlap of top-10 bibcodes
    overlap_at_20: float  # Jaccard overlap of top-20 bibcodes
    rank_correlation: float  # Kendall's tau on shared bibcodes
    tsvector_only: tuple[str, ...]  # bibcodes in tsvector but not bm25
    bm25_only: tuple[str, ...]  # bibcodes in bm25 but not tsvector


@dataclass
class EvalSummary:
    """Aggregated evaluation summary."""

    num_queries: int
    tsvector_mean_ms: float
    bm25_mean_ms: float
    mean_overlap_at_10: float
    mean_overlap_at_20: float
    mean_rank_correlation: float
    comparisons: list[QueryComparison] = field(default_factory=list)


# ---------------------------------------------------------------------------
# pg_search detection
# ---------------------------------------------------------------------------


def has_pg_search(conn: psycopg.Connection) -> bool:
    """Check if pg_search extension is installed and BM25 index exists."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS(
                SELECT 1 FROM pg_extension WHERE extname = 'pg_search'
            )
        """)
        ext_exists = cur.fetchone()[0]
        if not ext_exists:
            return False

        cur.execute("""
            SELECT EXISTS(
                SELECT 1 FROM pg_indexes WHERE indexname = 'idx_papers_bm25'
            )
        """)
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# BM25 search via pg_search
# ---------------------------------------------------------------------------


def bm25_search(
    conn: psycopg.Connection,
    query_text: str,
    *,
    limit: int = 20,
) -> QueryResult:
    """Full-text search using pg_search BM25 scoring.

    Uses the ParadeDB search syntax via the @@@ operator on the BM25 index.
    """
    t0 = time.perf_counter()

    sql = """
        SELECT p.bibcode, p.title, p.first_author, p.year,
               p.citation_count, p.abstract,
               paradedb.score(p.bibcode) AS rank
        FROM papers p
        WHERE p.bibcode @@@ paradedb.parse(%s)
        ORDER BY rank DESC
        LIMIT %s
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (query_text, limit))
        rows = cur.fetchall()

    latency = _elapsed_ms(t0)

    bibcodes = tuple(row["bibcode"] for row in rows)
    scores = tuple(float(row["rank"]) for row in rows)

    return QueryResult(
        query=query_text,
        backend="bm25",
        bibcodes=bibcodes,
        scores=scores,
        latency_ms=latency,
        result_count=len(rows),
    )


def tsvector_search(
    conn: psycopg.Connection,
    query_text: str,
    *,
    limit: int = 20,
) -> QueryResult:
    """Full-text search using tsvector with ts_rank_cd scoring.

    Wraps lexical_search and extracts bibcodes/scores for comparison.
    """
    t0 = time.perf_counter()
    result = lexical_search(conn, query_text, limit=limit)
    latency = _elapsed_ms(t0)

    bibcodes = tuple(p["bibcode"] for p in result.papers)
    scores = tuple(p["score"] for p in result.papers)

    return QueryResult(
        query=query_text,
        backend="tsvector",
        bibcodes=bibcodes,
        scores=scores,
        latency_ms=latency,
        result_count=result.total,
    )


# ---------------------------------------------------------------------------
# Ranking correlation
# ---------------------------------------------------------------------------


def jaccard_overlap(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets. Returns 0.0 if both empty."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def kendall_tau_on_shared(
    ranking_a: tuple[str, ...],
    ranking_b: tuple[str, ...],
) -> float:
    """Kendall's tau rank correlation on the intersection of two ranked lists.

    Returns a value in [-1, 1]. 1 means identical order, -1 means reversed.
    Returns 0.0 if fewer than 2 shared items.
    """
    shared = set(ranking_a) & set(ranking_b)
    if len(shared) < 2:
        return 0.0

    # Build rank maps (position in each list)
    rank_a = {bib: i for i, bib in enumerate(ranking_a) if bib in shared}
    rank_b = {bib: i for i, bib in enumerate(ranking_b) if bib in shared}

    items = sorted(shared)
    concordant = 0
    discordant = 0

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a_diff = rank_a[items[i]] - rank_a[items[j]]
            b_diff = rank_b[items[i]] - rank_b[items[j]]
            if a_diff * b_diff > 0:
                concordant += 1
            elif a_diff * b_diff < 0:
                discordant += 1
            # ties (diff==0) are neither concordant nor discordant

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 0.0
    return (concordant - discordant) / total_pairs


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_query(
    conn: psycopg.Connection,
    query_text: str,
    *,
    limit: int = 20,
    run_bm25: bool = True,
) -> QueryComparison | None:
    """Run a query against both backends and compare results."""
    tsv_result = tsvector_search(conn, query_text, limit=limit)

    if not run_bm25:
        return None

    bm25_result = bm25_search(conn, query_text, limit=limit)

    top10_tsv = set(tsv_result.bibcodes[:10])
    top10_bm25 = set(bm25_result.bibcodes[:10])
    top20_tsv = set(tsv_result.bibcodes[:20])
    top20_bm25 = set(bm25_result.bibcodes[:20])

    return QueryComparison(
        query=query_text,
        tsvector=tsv_result,
        bm25=bm25_result,
        overlap_at_10=round(jaccard_overlap(top10_tsv, top10_bm25), 4),
        overlap_at_20=round(jaccard_overlap(top20_tsv, top20_bm25), 4),
        rank_correlation=round(kendall_tau_on_shared(tsv_result.bibcodes, bm25_result.bibcodes), 4),
        tsvector_only=tuple(b for b in tsv_result.bibcodes if b not in top20_bm25),
        bm25_only=tuple(b for b in bm25_result.bibcodes if b not in top20_tsv),
    )


def run_eval(
    conn: psycopg.Connection,
    queries: list[str],
    *,
    limit: int = 20,
) -> EvalSummary:
    """Run the full evaluation across all queries."""
    bm25_available = has_pg_search(conn)
    if not bm25_available:
        logger.warning("pg_search not available — running tsvector-only evaluation")

    comparisons: list[QueryComparison] = []
    tsv_latencies: list[float] = []
    bm25_latencies: list[float] = []

    for query in queries:
        logger.info("Evaluating: %s", query)

        if bm25_available:
            comp = compare_query(conn, query, limit=limit, run_bm25=True)
            if comp is not None:
                comparisons.append(comp)
                tsv_latencies.append(comp.tsvector.latency_ms)
                bm25_latencies.append(comp.bm25.latency_ms)
                logger.info(
                    "  tsvector: %d results (%.1fms), bm25: %d results (%.1fms), "
                    "overlap@10=%.2f, tau=%.2f",
                    comp.tsvector.result_count,
                    comp.tsvector.latency_ms,
                    comp.bm25.result_count,
                    comp.bm25.latency_ms,
                    comp.overlap_at_10,
                    comp.rank_correlation,
                )
        else:
            tsv_result = tsvector_search(conn, query, limit=limit)
            tsv_latencies.append(tsv_result.latency_ms)
            logger.info(
                "  tsvector: %d results (%.1fms)",
                tsv_result.result_count,
                tsv_result.latency_ms,
            )

    def _safe_mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    return EvalSummary(
        num_queries=len(queries),
        tsvector_mean_ms=_safe_mean(tsv_latencies),
        bm25_mean_ms=_safe_mean(bm25_latencies),
        mean_overlap_at_10=_safe_mean([c.overlap_at_10 for c in comparisons]),
        mean_overlap_at_20=_safe_mean([c.overlap_at_20 for c in comparisons]),
        mean_rank_correlation=_safe_mean([c.rank_correlation for c in comparisons]),
        comparisons=comparisons,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(summary: EvalSummary, *, bm25_available: bool) -> str:
    """Generate a markdown evaluation report."""
    lines = [
        "# SciX Search Quality Evaluation",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Queries evaluated**: {summary.num_queries}",
        f"**pg_search BM25**: {'available' if bm25_available else 'NOT available'}",
        "",
    ]

    if not bm25_available:
        lines.extend(
            [
                "## tsvector-only Results",
                "",
                f"Mean latency: **{summary.tsvector_mean_ms}ms**",
                "",
                "pg_search extension is not installed. To enable BM25 comparison:",
                "1. Install ParadeDB pg_search extension",
                "2. Run migration 004_per_model_hnsw_and_pg_search.sql",
                "3. Re-run this evaluation",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| tsvector mean latency | {summary.tsvector_mean_ms}ms |",
            f"| BM25 mean latency | {summary.bm25_mean_ms}ms |",
            f"| Mean overlap @ 10 | {summary.mean_overlap_at_10:.2%} |",
            f"| Mean overlap @ 20 | {summary.mean_overlap_at_20:.2%} |",
            f"| Mean rank correlation (tau) | {summary.mean_rank_correlation:.3f} |",
            "",
            "## Per-Query Comparison",
            "",
            "| Query | tsvector (ms) | BM25 (ms) | Overlap@10 | Overlap@20 | Tau |",
            "|-------|--------------|-----------|------------|------------|-----|",
        ]
    )

    for comp in summary.comparisons:
        lines.append(
            f"| {comp.query} | {comp.tsvector.latency_ms:.1f} "
            f"| {comp.bm25.latency_ms:.1f} "
            f"| {comp.overlap_at_10:.2%} "
            f"| {comp.overlap_at_20:.2%} "
            f"| {comp.rank_correlation:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Overlap@K**: Jaccard similarity of top-K result sets (1.0 = identical sets).",
            "- **Tau**: Kendall's tau rank correlation on shared results (1.0 = identical ordering).",
            "- High overlap + low tau means both find the same papers but rank them differently.",
            "- Low overlap means the backends retrieve substantially different document sets.",
            "",
            "### BM25 vs ts_rank_cd",
            "",
            "- **ts_rank_cd**: Cover density ranking. Scores based on proximity of matching terms.",
            "  Does not account for document length or term frequency saturation.",
            "- **BM25**: Okapi BM25. Incorporates document length normalization (dl/avgdl) and",
            "  term frequency saturation (tf / (tf + k1)). Generally better for scientific text",
            "  where abstract length varies significantly (short letters vs long reviews).",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate tsvector vs pg_search BM25 search quality"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for markdown report (default: stdout)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of results per query (default: 20)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    conn = get_connection(args.dsn)
    bm25_available = has_pg_search(conn)

    logger.info("pg_search BM25 available: %s", bm25_available)

    summary = run_eval(conn, EVAL_QUERIES, limit=args.limit)
    report = generate_report(summary, bm25_available=bm25_available)

    if args.output:
        Path(args.output).write_text(report)
        logger.info("Report written to %s", args.output)
    else:
        print(report)

    conn.close()


if __name__ == "__main__":
    main()
