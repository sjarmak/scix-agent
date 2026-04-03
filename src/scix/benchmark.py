"""Benchmark validation against PRD requirements.

Implements the 5 benchmark categories from the SciX knowledge infrastructure PRD:
  1. Semantic search (latency + recall@10)
  2. Citation graph traversal (2-hop with fan-out)
  3. Faceted filtering (GIN index performance)
  4. Combined query (semantic + facets + citation expansion)
  5. Ingestion metrics (corpus statistics + HNSW memory)

Usage via CLI:
    python scripts/bench_prd.py
    python scripts/bench_prd.py --iterations 10 --output report.md
    python scripts/bench_prd.py --skip-semantic  # skip model-dependent benchmarks
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg

from scix.db import get_connection
from scix.search import (
    SearchFilters,
    get_citations,
    vector_search,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """Result from a single benchmark category."""

    name: str
    passed: bool | None  # None for informational benchmarks
    target: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    iterations: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionMetrics:
    """Corpus statistics from benchmark #5."""

    total_papers: int
    total_embeddings: int
    total_citation_edges: int
    papers_without_abstracts: int
    hnsw_index_bytes: int
    table_total_bytes: int
    system_ram_bytes: int
    hnsw_pct_of_ram: float


@dataclass(frozen=True)
class PRDBenchmarkReport:
    """Complete report from all 5 benchmark categories."""

    timestamp: str
    results: list[BenchmarkResult]
    ingestion: IngestionMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def percentile(data: list[float], pct: float) -> float:
    """Calculate percentile using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (pct / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return round(sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo]), 2)


def _select_test_papers(conn: psycopg.Connection, count: int = 5) -> list[dict[str, Any]]:
    """Select diverse papers with abstracts for benchmark queries."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT bibcode, title, abstract
            FROM papers
            WHERE abstract IS NOT NULL AND abstract != ''
              AND title IS NOT NULL
              AND citation_count > 5
            ORDER BY random()
            LIMIT %s
            """,
            (count,),
        )
        return [{"bibcode": r[0], "title": r[1], "abstract": r[2]} for r in cur.fetchall()]


def _select_citation_test_papers(
    conn: psycopg.Connection,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a typical and a highly-cited paper for graph benchmarks."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT bibcode, citation_count FROM papers
            WHERE citation_count BETWEEN 15 AND 30
            ORDER BY random() LIMIT 1
            """)
        typical = cur.fetchone()

        cur.execute("""
            SELECT bibcode, citation_count FROM papers
            WHERE citation_count > 1000
            ORDER BY citation_count DESC LIMIT 1
            """)
        highly_cited = cur.fetchone()

    return (
        {"bibcode": typical[0], "citation_count": typical[1]} if typical else {},
        {"bibcode": highly_cited[0], "citation_count": highly_cited[1]} if highly_cited else {},
    )


# ---------------------------------------------------------------------------
# Benchmark 1: Semantic Search
# ---------------------------------------------------------------------------


def bench_semantic_search(
    conn: psycopg.Connection,
    model: Any,
    tokenizer: Any,
    *,
    iterations: int = 10,
    limit: int = 10,
) -> BenchmarkResult:
    """Benchmark semantic search: embed abstracts, find top-k, measure recall@10.

    Recall@10 uses self-retrieval: each paper's own abstract is embedded and
    we check whether the paper appears in its own top-10 results.
    """
    from scix.embed import embed_batch, prepare_input

    papers = _select_test_papers(conn, count=5)
    if not papers:
        return BenchmarkResult(
            name="semantic_search",
            passed=None,
            target="p95 < 100ms",
            p50_ms=0,
            p95_ms=0,
            p99_ms=0,
            iterations=0,
            details={"error": "no papers with abstracts found"},
        )

    # Pre-embed all test papers
    inputs = []
    for p in papers:
        inp = prepare_input(p["bibcode"], p["title"], p["abstract"])
        if inp:
            inputs.append((p["bibcode"], inp.text))

    texts = [t for _, t in inputs]
    bibcodes = [b for b, _ in inputs]
    embeddings = embed_batch(model, tokenizer, texts)

    # Run iterations
    latencies: list[float] = []
    recall_hits = 0
    recall_total = len(bibcodes)

    for iteration in range(iterations):
        for bibcode, embedding in zip(bibcodes, embeddings):
            t0 = time.perf_counter()
            result = vector_search(conn, embedding, limit=limit)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            latencies.append(elapsed)

            # Check recall on first iteration only (deterministic)
            if iteration == 0:
                result_bibcodes = {p.get("bibcode") for p in result.papers}
                if bibcode in result_bibcodes:
                    recall_hits += 1

    p95 = percentile(latencies, 95)
    return BenchmarkResult(
        name="semantic_search",
        passed=p95 < 100,
        target="p95 < 100ms",
        p50_ms=percentile(latencies, 50),
        p95_ms=p95,
        p99_ms=percentile(latencies, 99),
        iterations=len(latencies),
        details={
            "recall_at_10": f"{recall_hits}/{recall_total}",
            "recall_fraction": round(recall_hits / max(recall_total, 1), 2),
            "query_count": len(bibcodes),
        },
    )


# ---------------------------------------------------------------------------
# Benchmark 2: Citation Graph 2-Hop Traversal
# ---------------------------------------------------------------------------


_TWO_HOP_CTE = """
WITH RECURSIVE citation_hops AS (
    -- Hop 1: papers that cite the root paper
    SELECT source_bibcode AS bibcode, 1 AS depth
    FROM citation_edges
    WHERE target_bibcode = %(root)s

    UNION ALL

    -- Hop 2: papers that cite the hop-1 papers
    SELECT ce.source_bibcode AS bibcode, ch.depth + 1
    FROM citation_hops ch
    JOIN citation_edges ce ON ce.target_bibcode = ch.bibcode
    WHERE ch.depth < 2
)
SELECT bibcode, MIN(depth) AS min_depth, COUNT(*) AS path_count
FROM citation_hops
WHERE bibcode != %(root)s
GROUP BY bibcode
ORDER BY min_depth, path_count DESC
LIMIT %(fan_out_limit)s
"""


def _run_two_hop(
    conn: psycopg.Connection,
    bibcode: str,
    fan_out_limit: int = 500,
) -> tuple[float, int, int, int]:
    """Execute 2-hop citation CTE, return (latency_ms, total, hop1_count, hop2_count)."""
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '30s'")
        cur.execute(
            _TWO_HOP_CTE,
            {"root": bibcode, "fan_out_limit": fan_out_limit},
        )
        rows = cur.fetchall()
    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    hop1 = sum(1 for r in rows if r[1] == 1)
    hop2 = sum(1 for r in rows if r[1] == 2)
    return elapsed, len(rows), hop1, hop2


def bench_citation_traversal(
    conn: psycopg.Connection,
    *,
    iterations: int = 5,
    fan_out_limit: int = 500,
) -> BenchmarkResult:
    """Benchmark 2-hop citation graph traversal on typical and highly-cited papers."""
    typical, highly_cited = _select_citation_test_papers(conn)

    if not typical or not highly_cited:
        return BenchmarkResult(
            name="citation_2hop",
            passed=None,
            target="informational",
            p50_ms=0,
            p95_ms=0,
            p99_ms=0,
            iterations=0,
            details={"error": "could not find test papers"},
        )

    details: dict[str, Any] = {}

    # Benchmark both cases
    for label, paper in [("typical", typical), ("highly_cited", highly_cited)]:
        latencies: list[float] = []
        for _ in range(iterations):
            elapsed, total, hop1, hop2 = _run_two_hop(conn, paper["bibcode"], fan_out_limit)
            latencies.append(elapsed)

        details[label] = {
            "bibcode": paper["bibcode"],
            "citation_count": paper["citation_count"],
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "total_results": total,
            "hop1_count": hop1,
            "hop2_count": hop2,
            "fan_out_ratio": round(hop2 / max(hop1, 1), 1),
        }

    # Use highly-cited p95 as the representative latency (worst case)
    all_latencies = [
        details["typical"]["p95_ms"],
        details["highly_cited"]["p95_ms"],
    ]

    return BenchmarkResult(
        name="citation_2hop",
        passed=None,  # informational, no target
        target="informational (fan-out limit prevents explosion)",
        p50_ms=min(all_latencies),
        p95_ms=max(all_latencies),
        p99_ms=max(all_latencies),
        iterations=iterations * 2,
        details=details,
    )


# ---------------------------------------------------------------------------
# Benchmark 3: Faceted Filtering
# ---------------------------------------------------------------------------


_FACET_FILTER_SQL = """
SELECT count(*)
FROM papers
WHERE year = 2023
  AND doctype = 'article'
  AND arxiv_class @> ARRAY['astro-ph.EP']
"""


def bench_faceted_filtering(
    conn: psycopg.Connection,
    *,
    iterations: int = 20,
) -> BenchmarkResult:
    """Benchmark faceted filtering with GIN indexes."""
    latencies: list[float] = []
    result_count = 0

    for _ in range(iterations):
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(_FACET_FILTER_SQL)
            result_count = cur.fetchone()[0]
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(elapsed)

    p95 = percentile(latencies, 95)
    return BenchmarkResult(
        name="faceted_filtering",
        passed=p95 < 50,
        target="p95 < 50ms",
        p50_ms=percentile(latencies, 50),
        p95_ms=p95,
        p99_ms=percentile(latencies, 99),
        iterations=iterations,
        details={
            "filter": "year=2023 AND doctype='article' AND arxiv_class @> '{astro-ph.EP}'",
            "matching_papers": result_count,
        },
    )


# ---------------------------------------------------------------------------
# Benchmark 4: Combined Query
# ---------------------------------------------------------------------------


def bench_combined_query(
    conn: psycopg.Connection,
    model: Any,
    tokenizer: Any,
    *,
    iterations: int = 5,
) -> BenchmarkResult:
    """Benchmark combined query: semantic search + facets + 1-hop citation expansion."""
    from scix.embed import embed_batch

    query_text = "exoplanet atmospheres spectroscopy"
    embeddings = embed_batch(model, tokenizer, [query_text])
    query_embedding = embeddings[0]

    filters = SearchFilters(year_min=2022, year_max=2024, doctype="article")

    latencies: list[float] = []
    search_count = 0
    citation_count = 0

    for _ in range(iterations):
        t0 = time.perf_counter()

        # Step 1: filtered semantic search
        search_result = vector_search(conn, query_embedding, filters=filters, limit=10)

        # Step 2: expand top-5 by 1-hop citations
        expanded = []
        for paper in search_result.papers[:5]:
            bib = paper.get("bibcode")
            if bib:
                cites = get_citations(conn, bib, limit=20)
                expanded.extend(cites.papers)

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(elapsed)
        search_count = len(search_result.papers)
        citation_count = len(expanded)

    p95 = percentile(latencies, 95)
    return BenchmarkResult(
        name="combined_query",
        passed=p95 < 500,
        target="p95 < 500ms",
        p50_ms=percentile(latencies, 50),
        p95_ms=p95,
        p99_ms=percentile(latencies, 99),
        iterations=iterations,
        details={
            "query": query_text,
            "filters": "year=[2022,2024], doctype=article",
            "search_results": search_count,
            "expanded_citations": citation_count,
        },
    )


# ---------------------------------------------------------------------------
# Benchmark 5: Ingestion Metrics
# ---------------------------------------------------------------------------


def collect_ingestion_metrics(conn: psycopg.Connection) -> IngestionMetrics:
    """Collect corpus statistics and HNSW memory consumption."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM papers")
        total_papers = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM paper_embeddings WHERE model_name = 'specter2'")
        total_embeddings = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM citation_edges")
        total_edges = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM papers WHERE abstract IS NULL OR abstract = ''")
        no_abstract = cur.fetchone()[0]

        # HNSW index size — find dynamically
        cur.execute("""
            SELECT indexname, pg_relation_size(quote_ident(indexname))
            FROM pg_indexes
            WHERE tablename = 'paper_embeddings'
              AND indexdef LIKE '%%hnsw%%'
            ORDER BY pg_relation_size(quote_ident(indexname)) DESC
            LIMIT 1
            """)
        row = cur.fetchone()
        hnsw_bytes = row[1] if row else 0

        cur.execute("SELECT pg_total_relation_size('paper_embeddings')")
        table_bytes = cur.fetchone()[0]

    # System RAM from /proc/meminfo
    system_ram = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    system_ram = int(line.split()[1]) * 1024  # kB to bytes
                    break
    except OSError:
        pass

    hnsw_pct = round((hnsw_bytes / max(system_ram, 1)) * 100, 1) if system_ram else 0

    return IngestionMetrics(
        total_papers=total_papers,
        total_embeddings=total_embeddings,
        total_citation_edges=total_edges,
        papers_without_abstracts=no_abstract,
        hnsw_index_bytes=hnsw_bytes,
        table_total_bytes=table_bytes,
        system_ram_bytes=system_ram,
        hnsw_pct_of_ram=hnsw_pct,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _fmt_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def generate_markdown_report(report: PRDBenchmarkReport) -> str:
    """Generate a markdown report from benchmark results."""
    lines = [
        "# SciX PRD Benchmark Validation Report",
        "",
        f"**Timestamp**: {report.timestamp}",
        "",
        "## Benchmark Results",
        "",
        "| # | Benchmark | Target | p50 (ms) | p95 (ms) | p99 (ms) | Iterations | Pass |",
        "|---|-----------|--------|----------|----------|----------|------------|------|",
    ]

    for i, r in enumerate(report.results, 1):
        status = "N/A" if r.passed is None else ("PASS" if r.passed else "**FAIL**")
        lines.append(
            f"| {i} | {r.name} | {r.target} | {r.p50_ms} | {r.p95_ms} "
            f"| {r.p99_ms} | {r.iterations} | {status} |"
        )

    # Detail sections
    for i, r in enumerate(report.results, 1):
        lines.extend(["", f"### {i}. {r.name}", ""])
        for k, v in r.details.items():
            if isinstance(v, dict):
                lines.append(f"**{k}**:")
                for sk, sv in v.items():
                    lines.append(f"  - {sk}: {sv}")
            else:
                lines.append(f"- {k}: {v}")

    # Ingestion metrics
    m = report.ingestion
    lines.extend(
        [
            "",
            "## Ingestion Metrics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total papers | {m.total_papers:,} |",
            f"| Total embeddings (SPECTER2) | {m.total_embeddings:,} |",
            f"| Total citation edges | {m.total_citation_edges:,} |",
            f"| Papers without abstracts | {m.papers_without_abstracts:,} |",
            f"| HNSW index size | {_fmt_bytes(m.hnsw_index_bytes)} |",
            f"| paper_embeddings total size | {_fmt_bytes(m.table_total_bytes)} |",
            f"| System RAM | {_fmt_bytes(m.system_ram_bytes)} |",
            f"| HNSW as % of RAM | {m.hnsw_pct_of_ram}% |",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_prd_benchmarks(
    dsn: str | None = None,
    *,
    iterations: int = 10,
    device: str = "cpu",
    skip_semantic: bool = False,
) -> PRDBenchmarkReport:
    """Run all 5 PRD benchmarks and return a report."""
    conn = get_connection(dsn)
    try:
        results: list[BenchmarkResult] = []

        model, tokenizer = None, None
        if not skip_semantic:
            from scix.embed import load_model

            logger.info("Loading SPECTER2 model on %s...", device)
            model, tokenizer = load_model("specter2", device=device)

        # 1. Semantic search
        if not skip_semantic:
            logger.info("Benchmark 1: Semantic search...")
            results.append(bench_semantic_search(conn, model, tokenizer, iterations=iterations))
            logger.info(
                "  p95=%.1fms, recall=%s",
                results[-1].p95_ms,
                results[-1].details.get("recall_at_10"),
            )
        else:
            logger.info("Skipping semantic search (--skip-semantic)")

        # 2. Citation graph traversal
        logger.info("Benchmark 2: Citation 2-hop traversal...")
        results.append(bench_citation_traversal(conn, iterations=max(iterations // 2, 3)))
        for label in ("typical", "highly_cited"):
            d = results[-1].details.get(label, {})
            logger.info(
                "  %s: p95=%.1fms, results=%s, fan-out=%.1fx",
                label,
                d.get("p95_ms", 0),
                d.get("total_results", 0),
                d.get("fan_out_ratio", 0),
            )

        # 3. Faceted filtering
        logger.info("Benchmark 3: Faceted filtering...")
        results.append(bench_faceted_filtering(conn, iterations=max(iterations * 2, 20)))
        logger.info(
            "  p95=%.1fms, matches=%s",
            results[-1].p95_ms,
            results[-1].details.get("matching_papers"),
        )

        # 4. Combined query
        if not skip_semantic:
            logger.info("Benchmark 4: Combined query...")
            results.append(bench_combined_query(conn, model, tokenizer, iterations=iterations))
            logger.info("  p95=%.1fms", results[-1].p95_ms)
        else:
            logger.info("Skipping combined query (--skip-semantic)")

        # 5. Ingestion metrics
        logger.info("Benchmark 5: Collecting ingestion metrics...")
        ingestion = collect_ingestion_metrics(conn)
        logger.info(
            "  papers=%s, edges=%s, HNSW=%s (%.1f%% of RAM)",
            f"{ingestion.total_papers:,}",
            f"{ingestion.total_citation_edges:,}",
            _fmt_bytes(ingestion.hnsw_index_bytes),
            ingestion.hnsw_pct_of_ram,
        )
    finally:
        conn.close()

    return PRDBenchmarkReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        results=results,
        ingestion=ingestion,
    )
