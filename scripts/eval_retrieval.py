#!/usr/bin/env python3
"""50-query retrieval evaluation: SPECTER2 vs BM25 vs Hybrid RRF.

Uses citation-based ground truth: for each query paper, its direct
citation neighborhood (references + citers) forms the relevant set.
Query text is the paper's title (standard approach in scientific IR eval,
cf. SPECTER, SciRepEval).

Systems compared:
  1. BM25 (lexical_search) — full 32M-paper corpus
  2. SPECTER2 (vector_search) — embedded paper subset via HNSW
  3. Hybrid RRF (BM25 + SPECTER2 fused via Reciprocal Rank Fusion, k=60)

Produces paper-ready markdown tables for Section 4.4 of the ADASS paper.

Usage:
    python3 scripts/eval_retrieval.py
    python3 scripts/eval_retrieval.py --output results/eval_retrieval.md
    python3 scripts/eval_retrieval.py --num-queries 50 --limit 20 -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg
from psycopg.rows import dict_row

from scix.db import get_connection
from scix.embed import embed_batch, load_model
from scix.ir_metrics import (
    EvalReport,
    RetrievalScore,
    aggregate_scores,
    compute_retrieval_score,
)
from scix.search import (
    SearchResult,
    hybrid_search,
    lexical_search,
    rrf_fuse,
    vector_search,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query selection: diverse, well-cited papers from the embedded set
# ---------------------------------------------------------------------------

# SQL to select query papers: well-cited, with abstracts, from diverse fields.
# Stratifies by decade to ensure temporal diversity.
_SELECT_QUERY_PAPERS_SQL = """
WITH embedded_bibs AS (
    SELECT bibcode FROM paper_embeddings WHERE model_name = 'specter2'
),
intra_cites AS (
    SELECT ce.source_bibcode AS bibcode, count(*) AS intra_cite_count
    FROM citation_edges ce
    JOIN embedded_bibs e1 ON ce.source_bibcode = e1.bibcode
    JOIN embedded_bibs e2 ON ce.target_bibcode = e2.bibcode
    GROUP BY ce.source_bibcode
    HAVING count(*) >= 3
),
candidates AS (
    SELECT p.bibcode, p.title, p.abstract, p.year, p.citation_count,
           p.arxiv_class, p.first_author,
           ic.intra_cite_count,
           (p.year / 10) * 10 AS decade,
           row_number() OVER (
               PARTITION BY (p.year / 10) * 10
               ORDER BY p.citation_count DESC
           ) AS rank_in_decade
    FROM intra_cites ic
    JOIN papers p ON p.bibcode = ic.bibcode
    WHERE p.abstract IS NOT NULL AND p.abstract != ''
      AND p.title IS NOT NULL
      AND p.citation_count > 20
)
SELECT bibcode, title, abstract, year, citation_count, arxiv_class,
       first_author, intra_cite_count, decade
FROM candidates
WHERE rank_in_decade <= %(per_decade)s
ORDER BY year, citation_count DESC
LIMIT %(total)s
"""


@dataclass(frozen=True)
class QueryPaper:
    """A paper selected as an evaluation query."""

    bibcode: str
    title: str
    abstract: str
    year: int
    citation_count: int
    arxiv_class: list[str]
    first_author: str
    intra_cite_count: int


def select_query_papers(
    conn: psycopg.Connection,
    num_queries: int = 50,
) -> list[QueryPaper]:
    """Select diverse, well-cited papers for evaluation queries."""
    # Allow generous per-decade allocation, then trim to target
    per_decade = max(num_queries // 5, 15)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            _SELECT_QUERY_PAPERS_SQL,
            {"per_decade": per_decade, "total": num_queries * 3},
        )
        rows = cur.fetchall()

    if not rows:
        logger.error("No suitable query papers found — is the embedding index populated?")
        return []

    # Diversify: greedily pick papers maximizing field coverage
    selected: list[QueryPaper] = []
    seen_fields: dict[str, int] = {}

    for row in rows:
        classes = row["arxiv_class"] or []
        # Score: prefer papers from underrepresented fields
        field_score = sum(1.0 / (seen_fields.get(c, 0) + 1) for c in classes) if classes else 0.5
        row["_diversity_score"] = field_score

    # Sort by diversity score descending, then citation count
    rows.sort(key=lambda r: (-r["_diversity_score"], -r["citation_count"]))

    for row in rows:
        if len(selected) >= num_queries:
            break

        paper = QueryPaper(
            bibcode=row["bibcode"],
            title=row["title"],
            abstract=row["abstract"],
            year=row["year"],
            citation_count=row["citation_count"],
            arxiv_class=row["arxiv_class"] or [],
            first_author=row["first_author"] or "",
            intra_cite_count=row["intra_cite_count"],
        )
        selected.append(paper)

        for c in paper.arxiv_class:
            seen_fields[c] = seen_fields.get(c, 0) + 1

    logger.info(
        "Selected %d query papers (years %d–%d, median citations: %d)",
        len(selected),
        min(p.year for p in selected) if selected else 0,
        max(p.year for p in selected) if selected else 0,
        sorted(p.citation_count for p in selected)[len(selected) // 2] if selected else 0,
    )
    return selected


# ---------------------------------------------------------------------------
# Ground truth: citation neighborhood
# ---------------------------------------------------------------------------


def get_citation_neighborhood(
    conn: psycopg.Connection,
    bibcode: str,
) -> dict[str, float]:
    """Get citation-based relevance map for a query paper.

    Returns dict of bibcode -> relevance grade:
      - Grade 2.0: Direct citations (papers that cite this paper or that this paper cites)
      - The query paper itself is excluded from the relevant set.
    """
    with conn.cursor() as cur:
        # Papers that cite this paper (incoming citations)
        cur.execute(
            "SELECT source_bibcode FROM citation_edges WHERE target_bibcode = %s",
            (bibcode,),
        )
        citers = {row[0] for row in cur.fetchall()}

        # Papers this paper cites (outgoing references)
        cur.execute(
            "SELECT target_bibcode FROM citation_edges WHERE source_bibcode = %s",
            (bibcode,),
        )
        references = {row[0] for row in cur.fetchall()}

    # All direct citation neighbors get grade 2.0
    neighborhood: dict[str, float] = {}
    for bib in citers | references:
        if bib != bibcode:
            neighborhood[bib] = 2.0

    return neighborhood


# ---------------------------------------------------------------------------
# Run retrieval for each system
# ---------------------------------------------------------------------------


def run_bm25(
    conn: psycopg.Connection,
    query_text: str,
    limit: int = 20,
) -> tuple[list[str], float]:
    """Run BM25 lexical search, return (bibcode_list, latency_ms)."""
    result = lexical_search(conn, query_text, limit=limit)
    bibcodes = [p["bibcode"] for p in result.papers]
    latency = result.timing_ms.get("lexical_ms", 0.0)
    return bibcodes, latency


def run_specter2(
    conn: psycopg.Connection,
    query_embedding: list[float],
    limit: int = 20,
) -> tuple[list[str], float]:
    """Run SPECTER2 vector search, return (bibcode_list, latency_ms)."""
    result = vector_search(
        conn, query_embedding, model_name="specter2", limit=limit
    )
    bibcodes = [p["bibcode"] for p in result.papers]
    latency = result.timing_ms.get("vector_ms", 0.0)
    return bibcodes, latency


def run_hybrid_rrf(
    conn: psycopg.Connection,
    query_text: str,
    query_embedding: list[float],
    limit: int = 20,
) -> tuple[list[str], float]:
    """Run hybrid BM25 + SPECTER2 via RRF, return (bibcode_list, latency_ms)."""
    result = hybrid_search(
        conn,
        query_text,
        query_embedding,
        model_name="specter2",
        top_n=limit,
    )
    bibcodes = [p["bibcode"] for p in result.papers]
    latency = result.timing_ms.get("total_ms", 0.0)
    return bibcodes, latency


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    conn: psycopg.Connection,
    query_papers: list[QueryPaper],
    model: Any,
    tokenizer: Any,
    *,
    limit: int = 20,
) -> dict[str, EvalReport]:
    """Run the full evaluation across all query papers and systems.

    Returns dict of system_name -> EvalReport.
    """
    systems = ("bm25", "specter2", "hybrid_rrf")
    scores: dict[str, list[RetrievalScore]] = {s: [] for s in systems}

    for i, qp in enumerate(query_papers):
        query_id = qp.bibcode
        query_text = qp.title
        logger.info(
            "[%d/%d] %s — %s (cites=%d)",
            i + 1, len(query_papers), qp.bibcode, qp.title[:60], qp.citation_count,
        )

        # Ground truth
        relevance_map = get_citation_neighborhood(conn, qp.bibcode)
        if not relevance_map:
            logger.warning("  Skipping — no citation neighbors found")
            continue

        # Embed query with SPECTER2
        embedding = embed_batch(model, tokenizer, [query_text])[0]

        # Run each system
        bm25_ids, bm25_ms = run_bm25(conn, query_text, limit=limit)
        spec_ids, spec_ms = run_specter2(conn, embedding, limit=limit)
        hybrid_ids, hybrid_ms = run_hybrid_rrf(conn, query_text, embedding, limit=limit)

        # Exclude the query paper itself from retrieved results
        bm25_ids = [b for b in bm25_ids if b != qp.bibcode]
        spec_ids = [b for b in spec_ids if b != qp.bibcode]
        hybrid_ids = [b for b in hybrid_ids if b != qp.bibcode]

        # Compute scores
        scores["bm25"].append(
            compute_retrieval_score(query_id, "bm25", bm25_ids, relevance_map, bm25_ms)
        )
        scores["specter2"].append(
            compute_retrieval_score(query_id, "specter2", spec_ids, relevance_map, spec_ms)
        )
        scores["hybrid_rrf"].append(
            compute_retrieval_score(query_id, "hybrid_rrf", hybrid_ids, relevance_map, hybrid_ms)
        )

        logger.info(
            "  BM25: nDCG=%.3f R@10=%.3f | SPECTER2: nDCG=%.3f R@10=%.3f | "
            "Hybrid: nDCG=%.3f R@10=%.3f | relevant=%d",
            scores["bm25"][-1].ndcg_at_10, scores["bm25"][-1].recall_at_10,
            scores["specter2"][-1].ndcg_at_10, scores["specter2"][-1].recall_at_10,
            scores["hybrid_rrf"][-1].ndcg_at_10, scores["hybrid_rrf"][-1].recall_at_10,
            len(relevance_map),
        )

    return {name: aggregate_scores(name, score_list) for name, score_list in scores.items()}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_paper_table(reports: dict[str, EvalReport]) -> str:
    """Generate markdown tables suitable for Section 4.4 of the ADASS paper."""
    lines = [
        "# Retrieval Evaluation Results",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Queries**: {next(iter(reports.values())).num_queries}",
        "**Ground truth**: Citation neighborhood (direct references + citers)",
        "**Query type**: Paper title",
        "",
        "## System Comparison",
        "",
        "| System | nDCG@10 | Recall@10 | Recall@20 | P@10 | MRR | Latency (ms) |",
        "|--------|---------|-----------|-----------|------|-----|-------------|",
    ]

    for name in ("bm25", "specter2", "hybrid_rrf"):
        r = reports[name]
        label = {"bm25": "BM25 (lexical)", "specter2": "SPECTER2 (dense)", "hybrid_rrf": "Hybrid RRF"}[name]
        lines.append(
            f"| {label} | {r.mean_ndcg_at_10:.4f} | {r.mean_recall_at_10:.4f} "
            f"| {r.mean_recall_at_20:.4f} | {r.mean_precision_at_10:.4f} "
            f"| {r.mean_mrr:.4f} | {r.mean_latency_ms:.1f} |"
        )

    # Improvement analysis
    if "hybrid_rrf" in reports and "bm25" in reports and "specter2" in reports:
        hybrid = reports["hybrid_rrf"]
        bm25 = reports["bm25"]
        spec = reports["specter2"]

        best_single_ndcg = max(bm25.mean_ndcg_at_10, spec.mean_ndcg_at_10)
        best_single_recall = max(bm25.mean_recall_at_10, spec.mean_recall_at_10)

        if best_single_ndcg > 0:
            ndcg_lift = (hybrid.mean_ndcg_at_10 - best_single_ndcg) / best_single_ndcg * 100
        else:
            ndcg_lift = 0.0

        if best_single_recall > 0:
            recall_lift = (hybrid.mean_recall_at_10 - best_single_recall) / best_single_recall * 100
        else:
            recall_lift = 0.0

        lines.extend([
            "",
            "## Hybrid RRF Improvement over Best Single System",
            "",
            f"- nDCG@10: {ndcg_lift:+.1f}%",
            f"- Recall@10: {recall_lift:+.1f}%",
            "",
            "RRF fusion combines the precision of dense retrieval (SPECTER2) with",
            "the broad coverage of lexical retrieval (BM25). Papers in the literature",
            "report 49–67% error reduction with hybrid approaches (cf. Section 4.4).",
        ])

    # Per-query breakdown
    lines.extend([
        "",
        "## Per-Query Results (top 10 by nDCG improvement from fusion)",
        "",
        "| Year | Bibcode | nDCG(BM25) | nDCG(S2) | nDCG(RRF) | Relevant |",
        "|------|---------|------------|----------|-----------|----------|",
    ])

    # Find queries where hybrid helped most
    hybrid_scores = {s.query_id: s for s in reports["hybrid_rrf"].per_query}
    bm25_scores = {s.query_id: s for s in reports["bm25"].per_query}
    spec_scores = {s.query_id: s for s in reports["specter2"].per_query}

    improvements = []
    for qid in hybrid_scores:
        h = hybrid_scores[qid].ndcg_at_10
        best_single = max(
            bm25_scores.get(qid, RetrievalScore("", "", 0, 0, 0, 0, 0, 0, 0, 0)).ndcg_at_10,
            spec_scores.get(qid, RetrievalScore("", "", 0, 0, 0, 0, 0, 0, 0, 0)).ndcg_at_10,
        )
        improvements.append((qid, h - best_single))

    improvements.sort(key=lambda x: -x[1])

    for qid, _ in improvements[:10]:
        h = hybrid_scores[qid]
        b = bm25_scores.get(qid)
        s = spec_scores.get(qid)
        # Extract year from bibcode (format: YYYY...)
        year = qid[:4] if len(qid) >= 4 else "?"
        lines.append(
            f"| {year} | `{qid}` | {b.ndcg_at_10:.3f} | {s.ndcg_at_10:.3f} "
            f"| {h.ndcg_at_10:.3f} | {h.num_relevant} |"
        )

    # Methodology note
    lines.extend([
        "",
        "## Methodology",
        "",
        "- **Query selection**: 50 well-cited papers from the SPECTER2-embedded subset,",
        "  stratified by decade and diversified across arXiv classes.",
        "- **Ground truth**: Direct citation neighborhood — papers that cite the query",
        "  paper or are cited by it. All neighbors receive relevance grade 2.0 (binary).",
        "- **Metrics**: Standard IR metrics (nDCG@10, Recall@K, Precision@10, MRR).",
        "- **RRF constant**: k=60 (standard setting).",
        "- **Note**: SPECTER2 retrieves from the ~20K embedded subset; BM25 searches",
        "  the full 32M-paper corpus. Hybrid RRF fuses both ranked lists.",
    ])

    return "\n".join(lines)


def save_raw_results(
    reports: dict[str, EvalReport],
    query_papers: list[QueryPaper],
    output_path: Path,
) -> None:
    """Save raw per-query results as JSONL for reproducibility."""
    jsonl_path = output_path.with_suffix(".jsonl")
    with open(jsonl_path, "w") as f:
        for name, report in reports.items():
            for score in report.per_query:
                # Find matching query paper
                qp = next((q for q in query_papers if q.bibcode == score.query_id), None)
                record = {
                    "system": name,
                    "query_id": score.query_id,
                    "query_title": qp.title if qp else "",
                    "query_year": qp.year if qp else 0,
                    "ndcg_at_10": score.ndcg_at_10,
                    "recall_at_10": score.recall_at_10,
                    "recall_at_20": score.recall_at_20,
                    "precision_at_10": score.precision_at_10,
                    "mrr": score.mrr,
                    "latency_ms": score.latency_ms,
                    "num_retrieved": score.num_retrieved,
                    "num_relevant": score.num_relevant,
                }
                f.write(json.dumps(record) + "\n")

    logger.info("Raw results saved to %s", jsonl_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="50-query retrieval evaluation: SPECTER2 vs BM25 vs Hybrid RRF"
    )
    parser.add_argument(
        "--output", "-o",
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
        "--num-queries", "-n",
        type=int,
        default=50,
        help="Number of evaluation queries (default: 50)",
    )
    parser.add_argument(
        "--limit", "-k",
        type=int,
        default=20,
        help="Number of results retrieved per system (default: 20)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for SPECTER2 model (default: cpu)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load SPECTER2 model
    logger.info("Loading SPECTER2 model on %s...", args.device)
    model, tokenizer = load_model("specter2", device=args.device)

    conn = get_connection(args.dsn)
    try:
        # Select query papers
        query_papers = select_query_papers(conn, num_queries=args.num_queries)
        if not query_papers:
            logger.error("No query papers found — aborting")
            sys.exit(1)

        # Run evaluation
        reports = run_evaluation(
            conn, query_papers, model, tokenizer, limit=args.limit,
        )

        # Generate report
        report_text = generate_paper_table(reports)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text)
            logger.info("Report written to %s", output_path)

            # Also save raw JSONL
            save_raw_results(reports, query_papers, output_path)
        else:
            print(report_text)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
