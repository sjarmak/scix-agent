#!/usr/bin/env python3
"""Coverage bias analysis for the SciX corpus.

Examines the harvested ADS metadata corpus for biases and data quality issues:
- Full-text (body) vs abstract-only distribution across multiple dimensions
- Missing data rates per field
- Citation network completeness (% of referenced papers in corpus)
- Doctype and database (discipline) distribution
- Overall corpus summary statistics

Outputs a markdown report with tables and matplotlib figures.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Add src/ to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psycopg

from scix.db import DEFAULT_DSN, get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributionRow:
    """A single row in a distribution comparison."""

    label: str
    total: int
    with_body: int
    without_body: int
    pct_with_body: float


@dataclass(frozen=True)
class FieldCompletenessRow:
    """Missing data rate for a single column."""

    field: str
    total: int
    non_null: int
    null_count: int
    pct_populated: float


@dataclass(frozen=True)
class CitationCompletenessResult:
    """Citation network completeness statistics."""

    total_edges: int
    edges_target_in_corpus: int
    edges_target_missing: int
    pct_target_in_corpus: float
    unique_targets: int
    unique_targets_in_corpus: int
    unique_targets_missing: int
    pct_unique_in_corpus: float


@dataclass(frozen=True)
class CorpusSummary:
    """Top-level corpus statistics."""

    total_papers: int
    total_with_body: int
    total_citation_edges: int
    total_embeddings: int
    year_min: int | None
    year_max: int | None
    median_citation_count: float | None
    median_reference_count: float | None


# ---------------------------------------------------------------------------
# Distribution query functions
# ---------------------------------------------------------------------------


def get_year_distribution(conn: psycopg.Connection) -> list[DistributionRow]:
    """Get year-wise distribution of full-text vs abstract-only papers."""
    query = """
        SELECT
            year,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE body IS NOT NULL) AS with_body,
            COUNT(*) FILTER (WHERE body IS NULL) AS without_body
        FROM papers
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    return [
        DistributionRow(
            label=str(row[0]),
            total=row[1],
            with_body=row[2],
            without_body=row[3],
            pct_with_body=round(100.0 * row[2] / row[1], 2) if row[1] > 0 else 0.0,
        )
        for row in rows
    ]


def get_arxiv_distribution(conn: psycopg.Connection) -> list[DistributionRow]:
    """Get arxiv_class distribution of full-text vs abstract-only papers.

    Unnests the arxiv_class array so each class is counted independently.
    Returns top 20 classes by total count.
    """
    query = """
        SELECT
            cls,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE body IS NOT NULL) AS with_body,
            COUNT(*) FILTER (WHERE body IS NULL) AS without_body
        FROM papers, LATERAL unnest(arxiv_class) AS cls
        GROUP BY cls
        ORDER BY total DESC
        LIMIT 20
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    return [
        DistributionRow(
            label=str(row[0]),
            total=row[1],
            with_body=row[2],
            without_body=row[3],
            pct_with_body=round(100.0 * row[2] / row[1], 2) if row[1] > 0 else 0.0,
        )
        for row in rows
    ]


def get_citation_distribution(conn: psycopg.Connection) -> list[DistributionRow]:
    """Get citation count distribution of full-text vs abstract-only papers.

    Buckets citations into ranges: 0, 1-5, 6-20, 21-100, 101-500, 500+.
    """
    query = """
        SELECT
            CASE
                WHEN citation_count = 0 THEN '0'
                WHEN citation_count BETWEEN 1 AND 5 THEN '1-5'
                WHEN citation_count BETWEEN 6 AND 20 THEN '6-20'
                WHEN citation_count BETWEEN 21 AND 100 THEN '21-100'
                WHEN citation_count BETWEEN 101 AND 500 THEN '101-500'
                ELSE '500+'
            END AS bucket,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE body IS NOT NULL) AS with_body,
            COUNT(*) FILTER (WHERE body IS NULL) AS without_body
        FROM papers
        WHERE citation_count IS NOT NULL
        GROUP BY bucket
        ORDER BY MIN(citation_count)
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    return [
        DistributionRow(
            label=str(row[0]),
            total=row[1],
            with_body=row[2],
            without_body=row[3],
            pct_with_body=round(100.0 * row[2] / row[1], 2) if row[1] > 0 else 0.0,
        )
        for row in rows
    ]


def get_journal_distribution(conn: psycopg.Connection) -> list[DistributionRow]:
    """Get journal (pub) distribution of full-text vs abstract-only papers.

    Returns top 20 journals by total count.
    """
    query = """
        SELECT
            pub,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE body IS NOT NULL) AS with_body,
            COUNT(*) FILTER (WHERE body IS NULL) AS without_body
        FROM papers
        WHERE pub IS NOT NULL
        GROUP BY pub
        ORDER BY total DESC
        LIMIT 20
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    return [
        DistributionRow(
            label=str(row[0]),
            total=row[1],
            with_body=row[2],
            without_body=row[3],
            pct_with_body=round(100.0 * row[2] / row[1], 2) if row[1] > 0 else 0.0,
        )
        for row in rows
    ]


def get_doctype_distribution(conn: psycopg.Connection) -> list[DistributionRow]:
    """Get doctype distribution of full-text vs abstract-only papers."""
    query = """
        SELECT
            doctype,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE body IS NOT NULL) AS with_body,
            COUNT(*) FILTER (WHERE body IS NULL) AS without_body
        FROM papers
        WHERE doctype IS NOT NULL
        GROUP BY doctype
        ORDER BY total DESC
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    return [
        DistributionRow(
            label=str(row[0]),
            total=row[1],
            with_body=row[2],
            without_body=row[3],
            pct_with_body=round(100.0 * row[2] / row[1], 2) if row[1] > 0 else 0.0,
        )
        for row in rows
    ]


def get_database_distribution(conn: psycopg.Connection) -> list[DistributionRow]:
    """Get database (discipline) distribution of full-text vs abstract-only papers.

    The ADS 'database' array field indicates which databases index the paper
    (e.g. 'astronomy', 'physics', 'general'). A single paper can appear in
    multiple databases, so we unnest and count independently.
    """
    query = """
        SELECT
            db,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE body IS NOT NULL) AS with_body,
            COUNT(*) FILTER (WHERE body IS NULL) AS without_body
        FROM papers, LATERAL unnest(database) AS db
        GROUP BY db
        ORDER BY total DESC
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    return [
        DistributionRow(
            label=str(row[0]),
            total=row[1],
            with_body=row[2],
            without_body=row[3],
            pct_with_body=round(100.0 * row[2] / row[1], 2) if row[1] > 0 else 0.0,
        )
        for row in rows
    ]


def get_field_completeness(conn: psycopg.Connection) -> list[FieldCompletenessRow]:
    """Get missing data rates for key columns in the papers table.

    Checks both scalar columns (IS NULL) and array columns (IS NULL OR empty).
    """
    # Columns to check: (sql_column, is_array)
    fields: list[tuple[str, bool]] = [
        ("title", False),
        ("abstract", False),
        ("body", False),
        ("year", False),
        ("doctype", False),
        ("pub", False),
        ("first_author", False),
        ("citation_count", False),
        ("read_count", False),
        ("reference_count", False),
        ("pubdate", False),
        ("lang", False),
        ("copyright", False),
        ("authors", True),
        ("affiliations", True),
        ("keywords", True),
        ("arxiv_class", True),
        ("database", True),
        ("doi", True),
        ("bibstem", True),
        ("bibgroup", True),
        ("orcid_pub", True),
        ("orcid_user", True),
    ]

    # Build a single query that counts non-null for each field
    select_parts = ["COUNT(*) AS total"]
    for field_name, is_array in fields:
        if is_array:
            select_parts.append(
                f"COUNT(*) FILTER (WHERE {field_name} IS NOT NULL "
                f"AND array_length({field_name}, 1) > 0) AS {field_name}_populated"
            )
        else:
            select_parts.append(
                f"COUNT(*) FILTER (WHERE {field_name} IS NOT NULL) AS {field_name}_populated"
            )

    query = f"SELECT {', '.join(select_parts)} FROM papers"

    with conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()

    if row is None:
        return []

    total = row[0]
    results: list[FieldCompletenessRow] = []
    for i, (field_name, _) in enumerate(fields):
        non_null = row[i + 1]
        null_count = total - non_null
        pct = round(100.0 * non_null / total, 2) if total > 0 else 0.0
        results.append(
            FieldCompletenessRow(
                field=field_name,
                total=total,
                non_null=non_null,
                null_count=null_count,
                pct_populated=pct,
            )
        )

    return results


def get_citation_completeness(
    conn: psycopg.Connection,
    sample_size: int = 100_000,
) -> CitationCompletenessResult:
    """Measure citation network completeness.

    Checks what fraction of cited papers (targets of citation edges) exist in
    the papers table. Uses sampling for efficiency on large tables:
    - Total edge count from pg_class estimate or exact count
    - Random sample of edges to estimate target-in-corpus rate

    Args:
        conn: Database connection.
        sample_size: Number of edges to sample for completeness estimation.
            Set to 0 to force exact (slow) computation.
    """
    # Get total edge count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM citation_edges")
        row = cur.fetchone()
    total_edges = row[0] if row is not None else 0

    if total_edges == 0:
        return CitationCompletenessResult(
            total_edges=0,
            edges_target_in_corpus=0,
            edges_target_missing=0,
            pct_target_in_corpus=0.0,
            unique_targets=0,
            unique_targets_in_corpus=0,
            unique_targets_missing=0,
            pct_unique_in_corpus=0.0,
        )

    # Sample-based estimation: take a random sample of edges and check targets
    # TABLESAMPLE SYSTEM is block-level sampling -- very fast on large tables
    if sample_size > 0 and total_edges > sample_size * 2:
        # Estimate sample percentage to get approximately sample_size rows
        sample_pct = min(100.0 * sample_size / total_edges * 1.2, 100.0)
        with conn.cursor() as cur:
            cur.execute(f"""
                WITH sample AS (
                    SELECT target_bibcode
                    FROM citation_edges TABLESAMPLE SYSTEM ({sample_pct})
                    LIMIT {sample_size}
                )
                SELECT
                    COUNT(*) AS sampled_edges,
                    COUNT(*) FILTER (
                        WHERE EXISTS (SELECT 1 FROM papers p WHERE p.bibcode = sample.target_bibcode)
                    ) AS edges_in_corpus,
                    COUNT(DISTINCT target_bibcode) AS unique_targets,
                    COUNT(DISTINCT target_bibcode) FILTER (
                        WHERE EXISTS (SELECT 1 FROM papers p WHERE p.bibcode = sample.target_bibcode)
                    ) AS unique_in_corpus
                FROM sample
            """)
            row = cur.fetchone()

        if row is None or row[0] == 0:
            return CitationCompletenessResult(
                total_edges=total_edges,
                edges_target_in_corpus=0,
                edges_target_missing=total_edges,
                pct_target_in_corpus=0.0,
                unique_targets=0,
                unique_targets_in_corpus=0,
                unique_targets_missing=0,
                pct_unique_in_corpus=0.0,
            )

        sampled = row[0]
        sampled_in = row[1]
        sampled_unique = row[2]
        sampled_unique_in = row[3]

        # Extrapolate from sample
        pct_edges = round(100.0 * sampled_in / sampled, 2)
        pct_unique = (
            round(100.0 * sampled_unique_in / sampled_unique, 2) if sampled_unique > 0 else 0.0
        )

        edges_in_est = round(total_edges * sampled_in / sampled)
        # Estimate unique targets from sample ratio
        unique_est = round(total_edges * sampled_unique / sampled)
        unique_in_est = (
            round(unique_est * sampled_unique_in / sampled_unique) if sampled_unique > 0 else 0
        )

        return CitationCompletenessResult(
            total_edges=total_edges,
            edges_target_in_corpus=edges_in_est,
            edges_target_missing=total_edges - edges_in_est,
            pct_target_in_corpus=pct_edges,
            unique_targets=unique_est,
            unique_targets_in_corpus=unique_in_est,
            unique_targets_missing=unique_est - unique_in_est,
            pct_unique_in_corpus=pct_unique,
        )

    # Exact computation for small tables or when sample_size=0
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (
                    WHERE EXISTS (SELECT 1 FROM papers p WHERE p.bibcode = ce.target_bibcode)
                ) AS in_corpus,
                COUNT(DISTINCT target_bibcode) AS unique_targets,
                COUNT(DISTINCT target_bibcode) FILTER (
                    WHERE EXISTS (SELECT 1 FROM papers p WHERE p.bibcode = ce.target_bibcode)
                ) AS unique_in_corpus
            FROM citation_edges ce
        """)
        row = cur.fetchone()

    if row is None:
        return CitationCompletenessResult(
            total_edges=total_edges,
            edges_target_in_corpus=0,
            edges_target_missing=total_edges,
            pct_target_in_corpus=0.0,
            unique_targets=0,
            unique_targets_in_corpus=0,
            unique_targets_missing=0,
            pct_unique_in_corpus=0.0,
        )

    return CitationCompletenessResult(
        total_edges=row[0],
        edges_target_in_corpus=row[1],
        edges_target_missing=row[0] - row[1],
        pct_target_in_corpus=round(100.0 * row[1] / row[0], 2) if row[0] > 0 else 0.0,
        unique_targets=row[2],
        unique_targets_in_corpus=row[3],
        unique_targets_missing=row[2] - row[3],
        pct_unique_in_corpus=round(100.0 * row[3] / row[2], 2) if row[2] > 0 else 0.0,
    )


def get_corpus_summary(conn: psycopg.Connection) -> CorpusSummary:
    """Get top-level corpus statistics."""
    query = """
        SELECT
            COUNT(*) AS total_papers,
            COUNT(*) FILTER (WHERE body IS NOT NULL) AS total_with_body,
            MIN(year) AS year_min,
            MAX(year) AS year_max,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY citation_count)
                FILTER (WHERE citation_count IS NOT NULL) AS median_citations,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY reference_count)
                FILTER (WHERE reference_count IS NOT NULL) AS median_references
        FROM papers
    """
    with conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()

    # Count edges and embeddings separately (may be large)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM citation_edges")
        edge_count = cur.fetchone()[0]  # type: ignore[index]

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM paper_embeddings")
        embed_count = cur.fetchone()[0]  # type: ignore[index]

    if row is None:
        return CorpusSummary(
            total_papers=0,
            total_with_body=0,
            total_citation_edges=0,
            total_embeddings=0,
            year_min=None,
            year_max=None,
            median_citation_count=None,
            median_reference_count=None,
        )

    return CorpusSummary(
        total_papers=row[0],
        total_with_body=row[1],
        total_citation_edges=edge_count,
        total_embeddings=embed_count,
        year_min=row[2],
        year_max=row[3],
        median_citation_count=float(row[4]) if row[4] is not None else None,
        median_reference_count=float(row[5]) if row[5] is not None else None,
    )


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def _get_plt():
    """Lazily import matplotlib.pyplot with Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _make_bar_chart(
    rows: Sequence[DistributionRow],
    title: str,
    xlabel: str,
    filename: str,
    figures_dir: Path,
    rotate_labels: bool = False,
) -> Path:
    """Create a grouped bar chart comparing full-text vs abstract-only counts."""
    labels = [r.label for r in rows]
    with_body = [r.with_body for r in rows]
    without_body = [r.without_body for r in rows]

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(labels))
    width = 0.35

    ax.bar([i - width / 2 for i in x], with_body, width, label="Full text", color="#2196F3")
    ax.bar([i + width / 2 for i in x], without_body, width, label="Abstract only", color="#FF9800")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Paper count")
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        labels, rotation=45 if rotate_labels else 0, ha="right" if rotate_labels else "center"
    )
    ax.legend()
    fig.tight_layout()

    filepath = figures_dir / filename
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", filepath)
    return filepath


def _make_pct_chart(
    rows: Sequence[DistributionRow],
    title: str,
    xlabel: str,
    filename: str,
    figures_dir: Path,
    rotate_labels: bool = False,
) -> Path:
    """Create a bar chart showing % with full text per category."""
    labels = [r.label for r in rows]
    pcts = [r.pct_with_body for r in rows]

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, pcts, color="#4CAF50")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% with full text")
    ax.set_title(title)
    if rotate_labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(y=sum(pcts) / len(pcts) if pcts else 0, color="red", linestyle="--", label="Average")
    ax.legend()
    fig.tight_layout()

    filepath = figures_dir / filename
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", filepath)
    return filepath


def _make_horizontal_bar(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    filename: str,
    figures_dir: Path,
    color: str = "#4CAF50",
    reference_line: float | None = None,
) -> Path:
    """Create a horizontal bar chart (useful for field completeness)."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))
    y_pos = range(len(labels))
    ax.barh(list(y_pos), list(values), color=color)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if reference_line is not None:
        ax.axvline(x=reference_line, color="red", linestyle="--", label=f"{reference_line}%")
        ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()

    filepath = figures_dir / filename
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", filepath)
    return filepath


def generate_figures(
    year_dist: Sequence[DistributionRow],
    arxiv_dist: Sequence[DistributionRow],
    citation_dist: Sequence[DistributionRow],
    journal_dist: Sequence[DistributionRow],
    figures_dir: Path,
    doctype_dist: Sequence[DistributionRow] | None = None,
    database_dist: Sequence[DistributionRow] | None = None,
    field_completeness: Sequence[FieldCompletenessRow] | None = None,
) -> dict[str, Path]:
    """Generate all charts and return a mapping of name to file path."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if year_dist:
        paths["year_counts"] = _make_bar_chart(
            year_dist,
            "Full Text vs Abstract Only by Year",
            "Year",
            "year_distribution.png",
            figures_dir,
        )
        paths["year_pct"] = _make_pct_chart(
            year_dist,
            "% Full Text Coverage by Year",
            "Year",
            "year_pct.png",
            figures_dir,
        )

    if arxiv_dist:
        paths["arxiv_counts"] = _make_bar_chart(
            arxiv_dist,
            "Full Text vs Abstract Only by arXiv Class (Top 20)",
            "arXiv Class",
            "arxiv_distribution.png",
            figures_dir,
            rotate_labels=True,
        )
        paths["arxiv_pct"] = _make_pct_chart(
            arxiv_dist,
            "% Full Text Coverage by arXiv Class (Top 20)",
            "arXiv Class",
            "arxiv_pct.png",
            figures_dir,
            rotate_labels=True,
        )

    if citation_dist:
        paths["citation_counts"] = _make_bar_chart(
            citation_dist,
            "Full Text vs Abstract Only by Citation Count",
            "Citation Bucket",
            "citation_distribution.png",
            figures_dir,
        )
        paths["citation_pct"] = _make_pct_chart(
            citation_dist,
            "% Full Text Coverage by Citation Count",
            "Citation Bucket",
            "citation_pct.png",
            figures_dir,
        )

    if journal_dist:
        paths["journal_counts"] = _make_bar_chart(
            journal_dist,
            "Full Text vs Abstract Only by Journal (Top 20)",
            "Journal",
            "journal_distribution.png",
            figures_dir,
            rotate_labels=True,
        )
        paths["journal_pct"] = _make_pct_chart(
            journal_dist,
            "% Full Text Coverage by Journal (Top 20)",
            "Journal",
            "journal_pct.png",
            figures_dir,
            rotate_labels=True,
        )

    if doctype_dist:
        paths["doctype_counts"] = _make_bar_chart(
            doctype_dist,
            "Full Text vs Abstract Only by Document Type",
            "Document Type",
            "doctype_distribution.png",
            figures_dir,
            rotate_labels=True,
        )
        paths["doctype_pct"] = _make_pct_chart(
            doctype_dist,
            "% Full Text Coverage by Document Type",
            "Document Type",
            "doctype_pct.png",
            figures_dir,
            rotate_labels=True,
        )

    if database_dist:
        paths["database_counts"] = _make_bar_chart(
            database_dist,
            "Full Text vs Abstract Only by Database (Discipline)",
            "Database",
            "database_distribution.png",
            figures_dir,
        )
        paths["database_pct"] = _make_pct_chart(
            database_dist,
            "% Full Text Coverage by Database (Discipline)",
            "Database",
            "database_pct.png",
            figures_dir,
        )

    if field_completeness:
        paths["field_completeness"] = _make_horizontal_bar(
            labels=[r.field for r in field_completeness],
            values=[r.pct_populated for r in field_completeness],
            title="Field Completeness (% Populated)",
            xlabel="% of papers with non-null value",
            filename="field_completeness.png",
            figures_dir=figures_dir,
            reference_line=80.0,
        )

    return paths


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _format_table(rows: Sequence[DistributionRow], label_header: str) -> str:
    """Format a list of DistributionRows as a markdown table."""
    lines = [
        f"| {label_header} | Total | Full Text | Abstract Only | % Full Text |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.label} | {r.total:,} | {r.with_body:,} | {r.without_body:,} | {r.pct_with_body:.1f}% |"
        )
    return "\n".join(lines)


def _format_completeness_table(rows: Sequence[FieldCompletenessRow]) -> str:
    """Format field completeness rows as a markdown table."""
    lines = [
        "| Field | Total | Populated | Missing | % Populated |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.field} | {r.total:,} | {r.non_null:,} | {r.null_count:,} | {r.pct_populated:.1f}% |"
        )
    return "\n".join(lines)


def generate_report(
    year_dist: Sequence[DistributionRow],
    arxiv_dist: Sequence[DistributionRow],
    citation_dist: Sequence[DistributionRow],
    journal_dist: Sequence[DistributionRow],
    output_path: Path | None = None,
    figures_dir: Path | None = None,
    doctype_dist: Sequence[DistributionRow] | None = None,
    database_dist: Sequence[DistributionRow] | None = None,
    field_completeness: Sequence[FieldCompletenessRow] | None = None,
    citation_completeness: CitationCompletenessResult | None = None,
    corpus_summary: CorpusSummary | None = None,
) -> str:
    """Generate a markdown report with corpus-wide coverage and bias analysis.

    Args:
        year_dist: Year-wise distribution data.
        arxiv_dist: arXiv class distribution data.
        citation_dist: Citation count bucket distribution data.
        journal_dist: Journal distribution data.
        output_path: If provided, write the report to this file.
        figures_dir: If provided, generate charts in this directory.
        doctype_dist: Document type distribution data.
        database_dist: Database (discipline) distribution data.
        field_completeness: Missing data rates per field.
        citation_completeness: Citation network completeness stats.
        corpus_summary: Top-level corpus statistics.

    Returns:
        The markdown report as a string.
    """
    sections: list[str] = []

    sections.append("# SciX Corpus Coverage Bias Analysis")
    sections.append("")
    sections.append(
        "This report analyzes the SciX corpus for coverage biases, data quality issues, "
        "and completeness across multiple dimensions. It compares the distribution of "
        "papers with full text (body IS NOT NULL) vs abstract-only papers and examines "
        "field-level data quality."
    )

    # Corpus summary
    sections.append("")
    sections.append("## Corpus Summary")
    sections.append("")
    if corpus_summary is not None:
        pct_body = (
            round(100.0 * corpus_summary.total_with_body / corpus_summary.total_papers, 1)
            if corpus_summary.total_papers > 0
            else 0.0
        )
        sections.append(f"- **Total papers**: {corpus_summary.total_papers:,}")
        sections.append(f"- **With full text**: {corpus_summary.total_with_body:,} ({pct_body}%)")
        sections.append(f"- **Citation edges**: {corpus_summary.total_citation_edges:,}")
        sections.append(f"- **Embeddings**: {corpus_summary.total_embeddings:,}")
        if corpus_summary.year_min is not None and corpus_summary.year_max is not None:
            sections.append(
                f"- **Year range**: {corpus_summary.year_min} -- {corpus_summary.year_max}"
            )
        if corpus_summary.median_citation_count is not None:
            sections.append(
                f"- **Median citation count**: {corpus_summary.median_citation_count:.0f}"
            )
        if corpus_summary.median_reference_count is not None:
            sections.append(
                f"- **Median reference count**: {corpus_summary.median_reference_count:.0f}"
            )
    else:
        # Fallback: derive from year_dist
        total_papers = sum(r.total for r in year_dist) if year_dist else 0
        total_with_body = sum(r.with_body for r in year_dist) if year_dist else 0
        overall_pct = round(100.0 * total_with_body / total_papers, 1) if total_papers > 0 else 0.0
        sections.append(f"- **Total papers**: {total_papers:,}")
        sections.append(f"- **With full text**: {total_with_body:,} ({overall_pct}%)")
        sections.append(
            f"- **Abstract only**: {total_papers - total_with_body:,} "
            f"({100.0 - overall_pct:.1f}%)"
        )

    # Field completeness
    if field_completeness:
        sections.append("")
        sections.append("## Field Completeness (Missing Data Rates)")
        sections.append("")
        sections.append(
            "Shows what percentage of papers have non-null (and non-empty for arrays) "
            "values for each field."
        )
        sections.append("")
        sections.append(_format_completeness_table(field_completeness))
        if figures_dir:
            sections.append("")
            sections.append("![Field Completeness](figures/field_completeness.png)")

    # Citation network completeness
    if citation_completeness is not None:
        sections.append("")
        sections.append("## Citation Network Completeness")
        sections.append("")
        sections.append(
            "Measures what fraction of cited papers (targets of citation edges) "
            "exist in the corpus."
        )
        sections.append("")
        cc = citation_completeness
        sections.append(f"- **Total citation edges**: {cc.total_edges:,}")
        sections.append(
            f"- **Edges with target in corpus**: {cc.edges_target_in_corpus:,} "
            f"({cc.pct_target_in_corpus:.1f}%)"
        )
        sections.append(
            f"- **Edges with target missing**: {cc.edges_target_missing:,} "
            f"({100.0 - cc.pct_target_in_corpus:.1f}%)"
        )
        sections.append(f"- **Unique cited papers**: {cc.unique_targets:,}")
        sections.append(
            f"- **Unique cited in corpus**: {cc.unique_targets_in_corpus:,} "
            f"({cc.pct_unique_in_corpus:.1f}%)"
        )
        sections.append(
            f"- **Unique cited missing**: {cc.unique_targets_missing:,} "
            f"({100.0 - cc.pct_unique_in_corpus:.1f}%)"
        )

    # Year distribution
    sections.append("")
    sections.append("## Year Distribution")
    sections.append("")
    sections.append(_format_table(year_dist, "Year"))
    if figures_dir:
        sections.append("")
        sections.append("![Year Distribution](figures/year_distribution.png)")
        sections.append("")
        sections.append("![Year Coverage %](figures/year_pct.png)")

    # arXiv class distribution
    sections.append("")
    sections.append("## arXiv Class Distribution (Top 20)")
    sections.append("")
    sections.append(_format_table(arxiv_dist, "arXiv Class"))
    if figures_dir:
        sections.append("")
        sections.append("![arXiv Distribution](figures/arxiv_distribution.png)")
        sections.append("")
        sections.append("![arXiv Coverage %](figures/arxiv_pct.png)")

    # Doctype distribution
    if doctype_dist:
        sections.append("")
        sections.append("## Document Type Distribution")
        sections.append("")
        sections.append(_format_table(doctype_dist, "Doctype"))
        if figures_dir:
            sections.append("")
            sections.append("![Doctype Distribution](figures/doctype_distribution.png)")
            sections.append("")
            sections.append("![Doctype Coverage %](figures/doctype_pct.png)")

    # Database (discipline) distribution
    if database_dist:
        sections.append("")
        sections.append("## Database (Discipline) Distribution")
        sections.append("")
        sections.append(
            "The ADS `database` field indicates which databases index a paper "
            "(e.g. astronomy, physics, general). Papers may appear in multiple databases."
        )
        sections.append("")
        sections.append(_format_table(database_dist, "Database"))
        if figures_dir:
            sections.append("")
            sections.append("![Database Distribution](figures/database_distribution.png)")
            sections.append("")
            sections.append("![Database Coverage %](figures/database_pct.png)")

    # Citation count distribution
    sections.append("")
    sections.append("## Citation Count Distribution")
    sections.append("")
    sections.append(_format_table(citation_dist, "Citations"))
    if figures_dir:
        sections.append("")
        sections.append("![Citation Distribution](figures/citation_distribution.png)")
        sections.append("")
        sections.append("![Citation Coverage %](figures/citation_pct.png)")

    # Journal distribution
    sections.append("")
    sections.append("## Journal Distribution (Top 20)")
    sections.append("")
    sections.append(_format_table(journal_dist, "Journal"))
    if figures_dir:
        sections.append("")
        sections.append("![Journal Distribution](figures/journal_distribution.png)")
        sections.append("")
        sections.append("![Journal Coverage %](figures/journal_pct.png)")

    report = "\n".join(sections) + "\n"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        logger.info("Report written to %s", output_path)

    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    dsn: str | None = None,
    output_dir: Path | None = None,
) -> str:
    """Run the full coverage bias analysis pipeline.

    Args:
        dsn: PostgreSQL DSN. Defaults to SCIX_DSN env var or 'dbname=scix'.
        output_dir: Base directory for output. Defaults to project docs/.

    Returns:
        The markdown report as a string.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "docs"

    figures_dir = output_dir / "figures"
    report_path = output_dir / "full_text_coverage_analysis.md"

    conn = get_connection(dsn)
    try:
        logger.info("Querying corpus summary...")
        corpus_summary = get_corpus_summary(conn)

        logger.info("Querying year distribution...")
        year_dist = get_year_distribution(conn)

        logger.info("Querying arXiv class distribution...")
        arxiv_dist = get_arxiv_distribution(conn)

        logger.info("Querying citation count distribution...")
        citation_dist = get_citation_distribution(conn)

        logger.info("Querying journal distribution...")
        journal_dist = get_journal_distribution(conn)

        logger.info("Querying doctype distribution...")
        doctype_dist = get_doctype_distribution(conn)

        logger.info("Querying database (discipline) distribution...")
        database_dist = get_database_distribution(conn)

        logger.info("Querying field completeness...")
        field_completeness = get_field_completeness(conn)

        logger.info("Querying citation network completeness...")
        citation_completeness = get_citation_completeness(conn)
    finally:
        conn.close()

    logger.info("Generating figures...")
    generate_figures(
        year_dist,
        arxiv_dist,
        citation_dist,
        journal_dist,
        figures_dir,
        doctype_dist=doctype_dist,
        database_dist=database_dist,
        field_completeness=field_completeness,
    )

    logger.info("Generating report...")
    report = generate_report(
        year_dist,
        arxiv_dist,
        citation_dist,
        journal_dist,
        output_path=report_path,
        figures_dir=figures_dir,
        doctype_dist=doctype_dist,
        database_dist=database_dist,
        field_completeness=field_completeness,
        citation_completeness=citation_completeness,
        corpus_summary=corpus_summary,
    )

    logger.info("Analysis complete.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze full-text coverage bias in the SciX corpus"
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for report and figures (default: docs/)",
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

    run_analysis(dsn=args.dsn, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
