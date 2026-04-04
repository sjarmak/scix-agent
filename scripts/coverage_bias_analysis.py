#!/usr/bin/env python3
"""Coverage bias analysis: compare full-text vs abstract-only paper distributions.

Queries the DB to compare distributions between full-text (body IS NOT NULL)
and abstract-only papers across: arxiv_class, year, citation_count, and journal (pub).
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


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


def generate_figures(
    year_dist: Sequence[DistributionRow],
    arxiv_dist: Sequence[DistributionRow],
    citation_dist: Sequence[DistributionRow],
    journal_dist: Sequence[DistributionRow],
    figures_dir: Path,
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


def generate_report(
    year_dist: Sequence[DistributionRow],
    arxiv_dist: Sequence[DistributionRow],
    citation_dist: Sequence[DistributionRow],
    journal_dist: Sequence[DistributionRow],
    output_path: Path | None = None,
    figures_dir: Path | None = None,
) -> str:
    """Generate a markdown report comparing full-text and abstract-only distributions.

    Args:
        year_dist: Year-wise distribution data.
        arxiv_dist: arXiv class distribution data.
        citation_dist: Citation count bucket distribution data.
        journal_dist: Journal distribution data.
        output_path: If provided, write the report to this file.
        figures_dir: If provided, generate charts in this directory.

    Returns:
        The markdown report as a string.
    """
    sections: list[str] = []

    sections.append("# Full-Text Coverage Bias Analysis")
    sections.append("")
    sections.append(
        "This report compares the distribution of papers with full text "
        "(body IS NOT NULL) vs abstract-only papers across multiple dimensions."
    )

    # Summary stats
    total_papers = sum(r.total for r in year_dist) if year_dist else 0
    total_with_body = sum(r.with_body for r in year_dist) if year_dist else 0
    overall_pct = round(100.0 * total_with_body / total_papers, 1) if total_papers > 0 else 0.0

    sections.append("")
    sections.append("## Summary")
    sections.append("")
    sections.append(f"- **Total papers**: {total_papers:,}")
    sections.append(f"- **With full text**: {total_with_body:,} ({overall_pct}%)")
    sections.append(
        f"- **Abstract only**: {total_papers - total_with_body:,} ({100.0 - overall_pct:.1f}%)"
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
        logger.info("Querying year distribution...")
        year_dist = get_year_distribution(conn)

        logger.info("Querying arXiv class distribution...")
        arxiv_dist = get_arxiv_distribution(conn)

        logger.info("Querying citation count distribution...")
        citation_dist = get_citation_distribution(conn)

        logger.info("Querying journal distribution...")
        journal_dist = get_journal_distribution(conn)
    finally:
        conn.close()

    logger.info("Generating figures...")
    generate_figures(year_dist, arxiv_dist, citation_dist, journal_dist, figures_dir)

    logger.info("Generating report...")
    report = generate_report(
        year_dist,
        arxiv_dist,
        citation_dist,
        journal_dist,
        output_path=report_path,
        figures_dir=figures_dir,
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
