#!/usr/bin/env python3
"""Cross-validate ADS citation graph against OpenAlex and produce a report.

Two modes:
  --populate   Populate the citation_diff table from citation_edges + works_references.
  --report     Generate a JSON/markdown summary from the populated citation_diff table.

Both modes accept --dsn for the database connection and --dry-run for safety.

Usage:
    # Populate citation_diff from the two source tables
    python scripts/analyze_citation_diff.py --populate --dsn "dbname=scix"

    # Dry run — show the INSERT query without executing
    python scripts/analyze_citation_diff.py --populate --dry-run

    # Generate JSON report from populated table
    python scripts/analyze_citation_diff.py --report --dsn "dbname=scix" --pretty

    # Generate markdown report
    python scripts/analyze_citation_diff.py --report --format markdown --output docs/analysis/2026-04_citation_diff.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import psycopg

# Ensure src/ is importable when running as a script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes for report output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverallStats:
    """Top-level summary of citation edge cross-validation."""

    total_edges: int
    ads_only: int
    openalex_only: int
    both: int
    ads_coverage_pct: float  # (both / (both + openalex_only)) * 100
    openalex_coverage_pct: float  # (both / (both + ads_only)) * 100
    overlap_pct: float  # (both / total) * 100


@dataclass(frozen=True)
class YearBucket:
    """Per-year edge coverage breakdown."""

    pub_year: int | None
    total_edges: int
    both_count: int
    ads_only_count: int
    openalex_only_count: int
    overlap_pct: float


@dataclass(frozen=True)
class JournalBucket:
    """Per-journal edge coverage breakdown."""

    journal: str | None
    total_edges: int
    both_count: int
    ads_only_count: int
    openalex_only_count: int
    overlap_pct: float


# ---------------------------------------------------------------------------
# Population: join ADS + OpenAlex edges into citation_diff
# ---------------------------------------------------------------------------

POPULATE_ADS_SQL = """\
INSERT INTO citation_diff (source_bibcode, target_bibcode, in_ads, in_openalex, source_attrs)
SELECT
    source_bibcode,
    target_bibcode,
    true,
    false,
    jsonb_build_object('match_type', 'ads_only')
FROM citation_edges
ON CONFLICT (source_bibcode, target_bibcode) DO UPDATE SET
    in_ads = true,
    source_attrs = jsonb_set(
        COALESCE(citation_diff.source_attrs, '{}'::jsonb),
        '{match_type}',
        CASE
            WHEN citation_diff.in_openalex THEN '"both"'::jsonb
            ELSE '"ads_only"'::jsonb
        END
    )
"""

POPULATE_OPENALEX_SQL = """\
INSERT INTO citation_diff (source_bibcode, target_bibcode, in_ads, in_openalex, source_attrs)
SELECT
    src_xw.bibcode   AS source_bibcode,
    tgt_xw.bibcode   AS target_bibcode,
    false,
    true,
    jsonb_build_object('match_type', 'openalex_only')
FROM works_references wr
JOIN papers_external_ids src_xw
    ON src_xw.openalex_id = wr.source_openalex_id
JOIN papers_external_ids tgt_xw
    ON tgt_xw.openalex_id = wr.referenced_openalex_id
WHERE src_xw.bibcode IS NOT NULL
  AND tgt_xw.bibcode IS NOT NULL
ON CONFLICT (source_bibcode, target_bibcode) DO UPDATE SET
    in_openalex = true,
    source_attrs = jsonb_set(
        COALESCE(citation_diff.source_attrs, '{}'::jsonb),
        '{match_type}',
        CASE
            WHEN citation_diff.in_ads THEN '"both"'::jsonb
            ELSE '"openalex_only"'::jsonb
        END
    )
"""

REFRESH_VIEWS_SQL = [
    "REFRESH MATERIALIZED VIEW CONCURRENTLY citation_diff_by_year",
    "REFRESH MATERIALIZED VIEW CONCURRENTLY citation_diff_by_journal",
]


def populate(conn: psycopg.Connection, *, dry_run: bool = False) -> None:
    """Populate citation_diff from citation_edges + works_references.

    Uses a two-pass approach:
    1. Insert all ADS citation edges (in_ads=true, in_openalex=false)
    2. Upsert all OpenAlex edges mapped to bibcodes (sets in_openalex=true)
    Then refreshes the materialized views.
    """
    if dry_run:
        logger.info("DRY RUN — would execute ADS pass:\n%s", POPULATE_ADS_SQL)
        logger.info("DRY RUN — would execute OpenAlex pass:\n%s", POPULATE_OPENALEX_SQL)
        for sql in REFRESH_VIEWS_SQL:
            logger.info("DRY RUN — would execute: %s", sql)
        return

    with conn.cursor() as cur:
        logger.info("Pass 1: Inserting ADS citation edges...")
        cur.execute(POPULATE_ADS_SQL)
        ads_count = cur.rowcount
        logger.info("  ADS pass: %d rows affected", ads_count)
        conn.commit()

        logger.info("Pass 2: Upserting OpenAlex edges (mapped to bibcodes)...")
        cur.execute(POPULATE_OPENALEX_SQL)
        oa_count = cur.rowcount
        logger.info("  OpenAlex pass: %d rows affected", oa_count)
        conn.commit()

    # REFRESH MATERIALIZED VIEW CONCURRENTLY cannot run inside a transaction
    # block. Switch to autocommit mode, matching the pattern in scix.views.
    conn.autocommit = True
    logger.info("Refreshing materialized views (autocommit)...")
    with conn.cursor() as cur:
        for view_sql in REFRESH_VIEWS_SQL:
            cur.execute(view_sql)
    conn.autocommit = False

    logger.info("Population complete: ADS=%d, OpenAlex=%d", ads_count, oa_count)


# ---------------------------------------------------------------------------
# Report: read citation_diff and aggregate
# ---------------------------------------------------------------------------


def overall_stats(conn: psycopg.Connection) -> OverallStats:
    """Compute top-level summary from citation_diff."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*)                                               AS total,
                COUNT(*) FILTER (WHERE in_ads AND NOT in_openalex)     AS ads_only,
                COUNT(*) FILTER (WHERE NOT in_ads AND in_openalex)     AS oa_only,
                COUNT(*) FILTER (WHERE in_ads AND in_openalex)         AS both
            FROM citation_diff
        """)
        row = cur.fetchone()
        if row is None or row[0] == 0:
            return OverallStats(
                total_edges=0,
                ads_only=0,
                openalex_only=0,
                both=0,
                ads_coverage_pct=0.0,
                openalex_coverage_pct=0.0,
                overlap_pct=0.0,
            )
        total, ads_only, oa_only, both = row
    return OverallStats(
        total_edges=total,
        ads_only=ads_only,
        openalex_only=oa_only,
        both=both,
        ads_coverage_pct=round(both / max(both + oa_only, 1) * 100, 2),
        openalex_coverage_pct=round(both / max(both + ads_only, 1) * 100, 2),
        overlap_pct=round(both / max(total, 1) * 100, 2),
    )


def by_year(conn: psycopg.Connection) -> list[YearBucket]:
    """Per-year breakdown from the materialized view."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pub_year, total_edges, both_count,
                   ads_only_count, openalex_only_count, overlap_pct
            FROM citation_diff_by_year
            ORDER BY pub_year
        """)
        return [
            YearBucket(
                pub_year=r[0],
                total_edges=r[1],
                both_count=r[2],
                ads_only_count=r[3],
                openalex_only_count=r[4],
                overlap_pct=float(r[5]) if r[5] is not None else 0.0,
            )
            for r in cur.fetchall()
        ]


def by_journal(conn: psycopg.Connection, limit: int = 50) -> list[JournalBucket]:
    """Per-journal breakdown from the materialized view (top N by edge count)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT journal, total_edges, both_count,
                   ads_only_count, openalex_only_count, overlap_pct
            FROM citation_diff_by_journal
            ORDER BY total_edges DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [
            JournalBucket(
                journal=r[0],
                total_edges=r[1],
                both_count=r[2],
                ads_only_count=r[3],
                openalex_only_count=r[4],
                overlap_pct=float(r[5]) if r[5] is not None else 0.0,
            )
            for r in cur.fetchall()
        ]


def generate_report(conn: psycopg.Connection, *, journal_limit: int = 50) -> dict[str, Any]:
    """Generate the full citation diff report as a dict."""
    stats = overall_stats(conn)
    years = by_year(conn)
    journals = by_journal(conn, limit=journal_limit)
    return {
        "overall": asdict(stats),
        "by_year": [asdict(y) for y in years],
        "by_journal": [asdict(j) for j in journals],
    }


def format_markdown(report: dict[str, Any]) -> str:
    """Format the report dict as a markdown document for paper Section 3.3."""
    lines: list[str] = []
    lines.append("# Citation Graph Cross-Validation: ADS vs OpenAlex")
    lines.append("")
    lines.append("## Overall Statistics")
    lines.append("")

    ov = report["overall"]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total unique edges | {ov['total_edges']:,} |")
    lines.append(f"| In both ADS and OpenAlex | {ov['both']:,} |")
    lines.append(f"| ADS only | {ov['ads_only']:,} |")
    lines.append(f"| OpenAlex only | {ov['openalex_only']:,} |")
    lines.append(f"| ADS coverage (both / OA total) | {ov['ads_coverage_pct']:.2f}% |")
    lines.append(f"| OpenAlex coverage (both / ADS total) | {ov['openalex_coverage_pct']:.2f}% |")
    lines.append(f"| Overlap (both / total) | {ov['overlap_pct']:.2f}% |")
    lines.append("")

    years = report.get("by_year", [])
    if years:
        lines.append("## Per-Year Edge Coverage")
        lines.append("")
        lines.append("| Year | Total | Both | ADS Only | OA Only | Overlap % |")
        lines.append("|------|-------|------|----------|---------|-----------|")
        for y in years:
            yr = y["pub_year"] if y["pub_year"] is not None else "N/A"
            lines.append(
                f"| {yr} | {y['total_edges']:,} | {y['both_count']:,} | "
                f"{y['ads_only_count']:,} | {y['openalex_only_count']:,} | "
                f"{y['overlap_pct']:.2f}% |"
            )
        lines.append("")

    journals = report.get("by_journal", [])
    if journals:
        lines.append("## Per-Journal Edge Coverage (Top 50)")
        lines.append("")
        lines.append("| Journal | Total | Both | ADS Only | OA Only | Overlap % |")
        lines.append("|---------|-------|------|----------|---------|-----------|")
        for j in journals:
            jname = j["journal"] if j["journal"] else "N/A"
            lines.append(
                f"| {jname} | {j['total_edges']:,} | {j['both_count']:,} | "
                f"{j['ads_only_count']:,} | {j['openalex_only_count']:,} | "
                f"{j['overlap_pct']:.2f}% |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validate ADS vs OpenAlex citation graphs",
    )
    parser.add_argument(
        "--dsn",
        default=DEFAULT_DSN,
        help="PostgreSQL DSN (default: $SCIX_DSN or scix.db.DEFAULT_DSN)",
    )
    parser.add_argument(
        "--populate",
        action="store_true",
        help="Populate the citation_diff table from citation_edges + works_references",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a report from the populated citation_diff table",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show SQL queries without executing (with --populate)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format for --report (default: json)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write report to file instead of stdout",
    )
    parser.add_argument(
        "--journal-limit",
        type=int,
        default=50,
        help="Max journals in per-journal breakdown (default: 50)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--yes-production",
        action="store_true",
        help="Confirm write operations against production DSN",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.populate and not args.report:
        parser.error("At least one of --populate or --report is required")

    dsn = args.dsn
    if args.populate and is_production_dsn(dsn) and not args.dry_run:
        if not args.yes_production:
            parser.error(
                f"DSN points at production ({redact_dsn(dsn)}). "
                "Pass --yes-production to confirm, or use --dry-run to inspect SQL."
            )
        logger.warning(
            "Production write confirmed via --yes-production (%s)",
            redact_dsn(dsn),
        )

    conn = psycopg.connect(dsn)
    try:
        if args.populate:
            populate(conn, dry_run=args.dry_run)

        if args.report:
            report = generate_report(conn, journal_limit=args.journal_limit)

            if args.format == "markdown":
                output_text = format_markdown(report)
            else:
                indent = 2 if args.pretty else None
                output_text = json.dumps(report, indent=indent, default=str) + "\n"

            if args.output:
                out_path = os.path.abspath(args.output)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    f.write(output_text)
                logger.info("Report written to %s", out_path)
            else:
                sys.stdout.write(output_text)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
