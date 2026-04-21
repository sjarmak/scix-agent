#!/usr/bin/env python3
"""Read-only coverage report for ``papers_fulltext`` / ``papers_fulltext_failures``.

Emits operational-monitoring statistics for the structural full-text parsing
pipeline (PRD: ``docs/prd/prd_structural_fulltext_parsing.md``, D2):

- per-``source`` row counts (``ar5iv`` / ``arxiv_local`` / ``s2orc`` /
  ``ads_body`` / ``docling`` / ``abstract`` / ...)
- sections-per-paper histogram via ``jsonb_array_length(sections)`` with
  bucketing ``0 | 1 | 2-4 | 5-9 | 10-19 | 20+``
- count of section-empty rows (``jsonb_array_length(sections) = 0``)
- tier distribution (latex/xml/text/abstract/other — derived from source)
- median sections-per-paper for LaTeX-derived sources
  (``source IN ('ar5iv','arxiv_local')``) via ``PERCENTILE_CONT(0.5)``
- failure rate ``= count(papers_fulltext_failures) / (count(papers_fulltext) +
  count(papers_fulltext_failures))`` expressed as a percentage

Safety
------
The script is read-only — it sets the session to ``READ ONLY`` on connect
and asserts ``SHOW transaction_read_only = 'on'`` before running any
reporting SQL. Any attempted write raises ``psycopg.errors.ReadOnlySqlTransaction``.

Usage
-----
::

    # Against the default DSN (env ``SCIX_DSN`` or ``dbname=scix``):
    python scripts/report_fulltext_coverage.py

    # Against scix_test:
    python scripts/report_fulltext_coverage.py --dsn dbname=scix_test

    # JSON output:
    python scripts/report_fulltext_coverage.py --format json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import psycopg

REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from scix.db import DEFAULT_DSN, redact_dsn  # noqa: E402
from scix.sources.ar5iv import LATEX_DERIVED_SOURCES  # noqa: E402

logger = logging.getLogger("report_fulltext_coverage")


# ---------------------------------------------------------------------------
# Bucketing + tiering policy
# ---------------------------------------------------------------------------


# Ordered list of (bucket_ord, bucket_label).
BUCKET_LABELS: tuple[tuple[int, str], ...] = (
    (0, "0"),
    (1, "1"),
    (2, "2-4"),
    (3, "5-9"),
    (4, "10-19"),
    (5, "20+"),
)

_BUCKET_ORD_TO_LABEL: dict[int, str] = dict(BUCKET_LABELS)


# Tier assignment: derived class for each known source. Sources not listed
# fall into the ``other`` tier so new source tags surface in the report
# without requiring a code change.
_TIER_BY_SOURCE: dict[str, str] = {
    "ar5iv": "latex",
    "arxiv_local": "latex",
    "s2orc": "xml",
    "docling": "xml",
    "ads_body": "text",
    "abstract": "abstract",
}


def tier_for_source(source: str) -> str:
    """Return the tier label for a given papers_fulltext.source value."""
    return _TIER_BY_SOURCE.get(source, "other")


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageReport:
    """Structured report produced by :func:`collect_coverage`."""

    dsn_redacted: str
    total_rows: int
    by_source: dict[str, int]
    by_tier: dict[str, int]
    histogram: list[tuple[str, int]]
    section_empty: int
    latex_median_sections: float | None
    failure_rate_pct: float
    fulltext_rows: int
    failure_rows: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Ensure histogram round-trips as list of [label, count] in JSON.
        d["histogram"] = [list(item) for item in self.histogram]
        return d


# ---------------------------------------------------------------------------
# Read-only enforcement
# ---------------------------------------------------------------------------


def assert_readonly(conn: psycopg.Connection) -> None:
    """Assert the session is in a READ ONLY transaction mode.

    Runs ``SHOW transaction_read_only``; raises ``RuntimeError`` if the
    value is anything other than ``'on'``. This guard is the write-safety
    contract for the script — if it passes, any accidental INSERT/UPDATE/
    DELETE/DDL emitted later will be refused by PostgreSQL with
    ``ReadOnlySqlTransaction``.
    """
    with conn.cursor() as cur:
        cur.execute("SHOW transaction_read_only")
        row = cur.fetchone()
    if row is None or str(row[0]).lower() != "on":
        actual = None if row is None else row[0]
        raise RuntimeError(
            f"expected transaction_read_only=on, got {actual!r} — "
            "refusing to run coverage report against a writable session"
        )


def enter_readonly(conn: psycopg.Connection) -> None:
    """Put the connection into a READ ONLY session and verify it stuck."""
    with conn.cursor() as cur:
        cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
    assert_readonly(conn)


# ---------------------------------------------------------------------------
# SQL helpers — all read-only SELECTs.
# ---------------------------------------------------------------------------


def _count_by_source(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source, count(*) FROM papers_fulltext " "GROUP BY source ORDER BY source"
        )
        return {str(source): int(count) for (source, count) in cur.fetchall()}


def _histogram(conn: psycopg.Connection) -> list[tuple[str, int]]:
    """Return ordered histogram of sections-per-paper bucket counts.

    Missing buckets (no rows) are emitted as 0 so the output shape is stable.
    """
    with conn.cursor() as cur:
        cur.execute("""
            WITH b AS (
              SELECT CASE
                WHEN jsonb_array_length(sections) = 0 THEN 0
                WHEN jsonb_array_length(sections) = 1 THEN 1
                WHEN jsonb_array_length(sections) BETWEEN 2 AND 4 THEN 2
                WHEN jsonb_array_length(sections) BETWEEN 5 AND 9 THEN 3
                WHEN jsonb_array_length(sections) BETWEEN 10 AND 19 THEN 4
                ELSE 5
              END AS ord
              FROM papers_fulltext
            )
            SELECT ord, count(*)::bigint FROM b GROUP BY ord ORDER BY ord
            """)
        counts = {int(ord_): int(count) for (ord_, count) in cur.fetchall()}
    return [(label, counts.get(ord_, 0)) for (ord_, label) in BUCKET_LABELS]


def _section_empty(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM papers_fulltext " "WHERE jsonb_array_length(sections) = 0"
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def _latex_median(conn: psycopg.Connection) -> float | None:
    """Return PERCENTILE_CONT(0.5) of sections-per-paper for LaTeX sources.

    Returns ``None`` when no LaTeX-derived rows exist (PERCENTILE_CONT over
    zero rows emits a NULL result).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP "
            "(ORDER BY jsonb_array_length(sections))::float "
            "FROM papers_fulltext WHERE source = ANY(%s)",
            (list(LATEX_DERIVED_SOURCES),),
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return float(row[0])


def _count_rows(conn: psycopg.Connection, table: str) -> int:
    # table is a fixed identifier from this module, never user input.
    with conn.cursor() as cur:
        cur.execute(
            psycopg.sql.SQL("SELECT count(*) FROM {tbl}").format(tbl=psycopg.sql.Identifier(table))
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def failure_rate_pct(fulltext_rows: int, failure_rows: int) -> float:
    """Compute failure rate as a percentage; 0.0 when denominator is 0."""
    denom = fulltext_rows + failure_rows
    if denom <= 0:
        return 0.0
    return 100.0 * failure_rows / denom


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def collect_coverage(conn: psycopg.Connection, *, dsn_redacted: str) -> CoverageReport:
    """Run the full coverage collection against ``conn``."""
    assert_readonly(conn)

    by_source = _count_by_source(conn)
    hist = _histogram(conn)
    empty = _section_empty(conn)
    latex_med = _latex_median(conn)
    fulltext_rows = _count_rows(conn, "papers_fulltext")
    failure_rows = _count_rows(conn, "papers_fulltext_failures")

    by_tier: dict[str, int] = {}
    for source, count in by_source.items():
        by_tier[tier_for_source(source)] = by_tier.get(tier_for_source(source), 0) + count

    total_rows = sum(by_source.values())

    return CoverageReport(
        dsn_redacted=dsn_redacted,
        total_rows=total_rows,
        by_source=by_source,
        by_tier=by_tier,
        histogram=hist,
        section_empty=empty,
        latex_median_sections=latex_med,
        failure_rate_pct=failure_rate_pct(fulltext_rows, failure_rows),
        fulltext_rows=fulltext_rows,
        failure_rows=failure_rows,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_text(report: CoverageReport) -> str:
    """Render the report as a human-readable text table."""
    lines: list[str] = []
    lines.append(f"dsn: {report.dsn_redacted}")
    lines.append(f"total_rows: {report.total_rows}")
    lines.append("by_source:")
    if not report.by_source:
        lines.append("  (none)")
    for source in sorted(report.by_source):
        lines.append(f"  source={source} rows={report.by_source[source]}")
    lines.append("by_tier:")
    if not report.by_tier:
        lines.append("  (none)")
    for tier in sorted(report.by_tier):
        lines.append(f"  tier={tier} rows={report.by_tier[tier]}")
    lines.append("histogram (sections_per_paper):")
    for label, count in report.histogram:
        lines.append(f"  {label}: {count}")
    lines.append(f"section_empty: {report.section_empty}")
    if report.latex_median_sections is None:
        lines.append("latex_median_sections: n/a (no rows)")
    else:
        lines.append(f"latex_median_sections: {report.latex_median_sections:.2f}")
    lines.append(
        f"failure_rate_pct: {report.failure_rate_pct:.2f} "
        f"(failures={report.failure_rows}, fulltext={report.fulltext_rows})"
    )
    return "\n".join(lines) + "\n"


def format_json(report: CoverageReport) -> str:
    """Render the report as JSON."""
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only operational report of papers_fulltext / "
            "papers_fulltext_failures coverage."
        )
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_DSN", DEFAULT_DSN),
        help="PostgreSQL DSN (default: $SCIX_DSN or 'dbname=scix'). Read-only.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn_redacted = redact_dsn(args.dsn)
    logger.info("collecting fulltext coverage from %s", dsn_redacted)

    with psycopg.connect(args.dsn) as conn:
        conn.autocommit = True
        enter_readonly(conn)
        report = collect_coverage(conn, dsn_redacted=dsn_redacted)

    if args.format == "json":
        sys.stdout.write(format_json(report))
    else:
        sys.stdout.write(format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
