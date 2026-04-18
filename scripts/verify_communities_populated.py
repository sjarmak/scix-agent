#!/usr/bin/env python3
"""Verify post-PRD ``community-detection-v2`` community data populated state.

Read-only diagnostic that answers the zx8.1 question: did the
``community-detection-v2`` PRD actually populate community tables in the
target database, and where is the data?

Checks performed
----------------

1. Migrations 051 (``community_semantic_*`` columns on ``paper_metrics``)
   and 052 (``signal`` column + 3-col PK on ``communities``) — are the
   DDL artifacts present?
2. ``paper_metrics`` — row counts per community column
   (``community_id_{coarse,medium,fine}``,
   ``community_semantic_{coarse,medium,fine}``, ``community_taxonomic``).
3. ``communities`` — total rows, breakdown by ``signal`` (if column
   present) and ``resolution``.
4. ``paper_communities`` — whether the table exists (PRD never defined
   this table; it's in the bead scope because the question was asked).

Exit codes
----------

- ``0`` — all expected artifacts populated and healthy.
- ``1`` — at least one required artifact missing or empty. The JSON
  report still prints to stdout; the non-zero code is advisory so CI /
  operator scripts can gate on it.

Usage
-----

::

    # Against production (read-only; no writes). The default DSN is
    # ``dbname=scix`` — this script does no DML so it is safe.
    python scripts/verify_communities_populated.py

    # Against the test DB:
    SCIX_DSN="dbname=scix_test" python scripts/verify_communities_populated.py

    # Emit the report to a file:
    python scripts/verify_communities_populated.py --output report.json

The script never writes to the database. It has no ``--allow-prod`` flag
because there is no write path to guard.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import psycopg

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.db import redact_dsn  # noqa: E402

logger = logging.getLogger("verify_communities")


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnCheck:
    """Presence of a single column on a table."""

    table: str
    column: str
    present: bool


@dataclass(frozen=True)
class TableCheck:
    """Presence of a single table."""

    table: str
    present: bool


@dataclass(frozen=True)
class ColumnRowCount:
    """Row count for a single column (NOT NULL)."""

    table: str
    column: str
    non_null_count: int | None  # None if column missing


@dataclass(frozen=True)
class SignalResolutionCount:
    """Count of rows in ``communities`` by (signal, resolution)."""

    signal: str | None
    resolution: str
    rows: int


@dataclass
class VerificationReport:
    """Full diagnostic report."""

    dsn_redacted: str
    columns: list[ColumnCheck] = field(default_factory=list)
    tables: list[TableCheck] = field(default_factory=list)
    paper_metrics_total: int = 0
    paper_metrics_columns: list[ColumnRowCount] = field(default_factory=list)
    communities_total: int = 0
    communities_breakdown: list[SignalResolutionCount] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dsn_redacted": self.dsn_redacted,
            "columns": [asdict(c) for c in self.columns],
            "tables": [asdict(t) for t in self.tables],
            "paper_metrics_total": self.paper_metrics_total,
            "paper_metrics_columns": [asdict(c) for c in self.paper_metrics_columns],
            "communities_total": self.communities_total,
            "communities_breakdown": [asdict(b) for b in self.communities_breakdown],
            "problems": list(self.problems),
            "healthy": not self.problems,
        }


# Expected artifacts per the community-detection-v2 PRD.

_EXPECTED_PAPER_METRICS_COLS: tuple[str, ...] = (
    "community_id_coarse",
    "community_id_medium",
    "community_id_fine",
    "community_semantic_coarse",
    "community_semantic_medium",
    "community_semantic_fine",
    "community_taxonomic",
)

_EXPECTED_COMMUNITIES_COLS: tuple[str, ...] = (
    "community_id",
    "resolution",
    "label",
    "paper_count",
    "top_keywords",
    "signal",
)

_CHECKED_TABLES: tuple[str, ...] = (
    "paper_metrics",
    "communities",
    "paper_communities",  # PRD never declared; bead flagged the question
)


# ---------------------------------------------------------------------------
# SQL helpers (all read-only)
# ---------------------------------------------------------------------------


def _columns_present(
    conn: psycopg.Connection, table: str, columns: Sequence[str]
) -> set[str]:
    """Return the subset of ``columns`` that exist on ``public.<table>``."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
              FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name   = %s
               AND column_name  = ANY(%s)
            """,
            (table, list(columns)),
        )
        return {row[0] for row in cur.fetchall()}


def _tables_present(
    conn: psycopg.Connection, tables: Sequence[str]
) -> set[str]:
    """Return the subset of ``tables`` that exist in the ``public`` schema."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
              FROM information_schema.tables
             WHERE table_schema = 'public'
               AND table_name   = ANY(%s)
            """,
            (list(tables),),
        )
        return {row[0] for row in cur.fetchall()}


def _count_non_null(conn: psycopg.Connection, table: str, column: str) -> int:
    # Identifier quoting — ``table`` and ``column`` come from the static
    # expected-artifact tuples above, never user input.
    with conn.cursor() as cur:
        cur.execute(
            psycopg.sql.SQL('SELECT count(*) FROM {tbl} WHERE {col} IS NOT NULL').format(
                tbl=psycopg.sql.Identifier(table),
                col=psycopg.sql.Identifier(column),
            )
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def _count_total(conn: psycopg.Connection, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            psycopg.sql.SQL("SELECT count(*) FROM {tbl}").format(
                tbl=psycopg.sql.Identifier(table)
            )
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def _communities_breakdown(
    conn: psycopg.Connection, *, has_signal: bool
) -> list[SignalResolutionCount]:
    if has_signal:
        query = (
            "SELECT signal, resolution, count(*) "
            "FROM communities GROUP BY signal, resolution ORDER BY signal, resolution"
        )
    else:
        query = (
            "SELECT NULL::text AS signal, resolution, count(*) "
            "FROM communities GROUP BY resolution ORDER BY resolution"
        )
    with conn.cursor() as cur:
        cur.execute(query)
        return [
            SignalResolutionCount(signal=signal, resolution=resolution, rows=int(rows))
            for (signal, resolution, rows) in cur.fetchall()
        ]


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------


def verify(conn: psycopg.Connection, *, dsn_redacted: str) -> VerificationReport:
    """Run all diagnostics against ``conn`` and return a report."""
    report = VerificationReport(dsn_redacted=dsn_redacted)

    # 1. Table presence — single batched lookup.
    present_tables = _tables_present(conn, _CHECKED_TABLES)
    table_flags: dict[str, bool] = {
        table: table in present_tables for table in _CHECKED_TABLES
    }
    for table in _CHECKED_TABLES:
        report.tables.append(TableCheck(table=table, present=table_flags[table]))

    # 2. Column presence — one batched lookup per table.
    col_flags: dict[tuple[str, str], bool] = {}
    for table, expected_cols in (
        ("paper_metrics", _EXPECTED_PAPER_METRICS_COLS),
        ("communities", _EXPECTED_COMMUNITIES_COLS),
    ):
        if not table_flags.get(table):
            continue
        present_cols = _columns_present(conn, table, expected_cols)
        for col in expected_cols:
            flag = col in present_cols
            col_flags[(table, col)] = flag
            report.columns.append(
                ColumnCheck(table=table, column=col, present=flag)
            )

    # 3. Row counts on paper_metrics.
    if table_flags.get("paper_metrics"):
        report.paper_metrics_total = _count_total(conn, "paper_metrics")
        for col in _EXPECTED_PAPER_METRICS_COLS:
            if col_flags.get(("paper_metrics", col)):
                count = _count_non_null(conn, "paper_metrics", col)
            else:
                count = None
            report.paper_metrics_columns.append(
                ColumnRowCount(table="paper_metrics", column=col, non_null_count=count)
            )

    # 4. Communities breakdown.
    if table_flags.get("communities"):
        report.communities_total = _count_total(conn, "communities")
        report.communities_breakdown = _communities_breakdown(
            conn, has_signal=col_flags.get(("communities", "signal"), False)
        )

    # 5. Problem detection (policy over the collected evidence).
    report.problems = _detect_problems(report, table_flags, col_flags)
    return report


def _detect_problems(
    report: VerificationReport,
    table_flags: Mapping[str, bool],
    col_flags: Mapping[tuple[str, str], bool],
) -> list[str]:
    """Compare observed state to PRD expectations; return failure messages."""
    problems: list[str] = []

    # Required tables (``paper_communities`` is explicitly NOT expected per
    # PRD; it is only reported for completeness, never a problem).
    for table in ("paper_metrics", "communities"):
        if not table_flags.get(table):
            problems.append(f"table missing: {table}")

    # Migration 051 — three semantic columns + their btree indexes.
    for col in (
        "community_semantic_coarse",
        "community_semantic_medium",
        "community_semantic_fine",
    ):
        if not col_flags.get(("paper_metrics", col)):
            problems.append(
                f"migration 051 not applied: paper_metrics.{col} missing"
            )

    # Migration 052 — signal column on communities.
    if table_flags.get("communities") and not col_flags.get(
        ("communities", "signal")
    ):
        problems.append("migration 052 not applied: communities.signal missing")

    # Citation Leiden assignment coverage.
    citation_cols = {
        c.column: c.non_null_count for c in report.paper_metrics_columns
    }
    for col in (
        "community_id_coarse",
        "community_id_medium",
        "community_id_fine",
    ):
        count = citation_cols.get(col)
        if count is None or count == 0:
            problems.append(
                f"citation community empty: paper_metrics.{col} has 0 non-null rows"
            )

    # Semantic Leiden assignment coverage (only meaningful if column exists).
    for col in (
        "community_semantic_coarse",
        "community_semantic_medium",
        "community_semantic_fine",
    ):
        if col_flags.get(("paper_metrics", col)):
            count = citation_cols.get(col)
            if count is None or count == 0:
                problems.append(
                    f"semantic community empty: paper_metrics.{col} has 0 non-null rows"
                )

    # Communities labels populated.
    if table_flags.get("communities") and report.communities_total == 0:
        problems.append("communities table empty: 0 rows (M4 labels not generated)")

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify post-PRD community data populated state (read-only)."
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_DSN", "dbname=scix"),
        help="PostgreSQL DSN (default: $SCIX_DSN or dbname=scix). Read-only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report. Also prints to stdout.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging (JSON still emitted).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn_redacted = redact_dsn(args.dsn)
    logger.info("verifying communities state against %s", dsn_redacted)

    with psycopg.connect(args.dsn) as conn:
        conn.autocommit = True
        report = verify(conn, dsn_redacted=dsn_redacted)

    payload = report.to_dict()
    json_text = json.dumps(payload, indent=2, sort_keys=True)
    print(json_text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_text + "\n", encoding="utf-8")

    if report.problems:
        logger.warning("%d problem(s) detected", len(report.problems))
        for msg in report.problems:
            logger.warning("  - %s", msg)
        return 1
    logger.info("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
