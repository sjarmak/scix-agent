#!/usr/bin/env python3
"""Emit a read-only JSON report of community-assignment coverage.

For each paper in ``paper_metrics``, counts how many have at least one
non-null signal across the three community assignments:

- **citation**: ``community_id_coarse`` (Leiden on citation edges, M3)
- **semantic**: ``community_semantic_coarse`` (k-means on INDUS, M2)
- **taxonomic**: ``community_taxonomic`` (arXiv-class / bibgroup, Layer 0)

The script is strictly read-only — a single ``SELECT`` over
``paper_metrics``. It does not write anywhere in the database; safe to run
against any DSN without ``--allow-prod``.

Outputs:
    ``results/community_coverage.json`` (gitignored; created on every run)
    ``docs/prd/artifacts/community_coverage.sample.json`` (committed;
        written only when the DSN is **not** a production DSN so the
        committed sample always reflects the test-database shape)

Schema::

    {
      "total_papers":            <int>,
      "citation_covered":        <int | null>,
      "semantic_covered":        <int | null>,
      "taxonomic_covered":       <int | null>,
      "union_covered":           <int>,
      "union_coverage_fraction": <float | null>
    }

A column count of ``null`` means that column does not (yet) exist on the
target DB — e.g. the semantic or taxonomic migration has not been applied.
The script logs a warning, skips that column in both the per-column count
and the UNION filter, and records ``null`` in the JSON.

Usage::

    SCIX_TEST_DSN="dbname=scix_test" \\
    python scripts/report_community_coverage.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import psycopg

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("report_community_coverage")


DEFAULT_OUTPUT = _REPO_ROOT / "results" / "community_coverage.json"
SAMPLE_ARTIFACT = _REPO_ROOT / "docs" / "prd" / "artifacts" / "community_coverage.sample.json"


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalColumn:
    """Mapping from a JSON output key to a ``paper_metrics`` column."""

    key: str  # JSON key, e.g. "citation_covered"
    column: str  # SQL column, e.g. "community_id_coarse"


# Order here determines the UNION-coverage filter and output JSON order.
SIGNAL_COLUMNS: tuple[SignalColumn, ...] = (
    SignalColumn(key="citation_covered", column="community_id_coarse"),
    SignalColumn(key="semantic_covered", column="community_semantic_coarse"),
    SignalColumn(key="taxonomic_covered", column="community_taxonomic"),
)

PAPER_METRICS_TABLE = "paper_metrics"


# ---------------------------------------------------------------------------
# Schema introspection — detect missing columns before running the count
# ---------------------------------------------------------------------------


def detect_present_columns(
    conn: psycopg.Connection,
    table: str,
    signals: Sequence[SignalColumn],
) -> dict[str, bool]:
    """Return {column_name: exists_bool} for every signal column.

    Uses ``information_schema.columns`` so we can emit a warning and leave
    the counter at ``null`` instead of crashing when a migration has not
    been applied yet.
    """
    wanted = [s.column for s in signals]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
              FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = %s
               AND column_name = ANY(%s)
            """,
            (table, wanted),
        )
        present = {row[0] for row in cur.fetchall()}

    return {col: (col in present) for col in wanted}


# ---------------------------------------------------------------------------
# Coverage query — a single SELECT over paper_metrics
# ---------------------------------------------------------------------------


def _build_coverage_sql(present_columns: Sequence[str]) -> str:
    """Compose one SELECT that counts each present signal and the UNION.

    ``present_columns`` must already be restricted to columns that exist
    on the target DB. Column names are internal constants (not user input)
    so direct interpolation is safe; even so we validate against the
    allowlist in ``SIGNAL_COLUMNS`` as a defence-in-depth check.
    """
    allowed = {s.column for s in SIGNAL_COLUMNS}
    for col in present_columns:
        if col not in allowed:
            raise ValueError(f"refusing to interpolate unknown column {col!r}")

    select_parts = ["COUNT(*) AS total_papers"]
    for col in present_columns:
        select_parts.append(f"COUNT({col}) AS {col}_nn")

    if present_columns:
        union_filter = " OR ".join(f"{col} IS NOT NULL" for col in present_columns)
        select_parts.append(f"COUNT(*) FILTER (WHERE {union_filter}) AS union_covered")
    else:
        # No signal columns exist yet — UNION coverage is trivially zero.
        select_parts.append("0 AS union_covered")

    return "SELECT " + ", ".join(select_parts) + f" FROM {PAPER_METRICS_TABLE}"


def collect_coverage(conn: psycopg.Connection) -> dict[str, object]:
    """Run the coverage SELECT and assemble the JSON payload.

    Columns that do not exist on the target DB are reported as ``null`` in
    the output and a warning is logged. The fraction is ``null`` when there
    are zero papers, matching ``NULLIF(total_papers, 0)`` semantics.
    """
    presence = detect_present_columns(conn, PAPER_METRICS_TABLE, SIGNAL_COLUMNS)
    present_cols = [col for col, ok in presence.items() if ok]
    missing_cols = [col for col, ok in presence.items() if not ok]
    for col in missing_cols:
        logger.warning(
            "%s.%s does not exist on the target DB — "
            "reporting null for this signal (migration not yet applied?)",
            PAPER_METRICS_TABLE,
            col,
        )

    sql = _build_coverage_sql(present_cols)
    logger.info("coverage query: %s", sql)
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("coverage query returned no rows — unexpected")

    # psycopg returns columns in SELECT order: total, then each present
    # column in its SIGNAL_COLUMNS order, then union_covered.
    total_papers = int(row[0])
    counts: dict[str, Optional[int]] = {}
    idx = 1
    for signal in SIGNAL_COLUMNS:
        if presence[signal.column]:
            counts[signal.key] = int(row[idx])
            idx += 1
        else:
            counts[signal.key] = None
    union_covered = int(row[idx])

    fraction: Optional[float] = union_covered / total_papers if total_papers > 0 else None

    payload: dict[str, object] = {"total_papers": total_papers}
    for signal in SIGNAL_COLUMNS:
        payload[signal.key] = counts[signal.key]
    payload["union_covered"] = union_covered
    payload["union_coverage_fraction"] = fraction
    return payload


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(
    payload: dict[str, object],
    output_path: Path,
    dsn: str,
    sample_path: Path = SAMPLE_ARTIFACT,
) -> list[Path]:
    """Write the payload to ``output_path`` and to the committed sample.

    The committed sample is only written when the target is NOT a
    production DSN — this matches M2's pattern (``docs/prd/artifacts`` is
    checked in, ``results/`` is gitignored) and keeps prod counts out of
    the repo.
    """
    written: list[Path] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    output_path.write_text(serialized, encoding="utf-8")
    logger.info("wrote %s", output_path)
    written.append(output_path)

    if not is_production_dsn(dsn):
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(serialized, encoding="utf-8")
        logger.info("wrote sample artifact %s", sample_path)
        written.append(sample_path)
    else:
        logger.info(
            "skipping sample artifact write (%s) — production DSN",
            sample_path,
        )
    return written


# ---------------------------------------------------------------------------
# DSN resolution — prefer SCIX_TEST_DSN, fall back to SCIX_DSN/DEFAULT_DSN
# ---------------------------------------------------------------------------


def _resolve_dsn(cli_dsn: Optional[str]) -> str:
    """Resolve the DSN to connect against.

    Precedence: ``--dsn`` CLI flag → ``SCIX_TEST_DSN`` → ``DEFAULT_DSN``.
    The script is read-only so we DO allow ``DEFAULT_DSN`` (which usually
    points at production) as the last-resort fallback — callers who need
    coverage numbers for the real corpus should not have to jump through
    ``--allow-prod`` hoops for a SELECT.
    """
    if cli_dsn:
        return cli_dsn
    test_dsn = os.environ.get("SCIX_TEST_DSN")
    if test_dsn:
        return test_dsn
    return DEFAULT_DSN


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN. Defaults to SCIX_TEST_DSN env, then SCIX_DSN.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=("Path for the coverage JSON (default: results/community_coverage.json)."),
    )
    parser.add_argument(
        "--sample-path",
        default=str(SAMPLE_ARTIFACT),
        help=(
            "Committed-sample path (default: "
            "docs/prd/artifacts/community_coverage.sample.json). "
            "Skipped when the DSN points at production."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    dsn = _resolve_dsn(args.dsn)

    logger.info("connecting to %s (read-only)", redact_dsn(dsn))

    # autocommit=True keeps the single SELECT out of an implicit transaction
    # so we never hold locks or snapshot state beyond the query itself.
    with psycopg.connect(dsn) as conn:
        conn.autocommit = True
        payload = collect_coverage(conn)

    logger.info(
        "coverage: total=%d citation=%s semantic=%s taxonomic=%s " "union=%d fraction=%s",
        payload["total_papers"],
        payload["citation_covered"],
        payload["semantic_covered"],
        payload["taxonomic_covered"],
        payload["union_covered"],
        (
            f"{payload['union_coverage_fraction']:.4f}"
            if payload["union_coverage_fraction"] is not None
            else "n/a"
        ),
    )

    write_outputs(
        payload,
        output_path=Path(args.output),
        dsn=dsn,
        sample_path=Path(args.sample_path),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
