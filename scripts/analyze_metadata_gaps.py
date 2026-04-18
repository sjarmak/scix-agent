#!/usr/bin/env python3
"""Analyze ADS metadata coverage gaps across the astronomy cohort.

Reads the ``papers`` table, filters astronomy papers (``arxiv_class`` starts with
``astro-ph`` OR the ADS collection ``astronomy`` is present in ``database``),
computes per-field coverage for ``facility`` (instruments), ``data`` (datasets),
and ``keyword_norm`` (software), and identifies the gap cohort — astronomy
papers missing at least one of those three entity-bearing fields.

Output: JSON report at ``results/metadata_gap_report.json`` plus an optional
sidecar bibcode list when the cohort is large.

Safety:
- Against the production DSN, the script refuses to run a full-corpus scan
  unless ``--sample-size N`` is provided OR ``SCIX_ALLOW_FULL_SCAN=1`` is set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running as a plain script: make src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import DEFAULT_DSN, get_connection, is_production_dsn, redact_dsn  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Three array fields we treat as entity-bearing metadata.
ENTITY_FIELDS: tuple[str, ...] = ("facility", "data", "keyword_norm")

# Map each field -> the entity class it populates in downstream extraction.
FIELD_TO_ENTITY_TYPE: dict[str, str] = {
    "facility": "instruments",
    "data": "datasets",
    "keyword_norm": "software",
}

# Cohort predicate reused by fetch_cohort_rows.  Exposed for tests / audit.
#
# NOTE on percent-sign escaping: psycopg3 treats ``%`` as the start of a
# placeholder when a query string is passed alongside parameters to
# ``cur.execute(query, params)``.  A bare ``ILIKE 'astro-ph%'`` therefore
# raises ``ProgrammingError: only '%s', '%b', '%t' are allowed as
# placeholders, got '%''``.  We escape the LIKE wildcard as ``%%``; psycopg3
# unescapes it back to a single ``%`` before sending to the server on BOTH
# the parameterised LIMIT path AND the streaming (no-params) path.
ASTRONOMY_COHORT_SQL: str = (
    "(EXISTS (SELECT 1 FROM unnest(arxiv_class) c WHERE c ILIKE 'astro-ph%%'))\n"
    "OR ('astronomy' = ANY(database))"
)

# PRD acceptance criterion: gap cohort expected to be < 40% of astronomy papers.
GAP_COHORT_EXPECTED_UPPER_BOUND: float = 0.40

# Default cap on inline bibcodes in the JSON report.  Above this we spill to
# a sidecar text file.
DEFAULT_COHORT_BIBCODE_CAP: int = 1000

DEFAULT_OUTPUT_PATH: str = "results/metadata_gap_report.json"
DEFAULT_COHORT_SIDECAR_PATH: str = "results/metadata_gap_cohort_bibcodes.txt"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldCoverage:
    """Per-field populated/total/coverage-pct triple."""

    field: str
    populated: int
    total: int
    coverage_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "populated": self.populated,
            "total": self.total,
            "coverage_pct": self.coverage_pct,
        }


@dataclass(frozen=True)
class GapEntityRow:
    """One row in the top-N gap ranking."""

    entity_type: str
    field: str
    gap_count: int
    gap_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "field": self.field,
            "gap_count": self.gap_count,
            "gap_ratio": self.gap_ratio,
        }


@dataclass(frozen=True)
class GapReport:
    """Full metadata gap analysis report."""

    total_astronomy_papers: int
    gap_cohort_count: int
    gap_cohort_ratio: float
    gap_cohort_bibcodes_sample: list[str]
    gap_cohort_bibcodes_sidecar_path: str | None
    top_gap_entity_types: list[GapEntityRow]
    per_field_coverage: dict[str, FieldCoverage]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_astronomy_papers": self.total_astronomy_papers,
            "gap_cohort_count": self.gap_cohort_count,
            "gap_cohort_ratio": self.gap_cohort_ratio,
            "gap_cohort_bibcodes_sample": list(self.gap_cohort_bibcodes_sample),
            "gap_cohort_bibcodes_sidecar_path": self.gap_cohort_bibcodes_sidecar_path,
            "top_gap_entity_types": [row.to_dict() for row in self.top_gap_entity_types],
            "per_field_coverage": {k: v.to_dict() for k, v in self.per_field_coverage.items()},
            "meta": dict(self.meta),
        }


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


def _is_populated(value: Any) -> bool:
    """Return True if the array-field value has at least one non-empty string."""
    if value is None:
        return False
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, str) and item.strip():
            return True
    return False


def is_row_in_gap_cohort(row: Mapping[str, Any]) -> bool:
    """A paper is in the gap cohort iff any of the three entity fields is empty/missing."""
    return any(not _is_populated(row.get(field)) for field in ENTITY_FIELDS)


def compute_field_coverage(rows: list[Mapping[str, Any]]) -> dict[str, FieldCoverage]:
    """Count populated rows per entity field and report coverage percentage."""
    total = len(rows)
    result: dict[str, FieldCoverage] = {}
    for field in ENTITY_FIELDS:
        populated = sum(1 for r in rows if _is_populated(r.get(field)))
        pct = (100.0 * populated / total) if total > 0 else 0.0
        result[field] = FieldCoverage(
            field=field,
            populated=populated,
            total=total,
            coverage_pct=round(pct, 4),
        )
    return result


def compute_gap_ranking(
    rows: list[Mapping[str, Any]],
    top_n: int = 10,
) -> list[GapEntityRow]:
    """Rank entity types by the number of astronomy papers missing that field."""
    total = len(rows)
    ranking: list[GapEntityRow] = []
    for field in ENTITY_FIELDS:
        missing = sum(1 for r in rows if not _is_populated(r.get(field)))
        ratio = (missing / total) if total > 0 else 0.0
        ranking.append(
            GapEntityRow(
                entity_type=FIELD_TO_ENTITY_TYPE[field],
                field=field,
                gap_count=missing,
                gap_ratio=round(ratio, 4),
            )
        )
    ranking.sort(key=lambda r: r.gap_count, reverse=True)
    return ranking[:top_n]


def build_report(
    rows: list[Mapping[str, Any]],
    *,
    sample_size: int | None,
    dsn_redacted: str,
    cohort_bibcode_cap: int = DEFAULT_COHORT_BIBCODE_CAP,
    sidecar_path: str | None = DEFAULT_COHORT_SIDECAR_PATH,
) -> GapReport:
    """Assemble the full report from an already-materialized row list."""
    total = len(rows)
    gap_bibcodes: list[str] = [
        str(r["bibcode"]) for r in rows if is_row_in_gap_cohort(r) and r.get("bibcode")
    ]
    gap_count = len(gap_bibcodes)
    gap_ratio = (gap_count / total) if total > 0 else 0.0

    if gap_count > cohort_bibcode_cap:
        sample = gap_bibcodes[:cohort_bibcode_cap]
        sidecar = sidecar_path
    else:
        sample = list(gap_bibcodes)
        sidecar = None

    meta = {
        "sample_size": sample_size,
        "dsn_redacted": dsn_redacted,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gap_ratio_under_40pct": gap_ratio < GAP_COHORT_EXPECTED_UPPER_BOUND,
        "gap_cohort_full_size": gap_count,
    }

    return GapReport(
        total_astronomy_papers=total,
        gap_cohort_count=gap_count,
        gap_cohort_ratio=round(gap_ratio, 4),
        gap_cohort_bibcodes_sample=sample,
        gap_cohort_bibcodes_sidecar_path=sidecar,
        top_gap_entity_types=compute_gap_ranking(rows),
        per_field_coverage=compute_field_coverage(rows),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------


def safety_guard(
    dsn: str | None,
    sample_size: int | None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Refuse to full-scan production unless the caller opts in explicitly.

    Exits with status 2 on violation.  Any other configuration returns silently.
    """
    env = env if env is not None else os.environ
    if sample_size is not None:
        return
    if env.get("SCIX_ALLOW_FULL_SCAN") == "1":
        return
    if is_production_dsn(dsn):
        print(
            "ERROR: refusing full-corpus scan against production DSN "
            f"({redact_dsn(dsn) if dsn else '<default>'}).  "
            "Pass --sample-size N, or set SCIX_ALLOW_FULL_SCAN=1 to override.",
            file=sys.stderr,
        )
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# Database I/O
# ---------------------------------------------------------------------------


def _fetch_rows_with_limit(conn: Any, sample_size: int) -> list[dict[str, Any]]:
    """Fetch a bounded sample using a LIMIT query."""
    query = (
        "SELECT bibcode, facility, data, keyword_norm\n"
        "FROM papers\n"
        f"WHERE {ASTRONOMY_COHORT_SQL}\n"
        "LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(query, (sample_size,))
        fetched = cur.fetchall()
    return [
        {
            "bibcode": bibcode,
            "facility": facility,
            "data": data,
            "keyword_norm": keyword_norm,
        }
        for bibcode, facility, data, keyword_norm in fetched
    ]


def _fetch_rows_streaming(conn: Any, batch_size: int = 50_000) -> list[dict[str, Any]]:
    """Stream the full astronomy cohort via a server-side named cursor."""
    query = (
        "SELECT bibcode, facility, data, keyword_norm\n"
        "FROM papers\n"
        f"WHERE {ASTRONOMY_COHORT_SQL}"
    )
    results: list[dict[str, Any]] = []
    with conn.cursor(name="gap_scan") as cur:
        cur.itersize = batch_size
        cur.execute(query)
        for bibcode, facility, data, keyword_norm in cur:
            results.append(
                {
                    "bibcode": bibcode,
                    "facility": facility,
                    "data": data,
                    "keyword_norm": keyword_norm,
                }
            )
    return results


def fetch_cohort_rows(conn: Any, sample_size: int | None) -> list[dict[str, Any]]:
    """Fetch astronomy papers.  Bounded by sample_size when provided, else full scan."""
    if sample_size is not None:
        logger.info("Fetching bounded sample of %d astronomy papers", sample_size)
        return _fetch_rows_with_limit(conn, sample_size)
    logger.warning("Streaming full astronomy cohort via server-side cursor")
    return _fetch_rows_streaming(conn)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_report(
    report: GapReport,
    output_path: Path,
    *,
    gap_bibcodes_full: list[str],
    cohort_bibcode_cap: int = DEFAULT_COHORT_BIBCODE_CAP,
) -> None:
    """Write the JSON report and (when cohort is large) the sidecar bibcode file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    logger.info("Wrote report to %s", output_path)

    sidecar_path = report.gap_cohort_bibcodes_sidecar_path
    if sidecar_path and len(gap_bibcodes_full) > cohort_bibcode_cap:
        sidecar = Path(sidecar_path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text("\n".join(gap_bibcodes_full) + "\n")
        logger.info("Wrote %d cohort bibcodes to %s", len(gap_bibcodes_full), sidecar)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ADS metadata coverage gaps in the astronomy cohort",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Bound the scan to N rows.  Required when running against the "
            "production DSN unless SCIX_ALLOW_FULL_SCAN=1 is set."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(DEFAULT_OUTPUT_PATH),
        help=f"Path for the JSON report (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (default: SCIX_DSN env var or 'dbname=scix')",
    )
    parser.add_argument(
        "--cohort-bibcode-cap",
        type=int,
        default=DEFAULT_COHORT_BIBCODE_CAP,
        help=(
            "Inline at most this many cohort bibcodes in the JSON.  "
            "A sidecar text file holds the full list when exceeded."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the report and log summary, but do not write files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    effective_dsn = args.dsn or DEFAULT_DSN
    safety_guard(effective_dsn, args.sample_size)

    conn = get_connection(args.dsn)
    try:
        conn.read_only = True  # psycopg sets the session to read-only
    except Exception:
        logger.debug("Could not flip connection to read-only; proceeding anyway")

    try:
        rows = fetch_cohort_rows(conn, args.sample_size)
    finally:
        conn.close()

    report = build_report(
        rows,
        sample_size=args.sample_size,
        dsn_redacted=redact_dsn(effective_dsn) if effective_dsn else "<none>",
        cohort_bibcode_cap=args.cohort_bibcode_cap,
    )

    gap_bibcodes_full: list[str] = [
        str(r["bibcode"]) for r in rows if is_row_in_gap_cohort(r) and r.get("bibcode")
    ]

    logger.info(
        "astronomy=%d  gap_cohort=%d  gap_ratio=%.4f  under_40pct=%s",
        report.total_astronomy_papers,
        report.gap_cohort_count,
        report.gap_cohort_ratio,
        report.meta["gap_ratio_under_40pct"],
    )
    if not report.meta["gap_ratio_under_40pct"]:
        logger.warning(
            "Gap cohort ratio %.4f >= %.2f — PRD expectation that gap cohort "
            "is <40%% of astronomy papers does not hold for this slice.",
            report.gap_cohort_ratio,
            GAP_COHORT_EXPECTED_UPPER_BOUND,
        )

    if args.dry_run:
        logger.info("--dry-run: skipping file writes")
    else:
        write_report(
            report,
            args.output_path,
            gap_bibcodes_full=gap_bibcodes_full,
            cohort_bibcode_cap=args.cohort_bibcode_cap,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
