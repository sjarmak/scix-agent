#!/usr/bin/env python3
"""Populate the ``co_mentions`` table from ``document_entities``.

Builds the entity↔entity co-occurrence summary defined in migration 063.
For every unordered pair (a, b) of entities that co-occur in at least two
papers, one row is written with:

    n_papers   — distinct supporting bibcodes
    first_year — earliest papers.year across the support set
    last_year  — latest   papers.year across the support set

Acceptance criteria (scix_experiments-dbl.5):
- Materialized view co_mentions populated  → done by this script
- Indexed for symmetric lookup by either entity → migration 063
- Refresh strategy documented → docs/prd/co_mentions.md
- 1 new MCP tool method or extension to entity_context that surfaces
  top-k co-mentions → see scix.search.get_top_co_mentions

Strategy
--------

The naive approach ``INSERT INTO co_mentions SELECT … FROM document_entities
de1 JOIN document_entities de2 …`` materializes ~150M pair instances in a
single hash aggregate. With 256MB work_mem that spills heavily and risks
busting the 30G scix-batch ceiling. Instead we chunk by ``papers.year``
and aggregate into a temporary staging table, then do a final
GROUP-BY-merge into ``co_mentions``.

Chunk shape: per year, build pair instances and aggregate within the year
into ``tmp_co_mentions_partial(a, b, n_papers_partial, min_year_partial,
max_year_partial)``. After all years are processed the merge step computes::

    SELECT a, b, SUM(n_papers_partial),
           MIN(min_year_partial), MAX(max_year_partial)
    FROM tmp_co_mentions_partial
    GROUP BY a, b
    HAVING SUM(n_papers_partial) >= 2;

n_papers across years is additive because each (paper, a, b) appears in
exactly one year-chunk. There is no double counting risk.

Refresh kinds
-------------

* ``full``  — TRUNCATE co_mentions and rebuild from scratch. The default.
* ``pilot`` — process a single year only, write to co_mentions
              non-destructively, log to co_mention_runs as 'pilot'. Used
              for smoke tests / runtime estimation.

Incremental refresh is not implemented in this script; see
docs/prd/co_mentions.md §"Refresh strategy" for why and the documented
escape hatch (rebuild monthly, document staleness).

Usage
-----

::

    # Pilot on a single recent year (small, fast)
    SCIX_TEST_DSN="dbname=scix_test" \\
    python scripts/populate_co_mentions.py --refresh-kind pilot --pilot-year 2025

    # Full rebuild against test DB
    SCIX_TEST_DSN="dbname=scix_test" \\
    python scripts/populate_co_mentions.py

    # Full rebuild against production (must run via scix-batch)
    scix-batch python scripts/populate_co_mentions.py --allow-prod

The ``--allow-prod`` flag is required when targeting the production scix
DB and refuses to run outside a systemd scope (mirrors
recompute_citation_communities.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("populate_co_mentions")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Postgres knob raised for hash aggregates over pair instances. Per-session
# only; reverts at connection close.
SESSION_WORK_MEM = "4GB"

# Minimum support — enforced at write time and by the table CHECK constraint.
DEFAULT_MIN_N_PAPERS = 2

# Year-chunked work table. Created as TEMP to auto-drop on commit.
STAGING_TABLE_DDL = """
    CREATE TEMP TABLE tmp_co_mentions_partial (
        entity_a_id        INTEGER NOT NULL,
        entity_b_id        INTEGER NOT NULL,
        n_papers_partial   INTEGER NOT NULL,
        min_year_partial   SMALLINT,
        max_year_partial   SMALLINT
    )
"""

# Partial aggregate for one year. We group within the year so the per-year
# pair-instance volume collapses before we union years together.
#
# DISTINCT bibcode is necessary because document_entities can have multiple
# rows for the same (bibcode, entity_id, link_type, tier) — we only want to
# count each paper once toward a pair.
INSERT_PARTIAL_FOR_YEAR_SQL = """
    INSERT INTO tmp_co_mentions_partial
        (entity_a_id, entity_b_id, n_papers_partial,
         min_year_partial, max_year_partial)
    SELECT
        de1.entity_id AS entity_a_id,
        de2.entity_id AS entity_b_id,
        COUNT(DISTINCT de1.bibcode) AS n_papers_partial,
        %(year)s::smallint AS min_year_partial,
        %(year)s::smallint AS max_year_partial
    FROM document_entities de1
    JOIN document_entities de2
      ON de1.bibcode = de2.bibcode
     AND de1.entity_id < de2.entity_id
    JOIN papers p
      ON p.bibcode = de1.bibcode
     AND p.year = %(year)s
    GROUP BY de1.entity_id, de2.entity_id
"""

# Same shape as above but for papers.year IS NULL — runs once per
# refresh and contributes pairs without a year envelope.
INSERT_PARTIAL_FOR_NULL_YEAR_SQL = """
    INSERT INTO tmp_co_mentions_partial
        (entity_a_id, entity_b_id, n_papers_partial,
         min_year_partial, max_year_partial)
    SELECT
        de1.entity_id AS entity_a_id,
        de2.entity_id AS entity_b_id,
        COUNT(DISTINCT de1.bibcode) AS n_papers_partial,
        NULL::smallint AS min_year_partial,
        NULL::smallint AS max_year_partial
    FROM document_entities de1
    JOIN document_entities de2
      ON de1.bibcode = de2.bibcode
     AND de1.entity_id < de2.entity_id
    JOIN papers p
      ON p.bibcode = de1.bibcode
     AND p.year IS NULL
    GROUP BY de1.entity_id, de2.entity_id
"""

# Final merge from staging into co_mentions. Filters pairs with cross-year
# support below min_n_papers.
MERGE_SQL = """
    INSERT INTO co_mentions
        (entity_a_id, entity_b_id, n_papers, first_year, last_year)
    SELECT
        entity_a_id,
        entity_b_id,
        SUM(n_papers_partial)::integer AS n_papers,
        MIN(min_year_partial) AS first_year,
        MAX(max_year_partial) AS last_year
    FROM tmp_co_mentions_partial
    GROUP BY entity_a_id, entity_b_id
    HAVING SUM(n_papers_partial) >= %(min_n_papers)s
"""

DISTINCT_YEARS_SQL = """
    SELECT DISTINCT year
    FROM papers
    WHERE year IS NOT NULL
    ORDER BY year
"""

HAS_NULL_YEAR_SQL = "SELECT EXISTS (SELECT 1 FROM papers WHERE year IS NULL)"

LOG_RUN_INSERT = """
    INSERT INTO co_mention_runs
        (refresh_kind, n_papers_input, min_n_papers, git_sha, notes)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id
"""

LOG_RUN_FINISH = """
    UPDATE co_mention_runs
       SET finished_at = now(),
           n_pairs_output = %s,
           notes = %s
     WHERE id = %s
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class ChunkResult:
    year: int | None
    pairs_inserted: int
    elapsed_s: float


@dataclass
class RunMeta:
    refresh_kind: str
    started_at_iso: str
    finished_at_iso: str | None = None
    n_papers_input: int = 0
    n_pairs_output: int = 0
    chunk_results: list[ChunkResult] = field(default_factory=list)
    git_sha: str | None = None

    def to_dict(self) -> dict:
        return {
            "refresh_kind": self.refresh_kind,
            "started_at": self.started_at_iso,
            "finished_at": self.finished_at_iso,
            "n_papers_input": self.n_papers_input,
            "n_pairs_output": self.n_pairs_output,
            "chunk_results": [
                {
                    "year": cr.year,
                    "pairs_inserted": cr.pairs_inserted,
                    "elapsed_s": cr.elapsed_s,
                }
                for cr in self.chunk_results
            ],
            "git_sha": self.git_sha,
        }


def _resolve_dsn(cli_dsn: str | None) -> str:
    if cli_dsn:
        return cli_dsn
    test_dsn = os.environ.get("SCIX_TEST_DSN")
    if test_dsn:
        return test_dsn
    return DEFAULT_DSN


def _git_sha() -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def _count_input_papers(conn: psycopg.Connection, year: int | None) -> int:
    sql = (
        "SELECT COUNT(*) FROM papers WHERE year = %s"
        if year is not None
        else "SELECT COUNT(*) FROM papers"
    )
    args: tuple = (year,) if year is not None else ()
    with conn.cursor() as cur:
        cur.execute(sql, args)
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


# ---------------------------------------------------------------------------
# Chunked population
# ---------------------------------------------------------------------------


def _populate_chunk(
    conn: psycopg.Connection,
    year: int | None,
) -> ChunkResult:
    """Insert per-year partial aggregates into the staging temp table."""
    t0 = time.perf_counter()
    sql = INSERT_PARTIAL_FOR_YEAR_SQL if year is not None else INSERT_PARTIAL_FOR_NULL_YEAR_SQL
    with conn.cursor() as cur:
        cur.execute(sql, {"year": year})
        n_inserted = cur.rowcount
    elapsed = time.perf_counter() - t0
    logger.info(
        "  year=%s pairs_inserted=%d elapsed=%.1fs",
        year if year is not None else "NULL",
        n_inserted,
        elapsed,
    )
    return ChunkResult(year=year, pairs_inserted=n_inserted, elapsed_s=elapsed)


def run(
    dsn: str,
    refresh_kind: str,
    min_n_papers: int,
    pilot_year: int | None,
    log_dir: Path,
) -> RunMeta:
    started_at = time.time()
    started_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at))
    git_sha = _git_sha()
    meta = RunMeta(
        refresh_kind=refresh_kind,
        started_at_iso=started_at_iso,
        git_sha=git_sha,
    )

    log_dir.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(dsn, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL work_mem = '{SESSION_WORK_MEM}'")
            cur.execute("SET LOCAL synchronous_commit = OFF")

            cur.execute(STAGING_TABLE_DDL)

            # Decide year set
            if refresh_kind == "pilot":
                if pilot_year is None:
                    raise SystemExit("pilot refresh requires --pilot-year (an integer year)")
                years: list[int | None] = [pilot_year]
            else:
                cur.execute(DISTINCT_YEARS_SQL)
                years = [int(r[0]) for r in cur.fetchall()]
                cur.execute(HAS_NULL_YEAR_SQL)
                has_null_year = bool(cur.fetchone()[0])  # type: ignore[index]
                if has_null_year:
                    years.append(None)

            logger.info(
                "Processing %d year-chunks (refresh_kind=%s)",
                len(years),
                refresh_kind,
            )

            # Count input papers across the year set
            for y in years:
                meta.n_papers_input += _count_input_papers(conn, y)

            cur.execute(
                LOG_RUN_INSERT,
                (
                    refresh_kind,
                    meta.n_papers_input,
                    min_n_papers,
                    git_sha,
                    f"started_at={started_at_iso}",
                ),
            )
            run_row = cur.fetchone()
            assert run_row is not None
            run_id = int(run_row[0])

            for y in years:
                cr = _populate_chunk(conn, y)
                meta.chunk_results.append(cr)

            if refresh_kind == "full":
                logger.info("TRUNCATE co_mentions")
                cur.execute("TRUNCATE TABLE co_mentions")

            logger.info("Merging staging → co_mentions (min_n_papers=%d)", min_n_papers)
            t_merge = time.perf_counter()
            cur.execute(MERGE_SQL, {"min_n_papers": min_n_papers})
            merge_pairs = cur.rowcount
            logger.info(
                "Merge inserted %d pair rows in %.1fs",
                merge_pairs,
                time.perf_counter() - t_merge,
            )
            meta.n_pairs_output = merge_pairs

            cur.execute(
                LOG_RUN_FINISH,
                (
                    merge_pairs,
                    f"chunks={len(meta.chunk_results)},"
                    f" elapsed_total_s={time.time() - started_at:.1f}",
                    run_id,
                ),
            )
            conn.commit()

    finished_at = time.time()
    meta.finished_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished_at))
    out_path = log_dir / f"run_meta_{started_at_iso.replace(':', '')}.json"
    out_path.write_text(json.dumps(meta.to_dict(), indent=2))
    logger.info("Wrote run metadata: %s", out_path)
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate co_mentions from document_entities.",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (defaults to SCIX_TEST_DSN env, then SCIX_DSN).",
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required to write to production DSN (dbname=scix).",
    )
    parser.add_argument(
        "--refresh-kind",
        choices=("full", "pilot"),
        default="full",
        help="full = TRUNCATE+rebuild; pilot = single year, append-only.",
    )
    parser.add_argument(
        "--pilot-year",
        type=int,
        default=None,
        help="Year to process when --refresh-kind=pilot.",
    )
    parser.add_argument(
        "--min-n-papers",
        type=int,
        default=DEFAULT_MIN_N_PAPERS,
        help=(
            "Minimum support count for a pair to be written to co_mentions"
            f" (default: {DEFAULT_MIN_N_PAPERS}, also enforced by CHECK)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=_REPO_ROOT / "logs" / "co_mentions",
        help="Directory for run_meta_*.json (default: logs/co_mentions/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dsn = _resolve_dsn(args.dsn)

    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "Refusing to write to production DSN %s — pass --allow-prod",
            redact_dsn(dsn),
        )
        return 2
    if args.allow_prod and not os.environ.get("INVOCATION_ID"):
        logger.error(
            "Refusing to run --allow-prod outside a systemd scope. "
            "Invoke via: scix-batch python %s <args...>",
            Path(sys.argv[0]).name,
        )
        return 2

    if args.min_n_papers < 2:
        logger.error("min-n-papers must be >= 2 (table CHECK constraint enforces this)")
        return 2

    if args.refresh_kind == "pilot" and args.pilot_year is None:
        logger.error("--refresh-kind=pilot requires --pilot-year")
        return 2

    logger.info("DSN: %s", redact_dsn(dsn))
    meta = run(
        dsn=dsn,
        refresh_kind=args.refresh_kind,
        min_n_papers=args.min_n_papers,
        pilot_year=args.pilot_year,
        log_dir=args.log_dir,
    )
    logger.info(
        "Summary: kind=%s, n_papers_input=%d, n_pairs_output=%d, chunks=%d",
        meta.refresh_kind,
        meta.n_papers_input,
        meta.n_pairs_output,
        len(meta.chunk_results),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
