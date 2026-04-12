#!/usr/bin/env python3
"""Catch-up entity linker — PRD §M10 / u13.

Companion to ``scripts/link_incremental.py``. When the incremental run
trips its circuit breaker, a chunk of recent papers advances past the
watermark with zero (or partial) entity links. Operators run this script
off-peak to backfill those papers.

Target set:
    All papers whose ``entry_date::timestamptz`` is less than or equal to
    the latest ``link_runs.max_entry_date`` AND that have zero tier-1
    and zero tier-2 rows in ``document_entities``.

The catchup runner deliberately has **no circuit breaker** — when it
runs, it runs to completion (bounded only by ``--limit``). That's the
whole point.

Usage::

    python scripts/link_catchup.py
    SCIX_TEST_DSN=dbname=scix_test \\
      python scripts/link_catchup.py --limit 1000 -v
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional

import psycopg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scix.circuit_breaker import CircuitBreaker  # noqa: E402
from scix.db import DEFAULT_DSN, get_connection  # noqa: E402

import link_incremental  # noqa: E402
import link_tier2  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatchupResult:
    papers_in_scope: int
    tier1_rows: int
    tier2_rows: int


def _seed_catchup_scope(
    conn: psycopg.Connection,
    *,
    limit: Optional[int],
) -> int:
    """Create ``_u13_incremental_bibcodes`` with papers missing any links.

    Uses the same temp-table name as ``link_incremental`` so the scoped
    tier-1 SQL and tier-2 runner both work without modification.
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _u13_incremental_bibcodes")
        cur.execute(
            "CREATE TEMP TABLE _u13_incremental_bibcodes ("
            "    bibcode TEXT PRIMARY KEY"
            ") ON COMMIT DROP"
        )

        # Papers with no tier-1 / tier-2 document_entities rows and whose
        # entry_date is <= the latest watermark (i.e. they were *meant*
        # to be linked by a previous incremental run that tripped).
        sql = (
            "INSERT INTO _u13_incremental_bibcodes (bibcode) "
            "SELECT p.bibcode FROM papers p "
            "WHERE NULLIF(p.entry_date, '') IS NOT NULL "
            "  AND NOT EXISTS ("
            "        SELECT 1 FROM document_entities de "
            "         WHERE de.bibcode = p.bibcode "
            "           AND de.tier IN (1, 2)"
            "    ) "
        )
        # Scope by latest watermark so we don't scan the whole corpus
        # when catchup is invoked against a database where most papers
        # never had links (early bootstrap).
        with conn.cursor() as wm_cur:
            wm_cur.execute(
                "SELECT max(max_entry_date) FROM link_runs " "WHERE max_entry_date IS NOT NULL"
            )
            wm_row = wm_cur.fetchone()
        latest_wm = wm_row[0] if wm_row else None
        params: list = []
        if latest_wm is not None:
            sql += " AND p.entry_date::timestamptz <= %s "
            params.append(latest_wm)
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        cur.execute(sql, params)

        cur.execute("SELECT count(*) FROM _u13_incremental_bibcodes")
        count = int(cur.fetchone()[0])

    return count


def run_catchup(
    conn: psycopg.Connection,
    *,
    limit: Optional[int] = None,
    automaton_path: pathlib.Path = link_incremental.DEFAULT_AC_AUTOMATON_PATH,
    max_per_entity: int = link_tier2.DEFAULT_MAX_PER_ENTITY,
) -> CatchupResult:
    """Run tier-1 + tier-2 against all papers missing both tier-1 and
    tier-2 links (bounded by ``limit`` and the latest watermark).
    """
    count = _seed_catchup_scope(conn, limit=limit)
    logger.info("catchup scope: %d papers", count)
    if count == 0:
        conn.commit()
        return CatchupResult(papers_in_scope=0, tier1_rows=0, tier2_rows=0)

    # No circuit breaker — use an effectively-infinite budget. This keeps
    # the tier-2 runner happy (it calls breaker.check() on every batch)
    # without actually enforcing a timeout.
    breaker = CircuitBreaker(budget_seconds=float("inf"))
    breaker.start()

    tier1_rows = link_incremental.run_tier1_scoped(conn)
    logger.info("catchup tier-1 inserted %d rows", tier1_rows)

    automaton = link_incremental._load_or_build_automaton(conn, automaton_path=automaton_path)
    tier2_rows = link_incremental.run_tier2_scoped(
        conn,
        breaker=breaker,
        automaton=automaton,
        max_per_entity=max_per_entity,
    )
    logger.info("catchup tier-2 inserted %d rows", tier2_rows)

    conn.commit()

    return CatchupResult(
        papers_in_scope=count,
        tier1_rows=tier1_rows,
        tier2_rows=tier2_rows,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-url", type=str, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max papers to process in one catchup pass",
    )
    parser.add_argument(
        "--automaton-path",
        type=str,
        default=str(link_incremental.DEFAULT_AC_AUTOMATON_PATH),
    )
    parser.add_argument(
        "--max-per-entity",
        type=int,
        default=link_tier2.DEFAULT_MAX_PER_ENTITY,
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = args.db_url or os.environ.get("SCIX_TEST_DSN") or DEFAULT_DSN
    conn = get_connection(dsn)
    try:
        result = run_catchup(
            conn,
            limit=args.limit,
            automaton_path=pathlib.Path(args.automaton_path),
            max_per_entity=args.max_per_entity,
        )
    finally:
        conn.close()

    print(
        f"link_catchup: scope={result.papers_in_scope} "
        f"tier1={result.tier1_rows} tier2={result.tier2_rows}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
