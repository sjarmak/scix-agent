#!/usr/bin/env python3
"""Incremental daily-sync entity linker — PRD §M10 / u13.

Reads the last watermark from ``link_runs``, selects all papers with
``entry_date > watermark``, runs Tier 1 (keyword match) and Tier 2
(Aho-Corasick abstract match) against that scoped set inside a
5-minute circuit-breaker budget, and appends a new ``link_runs`` row.

On circuit-breaker trip the run drops into graceful-degradation mode:

* the remaining tier-1 / tier-2 work is skipped,
* the watermark still advances (so tomorrow's run doesn't re-process the
  same papers), and
* ``link_runs.status`` is written as ``'tripped'``.

``scripts/link_catchup.py`` is the companion off-peak job that picks up
papers a tripped run skipped.

Two consecutive tripped runs (checked against the two most recent
``link_runs`` rows) emit a row into ``alerts`` with ``severity='page'``.

A separate staleness check compares ``now()`` to the latest watermark
and fires ``severity='page'`` alert when the gap exceeds 24 hours.

Usage::

    python scripts/link_incremental.py                          # prod DSN
    SCIX_TEST_DSN=dbname=scix_test \\
      python scripts/link_incremental.py --budget-seconds 300 -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import psycopg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scix.aho_corasick import (  # noqa: E402
    AhocorasickAutomaton,
    build_automaton,
    link_abstract,
)
from scix.circuit_breaker import CircuitBreaker, CircuitBreakerOpen  # noqa: E402
from scix.db import DEFAULT_DSN, get_connection  # noqa: E402

import link_tier2  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# PRD §M10 budget: 5 minutes of wall-clock time across both tiers.
DEFAULT_BUDGET_SECONDS: float = 300.0

# Default lookback floor for the very first run (no rows in link_runs).
EPOCH_FLOOR = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Paper batch size when iterating the tier-2 scoped set.
INCREMENTAL_BATCH_SIZE: int = 256

# Optional pre-built automaton pickle. If present, the incremental runner
# loads it instead of rebuilding from the DB. Built by a separate pipeline.
DEFAULT_AC_AUTOMATON_PATH: pathlib.Path = REPO_ROOT / "data" / "entities" / "ac_automaton.pkl"

# Watermark staleness threshold. Beyond this gap we page operators.
STALENESS_THRESHOLD = timedelta(hours=24)


# ---------------------------------------------------------------------------
# Scoped tier-1 SQL
# ---------------------------------------------------------------------------
#
# Adapter: the base tier-1 SQL in scripts/link_tier1.py scans every paper.
# For incremental runs we scope the paper side to a temp "incremental"
# bibcode set. Writes are still `ON CONFLICT DO NOTHING`-safe so re-runs
# over overlapping sets are idempotent.
#
# The temp table ``_u13_incremental_bibcodes`` is created by
# :func:`_seed_incremental_scope` earlier in the same transaction.

_INCREMENTAL_TIER1_SQL = """
WITH canonical_matches AS (
    SELECT
        p.bibcode                 AS bibcode,
        e.id                      AS entity_id,
        k                         AS matched_keyword,
        'canonical'::text         AS match_source
    FROM _u13_incremental_bibcodes s
    JOIN papers p ON p.bibcode = s.bibcode
    CROSS JOIN LATERAL unnest(p.keywords) AS k
    JOIN entities e ON lower(e.canonical_name) = lower(k)
    WHERE p.keywords IS NOT NULL
),
alias_matches AS (
    SELECT
        p.bibcode                 AS bibcode,
        e.id                      AS entity_id,
        k                         AS matched_keyword,
        'alias'::text             AS match_source
    FROM _u13_incremental_bibcodes s
    JOIN papers p ON p.bibcode = s.bibcode
    CROSS JOIN LATERAL unnest(p.keywords) AS k
    JOIN entity_aliases ea ON lower(ea.alias) = lower(k)
    JOIN entities e ON e.id = ea.entity_id
    WHERE p.keywords IS NOT NULL
),
all_matches AS (
    SELECT * FROM canonical_matches
    UNION
    SELECT * FROM alias_matches
),
collapsed AS (
    SELECT DISTINCT ON (bibcode, entity_id)
        bibcode, entity_id, matched_keyword, match_source
    FROM all_matches
    ORDER BY bibcode, entity_id, match_source
)
INSERT INTO document_entities (  -- noqa: resolver-lint (u13 incremental; scripts/ is outside AST-lint scope)
    bibcode, entity_id, link_type, tier, tier_version,
    confidence, match_method, evidence
)
SELECT
    bibcode,
    entity_id,
    'keyword_match'                                                       AS link_type,
    1::smallint                                                           AS tier,
    1                                                                     AS tier_version,
    1.0::real                                                             AS confidence,
    'keyword_exact_lower'                                                 AS match_method,
    jsonb_build_object('keyword', matched_keyword, 'match_source', match_source) AS evidence
FROM collapsed
ON CONFLICT (bibcode, entity_id, link_type, tier) DO NOTHING
"""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IncrementalResult:
    """Summary of one incremental run."""

    run_id: int
    new_watermark: Optional[datetime]
    papers_in_scope: int
    tier1_rows: int
    tier2_rows: int
    status: str  # 'ok' | 'tripped' | 'failed'
    trip_count: int
    alerts_emitted: int


# ---------------------------------------------------------------------------
# Watermark + alert helpers
# ---------------------------------------------------------------------------


def get_last_watermark(conn: psycopg.Connection) -> Optional[datetime]:
    """Return the most recent ``link_runs.max_entry_date``, or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT max_entry_date FROM link_runs "
            "WHERE max_entry_date IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT 1"
        )
        row = cur.fetchone()
    if row is None:
        return None
    return row[0]


def _recent_statuses(conn: psycopg.Connection, n: int = 2) -> list[str]:
    """Return the statuses of the ``n`` most recent link_runs rows."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM link_runs ORDER BY timestamp DESC LIMIT %s",
            (n,),
        )
        return [r[0] for r in cur.fetchall()]


def emit_alert(
    conn: psycopg.Connection,
    *,
    severity: str,
    source: str,
    message: str,
) -> int:
    """Insert a row into ``alerts``; return the generated id."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO alerts (severity, source, message) " "VALUES (%s, %s, %s) RETURNING id",
            (severity, source, message),
        )
        row = cur.fetchone()
    conn.commit()
    assert row is not None
    return int(row[0])


def check_watermark_staleness(
    conn: psycopg.Connection,
    *,
    threshold: timedelta = STALENESS_THRESHOLD,
    now: Optional[datetime] = None,
) -> Optional[int]:
    """Emit a page alert when ``now - latest_watermark`` exceeds threshold.

    Returns the alert id when fired, else None.
    """
    now = now or datetime.now(timezone.utc)
    last = get_last_watermark(conn)
    if last is None:
        # No watermark yet — treat as not stale (first run hasn't happened).
        return None
    if now - last > threshold:
        gap = now - last
        return emit_alert(
            conn,
            severity="page",
            source="watermark_staleness",
            message=(
                f"link_runs watermark is stale: last max_entry_date={last.isoformat()}, "
                f"gap={gap}, threshold={threshold}"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# Scope seeding
# ---------------------------------------------------------------------------


def _seed_incremental_scope(
    conn: psycopg.Connection,
    watermark: Optional[datetime],
) -> tuple[int, Optional[datetime]]:
    """Create (or replace) ``_u13_incremental_bibcodes`` temp table.

    Populates with every paper whose ``entry_date::timestamptz`` exceeds
    ``watermark`` (or everything if ``watermark`` is None). Returns the
    number of bibcodes seeded and the new max entry_date observed.
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _u13_incremental_bibcodes")
        cur.execute(
            "CREATE TEMP TABLE _u13_incremental_bibcodes ("
            "    bibcode TEXT PRIMARY KEY"
            ") ON COMMIT DROP"
        )
        if watermark is None:
            cur.execute(
                "INSERT INTO _u13_incremental_bibcodes (bibcode) "
                "SELECT bibcode FROM papers "
                "WHERE NULLIF(entry_date, '') IS NOT NULL"
            )
        else:
            cur.execute(
                "INSERT INTO _u13_incremental_bibcodes (bibcode) "
                "SELECT bibcode FROM papers "
                "WHERE NULLIF(entry_date, '') IS NOT NULL "
                "  AND entry_date::timestamptz > %s",
                (watermark,),
            )
        cur.execute("SELECT count(*) FROM _u13_incremental_bibcodes")
        count = int(cur.fetchone()[0])

        cur.execute(
            "SELECT max(entry_date::timestamptz) FROM papers p "
            "JOIN _u13_incremental_bibcodes s ON s.bibcode = p.bibcode "
            "WHERE NULLIF(p.entry_date, '') IS NOT NULL"
        )
        new_wm_row = cur.fetchone()
    new_wm = new_wm_row[0] if new_wm_row else None
    return count, new_wm


# ---------------------------------------------------------------------------
# Tier-1 / Tier-2 scoped runners
# ---------------------------------------------------------------------------


def run_tier1_scoped(conn: psycopg.Connection) -> int:
    """Run tier-1 keyword-match linker against the temp scope table."""
    with conn.cursor() as cur:
        cur.execute(_INCREMENTAL_TIER1_SQL)
        inserted = cur.rowcount or 0
    return int(inserted)


def _load_or_build_automaton(
    conn: psycopg.Connection,
    *,
    automaton_path: pathlib.Path = DEFAULT_AC_AUTOMATON_PATH,
) -> AhocorasickAutomaton:
    """Return an Aho-Corasick automaton — loaded from disk if available,
    otherwise built inline from ``curated_entity_core``.

    Per the PRD environment note, ``data/entities/ac_automaton.pkl`` will
    be produced by a separate pipeline in prod. In tests and during
    bootstrap the file won't exist, so we fall back to an inline build.
    """
    if automaton_path.exists():
        try:
            with automaton_path.open("rb") as fh:
                automaton = pickle.load(fh)
            logger.info("Loaded prebuilt automaton from %s", automaton_path)
            return automaton
        except (pickle.PickleError, EOFError, OSError) as exc:
            logger.warning(
                "Failed to load automaton from %s (%s); rebuilding inline",
                automaton_path,
                exc,
            )

    logger.info("Building automaton inline from curated_entity_core")
    rows = link_tier2.fetch_entity_rows(conn)
    return build_automaton(rows)


def run_tier2_scoped(
    conn: psycopg.Connection,
    *,
    breaker: CircuitBreaker,
    automaton: AhocorasickAutomaton,
    max_per_entity: int = link_tier2.DEFAULT_MAX_PER_ENTITY,
) -> int:
    """Run tier-2 Aho-Corasick linker against the temp scope table.

    Iterates ``_u13_incremental_bibcodes`` joined to ``papers`` and writes
    tier-2 rows via the same INSERT template the full tier-2 job uses.
    Checks the breaker on every batch so a budget trip stops mid-run.
    """
    # Fetch incremental papers with abstracts in one query; for
    # incremental runs the set is small (hundreds at most), so a single
    # materialized fetch is simpler than a server-side cursor.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p.bibcode, p.abstract "
            "  FROM _u13_incremental_bibcodes s "
            "  JOIN papers p ON p.bibcode = s.bibcode "
            " WHERE p.abstract IS NOT NULL"
        )
        papers: list[tuple[str, str]] = [(b, a or "") for b, a in cur.fetchall()]

    if not papers:
        return 0

    per_entity_count: dict[int, int] = {}
    demoted: set[int] = set()
    rows_inserted = 0

    with conn.cursor() as insert_cur:
        for start in range(0, len(papers), INCREMENTAL_BATCH_SIZE):
            breaker.check()  # raises CircuitBreakerOpen if budget gone
            batch = papers[start : start + INCREMENTAL_BATCH_SIZE]

            for bibcode, abstract in batch:
                hits = link_abstract(abstract, automaton)
                if not hits:
                    continue

                # Collapse to one hit per entity (earliest start)
                earliest: dict[int, "link_tier2.LinkCandidate"] = {}
                for cand in hits:
                    prev = earliest.get(cand.entity_id)
                    if prev is None or cand.start < prev.start:
                        earliest[cand.entity_id] = cand

                for cand in earliest.values():
                    entity_id = cand.entity_id
                    current = per_entity_count.get(entity_id, 0)
                    if current >= max_per_entity:
                        if entity_id not in demoted:
                            demoted.add(entity_id)
                            insert_cur.execute(
                                link_tier2._DEMOTE_SQL,
                                (entity_id,),
                            )
                        continue

                    evidence_json = json.dumps(
                        {
                            "matched_surface": cand.matched_surface,
                            "start": cand.start,
                            "end": cand.end,
                            "ambiguity_class": cand.ambiguity_class,
                            "is_alias": cand.is_alias,
                            "tier2_confidence_source": "aho_corasick+placeholder",
                            "source_run": "u13_incremental",
                        },
                        separators=(",", ":"),
                    )
                    insert_cur.execute(
                        link_tier2._INSERT_SQL,  # noqa: resolver-lint
                        {
                            "bibcode": bibcode,
                            "entity_id": entity_id,
                            "link_type": link_tier2.TIER2_LINK_TYPE,
                            "tier": link_tier2.TIER2_TIER,
                            "tier_version": link_tier2.TIER2_TIER_VERSION,
                            "confidence": cand.confidence,
                            "match_method": link_tier2.TIER2_MATCH_METHOD,
                            "evidence": evidence_json,
                        },
                    )
                    inserted = insert_cur.rowcount or 0
                    if inserted:
                        rows_inserted += inserted
                        per_entity_count[entity_id] = current + inserted

    return rows_inserted


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------


def _write_run_row(
    conn: psycopg.Connection,
    *,
    max_entry_date: Optional[datetime],
    rows_linked: int,
    status: str,
    trip_count: int,
    note: Optional[str] = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO link_runs "
            "(max_entry_date, rows_linked, status, trip_count, note) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING run_id",
            (max_entry_date, rows_linked, status, trip_count, note),
        )
        row = cur.fetchone()
    conn.commit()
    assert row is not None
    return int(row[0])


def run_incremental(
    conn: psycopg.Connection,
    *,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    automaton_path: pathlib.Path = DEFAULT_AC_AUTOMATON_PATH,
    max_per_entity: int = link_tier2.DEFAULT_MAX_PER_ENTITY,
    breaker: Optional[CircuitBreaker] = None,
) -> IncrementalResult:
    """Run one incremental tier-1 + tier-2 pass under a time budget.

    Parameters
    ----------
    conn:
        Open psycopg connection.
    budget_seconds:
        Wall-clock budget for the whole run. Breaker trips once exceeded.
    automaton_path:
        Optional pre-built Aho-Corasick pickle.
    max_per_entity:
        Per-entity linkage cap handed to the tier-2 pipeline.
    breaker:
        Optional pre-constructed :class:`CircuitBreaker` — tests inject
        one with ``budget_seconds=0.001`` to force a trip.
    """
    breaker = breaker or CircuitBreaker(budget_seconds=budget_seconds)
    breaker.start()

    watermark = get_last_watermark(conn)
    logger.info("last watermark: %s", watermark)

    count, new_watermark = _seed_incremental_scope(conn, watermark)
    logger.info("incremental scope: %d papers", count)

    # The temp table was created ON COMMIT DROP; keep the same transaction
    # alive until we're done writing tier-1 and tier-2 rows.

    alerts_emitted = 0
    tier1_rows = 0
    tier2_rows = 0
    status = "ok"
    note: Optional[str] = None

    try:
        # Tier 1
        breaker.check()
        tier1_rows = run_tier1_scoped(conn)
        logger.info("tier-1 inserted %d rows", tier1_rows)

        # Tier 2
        breaker.check()
        automaton = _load_or_build_automaton(conn, automaton_path=automaton_path)
        tier2_rows = run_tier2_scoped(
            conn,
            breaker=breaker,
            automaton=automaton,
            max_per_entity=max_per_entity,
        )
        logger.info("tier-2 inserted %d rows", tier2_rows)

    except CircuitBreakerOpen as exc:
        status = "tripped"
        note = str(exc)
        logger.warning("circuit breaker tripped: %s", exc)
        # Do NOT rollback the rows we already wrote — partial progress
        # stays in document_entities, catchup job handles the gap.

    # Commit tier-1 and tier-2 rows (if any).
    conn.commit()

    # After commit, the ON COMMIT DROP temp table is gone. Now write the
    # run row in a fresh transaction.
    run_id = _write_run_row(
        conn,
        max_entry_date=new_watermark if new_watermark is not None else watermark,
        rows_linked=tier1_rows + tier2_rows,
        status=status,
        trip_count=breaker.trip_count,
        note=note,
    )

    # 2-consecutive-trip paging check
    if status == "tripped":
        recent = _recent_statuses(conn, n=2)
        if len(recent) >= 2 and all(s == "tripped" for s in recent):
            emit_alert(
                conn,
                severity="page",
                source="incremental_sync",
                message=(
                    "circuit breaker tripped on 2 consecutive runs — "
                    "entity linking is falling behind"
                ),
            )
            alerts_emitted += 1

    # Standing watermark-staleness check
    if check_watermark_staleness(conn) is not None:
        alerts_emitted += 1

    return IncrementalResult(
        run_id=run_id,
        new_watermark=new_watermark if new_watermark is not None else watermark,
        papers_in_scope=count,
        tier1_rows=tier1_rows,
        tier2_rows=tier2_rows,
        status=status,
        trip_count=breaker.trip_count,
        alerts_emitted=alerts_emitted,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-url", type=str, default=None)
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=DEFAULT_BUDGET_SECONDS,
        help="Circuit-breaker wall-clock budget (default: 300s / 5min)",
    )
    parser.add_argument(
        "--max-per-entity",
        type=int,
        default=link_tier2.DEFAULT_MAX_PER_ENTITY,
    )
    parser.add_argument(
        "--automaton-path",
        type=str,
        default=str(DEFAULT_AC_AUTOMATON_PATH),
        help="Path to prebuilt Aho-Corasick pickle (optional)",
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
        result = run_incremental(
            conn,
            budget_seconds=args.budget_seconds,
            automaton_path=pathlib.Path(args.automaton_path),
            max_per_entity=args.max_per_entity,
        )
    finally:
        conn.close()

    print(
        f"link_incremental: run_id={result.run_id} "
        f"status={result.status} scope={result.papers_in_scope} "
        f"tier1={result.tier1_rows} tier2={result.tier2_rows} "
        f"alerts={result.alerts_emitted} trips={result.trip_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
