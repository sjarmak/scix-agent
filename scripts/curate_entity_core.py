#!/usr/bin/env python3
"""Curate the high-query-value entity core (M3.5.1).

Three-pass ranking against ``query_log`` + ``entities``:

  Pass 1 ("gap candidates")
      Queries with ``result_count = 0`` in the last ``window_days`` —
      surface the entity names the user is asking for but that we can't
      find. We attempt to bind the query string to an existing
      ``entities`` row via case-insensitive exact match on
      ``canonical_name`` or ``entity_aliases.alias``.

  Pass 2 ("Wikidata backfill gap closer") — N2 FUTURE WORK
      For gap candidates where no ``entities`` row exists yet, Wikidata
      backfill would add the entity. Implemented here as a no-op stub.

  Pass 3 ("unique + >=1 hit")
      Entities with ``ambiguity_class = 'unique'`` that received at
      least one query hit in the window.

Results are unioned, deduped on ``entity_id``, ranked by
``query_hits_14d DESC`` with ``source`` as a tiebreaker, and truncated
to the hard cap (``--max``, default 10,000).

Outputs:
  * ``build-artifacts/curated_core.csv`` — 7-column CSV
  * ``build-artifacts/curated_core_stratification.md`` — per-source counts
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scix import core_lifecycle  # noqa: E402
from scix.db import DEFAULT_DSN, get_connection  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MAX: int = 10_000
DEFAULT_WINDOW_DAYS: int = 14

CSV_COLUMNS = [
    "entity_id",
    "canonical_name",
    "source",
    "ambiguity_class",
    "pass_triggered",
    "query_hits_14d",
    "zero_result_hits_14d",
]


@dataclass(frozen=True)
class CoreRow:
    entity_id: int
    canonical_name: str
    source: str
    ambiguity_class: str
    pass_triggered: str
    query_hits_14d: int
    zero_result_hits_14d: int


def _pass1_gap_candidates(
    conn: psycopg.Connection,
    window_days: int,
    is_test: bool,
    session_id: str | None = None,
) -> list[CoreRow]:
    """Bind zero-result queries to existing entities via canonical/alias match."""
    session_clause = " AND session_id = %s" if session_id is not None else ""
    session_params: tuple = (session_id,) if session_id is not None else ()
    sql = f"""
        WITH zero_q AS (
            SELECT lower(trim(query)) AS q, COUNT(*) AS n
              FROM query_log
             WHERE ts >= now() - (%s::text || ' days')::interval
               AND result_count = 0
               AND query IS NOT NULL
               AND is_test = %s
               {session_clause}
             GROUP BY lower(trim(query))
        ),
        matched AS (
            SELECT e.id            AS entity_id,
                   e.canonical_name,
                   e.source,
                   COALESCE(e.ambiguity_class::text, 'unknown') AS ambiguity_class,
                   z.n              AS zero_hits
              FROM zero_q z
              JOIN entities e ON lower(e.canonical_name) = z.q
            UNION
            SELECT e.id            AS entity_id,
                   e.canonical_name,
                   e.source,
                   COALESCE(e.ambiguity_class::text, 'unknown') AS ambiguity_class,
                   z.n              AS zero_hits
              FROM zero_q z
              JOIN entity_aliases a ON lower(a.alias) = z.q
              JOIN entities e       ON e.id = a.entity_id
        )
        SELECT entity_id, canonical_name, source, ambiguity_class,
               SUM(zero_hits)::int AS zero_hits
          FROM matched
         GROUP BY entity_id, canonical_name, source, ambiguity_class
         ORDER BY zero_hits DESC
    """
    rows: list[CoreRow] = []
    with conn.cursor() as cur:
        cur.execute(sql, (window_days, is_test, *session_params))
        for entity_id, canonical, source, ambig, zero_hits in cur.fetchall():
            rows.append(
                CoreRow(
                    entity_id=int(entity_id),
                    canonical_name=canonical,
                    source=source,
                    ambiguity_class=ambig,
                    pass_triggered="pass1_gap",
                    query_hits_14d=0,
                    zero_result_hits_14d=int(zero_hits),
                )
            )
    return rows


def _pass2_wikidata_backfill() -> list[CoreRow]:
    """Pass 2: Wikidata gap closer. No-op stub — N2 future work.

    TODO(u07/N2): For gap candidates whose query string does not match any
    existing entities row, call the Wikidata resolver, create a new
    entities row, and emit it here with pass_triggered='pass2_wikidata'.
    """
    return []


def _pass3_unique_with_hits(
    conn: psycopg.Connection,
    window_days: int,
    is_test: bool,
    session_id: str | None = None,
) -> list[CoreRow]:
    session_clause = " AND session_id = %s" if session_id is not None else ""
    session_params: tuple = (session_id,) if session_id is not None else ()
    sql = f"""
        WITH hit_q AS (
            SELECT lower(trim(query)) AS q, COUNT(*) AS n
              FROM query_log
             WHERE ts >= now() - (%s::text || ' days')::interval
               AND result_count > 0
               AND query IS NOT NULL
               AND is_test = %s
               {session_clause}
             GROUP BY lower(trim(query))
        ),
        matched AS (
            SELECT e.id            AS entity_id,
                   e.canonical_name,
                   e.source,
                   e.ambiguity_class::text AS ambiguity_class,
                   h.n              AS hits
              FROM hit_q h
              JOIN entities e ON lower(e.canonical_name) = h.q
             WHERE e.ambiguity_class = 'unique'
            UNION
            SELECT e.id            AS entity_id,
                   e.canonical_name,
                   e.source,
                   e.ambiguity_class::text AS ambiguity_class,
                   h.n              AS hits
              FROM hit_q h
              JOIN entity_aliases a ON lower(a.alias) = h.q
              JOIN entities e       ON e.id = a.entity_id
             WHERE e.ambiguity_class = 'unique'
        )
        SELECT entity_id, canonical_name, source, ambiguity_class,
               SUM(hits)::int AS hits
          FROM matched
         GROUP BY entity_id, canonical_name, source, ambiguity_class
        HAVING SUM(hits) >= 1
         ORDER BY hits DESC
    """
    rows: list[CoreRow] = []
    with conn.cursor() as cur:
        cur.execute(sql, (window_days, is_test, *session_params))
        for entity_id, canonical, source, ambig, hits in cur.fetchall():
            rows.append(
                CoreRow(
                    entity_id=int(entity_id),
                    canonical_name=canonical,
                    source=source,
                    ambiguity_class=ambig,
                    pass_triggered="pass3_unique_with_hits",
                    query_hits_14d=int(hits),
                    zero_result_hits_14d=0,
                )
            )
    return rows


def _merge(rows: list[CoreRow]) -> list[CoreRow]:
    """Dedupe by entity_id, summing hit counts. First-seen pass wins the
    ``pass_triggered`` label (pass1 before pass3 in the caller's order)."""
    by_id: dict[int, CoreRow] = {}
    for r in rows:
        prev = by_id.get(r.entity_id)
        if prev is None:
            by_id[r.entity_id] = r
        else:
            by_id[r.entity_id] = CoreRow(
                entity_id=prev.entity_id,
                canonical_name=prev.canonical_name,
                source=prev.source,
                ambiguity_class=prev.ambiguity_class,
                pass_triggered=prev.pass_triggered,
                query_hits_14d=prev.query_hits_14d + r.query_hits_14d,
                zero_result_hits_14d=prev.zero_result_hits_14d + r.zero_result_hits_14d,
            )
    return list(by_id.values())


def _rank_and_cap(rows: list[CoreRow], max_n: int) -> list[CoreRow]:
    """Rank by (query_hits DESC, zero_result_hits DESC, source ASC) then cap."""
    ranked = sorted(
        rows,
        key=lambda r: (-r.query_hits_14d, -r.zero_result_hits_14d, r.source or ""),
    )
    return ranked[:max_n]


def _write_csv(rows: list[CoreRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow(
                [
                    r.entity_id,
                    r.canonical_name,
                    r.source,
                    r.ambiguity_class,
                    r.pass_triggered,
                    r.query_hits_14d,
                    r.zero_result_hits_14d,
                ]
            )


def _write_stratification(rows: list[CoreRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.source] = counts.get(r.source, 0) + 1
    total = sum(counts.values())
    lines: list[str] = []
    lines.append("# Curated Entity Core — Stratification")
    lines.append("")
    lines.append(f"Total rows: **{total}**")
    lines.append("")
    lines.append("| source | count | pct |")
    lines.append("|---|---:|---:|")
    for source in sorted(counts.keys()):
        c = counts[source]
        pct = (100.0 * c / total) if total else 0.0
        lines.append(f"| {source} | {c} | {pct:.1f}% |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _populate_core_table(rows: list[CoreRow], conn: psycopg.Connection) -> int:
    """Promote each curated row into ``curated_entity_core`` via
    :func:`core_lifecycle.promote`. Returns the number of rows promoted.

    Each promote() commits individually (incremental). On interruption,
    already-promoted rows persist; re-running is safe (upsert idempotent).
    """
    promoted = 0
    for r in rows:
        hits = r.query_hits_14d + r.zero_result_hits_14d
        core_lifecycle.promote(
            r.entity_id,
            query_hits_14d=hits,
            reason=f"curate_{r.pass_triggered}",
            conn=conn,
        )
        promoted += 1
    logger.info("Populated curated_entity_core: %d rows promoted", promoted)
    return promoted


def run_curation(
    conn: psycopg.Connection,
    csv_path: Path,
    strat_path: Path,
    window_days: int = DEFAULT_WINDOW_DAYS,
    max_n: int = DEFAULT_MAX,
    is_test: bool = False,
    populate: bool = False,
    session_id: str | None = None,
) -> list[CoreRow]:
    """Run the three-pass curation. Returns the final (ranked, capped) rows.

    When *populate* is True, also upsert results into the
    ``curated_entity_core`` table via :mod:`scix.core_lifecycle`.

    When *session_id* is set, pass1 and pass3 only consider query_log rows
    whose ``session_id`` matches — used to curate a reproducible snapshot
    from a seed bootstrap run without mixing in concurrent organic traffic.
    """
    logger.info(
        "Pass 1: gap candidates (window=%dd%s)",
        window_days,
        f", session={session_id}" if session_id else "",
    )
    p1 = _pass1_gap_candidates(conn, window_days, is_test, session_id)
    logger.info("  -> %d", len(p1))

    logger.info("Pass 2: Wikidata backfill (stub)")
    p2 = _pass2_wikidata_backfill()
    logger.info("  -> %d", len(p2))

    logger.info("Pass 3: unique + >=1 hit")
    p3 = _pass3_unique_with_hits(conn, window_days, is_test, session_id)
    logger.info("  -> %d", len(p3))

    merged = _merge(p1 + p2 + p3)
    capped = _rank_and_cap(merged, max_n)
    assert len(capped) <= max_n, f"hard cap violated: {len(capped)} > {max_n}"

    _write_csv(capped, csv_path)
    _write_stratification(capped, strat_path)
    logger.info("Wrote %d rows -> %s", len(capped), csv_path)

    if populate:
        _populate_core_table(capped, conn)

    return capped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--output", type=Path, default=Path("build-artifacts/curated_core.csv"))
    parser.add_argument(
        "--strat-output",
        type=Path,
        default=Path("build-artifacts/curated_core_stratification.md"),
    )
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--max", type=int, default=DEFAULT_MAX, dest="max_n")
    parser.add_argument(
        "--include-test-traffic",
        action="store_true",
        help="Treat rows with is_test=true as curation inputs (default: false)",
    )
    parser.add_argument(
        "--populate",
        action="store_true",
        help="Also upsert curated rows into curated_entity_core table",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Restrict pass1/pass3 to query_log rows tagged with this session_id",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dsn = args.dsn or os.environ.get("SCIX_TEST_DSN") or DEFAULT_DSN
    conn = get_connection(dsn)
    try:
        rows = run_curation(
            conn,
            csv_path=args.output,
            strat_path=args.strat_output,
            window_days=args.window_days,
            max_n=args.max_n,
            is_test=args.include_test_traffic,
            populate=args.populate,
            session_id=args.session_id,
        )
        print(f"Curated {len(rows)} rows")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
