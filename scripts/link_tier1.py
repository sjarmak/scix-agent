#!/usr/bin/env python3
"""Tier-1 keyword-match linker — PRD M3 / work unit u06.

Runs a single SQL pass joining ``papers.keywords`` against
``entities.canonical_name`` and ``entity_aliases.alias`` (case-insensitive),
writing new rows into ``document_entities`` with::

    tier = 1
    link_type = 'keyword_match'
    confidence = 1.0
    match_method = 'keyword_exact_lower'

Idempotent on rerun via ``ON CONFLICT (bibcode, entity_id, link_type, tier)
DO NOTHING``.

This is a transitional path: u03's resolver service will eventually own all
entity-link writes, but tier-1 keyword match is exempt for now (the linter
in u03 exempts this script explicitly).

Usage::

    python scripts/link_tier1.py --db-url "dbname=scix"
    python scripts/link_tier1.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys

import psycopg

from scix.db import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-pass SQL
# ---------------------------------------------------------------------------

# noqa: resolver-lint (transitional; u03's AST lint exempts tier-1 scripts)
TIER1_INSERT_SQL = """
WITH canonical_matches AS (
    SELECT
        p.bibcode                 AS bibcode,
        e.id                      AS entity_id,
        k                         AS matched_keyword,
        'canonical'::text         AS match_source
    FROM papers p
    CROSS JOIN LATERAL unnest(p.keywords) AS k
    JOIN entities e ON lower(e.canonical_name) = lower(k)
    WHERE p.keywords IS NOT NULL
      AND COALESCE(e.link_policy::text, 'open') = 'open'
),
alias_matches AS (
    SELECT
        p.bibcode                 AS bibcode,
        e.id                      AS entity_id,
        k                         AS matched_keyword,
        'alias'::text             AS match_source
    FROM papers p
    CROSS JOIN LATERAL unnest(p.keywords) AS k
    JOIN entity_aliases ea ON lower(ea.alias) = lower(k)
    JOIN entities e ON e.id = ea.entity_id
    WHERE p.keywords IS NOT NULL
      AND COALESCE(e.link_policy::text, 'open') = 'open'
),
all_matches AS (
    SELECT * FROM canonical_matches
    UNION
    SELECT * FROM alias_matches
),
-- Collapse duplicate (bibcode, entity_id) — if an entity matches on both
-- canonical and alias, or via two different keywords, we want a single row.
collapsed AS (
    SELECT DISTINCT ON (bibcode, entity_id)
        bibcode,
        entity_id,
        matched_keyword,
        match_source
    FROM all_matches
    ORDER BY bibcode, entity_id, match_source
)
INSERT INTO document_entities (  -- noqa: resolver-lint (transitional; u03's AST lint exempts tier-1 scripts)
    bibcode,
    entity_id,
    link_type,
    tier,
    tier_version,
    confidence,
    match_method,
    evidence
)
SELECT
    bibcode,
    entity_id,
    'keyword_match'                                                       AS link_type,
    1::smallint                                                           AS tier,
    2                                                                     AS tier_version,  -- bumped from 1: link_policy filter applied; DELETE old rows before re-running
    1.0::real                                                             AS confidence,
    'keyword_exact_lower'                                                 AS match_method,
    jsonb_build_object('keyword', matched_keyword, 'match_source', match_source) AS evidence
FROM collapsed
ON CONFLICT (bibcode, entity_id, link_type, tier) DO NOTHING
"""


def run_tier1_link(conn: psycopg.Connection, *, dry_run: bool = False) -> int:
    """Run the single-pass tier-1 keyword-match insert.

    Returns the number of rows actually inserted (rowcount after the
    ``ON CONFLICT DO NOTHING``).  When ``dry_run`` is True, the transaction
    is rolled back and the candidate count is returned instead.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM entities WHERE link_policy IS NULL")
        row = cur.fetchone()
        null_count: int = row[0] if row is not None else 0
        if null_count > 1000:
            logger.warning(
                "%d entities have NULL link_policy — run "
                "scripts/set_link_policy.py first for best precision",
                null_count,
            )

        cur.execute(TIER1_INSERT_SQL)
        inserted = cur.rowcount or 0

    if dry_run:
        logger.info("DRY RUN — rolling back %d candidate rows", inserted)
        conn.rollback()
    else:
        conn.commit()
        logger.info("Inserted %d tier-1 keyword-match rows", inserted)

    return inserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tier-1 keyword-match linker: join papers.keywords against "
            "entities/entity_aliases and write rows to document_entities."
        )
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Database DSN (default: SCIX_DSN env or 'dbname=scix').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run the SQL but roll back instead of committing.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = get_connection(args.db_url)
    try:
        inserted = run_tier1_link(conn, dry_run=args.dry_run)
    finally:
        conn.close()

    verb = "would insert" if args.dry_run else "inserted"
    print(f"tier-1 keyword-match: {verb} {inserted} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
