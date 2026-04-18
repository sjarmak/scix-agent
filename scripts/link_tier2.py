#!/usr/bin/env python3
"""Tier-2 Aho-Corasick abstract linker — PRD §M6 / §S2 / u09.

End-to-end pipeline:

1. Read the curated entity core (``curated_entity_core JOIN entities``)
   filtered on ``ambiguity_class IN ('unique','domain_safe','homograph')``.
2. Build a pickleable :class:`ahocorasick.Automaton` over every canonical
   name + alias surface form.
3. Stream papers with non-null abstracts (optionally restricted by a
   ``--bibcode-prefix`` for tests / shards).
4. Fan the abstracts out to a ``multiprocessing.Pool`` that runs
   :func:`scix.aho_corasick.link_abstract` per paper.
5. Collect candidates, enforce a per-entity linkage cap, write tier=2
   rows into ``document_entities`` and flip over-cap entities to
   ``link_policy = 'llm_only'``.

Usage::

    python scripts/link_tier2.py --allow-prod         # prod DSN (guard required)
    SCIX_TEST_DSN=dbname=scix_test \\
      python scripts/link_tier2.py --bibcode-prefix test_u09_ --workers 1

Transitional exemption: all ``INSERT INTO document_entities`` statements
in this file are marked ``# noqa: resolver-lint``. The real M13 resolver
(u03) only owns writes from within ``src/``; scripts under ``scripts/``
are outside the lint's scope by design, so the annotation is for parity
and discoverability only.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

import psycopg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.aho_corasick import (  # noqa: E402
    AhocorasickAutomaton,
    EntityRow,
    LinkCandidate,
    build_automaton,
    link_abstract,
)
from scix.db import DEFAULT_DSN, get_connection, is_production_dsn, redact_dsn  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Default per-entity linkage cap. Any entity whose Tier 2 output exceeds
# this count is demoted to ``link_policy='llm_only'`` and stops being
# written at tier=2. 25_000 is the PRD §M6 acceptance criterion.
DEFAULT_MAX_PER_ENTITY: int = 25_000

# Default worker count. One-per-CPU would be aggressive during tests;
# stay at 1 by default so fixtures are deterministic. Prod callers bump
# with ``--workers 16``.
DEFAULT_WORKERS: int = 1

# Batch size for the paper stream. The parent process owns the DB cursor
# and ships ``(bibcode, abstract)`` tuples to workers in chunks.
PAPER_BATCH_SIZE: int = 512

# Commit and sync the pipeline every N paper batches (~this many papers).
# Long single-transaction runs bloat WAL and lose all work on a crash.
# 0 means "commit only at end" (used by tests / dry-run).
DEFAULT_COMMIT_INTERVAL_BATCHES: int = 40  # ~20k papers per commit

TIER2_LINK_TYPE: str = "abstract_match"
TIER2_TIER: int = 2
TIER2_TIER_VERSION: int = 1
TIER2_MATCH_METHOD: str = "aho_corasick_abstract"

# ---------------------------------------------------------------------------
# Entity source
# ---------------------------------------------------------------------------

# The curated core (``curated_entity_core``) is a ~600-row operator-promoted
# subset. ``full`` widens the pool to every entity with a safe
# ambiguity_class (``unique``/``domain_safe``/``homograph``) that has not
# been demoted to ``link_policy='llm_only'``. Default stays ``curated`` so
# existing tests and callers are unaffected.
ENTITY_SOURCE_CURATED: str = "curated"
ENTITY_SOURCE_FULL: str = "full"
ENTITY_SOURCES: tuple[str, ...] = (ENTITY_SOURCE_CURATED, ENTITY_SOURCE_FULL)


# ---------------------------------------------------------------------------
# Entity fetching
# ---------------------------------------------------------------------------


def fetch_entity_rows(
    conn: psycopg.Connection,
    *,
    source: str = ENTITY_SOURCE_CURATED,
) -> list[EntityRow]:
    """Pull ``(entity_id, surface, canonical_name, ambiguity_class,
    is_alias)`` rows for the tier-2 automaton.

    Two queries are unioned client-side:

    * canonical surface (``is_alias=False``)
    * every alias (``is_alias=True``)

    The ``ambiguity_class`` filter excludes ``banned`` entities outright.

    Parameters
    ----------
    source
        ``"curated"`` (default) restricts to ``curated_entity_core`` —
        the operator-promoted subset used by the incremental runner and
        existing regression tests. ``"full"`` widens to every entity with
        a safe ambiguity_class that has not been demoted to
        ``link_policy='llm_only'``.
    """
    if source not in ENTITY_SOURCES:
        raise ValueError(f"unknown entity source: {source!r}; expected one of {ENTITY_SOURCES}")

    rows: list[EntityRow] = []

    if source == ENTITY_SOURCE_CURATED:
        canonical_sql = """
            SELECT e.id,
                   e.canonical_name AS surface,
                   e.canonical_name,
                   e.ambiguity_class::text
              FROM curated_entity_core c
              JOIN entities e ON e.id = c.entity_id
             WHERE e.ambiguity_class IN ('unique', 'domain_safe', 'homograph')
        """
        alias_sql = """
            SELECT e.id,
                   ea.alias AS surface,
                   e.canonical_name,
                   e.ambiguity_class::text
              FROM curated_entity_core c
              JOIN entities e ON e.id = c.entity_id
              JOIN entity_aliases ea ON ea.entity_id = e.id
             WHERE e.ambiguity_class IN ('unique', 'domain_safe', 'homograph')
        """
    else:  # ENTITY_SOURCE_FULL
        canonical_sql = """
            SELECT e.id,
                   e.canonical_name AS surface,
                   e.canonical_name,
                   e.ambiguity_class::text
              FROM entities e
             WHERE e.ambiguity_class IN ('unique', 'domain_safe', 'homograph')
               AND (e.link_policy IS NULL OR e.link_policy <> 'llm_only')
        """
        alias_sql = """
            SELECT e.id,
                   ea.alias AS surface,
                   e.canonical_name,
                   e.ambiguity_class::text
              FROM entities e
              JOIN entity_aliases ea ON ea.entity_id = e.id
             WHERE e.ambiguity_class IN ('unique', 'domain_safe', 'homograph')
               AND (e.link_policy IS NULL OR e.link_policy <> 'llm_only')
        """

    with conn.cursor() as cur:
        cur.execute(canonical_sql)
        for entity_id, surface, canonical, ambiguity in cur.fetchall():
            rows.append(
                EntityRow(
                    entity_id=int(entity_id),
                    surface=surface,
                    canonical_name=canonical,
                    ambiguity_class=ambiguity,
                    is_alias=False,
                )
            )
        cur.execute(alias_sql)
        for entity_id, surface, canonical, ambiguity in cur.fetchall():
            rows.append(
                EntityRow(
                    entity_id=int(entity_id),
                    surface=surface,
                    canonical_name=canonical,
                    ambiguity_class=ambiguity,
                    is_alias=True,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Paper streaming
# ---------------------------------------------------------------------------


def iter_paper_batches(
    conn: psycopg.Connection,
    bibcode_prefix: Optional[str],
    batch_size: int = PAPER_BATCH_SIZE,
) -> Iterator[list[tuple[str, str]]]:
    """Yield ``[(bibcode, abstract), ...]`` batches.

    A server-side cursor is used so the driver doesn't buffer the full
    papers table in memory. The optional ``bibcode_prefix`` filter is
    used by the test fixture to scope to ``test_u09_%``.
    """
    sql = "SELECT bibcode, abstract FROM papers WHERE abstract IS NOT NULL"
    params: list[str] = []
    if bibcode_prefix:
        sql += " AND bibcode LIKE %s"
        params.append(bibcode_prefix + "%")

    with conn.cursor(name="tier2_papers") as cur:
        cur.itersize = batch_size
        cur.execute(sql, params)
        batch: list[tuple[str, str]] = []
        for bibcode, abstract in cur:
            batch.append((bibcode, abstract or ""))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


# ---------------------------------------------------------------------------
# Worker: link one abstract
# ---------------------------------------------------------------------------

# Module-level automaton, populated by ``_worker_init`` in each forked
# child so the automaton is not re-pickled on every task.
_WORKER_AUTOMATON: Optional[AhocorasickAutomaton] = None


def _worker_init(automaton: AhocorasickAutomaton) -> None:
    """multiprocessing.Pool initializer. Stash the automaton per-worker."""
    global _WORKER_AUTOMATON
    _WORKER_AUTOMATON = automaton


def _worker_link(task: tuple[str, str]) -> tuple[str, list[LinkCandidate]]:
    """Run :func:`link_abstract` for a single ``(bibcode, abstract)``."""
    bibcode, abstract = task
    assert _WORKER_AUTOMATON is not None, "worker not initialized"
    hits = link_abstract(abstract, _WORKER_AUTOMATON)
    return bibcode, hits


def _link_serial(
    batch: Sequence[tuple[str, str]],
    automaton: AhocorasickAutomaton,
) -> Iterator[tuple[str, list[LinkCandidate]]]:
    """In-process fallback used when ``workers == 1``."""
    for bibcode, abstract in batch:
        yield bibcode, link_abstract(abstract, automaton)


# ---------------------------------------------------------------------------
# Writer: cap + insert
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier2Stats:
    """End-of-run summary emitted by :func:`run_tier2_link`."""

    papers_scanned: int
    candidates_generated: int
    rows_inserted: int
    entities_demoted: int
    entities_with_links: int


def _format_wall_time(seconds: float) -> str:
    """Format seconds into a human-readable ``1h 2m 3s`` string."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def write_tier2_summary(
    stats: Tier2Stats,
    output_path: pathlib.Path,
    *,
    wall_seconds: float,
    dry_run: bool = False,
) -> None:
    """Write a Markdown summary of the Tier-2 run to ``output_path``.

    Parameters
    ----------
    stats
        End-of-run stats from :func:`run_tier2_link`.
    output_path
        Path to the output ``.md`` file.
    wall_seconds
        Elapsed wall time in seconds.
    dry_run
        If True, the summary is marked as a dry run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wall_str = _format_wall_time(wall_seconds)
    mode_label = " (DRY RUN)" if dry_run else ""

    lines = [
        f"# Tier 2 Aho-Corasick Linker Summary{mode_label}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Papers scanned | {stats.papers_scanned:,} |",
        f"| Candidates generated | {stats.candidates_generated:,} |",
        f"| Rows inserted | {stats.rows_inserted:,} |",
        f"| Entities with links | {stats.entities_with_links:,} |",
        f"| Entities demoted | {stats.entities_demoted:,} |",
        f"| Wall time | {wall_str} |",
        "",
    ]
    output_path.write_text("\n".join(lines))
    logger.info("Summary written to %s", output_path)


# SQL lives as a module-level constant so the AST lint only scans it
# once. All three inserts/updates that touch document_entities or
# entities.link_policy are marked noqa.

_INSERT_SQL = """
    INSERT INTO document_entities (
        bibcode, entity_id, link_type, tier, tier_version,
        confidence, match_method, evidence
    )
    VALUES (
        %(bibcode)s, %(entity_id)s, %(link_type)s, %(tier)s, %(tier_version)s,
        %(confidence)s, %(match_method)s, %(evidence)s::jsonb
    )
    ON CONFLICT (bibcode, entity_id, link_type, tier) DO NOTHING
"""  # noqa: resolver-lint (transitional; tier-2 script owns its own writes per u09)

_DEMOTE_SQL = """
    UPDATE entities
       SET link_policy = 'llm_only'
     WHERE id = %s
"""


def _evidence_json(candidate: LinkCandidate) -> str:
    """Serialize a :class:`LinkCandidate` into JSON-encoded evidence.

    We use the stdlib json module via an inline helper because
    ``psycopg`` can accept Python dicts for jsonb columns, but we pass a
    string so the caller can assert on the exact shape in the test.
    """
    import json

    return json.dumps(
        {
            "matched_surface": candidate.matched_surface,
            "start": candidate.start,
            "end": candidate.end,
            "ambiguity_class": candidate.ambiguity_class,
            "is_alias": candidate.is_alias,
            "tier2_confidence_source": "aho_corasick+placeholder",
        },
        separators=(",", ":"),
    )


def _dedupe_candidates(candidates: Iterable[LinkCandidate]) -> list[LinkCandidate]:
    """Drop repeated (entity_id,) hits for the same paper, keeping the
    earliest position. Downstream PK only keys on (bibcode, entity_id,
    link_type, tier), so multi-mention within one abstract collapses.
    """
    seen: dict[int, LinkCandidate] = {}
    for cand in candidates:
        prev = seen.get(cand.entity_id)
        if prev is None or cand.start < prev.start:
            seen[cand.entity_id] = cand
    return list(seen.values())


def run_tier2_link(
    conn: psycopg.Connection,
    *,
    workers: int = DEFAULT_WORKERS,
    bibcode_prefix: Optional[str] = None,
    max_per_entity: int = DEFAULT_MAX_PER_ENTITY,
    dry_run: bool = False,
    entity_source: str = ENTITY_SOURCE_CURATED,
    commit_interval_batches: int = 0,
) -> Tier2Stats:
    """Run the full Tier-2 linkage pass against ``conn``.

    Parameters
    ----------
    conn
        Open psycopg connection.
    workers
        Parallelism for :func:`link_abstract`. 1 stays in-process; >1
        uses ``multiprocessing.Pool(fork)``.
    bibcode_prefix
        Optional LIKE prefix to scope to a bibcode shard (used by tests).
    max_per_entity
        Per-entity linkage cap. Entities exceeding the cap are demoted
        to ``link_policy='llm_only'`` and stop receiving writes.
    dry_run
        If True, the transaction is rolled back instead of committed.
    entity_source
        Which entity pool the automaton is built from. ``"curated"``
        (default) uses ``curated_entity_core``. ``"full"`` uses every
        entity with a safe ambiguity_class.
    commit_interval_batches
        Commit the write connection every N paper batches. 0 (default)
        commits only at the end — preserves existing test semantics.
        Long prod runs should set e.g. 40 to cap WAL size and survive
        crashes.

    Returns
    -------
    Tier2Stats
        Aggregate counts for the run.
    """
    logger.info("Fetching entity rows (source=%s)...", entity_source)
    entity_rows = fetch_entity_rows(conn, source=entity_source)
    logger.info("  -> %d surface forms", len(entity_rows))
    if not entity_rows:
        logger.warning("entity pool is empty; nothing to link")
        return Tier2Stats(0, 0, 0, 0, 0)

    automaton = build_automaton(entity_rows)
    logger.info("Built automaton over %d surfaces", len(automaton))

    per_entity_count: dict[int, int] = {}
    demoted: set[int] = set()
    entities_with_links: set[int] = set()

    papers_scanned = 0
    candidates_generated = 0
    rows_inserted = 0

    pool: Optional[mp.pool.Pool] = None
    if workers > 1:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(automaton,),
        )

    log_interval = 50_000  # log every N papers scanned

    # Writes go through a second connection in pipeline mode. Pipeline
    # mode batches execute() round-trips to the server (3-5x speedup
    # for per-row INSERTs) but is incompatible with the server-side
    # cursor used to stream papers, so we split read and write paths.
    write_conn = psycopg.connect(conn.info.dsn)
    batches_since_commit = 0
    try:
        with write_conn.pipeline(), write_conn.cursor() as insert_cur:
            for batch in iter_paper_batches(conn, bibcode_prefix):
                papers_scanned += len(batch)
                batches_since_commit += 1
                if papers_scanned % log_interval < len(batch):
                    logger.info(
                        "  progress: %d papers scanned, %d rows pending, %d demoted",
                        papers_scanned,
                        rows_inserted,
                        len(demoted),
                    )

                if pool is not None:
                    results = pool.map(_worker_link, batch)
                else:
                    results = list(_link_serial(batch, automaton))

                for bibcode, hits in results:
                    if not hits:
                        continue
                    deduped = _dedupe_candidates(hits)
                    candidates_generated += len(deduped)

                    for cand in deduped:
                        entity_id = cand.entity_id

                        # Cap enforcement. Once an entity crosses the
                        # threshold, it is demoted and we skip writes.
                        current = per_entity_count.get(entity_id, 0)
                        if current >= max_per_entity:
                            if entity_id not in demoted:
                                demoted.add(entity_id)
                                if not dry_run:
                                    insert_cur.execute(_DEMOTE_SQL, (entity_id,))
                            continue

                        insert_cur.execute(
                            _INSERT_SQL,  # noqa: resolver-lint
                            {
                                "bibcode": bibcode,
                                "entity_id": entity_id,
                                "link_type": TIER2_LINK_TYPE,
                                "tier": TIER2_TIER,
                                "tier_version": TIER2_TIER_VERSION,
                                "confidence": cand.confidence,
                                "match_method": TIER2_MATCH_METHOD,
                                "evidence": _evidence_json(cand),
                            },
                        )
                        # Under pipeline mode, rowcount isn't reliable
                        # until sync. Use optimistic per-candidate
                        # accounting: count every accepted candidate as
                        # inserted. ON CONFLICT DO NOTHING may drop
                        # some duplicates, which only shifts demotion
                        # a few rows earlier — acceptable drift.
                        rows_inserted += 1
                        per_entity_count[entity_id] = current + 1
                        entities_with_links.add(entity_id)

                if (
                    commit_interval_batches > 0
                    and not dry_run
                    and batches_since_commit >= commit_interval_batches
                ):
                    # commit() inside a pipeline block is legal — it
                    # flushes the pipeline, commits, and keeps pipeline
                    # mode active for subsequent executes.
                    write_conn.commit()
                    batches_since_commit = 0
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    try:
        if dry_run:
            write_conn.rollback()
            logger.info("DRY RUN — rolled back %d tier-2 rows", rows_inserted)
        else:
            write_conn.commit()
            logger.info(
                "Committed %d tier-2 rows across %d entities (%d demoted)",
                rows_inserted,
                len(entities_with_links),
                len(demoted),
            )
    finally:
        write_conn.close()

    return Tier2Stats(
        papers_scanned=papers_scanned,
        candidates_generated=candidates_generated,
        rows_inserted=rows_inserted,
        entities_demoted=len(demoted),
        entities_with_links=len(entities_with_links),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-url", type=str, default=None)
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help="multiprocessing.Pool size"
    )
    parser.add_argument(
        "--bibcode-prefix",
        type=str,
        default=None,
        help="Only link papers whose bibcode starts with this string",
    )
    parser.add_argument(
        "--max-per-entity",
        type=int,
        default=DEFAULT_MAX_PER_ENTITY,
        help="Per-entity linkage cap; over-cap entities are demoted to llm_only",
    )
    parser.add_argument(
        "--entity-source",
        choices=ENTITY_SOURCES,
        default=ENTITY_SOURCE_CURATED,
        help=(
            "Entity pool: 'curated' uses curated_entity_core (~600 rows); "
            "'full' widens to every entity with a safe ambiguity_class."
        ),
    )
    parser.add_argument(
        "--commit-interval-batches",
        type=int,
        default=0,
        help=(
            "Commit write connection every N paper batches (0 = commit "
            "only at end). Long prod runs should set 40 or so."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Allow running against the production database.",
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
    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "refusing to run against production DSN %s — pass --allow-prod to override",
            redact_dsn(dsn),
        )
        return 2

    conn = get_connection(dsn)
    t0 = time.monotonic()
    try:
        stats = run_tier2_link(
            conn,
            workers=args.workers,
            bibcode_prefix=args.bibcode_prefix,
            max_per_entity=args.max_per_entity,
            dry_run=args.dry_run,
            entity_source=args.entity_source,
            commit_interval_batches=args.commit_interval_batches,
        )
        wall_seconds = time.monotonic() - t0
    finally:
        conn.close()

    verb = "would insert" if args.dry_run else "inserted"
    print(
        f"tier-2 aho-corasick: scanned {stats.papers_scanned} papers, "
        f"{verb} {stats.rows_inserted} rows "
        f"({stats.entities_with_links} entities, {stats.entities_demoted} demoted)"
    )

    summary_path = REPO_ROOT / "build-artifacts" / "tier2_summary.md"
    write_tier2_summary(stats, summary_path, wall_seconds=wall_seconds, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
