#!/usr/bin/env python3
"""Section-grain Aho-Corasick entity linker — bead scix_experiments-67e.

Pairs with the wqr.9 section-embeddings work. After
``papers_fulltext.sections`` is materialized, this script re-runs the
Tier-2 surface-form automaton against each section's text and writes one
row per ``(bibcode, section_index, entity_id, link_type, tier)`` into
``section_entities`` (migration 063).

End-to-end pipeline
-------------------

1. Reuse :func:`scripts.link_tier2.fetch_entity_rows` to pull surface forms
   from the same entity pool the abstract Tier-2 run uses. The default
   pool excludes ``link_policy='llm_only'`` so entities demoted by the
   abstract pass don't get re-linked at section grain.
2. Build a single Aho-Corasick automaton via
   :func:`scix.aho_corasick.build_automaton` and fan it out to a
   ``multiprocessing.Pool``.
3. Stream papers from ``papers_fulltext`` (server-side cursor, batched).
4. For each paper, normalize the ``sections`` JSONB and run
   :func:`scix.section_linker.link_paper_sections` per section text.
5. Insert section_entities rows in pipeline mode with
   ``ON CONFLICT DO NOTHING`` so reruns are idempotent.

Usage
-----

::

    SCIX_TEST_DSN=dbname=scix_test \\
      python scripts/link_section_entities.py --bibcode-prefix test_67e_ --workers 1

    scix-batch python scripts/link_section_entities.py --allow-prod --workers 16

Long prod runs MUST set ``--commit-interval-batches`` so a crash doesn't
discard the entire pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import psycopg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import link_tier2  # noqa: E402  reuse fetch_entity_rows + ENTITY_SOURCES

from scix.aho_corasick import (  # noqa: E402
    AhocorasickAutomaton,
    build_automaton,
)
from scix.db import DEFAULT_DSN, get_connection, is_production_dsn, redact_dsn  # noqa: E402
from scix.section_linker import (  # noqa: E402
    SectionLinkCandidate,
    link_paper_sections,
    parse_sections,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_WORKERS: int = 1
PAPER_BATCH_SIZE: int = 256

# Mirror the abstract tier-2 commit cadence semantics: 0 = commit only at
# end (preserves test determinism); prod runs should pass --commit-interval-batches 40.
DEFAULT_COMMIT_INTERVAL_BATCHES: int = 0

LINK_TYPE: str = "section_match"
TIER: int = 2
TIER_VERSION: int = 1
MATCH_METHOD: str = "aho_corasick_section"


# ---------------------------------------------------------------------------
# Paper streaming
# ---------------------------------------------------------------------------


def iter_paper_batches(
    conn: psycopg.Connection,
    bibcode_prefix: Optional[str],
    batch_size: int = PAPER_BATCH_SIZE,
) -> Iterator[list[tuple[str, list[dict]]]]:
    """Yield ``[(bibcode, sections_jsonb), ...]`` batches.

    A server-side cursor streams ``papers_fulltext`` rows so we don't buffer
    the whole table client-side. Empty section arrays are skipped at the
    SQL level — this is a section-grain linker, no point fetching papers
    that have no sections to link.
    """
    sql = (
        "SELECT bibcode, sections "
        "FROM papers_fulltext "
        "WHERE jsonb_array_length(sections) > 0"
    )
    params: list[str] = []
    if bibcode_prefix:
        sql += " AND bibcode LIKE %s"
        params.append(bibcode_prefix + "%")

    with conn.cursor(name="section_linker_papers") as cur:
        cur.itersize = batch_size
        cur.execute(sql, params)
        batch: list[tuple[str, list[dict]]] = []
        for bibcode, sections_jsonb in cur:
            batch.append((bibcode, sections_jsonb or []))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

# Each forked worker stashes the automaton at module level so it doesn't
# travel across the pickle boundary on every task.
_WORKER_AUTOMATON: Optional[AhocorasickAutomaton] = None


def _worker_init(automaton: AhocorasickAutomaton) -> None:
    global _WORKER_AUTOMATON
    _WORKER_AUTOMATON = automaton


def _worker_link(
    task: tuple[str, list[dict]],
) -> tuple[str, list[SectionLinkCandidate]]:
    bibcode, sections_jsonb = task
    assert _WORKER_AUTOMATON is not None, "worker not initialized"
    sections = parse_sections(sections_jsonb)
    return bibcode, link_paper_sections(sections, _WORKER_AUTOMATON)


def _link_serial(
    batch: Sequence[tuple[str, list[dict]]],
    automaton: AhocorasickAutomaton,
) -> Iterator[tuple[str, list[SectionLinkCandidate]]]:
    """In-process fallback used when ``workers == 1``."""
    for bibcode, sections_jsonb in batch:
        sections = parse_sections(sections_jsonb)
        yield bibcode, link_paper_sections(sections, automaton)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionLinkStats:
    """End-of-run summary."""

    papers_scanned: int
    sections_scanned: int
    candidates_generated: int
    rows_inserted: int
    entities_with_links: int


def _evidence_json(candidate: SectionLinkCandidate) -> str:
    """Serialize a SectionLinkCandidate's evidence payload.

    Mirrors document_entities.evidence shape from
    :func:`scripts.link_tier2._evidence_json` so callers parsing both
    tables can use the same code path.
    """
    raw = candidate.candidate
    return json.dumps(
        {
            "matched_surface": raw.matched_surface,
            "start": raw.start,
            "end": raw.end,
            "ambiguity_class": raw.ambiguity_class,
            "is_alias": raw.is_alias,
            "tier2_confidence_source": "aho_corasick+placeholder",
        },
        separators=(",", ":"),
    )


_INSERT_SQL = """
    INSERT INTO section_entities (
        bibcode, section_index, entity_id, link_type, tier, tier_version,
        confidence, match_method, section_heading, section_role, evidence
    )
    VALUES (
        %(bibcode)s, %(section_index)s, %(entity_id)s, %(link_type)s,
        %(tier)s, %(tier_version)s, %(confidence)s, %(match_method)s,
        %(section_heading)s, %(section_role)s, %(evidence)s::jsonb
    )
    ON CONFLICT (bibcode, section_index, entity_id, link_type, tier)
    DO NOTHING
"""  # resolver-lint: bypass (transitional; section linker owns its own writes per 67e)


def run_section_link(
    conn: psycopg.Connection,
    *,
    workers: int = DEFAULT_WORKERS,
    bibcode_prefix: Optional[str] = None,
    dry_run: bool = False,
    entity_source: str = link_tier2.ENTITY_SOURCE_CURATED,
    commit_interval_batches: int = DEFAULT_COMMIT_INTERVAL_BATCHES,
) -> SectionLinkStats:
    """Run the full section-grain linkage pass against ``conn``.

    Parameters
    ----------
    conn
        Open psycopg connection.
    workers
        Parallelism for :func:`scix.section_linker.link_paper_sections`.
        ``1`` keeps everything in-process; ``>1`` uses
        ``multiprocessing.Pool(fork)``.
    bibcode_prefix
        Optional LIKE prefix to scope to a bibcode shard (used by tests
        and shard-level prod runs).
    dry_run
        If True, the write transaction is rolled back instead of committed.
    entity_source
        Which entity pool the automaton is built from. Forwarded to
        :func:`scripts.link_tier2.fetch_entity_rows`.
    commit_interval_batches
        Commit the write connection every N paper batches. ``0`` (default)
        commits only at the end. Long prod runs should pass ``40`` or so
        to cap WAL size and survive crashes.
    """
    logger.info("Fetching entity rows (source=%s)...", entity_source)
    entity_rows = link_tier2.fetch_entity_rows(conn, source=entity_source)
    logger.info("  -> %d surface forms", len(entity_rows))
    if not entity_rows:
        logger.warning("entity pool is empty; nothing to link")
        return SectionLinkStats(0, 0, 0, 0, 0)

    automaton = build_automaton(entity_rows)
    logger.info("Built automaton over %d surfaces", len(automaton))

    entities_with_links: set[int] = set()

    papers_scanned = 0
    sections_scanned = 0
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

    log_interval = 25_000

    write_conn = psycopg.connect(conn.info.dsn)
    batches_since_commit = 0
    try:
        with write_conn.pipeline(), write_conn.cursor() as insert_cur:
            for batch in iter_paper_batches(conn, bibcode_prefix):
                papers_scanned += len(batch)
                batches_since_commit += 1
                if papers_scanned % log_interval < len(batch):
                    logger.info(
                        "  progress: %d papers scanned, %d sections, "
                        "%d rows pending",
                        papers_scanned,
                        sections_scanned,
                        rows_inserted,
                    )

                if pool is not None:
                    results = pool.map(_worker_link, batch)
                else:
                    results = list(_link_serial(batch, automaton))

                for bibcode, hits in results:
                    if not hits:
                        continue
                    seen_section_indices: set[int] = set()
                    for cand in hits:
                        seen_section_indices.add(cand.section_index)
                        candidates_generated += 1
                        entities_with_links.add(cand.entity_id)
                        insert_cur.execute(
                            _INSERT_SQL,  # resolver-lint: bypass
                            {
                                "bibcode": bibcode,
                                "section_index": cand.section_index,
                                "entity_id": cand.entity_id,
                                "link_type": LINK_TYPE,
                                "tier": TIER,
                                "tier_version": TIER_VERSION,
                                "confidence": cand.candidate.confidence,
                                "match_method": MATCH_METHOD,
                                "section_heading": cand.section_heading,
                                "section_role": cand.section_role,
                                "evidence": _evidence_json(cand),
                            },
                        )
                        # Pipeline mode masks rowcount until sync; assume
                        # one row per candidate. ON CONFLICT DO NOTHING may
                        # drop a few duplicates, which is acceptable drift.
                        rows_inserted += 1
                    sections_scanned += len(seen_section_indices)

                if (
                    commit_interval_batches > 0
                    and not dry_run
                    and batches_since_commit >= commit_interval_batches
                ):
                    write_conn.commit()
                    batches_since_commit = 0
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    try:
        if dry_run:
            write_conn.rollback()
            logger.info("DRY RUN — rolled back %d section_entities rows", rows_inserted)
        else:
            write_conn.commit()
            logger.info(
                "Committed %d section_entities rows across %d entities",
                rows_inserted,
                len(entities_with_links),
            )
    finally:
        write_conn.close()

    return SectionLinkStats(
        papers_scanned=papers_scanned,
        sections_scanned=sections_scanned,
        candidates_generated=candidates_generated,
        rows_inserted=rows_inserted,
        entities_with_links=len(entities_with_links),
    )


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _format_wall_time(seconds: float) -> str:
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


def write_summary(
    stats: SectionLinkStats,
    output_path: pathlib.Path,
    *,
    wall_seconds: float,
    dry_run: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wall_str = _format_wall_time(wall_seconds)
    mode_label = " (DRY RUN)" if dry_run else ""
    lines = [
        f"# Section-grain Aho-Corasick Linker Summary{mode_label}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Papers scanned | {stats.papers_scanned:,} |",
        f"| Sections scanned | {stats.sections_scanned:,} |",
        f"| Candidates generated | {stats.candidates_generated:,} |",
        f"| Rows inserted | {stats.rows_inserted:,} |",
        f"| Entities with links | {stats.entities_with_links:,} |",
        f"| Wall time | {wall_str} |",
        "",
    ]
    output_path.write_text("\n".join(lines))
    logger.info("Summary written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-url", type=str, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="multiprocessing.Pool size",
    )
    parser.add_argument(
        "--bibcode-prefix",
        type=str,
        default=None,
        help="Only link papers whose bibcode starts with this string",
    )
    parser.add_argument(
        "--entity-source",
        choices=link_tier2.ENTITY_SOURCES,
        default=link_tier2.ENTITY_SOURCE_CURATED,
        help=(
            "Entity pool: 'curated' uses curated_entity_core (~600 rows); "
            "'full' widens to every entity with a safe ambiguity_class "
            "(except link_policy='llm_only')."
        ),
    )
    parser.add_argument(
        "--commit-interval-batches",
        type=int,
        default=DEFAULT_COMMIT_INTERVAL_BATCHES,
        help=(
            "Commit write connection every N paper batches (0 = commit "
            "only at end). Prod runs should pass 40 or so."
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

    # Self-enforce systemd-scope wrapping for prod runs (mirrors other
    # --allow-prod scripts; CLAUDE.md storage tiering / scix-batch policy).
    if args.allow_prod and not os.environ.get("SYSTEMD_SCOPE"):
        logger.error(
            "--allow-prod requires running inside a systemd scope; "
            "use scix-batch python scripts/link_section_entities.py ..."
        )
        return 2

    conn = get_connection(dsn)
    t0 = time.monotonic()
    try:
        stats = run_section_link(
            conn,
            workers=args.workers,
            bibcode_prefix=args.bibcode_prefix,
            dry_run=args.dry_run,
            entity_source=args.entity_source,
            commit_interval_batches=args.commit_interval_batches,
        )
        wall_seconds = time.monotonic() - t0
    finally:
        conn.close()

    verb = "would insert" if args.dry_run else "inserted"
    print(
        f"section-link aho-corasick: scanned {stats.papers_scanned} papers "
        f"({stats.sections_scanned} sections), "
        f"{verb} {stats.rows_inserted} rows "
        f"({stats.entities_with_links} entities)"
    )

    summary_path = REPO_ROOT / "build-artifacts" / "section_link_summary.md"
    write_summary(stats, summary_path, wall_seconds=wall_seconds, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
