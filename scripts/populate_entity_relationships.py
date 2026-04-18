#!/usr/bin/env python3
"""Populate public.entity_relationships from source hierarchies.

Per-source extractors (see :mod:`scix.entity_relationships`) turn the
hierarchy strings we already store in ``entities.properties`` into
directed edges.  Supported tiers:

* ``gcmd`` — ``gcmd_hierarchy`` path, scheme-scoped.  ``parent_of``.
* ``spase`` — SPASE ObservedRegion dot notation.  ``parent_of``.
* ``ssodnet`` — ``sso_class`` family tree.  Taxon->taxon ``parent_of``
  plus optional asteroid->leaf-class ``part_of`` edges.
* ``curated_flagship`` — hardcoded mission->instrument map.
  ``has_instrument``.

Idempotent: every insert is ``ON CONFLICT DO NOTHING`` against the
``(subject_entity_id, predicate, object_entity_id)`` unique key.  Each
source commits independently so a partial run can be resumed.

Usage::

    .venv/bin/python scripts/populate_entity_relationships.py \\
        --source gcmd spase curated_flagship --allow-prod
    .venv/bin/python scripts/populate_entity_relationships.py \\
        --source ssodnet --include-ssodnet-targets --allow-prod
    .venv/bin/python scripts/populate_entity_relationships.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from dataclasses import dataclass

import psycopg
from psycopg import sql

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.db import (  # noqa: E402
    DEFAULT_DSN,
    get_connection,
    is_production_dsn,
    redact_dsn,
)
from scix.entity_relationships import (  # noqa: E402
    EdgeCandidate,
    extract_curated_flagship_edges,
    extract_gcmd_edges,
    extract_spase_region_edges,
    extract_ssodnet_class_edges,
    parse_gcmd_hierarchy,
)
from scix.harvest_utils import HarvestRunLog  # noqa: E402

logger = logging.getLogger(__name__)

ALL_SOURCES = ("gcmd", "spase", "ssodnet", "curated_flagship")


@dataclass(frozen=True)
class SourceStats:
    source: str
    edges_emitted: int
    edges_inserted: int
    taxa_created: int = 0


# ---------------------------------------------------------------------------
# Shared bulk insert
# ---------------------------------------------------------------------------


_INSERT_SQL = sql.SQL("""
    INSERT INTO entity_relationships (
        subject_entity_id, predicate, object_entity_id,
        source, harvest_run_id, confidence, evidence
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (subject_entity_id, predicate, object_entity_id) DO NOTHING
    """)


def bulk_insert_edges(
    conn: psycopg.Connection,
    edges: list[EdgeCandidate],
    *,
    harvest_run_id: int,
    batch_size: int = 5000,
) -> int:
    """Insert resolved edges into entity_relationships.

    Every edge must have integer ``subject_id`` and ``object_id``.  The
    caller is responsible for resolving synthetic taxa to ids before
    passing them in.

    Returns the number of *new* rows inserted (on-conflict-skipped
    rows are counted via ``cur.rowcount`` per batch which reports the
    inserts, not the skipped ones — psycopg's behaviour).
    """
    if not edges:
        return 0

    inserted = 0
    # Ensure every edge has resolved ids — guard against programmer error
    for e in edges:
        if e.subject_id is None or e.object_id is None:
            raise ValueError(f"unresolved edge: subject_id/object_id must be set, got {e}")

    with conn.cursor() as cur:
        for start in range(0, len(edges), batch_size):
            batch = edges[start : start + batch_size]
            rows = [
                (
                    e.subject_id,
                    e.predicate,
                    e.object_id,
                    e.source,
                    harvest_run_id,
                    1.0,
                    json.dumps(dict(e.evidence)) if e.evidence else None,
                )
                for e in batch
            ]
            cur.executemany(_INSERT_SQL, rows)
            # executemany rowcount is total affected across the batch
            if cur.rowcount is not None and cur.rowcount >= 0:
                inserted += cur.rowcount
    return inserted


# ---------------------------------------------------------------------------
# GCMD
# ---------------------------------------------------------------------------


def populate_gcmd(conn: psycopg.Connection, *, harvest_run_id: int) -> SourceStats:
    """Extract and insert GCMD parent_of edges."""
    logger.info("gcmd: loading entities with gcmd_hierarchy...")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, canonical_name,
                   properties->>'gcmd_scheme' AS scheme,
                   properties->>'gcmd_hierarchy' AS hierarchy
              FROM entities
             WHERE source = 'gcmd'
               AND properties ? 'gcmd_hierarchy'
            """)
        rows = cur.fetchall()

    # Build (canonical_name, scheme) -> id lookup from the same set.
    # Register both the canonical name AND the trailing hierarchy
    # segment (short_name), because GCMD's parent segments are
    # expressed as short forms (e.g. "ATMOSPHERIC RADIATION") while
    # the entity's canonical is sometimes "PARENT > LEAF".  .setdefault
    # ensures the canonical mapping wins on collision.
    by_name: dict[tuple[str, str], int] = {}
    for entity_id, canonical, scheme, hierarchy in rows:
        if scheme is None:
            continue
        by_name[(canonical, scheme)] = entity_id
        segs = parse_gcmd_hierarchy(hierarchy)
        if segs:
            by_name.setdefault((segs[-1], scheme), entity_id)

    edges = list(extract_gcmd_edges(rows, by_name))
    logger.info("gcmd: emitted %d candidate edges", len(edges))

    inserted = bulk_insert_edges(conn, edges, harvest_run_id=harvest_run_id)
    conn.commit()
    logger.info("gcmd: inserted %d new edges (rest were ON CONFLICT)", inserted)
    return SourceStats(source="gcmd", edges_emitted=len(edges), edges_inserted=inserted)


# ---------------------------------------------------------------------------
# SPASE
# ---------------------------------------------------------------------------


def populate_spase(conn: psycopg.Connection, *, harvest_run_id: int) -> SourceStats:
    """Extract SPASE ObservedRegion parent_of edges."""
    logger.info("spase: loading ObservedRegion entities...")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, canonical_name
              FROM entities
             WHERE source = 'spase'
               AND entity_type = 'observable'
               AND properties->>'spase_list' = 'ObservedRegion'
            """)
        rows = cur.fetchall()

    by_name = {canonical: entity_id for entity_id, canonical in rows}
    edges = list(extract_spase_region_edges(rows, by_name))
    logger.info("spase: emitted %d candidate edges", len(edges))

    inserted = bulk_insert_edges(conn, edges, harvest_run_id=harvest_run_id)
    conn.commit()
    logger.info("spase: inserted %d new edges", inserted)
    return SourceStats(source="spase", edges_emitted=len(edges), edges_inserted=inserted)


# ---------------------------------------------------------------------------
# SsODNet
# ---------------------------------------------------------------------------


def _upsert_synthetic_taxa(conn: psycopg.Connection, taxon_names: list[str]) -> dict[str, int]:
    """Upsert SsODNet taxonomic classes as entities; return name->id map."""
    if not taxon_names:
        return {}

    # Insert missing taxa; rely on the existing unique constraint to
    # skip duplicates.  entity_type='taxon' is a new value — we use it
    # here to distinguish asteroid class nodes from the real asteroid
    # target entities (entity_type='target').
    name_to_id: dict[str, int] = {}
    with conn.cursor() as cur:
        # 1. Bulk insert (ignore conflicts) — psycopg has no executemany
        # RETURNING so we do a single multi-row INSERT.
        values_sql = ",".join(["(%s, %s, %s, %s)"] * len(taxon_names))
        params: list[object] = []
        for name in taxon_names:
            params.extend([name, "taxon", "planetary_science", "ssodnet"])
        cur.execute(
            f"""
            INSERT INTO entities
                (canonical_name, entity_type, discipline, source)
            VALUES {values_sql}
            ON CONFLICT (canonical_name, entity_type, source) DO NOTHING
            """,
            params,
        )

        # 2. Look up ids for every name we care about
        cur.execute(
            """
            SELECT canonical_name, id FROM entities
             WHERE source = 'ssodnet'
               AND entity_type = 'taxon'
               AND canonical_name = ANY(%s)
            """,
            (taxon_names,),
        )
        for name, eid in cur.fetchall():
            name_to_id[name] = eid

    return name_to_id


def populate_ssodnet(
    conn: psycopg.Connection,
    *,
    harvest_run_id: int,
    include_targets: bool,
    limit: int | None = None,
) -> SourceStats:
    """Extract SsODNet class tree edges.

    When ``include_targets`` is True (default) also emit one ``part_of``
    edge per asteroid to its leaf taxonomic class — roughly 1.48M edges
    for the current corpus.
    """
    logger.info(
        "ssodnet: loading target entities (include_targets=%s, limit=%s)...",
        include_targets,
        limit,
    )
    with conn.cursor() as cur:
        query = """
            SELECT id, canonical_name, properties->>'sso_class' AS sso_class
              FROM entities
             WHERE source = 'ssodnet'
               AND entity_type = 'target'
               AND properties ? 'sso_class'
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        cur.execute(query)
        rows = cur.fetchall()

    edges, taxa = extract_ssodnet_class_edges(rows, include_targets=include_targets)
    logger.info(
        "ssodnet: emitted %d candidate edges, %d synthetic taxa to upsert",
        len(edges),
        len(taxa),
    )

    if not taxa:
        return SourceStats(source="ssodnet", edges_emitted=0, edges_inserted=0)

    taxa_names = [t.canonical_name for t in taxa]
    name_to_id = _upsert_synthetic_taxa(conn, taxa_names)
    conn.commit()
    taxa_created = len(name_to_id)
    logger.info("ssodnet: upserted %d taxa", taxa_created)

    # Resolve name endpoints on edges
    resolved: list[EdgeCandidate] = []
    unresolved = 0
    for e in edges:
        subj = e.subject_id
        obj = e.object_id
        if subj is None and e.subject_name is not None:
            subj = name_to_id.get(e.subject_name)
        if obj is None and e.object_name is not None:
            obj = name_to_id.get(e.object_name)
        if subj is None or obj is None:
            unresolved += 1
            continue
        resolved.append(
            EdgeCandidate(
                subject_id=subj,
                object_id=obj,
                predicate=e.predicate,
                source=e.source,
                evidence=e.evidence,
            )
        )
    if unresolved:
        logger.warning("ssodnet: %d edges dropped — endpoint could not be resolved", unresolved)

    inserted = bulk_insert_edges(conn, resolved, harvest_run_id=harvest_run_id)
    conn.commit()
    logger.info("ssodnet: inserted %d new edges", inserted)
    return SourceStats(
        source="ssodnet",
        edges_emitted=len(resolved),
        edges_inserted=inserted,
        taxa_created=taxa_created,
    )


# ---------------------------------------------------------------------------
# Curated flagship
# ---------------------------------------------------------------------------


def populate_curated_flagship(conn: psycopg.Connection, *, harvest_run_id: int) -> SourceStats:
    """Extract curated flagship mission->instrument edges."""
    logger.info("curated_flagship: loading flagship missions and known instruments...")
    with conn.cursor() as cur:
        # Flagship missions live under source='curated_flagship_v1'
        cur.execute("""
            SELECT canonical_name, id FROM entities
             WHERE source = 'curated_flagship_v1'
               AND entity_type = 'mission'
            """)
        missions_by_name = {name: eid for name, eid in cur.fetchall()}

        # Instruments can come from any source (GCMD, AAS, SPASE) — pick
        # the first id per canonical_name.  We don't try to
        # disambiguate because the curated table uses canonical short
        # names that should be unique across the flagship suite.
        cur.execute("""
            SELECT DISTINCT ON (canonical_name) canonical_name, id
              FROM entities
             WHERE entity_type = 'instrument'
             ORDER BY canonical_name, id
            """)
        instruments_by_name = {name: eid for name, eid in cur.fetchall()}

    edges = list(extract_curated_flagship_edges(missions_by_name, instruments_by_name))
    logger.info("curated_flagship: emitted %d candidate edges", len(edges))

    inserted = bulk_insert_edges(conn, edges, harvest_run_id=harvest_run_id)
    conn.commit()
    logger.info("curated_flagship: inserted %d new edges", inserted)
    return SourceStats(
        source="curated_flagship",
        edges_emitted=len(edges),
        edges_inserted=inserted,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


SOURCE_DISPATCH = {
    "gcmd": populate_gcmd,
    "spase": populate_spase,
    "curated_flagship": populate_curated_flagship,
}


def run(
    conn: psycopg.Connection,
    *,
    sources: list[str],
    include_ssodnet_targets: bool,
    ssodnet_limit: int | None,
) -> list[SourceStats]:
    """Run the requested extractors and return per-source stats."""
    run_log = HarvestRunLog(conn, source="entity_relationships_populate")
    run_id = run_log.start(
        config={
            "sources": sources,
            "include_ssodnet_targets": include_ssodnet_targets,
            "ssodnet_limit": ssodnet_limit,
        }
    )

    all_stats: list[SourceStats] = []
    total_emitted = 0
    total_inserted = 0
    try:
        for src in sources:
            if src == "ssodnet":
                stats = populate_ssodnet(
                    conn,
                    harvest_run_id=run_id,
                    include_targets=include_ssodnet_targets,
                    limit=ssodnet_limit,
                )
            else:
                handler = SOURCE_DISPATCH[src]
                stats = handler(conn, harvest_run_id=run_id)
            all_stats.append(stats)
            total_emitted += stats.edges_emitted
            total_inserted += stats.edges_inserted
    finally:
        # Mark complete with counts — refresh_views=False because
        # downstream MVs don't depend on entity_relationships.
        run_log.complete(
            records_fetched=total_emitted,
            records_upserted=total_inserted,
            counts={s.source: s.edges_inserted for s in all_stats},
            refresh_views=False,
        )

    return all_stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        "--sources",
        dest="sources",
        nargs="+",
        choices=list(ALL_SOURCES) + ["all"],
        default=["all"],
        help="Which source extractors to run (default: all)",
    )
    parser.add_argument(
        "--include-ssodnet-targets",
        action="store_true",
        default=False,
        help=(
            "When ssodnet is in the source list, also emit one part_of "
            "edge per asteroid to its leaf class (~1.48M edges). Off by "
            "default to keep runs fast; turn on for full coverage."
        ),
    )
    parser.add_argument(
        "--ssodnet-limit",
        type=int,
        default=None,
        help="Optional cap on SsODNet target rows processed (for smoke tests).",
    )
    parser.add_argument("--db-url", dest="dsn", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--allow-prod", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = args.dsn or DEFAULT_DSN
    if is_production_dsn(dsn) and not args.allow_prod:
        logger.error(
            "refusing to run against production DSN %s — pass --allow-prod",
            redact_dsn(dsn),
        )
        return 2

    sources: list[str] = list(ALL_SOURCES) if "all" in args.sources else list(args.sources)
    logger.info(
        "populate_entity_relationships: dsn=%s sources=%s dry_run=%s",
        redact_dsn(dsn),
        sources,
        args.dry_run,
    )

    conn = get_connection(dsn)
    try:
        stats_list = run(
            conn,
            sources=sources,
            include_ssodnet_targets=args.include_ssodnet_targets,
            ssodnet_limit=args.ssodnet_limit,
        )
        if args.dry_run:
            conn.rollback()
            logger.info("DRY RUN — rolled back all DB writes")
    finally:
        conn.close()

    print("\nSummary:")
    for s in stats_list:
        extra = f" (taxa_created={s.taxa_created})" if s.taxa_created else ""
        print(
            f"  {s.source:20s} emitted={s.edges_emitted:>8d}  "
            f"inserted={s.edges_inserted:>8d}{extra}"
        )
    print(f"  total inserted: {sum(s.edges_inserted for s in stats_list):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
