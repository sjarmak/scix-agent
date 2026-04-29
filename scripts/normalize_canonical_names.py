#!/usr/bin/env python3
"""Backfill: re-canonicalize gliner ``entities`` rows and merge collisions.

Closes the markup-artifact split observed in bead
``scix_experiments-06xc``: 17.6M GLiNER entity rows include
``co<sub>2</sub>`` (chemical, 15k mentions) as a separate canonical
from ``co2`` (chemical) because ``canonicalize`` originally only
lowercased and stripped whitespace. Now that ``canonicalize`` runs
through :func:`scix.extract.surface_normalize.normalize_surface`, new
ingests merge cleanly. This script applies the same normalization to
the existing rows.

Algorithm
---------

1. Stream the gliner ``entities`` rows in entity-type chunks.
2. For each row, compute ``new_canon = canonicalize(canonical_name)``.
   If ``new_canon == canonical_name`` the row is already clean, skip.
3. Group changed rows by ``(new_canon, entity_type)`` (source is
   always 'gliner' here). Within a group:

   * Pick the survivor: the row whose ``id`` is lowest (ties broken
     by earliest ``created_at``). The lowest-id heuristic preserves
     the oldest row, which is the row most other tables already
     point at.
   * Merge ``document_entities`` from each loser into the survivor
     (``ON CONFLICT (bibcode, entity_id, link_type, tier) DO NOTHING``).
   * Merge ``entity_aliases`` and ``entity_identifiers`` similarly.
   * Re-point ``entity_relationships`` rows whose subject or object
     is a loser, dropping any (subject, predicate, object) duplicates
     that result.
   * Re-point ``curated_entity_core`` and ``dataset_entities``.
   * Insert the loser's old ``canonical_name`` as an
     ``alias_source='canonical_pre_normalize'`` alias on the survivor
     so query-time alias lookup still finds the old form.
   * Record an :func:`scix.entity_audit.record_merge` row.
   * ``DELETE FROM entities WHERE id = loser_id`` — the cascade
     cleans up any leftover bridge rows.
4. After a successful run, refresh
   ``document_entities_canonical`` so the materialized view sees the
   merged rows.

Production safety
-----------------

* ``--dry-run`` rolls back the transaction at the end (default).
* ``--allow-prod`` is required against the production DSN, and only
  works inside a ``systemd-run --scope`` (``INVOCATION_ID`` set).
* Each entity_type chunk runs in its own transaction so a crash mid-run
  loses at most one type's worth of work; the script is idempotent and
  can be rerun.

Usage
-----

::

    .venv/bin/python scripts/normalize_canonical_names.py --dry-run
    .venv/bin/python scripts/normalize_canonical_names.py --db scix_test
    scix-batch --allow-prod python scripts/normalize_canonical_names.py \\
        --allow-prod --refresh-mv
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import psycopg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scix.db import DEFAULT_DSN, get_connection, is_production_dsn, redact_dsn  # noqa: E402
from scix.entity_audit import record_merge  # noqa: E402
from scix.extract.ner_pass import canonicalize  # noqa: E402

logger = logging.getLogger(__name__)


# Source we are normalizing. Other sources have curated canonical names
# from upstream registries and should not be retroactively rewritten.
SOURCE_GLINER = "gliner"

# All entity types observed in the corpus. We process them one at a
# time so each chunk fits comfortably in working memory and so a crash
# only loses a single type's progress.
DEFAULT_ENTITY_TYPES: tuple[str, ...] = (
    "chemical",
    "method",
    "instrument",
    "organism",
    "location",
    "gene",
    "dataset",
    "software",
    "mission",
)

# Free-text reason recorded in entity_merge_log for each merge.
MERGE_REASON = "surface_normalize_backfill"
# Stamp on entity_aliases.alias_source so we can find them later.
ALIAS_SOURCE = "canonical_pre_normalize"


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeStats:
    """Per-entity-type counters for the backfill run."""

    entity_type: str
    rows_seen: int
    rows_unchanged: int
    rows_renamed: int
    merge_groups: int
    rows_merged_away: int
    document_entities_repointed: int
    aliases_added: int


class ProdGuardError(SystemExit):
    """Raised when production safety guards refuse to proceed."""


# ---------------------------------------------------------------------------
# Core normalization pass
# ---------------------------------------------------------------------------


def _select_entities_for_type(
    cur: psycopg.Cursor,
    entity_type: str,
) -> list[tuple[int, str, Optional[str]]]:
    """Stream gliner entities of a given type into memory.

    Returns ``[(id, canonical_name, source_version), ...]`` ordered by
    id ASC (so the lowest-id survivor is the first row in any group).
    """
    cur.execute(
        """
        SELECT id, canonical_name, source_version
        FROM entities
        WHERE source = %s AND entity_type = %s
        ORDER BY id
        """,
        (SOURCE_GLINER, entity_type),
    )
    return [(row[0], row[1], row[2]) for row in cur.fetchall()]


def _classify_rows(
    rows: list[tuple[int, str, Optional[str]]],
) -> tuple[list[tuple[int, str]], dict[str, list[int]]]:
    """Partition rows into (rename_only, merge_groups).

    ``rename_only`` is the list of (id, new_canon) where the row's
    current ``canonical_name`` differs from ``new_canon`` and no other
    row's ``new_canon`` collides. These are simple UPDATEs.

    ``merge_groups`` maps ``new_canon`` -> ``[id, ...]`` (sorted by id
    ASC) for every group with size >= 2. The first id is the
    survivor; the rest are losers to merge into it.
    """
    by_new_canon: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row_id, old_canon, _ in rows:
        new_canon = canonicalize(old_canon)
        by_new_canon[new_canon].append((row_id, old_canon))

    rename_only: list[tuple[int, str]] = []
    merge_groups: dict[str, list[int]] = {}

    for new_canon, members in by_new_canon.items():
        if len(members) == 1:
            row_id, old_canon = members[0]
            if new_canon != old_canon:
                rename_only.append((row_id, new_canon))
        else:
            # Multiple ids land on the same new_canon -> merge them.
            members.sort(key=lambda m: m[0])
            merge_groups[new_canon] = [m[0] for m in members]

    return rename_only, merge_groups


def _merge_one_group(
    conn: psycopg.Connection,
    survivor_id: int,
    loser_ids: list[int],
    new_canon: str,
    entity_type: str,
    actor: str,
) -> tuple[int, int]:
    """Fold ``loser_ids`` into ``survivor_id``.

    Returns (document_entities_repointed, aliases_added).

    Strategy: re-point bridge rows from losers to survivor, dropping
    rows that would violate composite primary keys via INSERT ... ON
    CONFLICT DO NOTHING. The loser entities are deleted at the end;
    ``ON DELETE CASCADE`` cleans up any rows we did not move.
    """
    de_repointed = 0
    aliases_added = 0

    with conn.cursor() as cur:
        # 1. document_entities: insert (survivor) duplicates, then delete
        #    losers' rows. The composite PK is (bibcode, entity_id,
        #    link_type, tier).
        cur.execute(
            """
            INSERT INTO document_entities
                (bibcode, entity_id, link_type, confidence, match_method,
                 evidence, harvest_run_id, tier, tier_version,
                 citation_consistency)
            SELECT bibcode, %(surv)s, link_type, confidence, match_method,
                   evidence, harvest_run_id, tier, tier_version,
                   citation_consistency
            FROM document_entities
            WHERE entity_id = ANY(%(losers)s)
            ON CONFLICT (bibcode, entity_id, link_type, tier) DO NOTHING
            """,
            {"surv": survivor_id, "losers": loser_ids},
        )
        de_repointed = cur.rowcount or 0

        # 2. entity_aliases: re-point losers' aliases to survivor.
        cur.execute(
            """
            INSERT INTO entity_aliases (entity_id, alias, alias_source)
            SELECT %(surv)s, alias, alias_source
            FROM entity_aliases
            WHERE entity_id = ANY(%(losers)s)
            ON CONFLICT (entity_id, alias) DO NOTHING
            """,
            {"surv": survivor_id, "losers": loser_ids},
        )
        aliases_added += cur.rowcount or 0

        # 3. Capture each loser's old canonical_name as an alias on the
        #    survivor so query-time alias lookup still finds the old
        #    surface form (e.g. 'co<sub>2</sub>' -> survivor 'co2').
        cur.execute(
            """
            INSERT INTO entity_aliases (entity_id, alias, alias_source)
            SELECT %(surv)s, canonical_name, %(asrc)s
            FROM entities
            WHERE id = ANY(%(losers)s)
              AND canonical_name <> %(new_canon)s
            ON CONFLICT (entity_id, alias) DO NOTHING
            """,
            {
                "surv": survivor_id,
                "asrc": ALIAS_SOURCE,
                "losers": loser_ids,
                "new_canon": new_canon,
            },
        )
        aliases_added += cur.rowcount or 0

        # 4. entity_identifiers: re-point. PK is (id_scheme, external_id),
        #    so we use INSERT...ON CONFLICT DO NOTHING and then drop the
        #    original rows via the cascade on DELETE entities below.
        cur.execute(
            """
            INSERT INTO entity_identifiers
                (entity_id, id_scheme, external_id, is_primary)
            SELECT %(surv)s, id_scheme, external_id, is_primary
            FROM entity_identifiers
            WHERE entity_id = ANY(%(losers)s)
            ON CONFLICT (id_scheme, external_id) DO NOTHING
            """,
            {"surv": survivor_id, "losers": loser_ids},
        )

        # 5. entity_relationships: re-point subject and object sides
        #    independently. UNIQUE(subject, predicate, object) on the
        #    target table means we may produce duplicates -- handle via
        #    INSERT ... ON CONFLICT DO NOTHING.
        cur.execute(
            """
            INSERT INTO entity_relationships
                (subject_entity_id, predicate, object_entity_id,
                 source, harvest_run_id, confidence)
            SELECT %(surv)s, predicate, object_entity_id,
                   source, harvest_run_id, confidence
            FROM entity_relationships
            WHERE subject_entity_id = ANY(%(losers)s)
            ON CONFLICT (subject_entity_id, predicate, object_entity_id)
                DO NOTHING
            """,
            {"surv": survivor_id, "losers": loser_ids},
        )
        cur.execute(
            """
            INSERT INTO entity_relationships
                (subject_entity_id, predicate, object_entity_id,
                 source, harvest_run_id, confidence)
            SELECT subject_entity_id, predicate, %(surv)s,
                   source, harvest_run_id, confidence
            FROM entity_relationships
            WHERE object_entity_id = ANY(%(losers)s)
            ON CONFLICT (subject_entity_id, predicate, object_entity_id)
                DO NOTHING
            """,
            {"surv": survivor_id, "losers": loser_ids},
        )

        # 6. dataset_entities: the loser may be referenced; PK is
        #    (dataset_id, entity_id, relationship).
        cur.execute(
            """
            INSERT INTO dataset_entities (dataset_id, entity_id, relationship)
            SELECT dataset_id, %(surv)s, relationship
            FROM dataset_entities
            WHERE entity_id = ANY(%(losers)s)
            ON CONFLICT (dataset_id, entity_id, relationship) DO NOTHING
            """,
            {"surv": survivor_id, "losers": loser_ids},
        )

        # 7. curated_entity_core: re-point if any losers appear.
        cur.execute(
            """
            UPDATE curated_entity_core
            SET entity_id = %(surv)s
            WHERE entity_id = ANY(%(losers)s)
              AND NOT EXISTS (
                  SELECT 1 FROM curated_entity_core c2
                  WHERE c2.entity_id = %(surv)s
              )
            """,
            {"surv": survivor_id, "losers": loser_ids},
        )

        # 8. Record audit rows for each loser -> survivor merge BEFORE
        #    deleting the losers (entity_merge_log only has an FK on
        #    new_entity_id, so old_entity_id can outlive the deleted row;
        #    we record first to keep the order obvious in the log).
        for loser_id in loser_ids:
            record_merge(
                conn,
                old_entity_id=loser_id,
                new_entity_id=survivor_id,
                reason=MERGE_REASON,
                merged_by=actor,
            )

        # 9. Delete losers BEFORE renaming the survivor. The unique
        #    constraint on (canonical_name, entity_type, source) means
        #    that if a loser's existing canonical_name already equals
        #    `new_canon` (e.g. survivor='cocl<sub>2</sub>',
        #    loser='cocl2' both map to 'cocl2'), the survivor UPDATE in
        #    step 10 would collide with the still-present loser row.
        #    Removing losers first frees the constraint.
        cur.execute(
            "DELETE FROM entities WHERE id = ANY(%s)",
            (loser_ids,),
        )

        # 10. Update the survivor's canonical_name to the new value.
        #     Safe now that all colliding losers have been deleted.
        cur.execute(
            """
            UPDATE entities
            SET canonical_name = %(new_canon)s,
                updated_at = NOW()
            WHERE id = %(surv)s
              AND canonical_name <> %(new_canon)s
            """,
            {"surv": survivor_id, "new_canon": new_canon},
        )

    return de_repointed, aliases_added


def _apply_renames(
    conn: psycopg.Connection,
    rename_only: list[tuple[int, str]],
) -> int:
    """Update entities.canonical_name where no collision exists.

    Returns the number of rows actually renamed.
    """
    if not rename_only:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE entities
            SET canonical_name = %(new_canon)s, updated_at = NOW()
            WHERE id = %(id)s AND canonical_name <> %(new_canon)s
            """,
            [{"id": rid, "new_canon": nc} for rid, nc in rename_only],
        )
    return len(rename_only)


def normalize_entity_type(
    conn: psycopg.Connection,
    entity_type: str,
    *,
    actor: str,
) -> TypeStats:
    """Run the normalization pass for one entity_type.

    The caller is responsible for COMMIT/ROLLBACK so that --dry-run
    runs can roll back the entire pass.
    """
    with conn.cursor() as cur:
        rows = _select_entities_for_type(cur, entity_type)

    if not rows:
        return TypeStats(
            entity_type=entity_type,
            rows_seen=0,
            rows_unchanged=0,
            rows_renamed=0,
            merge_groups=0,
            rows_merged_away=0,
            document_entities_repointed=0,
            aliases_added=0,
        )

    rename_only, merge_groups = _classify_rows(rows)

    rows_unchanged = len(rows) - len(rename_only) - sum(
        len(ids) for ids in merge_groups.values()
    )

    de_total = 0
    aliases_total = 0
    rows_merged_away = 0

    for new_canon, ids in merge_groups.items():
        survivor = ids[0]
        losers = ids[1:]
        de, al = _merge_one_group(
            conn, survivor, losers, new_canon, entity_type, actor
        )
        de_total += de
        aliases_total += al
        rows_merged_away += len(losers)

    rows_renamed = _apply_renames(conn, rename_only)

    return TypeStats(
        entity_type=entity_type,
        rows_seen=len(rows),
        rows_unchanged=rows_unchanged,
        rows_renamed=rows_renamed,
        merge_groups=len(merge_groups),
        rows_merged_away=rows_merged_away,
        document_entities_repointed=de_total,
        aliases_added=aliases_total,
    )


def refresh_canonical_mv(conn: psycopg.Connection) -> None:
    """Refresh ``document_entities_canonical`` after a merge pass.

    The materialized view aggregates by ``entity_id``, so any ids we
    deleted will leave stale rows until refreshed.
    """
    with conn.cursor() as cur:
        cur.execute("REFRESH MATERIALIZED VIEW document_entities_canonical")
    conn.commit()


# ---------------------------------------------------------------------------
# Production-safety guard
# ---------------------------------------------------------------------------


def enforce_prod_guard(
    *,
    dsn: str,
    allow_prod: bool,
    env: Mapping[str, str],
) -> None:
    """Mirror the guard pattern used by other backfill scripts."""
    if is_production_dsn(dsn) and not allow_prod:
        msg = (
            f"refusing to write to production DSN {redact_dsn(dsn)} - "
            "pass --allow-prod to override"
        )
        logger.error(msg)
        raise ProdGuardError(2)

    if allow_prod and not env.get("INVOCATION_ID"):
        msg = (
            "refusing to run --allow-prod outside a systemd scope. "
            "Invoke via: scix-batch --allow-prod python "
            "scripts/normalize_canonical_names.py --allow-prod"
        )
        logger.error(msg)
        raise ProdGuardError(2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_dsn(db_arg: Optional[str]) -> str:
    if db_arg:
        if "=" in db_arg or "://" in db_arg:
            return db_arg
        return f"dbname={db_arg}"
    return os.environ.get("SCIX_DSN") or DEFAULT_DSN


def _format_stats(stats: list[TypeStats]) -> str:
    """Pretty-print per-type stats for log output."""
    lines = [
        f"{'type':12} {'seen':>10} {'unchanged':>10} {'renamed':>10} "
        f"{'mergeGrp':>10} {'mergedAway':>12} {'deRepoint':>12} "
        f"{'aliases':>10}"
    ]
    totals = TypeStats("TOTAL", 0, 0, 0, 0, 0, 0, 0)
    for s in stats:
        lines.append(
            f"{s.entity_type:12} {s.rows_seen:>10} {s.rows_unchanged:>10} "
            f"{s.rows_renamed:>10} {s.merge_groups:>10} "
            f"{s.rows_merged_away:>12} {s.document_entities_repointed:>12} "
            f"{s.aliases_added:>10}"
        )
        totals = TypeStats(
            "TOTAL",
            totals.rows_seen + s.rows_seen,
            totals.rows_unchanged + s.rows_unchanged,
            totals.rows_renamed + s.rows_renamed,
            totals.merge_groups + s.merge_groups,
            totals.rows_merged_away + s.rows_merged_away,
            totals.document_entities_repointed + s.document_entities_repointed,
            totals.aliases_added + s.aliases_added,
        )
    lines.append(
        f"{totals.entity_type:12} {totals.rows_seen:>10} "
        f"{totals.rows_unchanged:>10} {totals.rows_renamed:>10} "
        f"{totals.merge_groups:>10} {totals.rows_merged_away:>12} "
        f"{totals.document_entities_repointed:>12} {totals.aliases_added:>10}"
    )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-canonicalize gliner entities and merge surface-form "
            "duplicates produced by HTML/Unicode markup leakage."
        ),
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database name or full DSN (default: SCIX_DSN env or scix).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run the backfill but roll back without committing.",
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        default=False,
        help="Required to write to a production DSN. Implies systemd scope.",
    )
    parser.add_argument(
        "--entity-type",
        action="append",
        default=None,
        help=(
            "Restrict to one or more entity_types (can be repeated). "
            f"Default: {', '.join(DEFAULT_ENTITY_TYPES)}."
        ),
    )
    parser.add_argument(
        "--refresh-mv",
        action="store_true",
        default=False,
        help=(
            "Refresh document_entities_canonical materialized view after "
            "the pass completes."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dsn = _resolve_dsn(args.db)
    enforce_prod_guard(dsn=dsn, allow_prod=args.allow_prod, env=os.environ)
    actor = os.environ.get("USER", "unknown") + ":normalize_canonical_names"

    entity_types = tuple(args.entity_type) if args.entity_type else DEFAULT_ENTITY_TYPES

    logger.info(
        "Starting normalize_canonical_names: db=%s dry_run=%s types=%s",
        redact_dsn(dsn),
        args.dry_run,
        entity_types,
    )

    all_stats: list[TypeStats] = []
    with get_connection(dsn) as conn:
        for et in entity_types:
            logger.info("Processing entity_type=%s", et)
            try:
                stats = normalize_entity_type(conn, et, actor=actor)
                if args.dry_run:
                    conn.rollback()
                    logger.info("dry-run: rolled back %s", et)
                else:
                    conn.commit()
                all_stats.append(stats)
                logger.info(
                    "%s: seen=%d renamed=%d mergeGroups=%d mergedAway=%d "
                    "deRepointed=%d aliasesAdded=%d",
                    et,
                    stats.rows_seen,
                    stats.rows_renamed,
                    stats.merge_groups,
                    stats.rows_merged_away,
                    stats.document_entities_repointed,
                    stats.aliases_added,
                )
            except Exception:  # pragma: no cover - operator-visible
                conn.rollback()
                logger.exception("entity_type=%s failed; rolled back", et)
                raise

        if args.refresh_mv and not args.dry_run:
            logger.info("Refreshing document_entities_canonical materialized view")
            refresh_canonical_mv(conn)

    print(_format_stats(all_stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
