#!/usr/bin/env python3
"""Materialize GCMD-rooted dataset entities from the CMR collection catalog.

GCMD vocabularies (instruments, platforms, observables, providers, projects)
are already loaded as ``entities`` rows of the corresponding ``entity_type``.
The GCMD ecosystem also exposes a *platform/dataset relationship feed*: every
NASA EOSDIS CMR collection records the GCMD-controlled platforms and
instruments it observes with. Each such collection therefore is a dataset
that lives inside the GCMD entity graph.

This script bridges that gap:

* Selects every ``datasets`` row sourced from CMR that has at least one
  GCMD-controlled platform or instrument link via the ``dataset_entities``
  bridge.
* Upserts a corresponding row into ``entities`` with
  ``entity_type='dataset'`` and ``source='gcmd'`` so the dataset side of the
  relationship is searchable as a first-class entity.
* Stores the CMR concept-id under ``entity_identifiers.id_scheme='cmr_concept_id'``
  and a ``gcmd_dataset_uuid`` synthetic identifier (UUID-shaped form derived
  from the concept-id) for cross-walks.
* Writes ``entity_relationships`` rows mirroring the platform / instrument
  links (predicates ``observed_by`` and ``measured_with``) so traversal from
  a GCMD platform to its datasets and back again works end-to-end.

The script reads from the current state of the ``datasets`` table — run
``harvest_eosdis_extend.py`` first to ensure full CMR coverage.

Usage::

    python scripts/harvest_gcmd_datasets.py --dry-run
    python scripts/harvest_gcmd_datasets.py
    python scripts/harvest_gcmd_datasets.py --batch-size 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.harvest_utils import (
    HarvestRunLog,
    upsert_entity,
    upsert_entity_alias,
    upsert_entity_identifier,
    upsert_entity_relationship,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARVEST_SOURCE = "gcmd_datasets"
ENTITY_SOURCE = "gcmd"
DISCIPLINE = "earth_science"

PRED_OBSERVED_BY = "observed_by"  # dataset -> platform
PRED_MEASURED_WITH = "measured_with"  # dataset -> instrument

# Map dataset_entities.relationship -> entity_relationships.predicate.
# Anything not in this map is skipped (we don't want to leak "has_instrument"
# style strings into the relationship graph from CMR's side).
RELATIONSHIP_PREDICATE: dict[str, str] = {
    "on_platform": PRED_OBSERVED_BY,
    "has_instrument": PRED_MEASURED_WITH,
}


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------


def _select_gcmd_linked_datasets(
    conn: Any,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return CMR datasets that have at least one GCMD-linked entity."""
    sql = """
        SELECT d.id, d.name, d.canonical_id, d.description, d.discipline,
               d.temporal_start, d.temporal_end, d.properties
        FROM datasets d
        WHERE d.source = 'cmr'
          AND EXISTS (
              SELECT 1
              FROM dataset_entities de
              JOIN entity_identifiers ei ON ei.entity_id = de.entity_id
              WHERE de.dataset_id = d.id
                AND ei.id_scheme = 'gcmd_uuid'
          )
        ORDER BY d.id
    """
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT %s"
        params = (limit,)

    rows: list[dict[str, Any]] = []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        for row in cur.fetchall():
            rows.append(dict(zip(cols, row, strict=True)))
    logger.info("Selected %d GCMD-linked CMR datasets", len(rows))
    return rows


def _select_gcmd_links_for_dataset(
    conn: Any,
    dataset_id: int,
) -> list[tuple[int, str]]:
    """Return [(entity_id, relationship)] of GCMD-linked entities."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT de.entity_id, de.relationship
            FROM dataset_entities de
            JOIN entity_identifiers ei ON ei.entity_id = de.entity_id
            WHERE de.dataset_id = %s
              AND ei.id_scheme = 'gcmd_uuid'
            """,
            (dataset_id,),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------


def _ingest_dataset(
    conn: Any,
    dataset: dict[str, Any],
    *,
    harvest_run_id: int,
) -> tuple[int, int, int]:
    """Upsert one dataset's entity, identifiers, aliases, and relationships.

    Returns ``(entity_id, alias_count, relationship_count)``.
    """
    name: str = dataset["name"] or dataset["canonical_id"]
    properties = dataset.get("properties") or {}

    entity_props: dict[str, Any] = {
        "cmr_concept_id": dataset["canonical_id"],
        "discipline": dataset.get("discipline") or DISCIPLINE,
        "short_name": properties.get("short_name", ""),
        "platforms": properties.get("platforms", []),
        "instruments": properties.get("instruments", []),
        "science_keywords": properties.get("science_keywords", []),
    }
    if dataset.get("temporal_start") is not None:
        entity_props["temporal_start"] = str(dataset["temporal_start"])
    if dataset.get("temporal_end") is not None:
        entity_props["temporal_end"] = str(dataset["temporal_end"])
    if dataset.get("description"):
        entity_props["description"] = dataset["description"]

    entity_id = upsert_entity(
        conn,
        canonical_name=name,
        entity_type="dataset",
        source=ENTITY_SOURCE,
        discipline=DISCIPLINE,
        harvest_run_id=harvest_run_id,
        properties=entity_props,
    )

    upsert_entity_identifier(
        conn,
        entity_id=entity_id,
        id_scheme="cmr_concept_id",
        external_id=dataset["canonical_id"],
        is_primary=True,
    )

    alias_count = 0
    short_name = properties.get("short_name", "").strip() if properties else ""
    if short_name and short_name != name:
        upsert_entity_alias(
            conn,
            entity_id=entity_id,
            alias=short_name,
            alias_source="cmr",
        )
        alias_count += 1

    rel_count = 0
    for linked_entity_id, relationship in _select_gcmd_links_for_dataset(
        conn, dataset["id"]
    ):
        predicate = RELATIONSHIP_PREDICATE.get(relationship)
        if predicate is None:
            continue
        upsert_entity_relationship(
            conn,
            subject_entity_id=entity_id,
            predicate=predicate,
            object_entity_id=linked_entity_id,
            source=ENTITY_SOURCE,
            harvest_run_id=harvest_run_id,
            confidence=1.0,
        )
        rel_count += 1

    return entity_id, alias_count, rel_count


def _ingest_batch(
    conn: Any,
    datasets: list[dict[str, Any]],
    *,
    harvest_run_id: int,
    batch_size: int,
) -> dict[str, int]:
    """Ingest datasets in batches, committing after each."""
    counts = {"entities": 0, "aliases": 0, "relationships": 0}

    for offset in range(0, len(datasets), batch_size):
        chunk = datasets[offset : offset + batch_size]
        for ds in chunk:
            _entity_id, alias_count, rel_count = _ingest_dataset(
                conn, ds, harvest_run_id=harvest_run_id
            )
            counts["entities"] += 1
            counts["aliases"] += alias_count
            counts["relationships"] += rel_count

        conn.commit()
        logger.info(
            "Committed %d/%d datasets (entities=%d aliases=%d relationships=%d)",
            offset + len(chunk),
            len(datasets),
            counts["entities"],
            counts["aliases"],
            counts["relationships"],
        )

    return counts


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_harvest(
    dsn: str | None = None,
    limit: int | None = None,
    batch_size: int = 1000,
    dry_run: bool = False,
) -> dict[str, int]:
    """Materialize GCMD-rooted dataset entities from CMR collections."""
    t0 = time.monotonic()
    conn = get_connection(dsn)
    try:
        datasets = _select_gcmd_linked_datasets(conn, limit=limit)

        if dry_run:
            counts = {
                "candidate_datasets": len(datasets),
            }
            logger.info("Dry run — %s in %.2fs", counts, time.monotonic() - t0)
            return counts

        if not datasets:
            logger.warning(
                "No GCMD-linked CMR datasets found. Run harvest_eosdis_extend.py "
                "first to ingest CMR collections."
            )
            return {"entities": 0, "aliases": 0, "relationships": 0}

        run_log = HarvestRunLog(conn, HARVEST_SOURCE)
        try:
            run_log.start(
                config={
                    "limit": limit,
                    "batch_size": batch_size,
                    "source_table": "datasets",
                }
            )
            counts = _ingest_batch(
                conn,
                datasets,
                harvest_run_id=run_log.run_id,
                batch_size=batch_size,
            )
            counts["candidate_datasets"] = len(datasets)
            run_log.complete(
                records_fetched=len(datasets),
                records_upserted=counts["entities"],
                counts=counts,
            )
            return counts
        except Exception as exc:
            try:
                run_log.fail(str(exc))
            except Exception:
                logger.warning("Failed to update harvest_run status to 'failed'")
            raise
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize GCMD-rooted dataset entities from CMR collections "
            "with GCMD-vocabulary platform/instrument links."
        ),
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database DSN (uses SCIX_DSN env var if omitted)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest the first N candidate datasets (debugging)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Rows per commit batch (default: 1000)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    counts = run_harvest(
        dsn=args.dsn,
        limit=args.limit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Dry run — GCMD-linked CMR datasets: {counts}")
    else:
        print(f"GCMD datasets harvest complete: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
