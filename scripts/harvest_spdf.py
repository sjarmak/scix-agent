#!/usr/bin/env python3
"""Harvest SPDF/CDAWeb datasets, observatories, and instruments.

Downloads dataset, observatory, and instrument metadata from the CDAWeb REST API
and stores them in the entity graph tables (migration 021):

- Datasets -> datasets table (source='spdf', canonical_id=CDAWeb dataset ID)
- Observatories -> entities table (entity_type='observatory', discipline='heliophysics')
- Instruments -> entities table (entity_type='instrument', discipline='heliophysics')
- SPASE ResourceIDs -> entity_identifiers (id_scheme='spase_resource_id')
- Instrument-Observatory links -> entity_relationships (predicate='at_observatory')
- Dataset-Instrument links -> dataset_entities (relationship='from_instrument')

Also writes to entity_dictionary via bulk_load() for backward compatibility.

Usage:
    python scripts/harvest_spdf.py --help
    python scripts/harvest_spdf.py --dry-run
    python scripts/harvest_spdf.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

CDAWEB_BASE = "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys"
SOURCE = "spdf"
DISCIPLINE = "heliophysics"


class FetchResult(NamedTuple):
    """Raw JSON responses from CDAWeb API."""

    datasets: list[dict[str, Any]]
    observatories: list[dict[str, Any]]
    instruments: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _make_client() -> ResilientClient:
    """Create a ResilientClient configured for CDAWeb."""
    return ResilientClient(
        max_retries=3,
        backoff_base=2.0,
        rate_limit=5.0,
        cache_dir=Path(".cache/spdf"),
        cache_ttl=86400.0,
        user_agent="scix-harvester/1.0",
        timeout=120.0,
    )


def fetch_datasets(client: ResilientClient) -> list[dict[str, Any]]:
    """Fetch all datasets from the CDAWeb REST API.

    Args:
        client: ResilientClient instance.

    Returns:
        List of dataset dicts from the API response.
    """
    url = f"{CDAWEB_BASE}/datasets"
    resp = client.get(url, headers={"Accept": "application/json"})
    data = resp.json()
    datasets = data.get("DatasetDescription", [])
    logger.info("Fetched %d datasets from CDAWeb", len(datasets))
    return datasets


def fetch_observatories(client: ResilientClient) -> list[dict[str, Any]]:
    """Fetch all observatories from the CDAWeb REST API.

    Args:
        client: ResilientClient instance.

    Returns:
        List of observatory group dicts from the API response.
    """
    url = f"{CDAWEB_BASE}/observatories"
    resp = client.get(url, headers={"Accept": "application/json"})
    data = resp.json()
    groups = data.get("ObservatoryGroupDescription", [])
    logger.info("Fetched %d observatory groups from CDAWeb", len(groups))
    return groups


def fetch_instruments(client: ResilientClient) -> list[dict[str, Any]]:
    """Fetch all instruments from the CDAWeb REST API.

    Args:
        client: ResilientClient instance.

    Returns:
        List of instrument type dicts from the API response.
    """
    url = f"{CDAWEB_BASE}/instruments"
    resp = client.get(url, headers={"Accept": "application/json"})
    data = resp.json()
    types = data.get("InstrumentTypeDescription", [])
    logger.info("Fetched %d instrument types from CDAWeb", len(types))
    return types


def fetch_all(client: ResilientClient) -> FetchResult:
    """Fetch datasets, observatories, and instruments from CDAWeb.

    Args:
        client: ResilientClient instance.

    Returns:
        FetchResult with all three collections.
    """
    return FetchResult(
        datasets=fetch_datasets(client),
        observatories=fetch_observatories(client),
        instruments=fetch_instruments(client),
    )


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def parse_observatories(
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse observatory groups into flat observatory records.

    Each ObservatoryGroupDescription has a Name and a list of
    ObservatoryDescription entries. We flatten to individual observatories.

    Args:
        groups: Raw observatory group dicts from the API.

    Returns:
        List of dicts with keys: name, group, spase_resource_id (optional).
    """
    observatories: list[dict[str, Any]] = []
    seen: set[str] = set()

    for group in groups:
        group_name = group.get("Name", "").strip()
        obs_list = group.get("ObservatoryDescription", [])
        if isinstance(obs_list, dict):
            obs_list = [obs_list]

        for obs in obs_list:
            name = obs.get("Name", "").strip()
            if not name or name in seen:
                continue
            seen.add(name)

            record: dict[str, Any] = {
                "name": name,
                "group": group_name,
            }
            spase_id = obs.get("ResourceId", "").strip()
            if spase_id:
                record["spase_resource_id"] = spase_id

            observatories.append(record)

    logger.info("Parsed %d unique observatories", len(observatories))
    return observatories


def parse_instruments(
    types: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse instrument types into flat instrument records.

    Each InstrumentTypeDescription has a Name and a list of
    InstrumentDescription entries.

    Args:
        types: Raw instrument type dicts from the API.

    Returns:
        List of dicts with keys: name, instrument_type, spase_resource_id (optional).
    """
    instruments: list[dict[str, Any]] = []
    seen: set[str] = set()

    for itype in types:
        type_name = itype.get("Name", "").strip()
        inst_list = itype.get("InstrumentDescription", [])
        if isinstance(inst_list, dict):
            inst_list = [inst_list]

        for inst in inst_list:
            name = inst.get("Name", "").strip()
            if not name or name in seen:
                continue
            seen.add(name)

            record: dict[str, Any] = {
                "name": name,
                "instrument_type": type_name,
            }
            spase_id = inst.get("ResourceId", "").strip()
            if spase_id:
                record["spase_resource_id"] = spase_id

            instruments.append(record)

    logger.info("Parsed %d unique instruments", len(instruments))
    return instruments


def parse_datasets(
    raw_datasets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse dataset descriptions into structured records.

    Args:
        raw_datasets: Raw dataset dicts from the API.

    Returns:
        List of dicts with keys: id, label, description, start_date, end_date,
            observatory_groups, instrument_types, spase_resource_id (optional).
    """
    datasets: list[dict[str, Any]] = []

    for ds in raw_datasets:
        ds_id = ds.get("Id", "").strip()
        if not ds_id:
            continue

        label = ds.get("Label", "").strip() or ds_id

        record: dict[str, Any] = {
            "id": ds_id,
            "label": label,
        }

        notes = ds.get("Notes", "").strip()
        if notes:
            record["description"] = notes

        time_interval = ds.get("TimeInterval", {})
        if time_interval:
            start = time_interval.get("Start", "").strip()
            end = time_interval.get("End", "").strip()
            if start:
                record["start_date"] = start[:10]  # YYYY-MM-DD
            if end:
                record["end_date"] = end[:10]

        # Observatory and instrument associations
        obs_groups = ds.get("ObservatoryGroup", [])
        if isinstance(obs_groups, str):
            obs_groups = [obs_groups]
        record["observatory_groups"] = [g.strip() for g in obs_groups if g.strip()]

        inst_types = ds.get("InstrumentType", [])
        if isinstance(inst_types, str):
            inst_types = [inst_types]
        record["instrument_types"] = [t.strip() for t in inst_types if t.strip()]

        spase_id = ds.get("ResourceId", "").strip()
        if spase_id:
            record["spase_resource_id"] = spase_id

        datasets.append(record)

    logger.info("Parsed %d datasets", len(datasets))
    return datasets


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------


def _create_harvest_run(conn: Any, config: dict[str, Any] | None = None) -> int:
    """Create a harvest_runs row with status='running'.

    Args:
        conn: Database connection.
        config: Optional config JSONB.

    Returns:
        The harvest_run id.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO harvest_runs (source, status, config)
            VALUES (%s, 'running', %s)
            RETURNING id
            """,
            (SOURCE, json.dumps(config or {})),
        )
        run_id: int = cur.fetchone()[0]
    conn.commit()
    return run_id


def _complete_harvest_run(
    conn: Any,
    run_id: int,
    *,
    records_fetched: int,
    records_upserted: int,
    counts: dict[str, int],
) -> None:
    """Mark a harvest_run as completed with counts.

    Args:
        conn: Database connection.
        run_id: The harvest_run id.
        records_fetched: Total records fetched from API.
        records_upserted: Total records written to DB.
        counts: Breakdown of counts by type.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE harvest_runs
            SET finished_at = now(),
                status = 'completed',
                records_fetched = %s,
                records_upserted = %s,
                counts = %s
            WHERE id = %s
            """,
            (records_fetched, records_upserted, json.dumps(counts), run_id),
        )
    conn.commit()


def _fail_harvest_run(conn: Any, run_id: int, error_message: str) -> None:
    """Mark a harvest_run as failed.

    Args:
        conn: Database connection.
        run_id: The harvest_run id.
        error_message: Error description.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE harvest_runs
            SET finished_at = now(),
                status = 'failed',
                error_message = %s
            WHERE id = %s
            """,
            (error_message, run_id),
        )
    conn.commit()


def _upsert_entity(
    conn: Any,
    *,
    canonical_name: str,
    entity_type: str,
    source: str,
    discipline: str,
    harvest_run_id: int,
    properties: dict[str, Any] | None = None,
) -> int:
    """Upsert an entity and return its id.

    Args:
        conn: Database connection.
        canonical_name: Entity canonical name.
        entity_type: Entity type string.
        source: Source identifier.
        discipline: Discipline tag.
        harvest_run_id: Associated harvest run.
        properties: Optional JSONB properties.

    Returns:
        The entity id.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, discipline, source,
                                  harvest_run_id, properties)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (canonical_name, entity_type, source) DO UPDATE SET
                discipline = EXCLUDED.discipline,
                harvest_run_id = EXCLUDED.harvest_run_id,
                properties = EXCLUDED.properties,
                updated_at = NOW()
            RETURNING id
            """,
            (
                canonical_name,
                entity_type,
                discipline,
                source,
                harvest_run_id,
                json.dumps(properties or {}),
            ),
        )
        return cur.fetchone()[0]


def _upsert_entity_identifier(
    conn: Any,
    *,
    entity_id: int,
    id_scheme: str,
    external_id: str,
    is_primary: bool = False,
) -> None:
    """Upsert an entity identifier.

    Args:
        conn: Database connection.
        entity_id: The entity's id.
        id_scheme: Identifier scheme (e.g. 'spase_resource_id').
        external_id: The external identifier value.
        is_primary: Whether this is the primary identifier.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entity_identifiers (entity_id, id_scheme, external_id, is_primary)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id_scheme, external_id) DO UPDATE SET
                entity_id = EXCLUDED.entity_id,
                is_primary = EXCLUDED.is_primary
            """,
            (entity_id, id_scheme, external_id, is_primary),
        )


def _upsert_entity_relationship(
    conn: Any,
    *,
    subject_entity_id: int,
    predicate: str,
    object_entity_id: int,
    source: str,
    harvest_run_id: int,
) -> None:
    """Upsert an entity relationship.

    Args:
        conn: Database connection.
        subject_entity_id: Subject entity id.
        predicate: Relationship predicate.
        object_entity_id: Object entity id.
        source: Source identifier.
        harvest_run_id: Associated harvest run.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entity_relationships
                (subject_entity_id, predicate, object_entity_id, source, harvest_run_id)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (subject_entity_id, predicate, object_entity_id) DO NOTHING
            """,
            (subject_entity_id, predicate, object_entity_id, source, harvest_run_id),
        )


def _upsert_dataset(
    conn: Any,
    *,
    name: str,
    discipline: str,
    source: str,
    canonical_id: str,
    description: str | None = None,
    temporal_start: str | None = None,
    temporal_end: str | None = None,
    properties: dict[str, Any] | None = None,
    harvest_run_id: int,
) -> int:
    """Upsert a dataset and return its id.

    Args:
        conn: Database connection.
        name: Dataset display name.
        discipline: Discipline tag.
        source: Source identifier.
        canonical_id: Canonical dataset identifier.
        description: Optional description text.
        temporal_start: Optional start date (YYYY-MM-DD).
        temporal_end: Optional end date (YYYY-MM-DD).
        properties: Optional JSONB properties.
        harvest_run_id: Associated harvest run.

    Returns:
        The dataset id.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO datasets (name, discipline, source, canonical_id, description,
                                  temporal_start, temporal_end, properties, harvest_run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, canonical_id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                temporal_start = EXCLUDED.temporal_start,
                temporal_end = EXCLUDED.temporal_end,
                properties = EXCLUDED.properties,
                harvest_run_id = EXCLUDED.harvest_run_id
            RETURNING id
            """,
            (
                name,
                discipline,
                source,
                canonical_id,
                description,
                temporal_start,
                temporal_end,
                json.dumps(properties or {}),
                harvest_run_id,
            ),
        )
        return cur.fetchone()[0]


def _upsert_dataset_entity(
    conn: Any,
    *,
    dataset_id: int,
    entity_id: int,
    relationship: str,
) -> None:
    """Upsert a dataset-entity bridge row.

    Args:
        conn: Database connection.
        dataset_id: The dataset id.
        entity_id: The entity id.
        relationship: Relationship type string.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO dataset_entities (dataset_id, entity_id, relationship)
            VALUES (%s, %s, %s)
            ON CONFLICT (dataset_id, entity_id, relationship) DO NOTHING
            """,
            (dataset_id, entity_id, relationship),
        )


def store_harvest(
    conn: Any,
    *,
    observatories: list[dict[str, Any]],
    instruments: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
) -> dict[str, int]:
    """Store all harvested data into DB tables.

    Args:
        conn: Database connection.
        observatories: Parsed observatory records.
        instruments: Parsed instrument records.
        datasets: Parsed dataset records.

    Returns:
        Dict of counts by category.
    """
    total_fetched = len(observatories) + len(instruments) + len(datasets)
    run_id = _create_harvest_run(conn, config={"source_url": CDAWEB_BASE})

    try:
        # --- Observatories ---
        obs_name_to_id: dict[str, int] = {}
        for obs in observatories:
            eid = _upsert_entity(
                conn,
                canonical_name=obs["name"],
                entity_type="observatory",
                source=SOURCE,
                discipline=DISCIPLINE,
                harvest_run_id=run_id,
                properties={"group": obs.get("group", "")},
            )
            obs_name_to_id[obs["name"]] = eid

            if "spase_resource_id" in obs:
                _upsert_entity_identifier(
                    conn,
                    entity_id=eid,
                    id_scheme="spase_resource_id",
                    external_id=obs["spase_resource_id"],
                    is_primary=True,
                )

        conn.commit()
        logger.info("Stored %d observatories", len(obs_name_to_id))

        # --- Instruments ---
        inst_name_to_id: dict[str, int] = {}
        # Track instrument -> observatory associations from API
        inst_observatory_map: dict[str, set[str]] = {}

        for inst in instruments:
            eid = _upsert_entity(
                conn,
                canonical_name=inst["name"],
                entity_type="instrument",
                source=SOURCE,
                discipline=DISCIPLINE,
                harvest_run_id=run_id,
                properties={"instrument_type": inst.get("instrument_type", "")},
            )
            inst_name_to_id[inst["name"]] = eid

            if "spase_resource_id" in inst:
                _upsert_entity_identifier(
                    conn,
                    entity_id=eid,
                    id_scheme="spase_resource_id",
                    external_id=inst["spase_resource_id"],
                    is_primary=True,
                )

        conn.commit()
        logger.info("Stored %d instruments", len(inst_name_to_id))

        # --- Datasets ---
        dataset_count = 0
        relationship_count = 0

        for ds in datasets:
            ds_props: dict[str, Any] = {
                "observatory_groups": ds.get("observatory_groups", []),
                "instrument_types": ds.get("instrument_types", []),
                "spdf_dataset_id": ds["id"],
            }
            if "spase_resource_id" in ds:
                ds_props["spase_resource_id"] = ds["spase_resource_id"]

            ds_id = _upsert_dataset(
                conn,
                name=ds["label"],
                discipline=DISCIPLINE,
                source=SOURCE,
                canonical_id=ds["id"],
                description=ds.get("description"),
                temporal_start=ds.get("start_date"),
                temporal_end=ds.get("end_date"),
                properties=ds_props,
                harvest_run_id=run_id,
            )
            dataset_count += 1

            # Dataset identifiers go in properties since entity_identifiers
            # requires an entity_id FK. Store SPDF dataset ID and SPASE
            # ResourceID in the dataset's properties JSONB.
            # The canonical_id field already stores the CDAWeb dataset ID.

            # Link datasets to instruments via dataset_entities
            # CDAWeb datasets reference instrument types, not individual instruments.
            # We match by instrument_type name against our instruments.
            for inst_type in ds.get("instrument_types", []):
                # Find instruments of this type
                for inst in instruments:
                    if inst.get("instrument_type") == inst_type:
                        inst_eid = inst_name_to_id.get(inst["name"])
                        if inst_eid is not None:
                            _upsert_dataset_entity(
                                conn,
                                dataset_id=ds_id,
                                entity_id=inst_eid,
                                relationship="from_instrument",
                            )
                            relationship_count += 1

        conn.commit()
        logger.info(
            "Stored %d datasets with %d instrument links",
            dataset_count,
            relationship_count,
        )

        # --- Instrument -> Observatory relationships ---
        # CDAWeb datasets associate observatory groups with instruments indirectly.
        # Build associations from dataset metadata: if a dataset references both
        # an observatory group and instrument type, those instruments are at those
        # observatories.
        obs_inst_pairs: set[tuple[int, int]] = set()
        for ds in datasets:
            for obs_group in ds.get("observatory_groups", []):
                for inst_type in ds.get("instrument_types", []):
                    # Match observatory by group name
                    for obs in observatories:
                        if obs.get("group") == obs_group:
                            obs_eid = obs_name_to_id.get(obs["name"])
                            if obs_eid is None:
                                continue
                            for inst in instruments:
                                if inst.get("instrument_type") == inst_type:
                                    inst_eid = inst_name_to_id.get(inst["name"])
                                    if inst_eid is not None:
                                        obs_inst_pairs.add((inst_eid, obs_eid))

        for inst_eid, obs_eid in obs_inst_pairs:
            _upsert_entity_relationship(
                conn,
                subject_entity_id=inst_eid,
                predicate="at_observatory",
                object_entity_id=obs_eid,
                source=SOURCE,
                harvest_run_id=run_id,
            )

        conn.commit()
        logger.info(
            "Created %d instrument-observatory relationships",
            len(obs_inst_pairs),
        )

        total_upserted = len(obs_name_to_id) + len(inst_name_to_id) + dataset_count
        counts = {
            "observatories": len(obs_name_to_id),
            "instruments": len(inst_name_to_id),
            "datasets": dataset_count,
            "dataset_instrument_links": relationship_count,
            "instrument_observatory_links": len(obs_inst_pairs),
        }

        _complete_harvest_run(
            conn,
            run_id,
            records_fetched=total_fetched,
            records_upserted=total_upserted,
            counts=counts,
        )

        return counts

    except Exception:
        _fail_harvest_run(conn, run_id, error_message=str(sys.exc_info()[1]))
        raise


def _build_dictionary_entries(
    observatories: list[dict[str, Any]],
    instruments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build entity_dictionary entries for backward compatibility.

    Observatories are stored as entity_type='instrument' in entity_dictionary
    since 'observatory' is not in ALLOWED_ENTITY_TYPES.

    Args:
        observatories: Parsed observatory records.
        instruments: Parsed instrument records.

    Returns:
        List of entry dicts compatible with bulk_load().
    """
    entries: list[dict[str, Any]] = []

    for obs in observatories:
        entries.append(
            {
                "canonical_name": obs["name"],
                "entity_type": "instrument",  # closest allowed type
                "source": SOURCE,
                "external_id": obs.get("spase_resource_id"),
                "aliases": [],
                "metadata": {
                    "spdf_type": "observatory",
                    "group": obs.get("group", ""),
                },
            }
        )

    for inst in instruments:
        entries.append(
            {
                "canonical_name": inst["name"],
                "entity_type": "instrument",
                "source": SOURCE,
                "external_id": inst.get("spase_resource_id"),
                "aliases": [],
                "metadata": {
                    "spdf_type": "instrument",
                    "instrument_type": inst.get("instrument_type", ""),
                },
            }
        )

    return entries


def run_harvest(
    dsn: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Run the full SPDF/CDAWeb harvest pipeline.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        dry_run: If True, fetch and parse without writing to DB.

    Returns:
        Dict of counts by category.
    """
    t0 = time.monotonic()

    client = _make_client()
    raw = fetch_all(client)

    observatories = parse_observatories(raw.observatories)
    instruments = parse_instruments(raw.instruments)
    datasets = parse_datasets(raw.datasets)

    if dry_run:
        counts = {
            "observatories": len(observatories),
            "instruments": len(instruments),
            "datasets": len(datasets),
        }
        elapsed = time.monotonic() - t0
        logger.info("Dry run complete in %.1fs: %s", elapsed, counts)
        return counts

    conn = get_connection(dsn)
    try:
        # Store in entity graph tables
        counts = store_harvest(
            conn,
            observatories=observatories,
            instruments=instruments,
            datasets=datasets,
        )

        # Backward-compat: write to entity_dictionary
        dict_entries = _build_dictionary_entries(observatories, instruments)
        dict_count = bulk_load(conn, dict_entries, discipline=DISCIPLINE)
        logger.info("Loaded %d entries into entity_dictionary (compat)", dict_count)

    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info("SPDF harvest complete in %.1fs: %s", elapsed, counts)
    return counts


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the SPDF harvest pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest SPDF/CDAWeb datasets, observatories, and instruments",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse without writing to the database",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    counts = run_harvest(dsn=args.dsn, dry_run=args.dry_run)

    if args.dry_run:
        print(f"Dry run — {sum(counts.values())} total: {counts}")
    else:
        print(f"SPDF harvest complete: {counts}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
