#!/usr/bin/env python3
"""Harvest the WCRP CMIP6 controlled vocabulary registry into the datasets table.

The Coupled Model Intercomparison Project Phase 6 (CMIP6) defines its
canonical models, experiments, institutions, and activities in the
WCRP-CMIP/CMIP6_CVs GitHub repository. We ingest two of those vocabularies
as ``datasets`` rows so they are first-class search targets:

* ``source='cmip6_source'`` — one row per registered climate model
  (``source_id``); ~130 entries.
* ``source='cmip6_experiment'`` — one row per registered experiment
  (``experiment_id``); ~320 entries.

For each row we also write a corresponding ``entities`` row of type
``dataset`` so the registry shows up in entity search alongside other
domain entities, and a ``cmip6_*`` identifier in ``entity_identifiers``.

Usage::

    python scripts/harvest_cmip6.py --dry-run
    python scripts/harvest_cmip6.py
    python scripts/harvest_cmip6.py --kind source --kind experiment
"""

from __future__ import annotations

import argparse
import json
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
)
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CV_BASE = "https://raw.githubusercontent.com/WCRP-CMIP/CMIP6_CVs/main"
DISCIPLINE = "earth_science"
HARVEST_SOURCE = "cmip6"  # used for harvest_runs.source

# Each CV entry tracked. (key in JSON, dataset source tag, canonical_id prefix)
KIND_CONFIG: dict[str, dict[str, str]] = {
    "source": {
        "json_url": f"{CV_BASE}/CMIP6_source_id.json",
        "json_key": "source_id",
        "dataset_source": "cmip6_source",
        "id_scheme": "cmip6_source_id",
    },
    "experiment": {
        "json_url": f"{CV_BASE}/CMIP6_experiment_id.json",
        "json_key": "experiment_id",
        "dataset_source": "cmip6_experiment",
        "id_scheme": "cmip6_experiment_id",
    },
}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _make_client() -> ResilientClient:
    return ResilientClient(
        max_retries=3,
        backoff_base=2.0,
        rate_limit=5.0,
        cache_dir=Path(".cache/cmip6"),
        cache_ttl=86400.0,
        user_agent="scix-harvester/1.0",
        timeout=60.0,
    )


def _fetch_cv(client: ResilientClient, url: str, json_key: str) -> dict[str, Any]:
    """Download a CMIP6 CV JSON file and return the inner registry dict."""
    resp = client.get(url, headers={"Accept": "application/json"})
    payload = resp.json()
    if not isinstance(payload, dict) or json_key not in payload:
        raise ValueError(f"Unexpected payload from {url}: missing key {json_key!r}")
    cv = payload[json_key]
    if not isinstance(cv, dict):
        raise ValueError(f"CV {json_key} from {url} is not a dict")
    logger.info("Fetched %d %s entries from %s", len(cv), json_key, url)
    return cv


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_source_record(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a CMIP6 source_id record into a dataset row."""
    label = raw.get("label", name).strip() or name
    long_name = raw.get("label_extended", "").strip()
    institutions = raw.get("institution_id", []) or []
    activities = raw.get("activity_participation", []) or []

    components = raw.get("model_component", {}) or {}
    component_summary: dict[str, str] = {}
    for comp_key, comp_val in components.items():
        if isinstance(comp_val, dict):
            desc = comp_val.get("description", "").strip()
            if desc and desc.lower() != "none":
                component_summary[comp_key] = desc

    description_parts: list[str] = []
    if long_name:
        description_parts.append(long_name)
    if institutions:
        description_parts.append(f"Institution: {', '.join(institutions)}")
    if activities:
        description_parts.append(f"MIPs: {', '.join(activities)}")
    description = ". ".join(description_parts) or None

    properties = {
        "label": label,
        "label_extended": long_name,
        "institution_id": institutions,
        "activity_participation": activities,
        "release_year": raw.get("release_year", ""),
        "model_components": component_summary,
        "license": (raw.get("license_info") or {}).get("id", ""),
    }

    return {
        "canonical_id": name,
        "name": long_name or label or name,
        "description": description,
        "properties": properties,
        "aliases": [label] if label and label != (long_name or label or name) else [],
    }


def _parse_experiment_record(name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a CMIP6 experiment_id record into a dataset row."""
    activity = raw.get("activity_id", []) or []
    parent_activity = raw.get("parent_activity_id", []) or []
    parent_experiment = raw.get("parent_experiment_id", []) or []
    description = (raw.get("description") or "").strip()
    long_name = (raw.get("experiment") or "").strip()

    start_year = raw.get("start_year", "").strip() if isinstance(raw.get("start_year"), str) else ""
    end_year = raw.get("end_year", "").strip() if isinstance(raw.get("end_year"), str) else ""

    properties = {
        "experiment_id": raw.get("experiment_id", name),
        "experiment": long_name,
        "activity_id": activity,
        "parent_activity_id": parent_activity,
        "parent_experiment_id": parent_experiment,
        "tier": raw.get("tier", ""),
        "start_year": start_year,
        "end_year": end_year,
        "min_number_yrs_per_sim": raw.get("min_number_yrs_per_sim", ""),
        "required_model_components": raw.get("required_model_components", []),
        "additional_allowed_model_components": raw.get(
            "additional_allowed_model_components", []
        ),
        "sub_experiment_id": raw.get("sub_experiment_id", []),
    }

    full_description = description
    if long_name and long_name not in full_description:
        full_description = f"{description}. {long_name}".strip(". ")

    temporal_start = f"{start_year}-01-01" if start_year.isdigit() and len(start_year) == 4 else None
    temporal_end = f"{end_year}-12-31" if end_year.isdigit() and len(end_year) == 4 else None

    return {
        "canonical_id": name,
        "name": long_name or description or name,
        "description": full_description or None,
        "properties": properties,
        "temporal_start": temporal_start,
        "temporal_end": temporal_end,
        "aliases": [name] if (long_name or description) and name not in (long_name, description) else [],
    }


PARSERS = {
    "source": _parse_source_record,
    "experiment": _parse_experiment_record,
}


# ---------------------------------------------------------------------------
# Database write
# ---------------------------------------------------------------------------


def _upsert_dataset(
    conn: Any,
    *,
    name: str,
    discipline: str,
    source: str,
    canonical_id: str,
    description: str | None,
    temporal_start: str | None,
    temporal_end: str | None,
    properties: dict[str, Any],
    harvest_run_id: int,
) -> int:
    """Upsert into ``datasets`` and return the row id."""
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
                json.dumps(properties),
                harvest_run_id,
            ),
        )
        return cur.fetchone()[0]


def _store_kind(
    conn: Any,
    kind: str,
    cv: dict[str, Any],
    *,
    harvest_run_id: int,
) -> dict[str, int]:
    """Persist one CMIP6 kind (source or experiment) to datasets + entities."""
    cfg = KIND_CONFIG[kind]
    parser = PARSERS[kind]

    dataset_count = 0
    entity_count = 0
    alias_count = 0

    for name, raw in cv.items():
        if not isinstance(raw, dict):
            logger.debug("Skipping non-dict entry %r for kind=%s", name, kind)
            continue
        record = parser(name, raw)

        ds_id = _upsert_dataset(
            conn,
            name=record["name"],
            discipline=DISCIPLINE,
            source=cfg["dataset_source"],
            canonical_id=record["canonical_id"],
            description=record.get("description"),
            temporal_start=record.get("temporal_start"),
            temporal_end=record.get("temporal_end"),
            properties=record["properties"],
            harvest_run_id=harvest_run_id,
        )
        dataset_count += 1

        # Mirror into entity graph so the registry is searchable as entities.
        entity_id = upsert_entity(
            conn,
            canonical_name=record["name"],
            entity_type="dataset",
            source=cfg["dataset_source"],
            discipline=DISCIPLINE,
            harvest_run_id=harvest_run_id,
            properties={
                "dataset_id": ds_id,
                "kind": kind,
                **record["properties"],
            },
        )
        entity_count += 1

        upsert_entity_identifier(
            conn,
            entity_id=entity_id,
            id_scheme=cfg["id_scheme"],
            external_id=record["canonical_id"],
            is_primary=True,
        )

        for alias in record.get("aliases", []):
            if alias and alias != record["name"]:
                upsert_entity_alias(
                    conn,
                    entity_id=entity_id,
                    alias=alias,
                    alias_source="cmip6",
                )
                alias_count += 1

    conn.commit()

    counts = {
        f"{kind}_datasets": dataset_count,
        f"{kind}_entities": entity_count,
        f"{kind}_aliases": alias_count,
    }
    logger.info("Stored CMIP6 %s registry: %s", kind, counts)
    return counts


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_harvest(
    dsn: str | None = None,
    kinds: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Harvest CMIP6 CVs and load them into datasets + entities."""
    target_kinds = list(kinds) if kinds else list(KIND_CONFIG.keys())
    invalid = [k for k in target_kinds if k not in KIND_CONFIG]
    if invalid:
        raise ValueError(f"Unknown CMIP6 kind(s): {invalid}")

    t0 = time.monotonic()
    client = _make_client()

    cvs: dict[str, dict[str, Any]] = {}
    for kind in target_kinds:
        cfg = KIND_CONFIG[kind]
        cvs[kind] = _fetch_cv(client, cfg["json_url"], cfg["json_key"])

    if dry_run:
        counts = {f"{k}_count": len(v) for k, v in cvs.items()}
        counts["total"] = sum(len(v) for v in cvs.values())
        logger.info("Dry run — %s in %.2fs", counts, time.monotonic() - t0)
        return counts

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, HARVEST_SOURCE)
    try:
        run_log.start(config={"kinds": target_kinds, "cv_base": CV_BASE})

        all_counts: dict[str, int] = {}
        total_records = 0
        for kind in target_kinds:
            kind_counts = _store_kind(
                conn,
                kind,
                cvs[kind],
                harvest_run_id=run_log.run_id,
            )
            all_counts.update(kind_counts)
            total_records += kind_counts.get(f"{kind}_datasets", 0)

        all_counts["total_datasets"] = total_records
        run_log.complete(
            records_fetched=sum(len(v) for v in cvs.values()),
            records_upserted=total_records,
            counts=all_counts,
        )
        return all_counts
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
        description="Harvest WCRP CMIP6 CVs into datasets + entities tables",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database DSN (uses SCIX_DSN env var if omitted)",
    )
    parser.add_argument(
        "--kind",
        action="append",
        choices=list(KIND_CONFIG.keys()),
        default=None,
        help="Limit to a specific CV kind (repeatable). Default: all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    counts = run_harvest(dsn=args.dsn, kinds=args.kind, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Dry run — CMIP6 CVs: {counts}")
    else:
        print(f"CMIP6 harvest complete: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
