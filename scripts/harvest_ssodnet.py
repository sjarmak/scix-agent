#!/usr/bin/env python3
"""Harvest SsODNet (Solar System Open Database Network) entities.

Downloads the ssoBFT Parquet file (bulk mode) or queries the SsODNet API
(seed mode) to populate the entity graph with solar system objects:

- Entities -> entities table (entity_type='target', source='ssodnet')
- SsODNet names -> entity_identifiers (id_scheme='ssodnet')
- SPK-IDs -> entity_identifiers (id_scheme='sbdb_spkid')
- Alternate designations -> entity_aliases
- Physical properties -> entities.properties JSONB

Bulk mode uses the staging schema (migration 022) for atomic promote.
Seed mode uses direct upsert helpers for a small set of well-known objects.

Usage:
    python scripts/harvest_ssodnet.py --help
    python scripts/harvest_ssodnet.py --mode seed --dry-run
    python scripts/harvest_ssodnet.py --mode bulk -v
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import tempfile
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

PARQUET_URL = "https://ssp.imcce.fr/data/ssoBFT-latest_Asteroid.parquet"
API_BASE = "https://ssp.imcce.fr/webservices/ssodnet/api/ssocard"
SOURCE = "ssodnet"
DISCIPLINE = "planetary_science"
ENTITY_TYPE = "target"

# Well-known objects for seed mode
WELL_KNOWN_OBJECTS: list[str] = [
    "Ceres",
    "Pallas",
    "Juno",
    "Vesta",
    "Astraea",
    "Hebe",
    "Iris",
    "Flora",
    "Metis",
    "Hygiea",
    "Eros",
    "Ida",
    "Gaspra",
    "Mathilde",
    "Itokawa",
    "Bennu",
    "Ryugu",
    "Psyche",
    "Lutetia",
    "Steins",
]

# ---------------------------------------------------------------------------
# Module-level client (lazy init)
# ---------------------------------------------------------------------------

_client: ResilientClient | None = None


def _get_client() -> ResilientClient:
    """Return a shared ResilientClient instance."""
    global _client
    if _client is None:
        _client = ResilientClient(
            user_agent="scix-experiments/1.0",
            max_retries=3,
            backoff_base=2.0,
            rate_limit=2.0,
            cache_dir=Path(".cache/ssodnet"),
            cache_ttl=86400.0,
            timeout=300.0,
        )
    return _client


# ---------------------------------------------------------------------------
# Parquet download with SHA-256 checksum
# ---------------------------------------------------------------------------


def download_parquet(
    url: str,
    dest_path: Path,
    client: ResilientClient | None = None,
) -> tuple[Path, str]:
    """Download a Parquet file and compute its SHA-256 checksum.

    Args:
        url: URL to download from.
        dest_path: Local file path to save to.
        client: Optional ResilientClient instance.

    Returns:
        Tuple of (dest_path, sha256_hex).
    """
    if client is None:
        client = _get_client()

    logger.info("Downloading Parquet from %s", url)
    response = client.get(url)

    content = response.content if hasattr(response, "content") else response.text.encode()
    sha256 = hashlib.sha256(content).hexdigest()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(content)

    logger.info(
        "Downloaded %d bytes to %s (SHA-256: %s)",
        len(content),
        dest_path,
        sha256[:16],
    )
    return dest_path, sha256


# ---------------------------------------------------------------------------
# Parquet reading
# ---------------------------------------------------------------------------


def read_parquet(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet file and return rows as dicts.

    Tries pyarrow first, falls back to pandas.

    Args:
        path: Path to the Parquet file.

    Returns:
        List of row dicts.
    """
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(str(path))
        records = table.to_pydict()
        # Convert columnar dict to list of row dicts
        keys = list(records.keys())
        num_rows = len(records[keys[0]]) if keys else 0
        rows = [{k: records[k][i] for k in keys} for i in range(num_rows)]
        logger.info("Read %d rows from %s using pyarrow", len(rows), path)
        return rows
    except ImportError:
        pass

    try:
        import pandas as pd

        df = pd.read_parquet(str(path))
        rows = df.to_dict(orient="records")
        logger.info("Read %d rows from %s using pandas", len(rows), path)
        return rows
    except ImportError:
        raise ImportError(
            "Neither pyarrow nor pandas is available. "
            "Install one of them: pip install pyarrow or pip install pandas"
        )


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------


def parse_sso_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a single ssoBFT row into a harvester record.

    Args:
        row: Dict from Parquet row.

    Returns:
        Parsed record dict or None if row is invalid.
    """
    sso_name = str(row.get("sso_name", "")).strip()
    if not sso_name:
        return None

    # Build properties from physical data
    properties: dict[str, Any] = {}
    for field in ("diameter", "albedo", "taxonomy_class"):
        val = row.get(field)
        if val is not None and val != "" and str(val) != "nan":
            # Map taxonomy_class to taxonomy in properties
            prop_key = "taxonomy" if field == "taxonomy_class" else field
            properties[prop_key] = val

    sso_number = row.get("sso_number")
    if sso_number is not None and str(sso_number).strip():
        properties["sso_number"] = sso_number

    # Build identifiers
    identifiers: list[dict[str, Any]] = [
        {"id_scheme": "ssodnet", "external_id": sso_name, "is_primary": True},
    ]
    spkid = row.get("spkid")
    if spkid is not None and str(spkid).strip() and str(spkid) != "nan":
        identifiers.append(
            {"id_scheme": "sbdb_spkid", "external_id": str(spkid).strip(), "is_primary": False}
        )

    # Build aliases from other_designations (pipe-separated)
    aliases: list[str] = []
    other_desig = row.get("other_designations", "")
    if other_desig and str(other_desig).strip() and str(other_desig) != "nan":
        for desig in str(other_desig).split("|"):
            desig = desig.strip()
            if desig and desig != sso_name:
                aliases.append(desig)

    # Add numbered designation as alias if present
    if sso_number is not None and str(sso_number).strip() and str(sso_number) != "nan":
        num_str = str(sso_number).strip()
        numbered_name = f"({num_str}) {sso_name}"
        if numbered_name not in aliases:
            aliases.append(numbered_name)

    return {
        "canonical_name": sso_name,
        "entity_type": ENTITY_TYPE,
        "source": SOURCE,
        "discipline": DISCIPLINE,
        "properties": properties,
        "identifiers": identifiers,
        "aliases": aliases,
    }


# ---------------------------------------------------------------------------
# Staging writes (bulk mode) using COPY protocol
# ---------------------------------------------------------------------------


def write_staging_entities(
    conn: Any,
    records: list[dict[str, Any]],
) -> dict[str, int]:
    """Write parsed records to staging tables using COPY protocol.

    Writes to staging.entities, staging.entity_identifiers, and
    staging.entity_aliases in sequence.

    Args:
        conn: Database connection (psycopg).
        records: List of parsed record dicts from parse_sso_record.

    Returns:
        Dict with counts: entities, identifiers, aliases.
    """
    entity_count = 0
    identifier_count = 0
    alias_count = 0

    # Build staging entity rows, tracking local IDs for identifier/alias FK
    entity_rows: list[tuple[int, str, str, str, str, str]] = []
    identifier_rows: list[tuple[int, str, str, bool]] = []
    alias_rows: list[tuple[int, str, str]] = []

    # Use a sequential local ID for staging table
    seen_entities: dict[str, int] = {}  # canonical_name -> local_id

    for rec in records:
        canonical_name = rec["canonical_name"]
        if canonical_name in seen_entities:
            continue

        entity_count += 1
        local_id = entity_count
        seen_entities[canonical_name] = local_id

        entity_rows.append(
            (
                local_id,
                canonical_name,
                rec["entity_type"],
                rec["discipline"],
                rec["source"],
                json.dumps(rec["properties"]),
            )
        )

        for ident in rec.get("identifiers", []):
            identifier_rows.append(
                (
                    local_id,
                    ident["id_scheme"],
                    ident["external_id"],
                    ident.get("is_primary", False),
                )
            )
            identifier_count += 1

        # Deduplicate aliases per entity
        seen_aliases: set[str] = set()
        for alias in rec.get("aliases", []):
            if alias not in seen_aliases:
                alias_rows.append((local_id, alias, SOURCE))
                alias_count += 1
                seen_aliases.add(alias)

    # Truncate staging tables first
    with conn.cursor() as cur:
        cur.execute("TRUNCATE staging.entity_aliases")
        cur.execute("TRUNCATE staging.entity_identifiers")
        cur.execute("TRUNCATE staging.entities")
    conn.commit()

    # COPY entities
    with conn.cursor() as cur:
        with cur.copy(
            "COPY staging.entities (id, canonical_name, entity_type, discipline, source, properties) "
            "FROM STDIN"
        ) as copy:
            for row in entity_rows:
                copy.write_row(row)
    conn.commit()
    logger.info("Staged %d entities", entity_count)

    # COPY identifiers
    with conn.cursor() as cur:
        with cur.copy(
            "COPY staging.entity_identifiers (entity_id, id_scheme, external_id, is_primary) "
            "FROM STDIN"
        ) as copy:
            for row in identifier_rows:
                copy.write_row(row)
    conn.commit()
    logger.info("Staged %d identifiers", identifier_count)

    # COPY aliases
    with conn.cursor() as cur:
        with cur.copy(
            "COPY staging.entity_aliases (entity_id, alias, alias_source) " "FROM STDIN"
        ) as copy:
            for row in alias_rows:
                copy.write_row(row)
    conn.commit()
    logger.info("Staged %d aliases", alias_count)

    return {
        "entities": entity_count,
        "identifiers": identifier_count,
        "aliases": alias_count,
    }


def promote_staging(conn: Any) -> int:
    """Call staging.promote_entities() to atomically move data to public tables.

    Args:
        conn: Database connection.

    Returns:
        Number of promoted entities.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT staging.promote_entities()")
        result = cur.fetchone()[0]
    conn.commit()
    logger.info("Promoted %d entities from staging to public", result)
    return result


# ---------------------------------------------------------------------------
# Bulk harvest pipeline
# ---------------------------------------------------------------------------


def run_bulk_harvest(
    dsn: str | None = None,
    parquet_url: str = PARQUET_URL,
    dry_run: bool = False,
) -> int:
    """Run the bulk harvest pipeline: download Parquet, stage, promote.

    Args:
        dsn: Database connection string.
        parquet_url: URL for the ssoBFT Parquet file.
        dry_run: If True, download and parse but skip DB writes.

    Returns:
        Number of entities loaded.
    """
    t0 = time.monotonic()

    # Download to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "ssoBFT.parquet"
        download_parquet(parquet_url, dest)

        # Read and parse
        rows = read_parquet(dest)

    records = []
    skipped = 0
    for row in rows:
        rec = parse_sso_record(row)
        if rec is not None:
            records.append(rec)
        else:
            skipped += 1

    logger.info("Parsed %d records (%d skipped)", len(records), skipped)

    if dry_run:
        logger.info("Dry run — would load %d entities", len(records))
        return len(records)

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, SOURCE)
    try:
        run_log.start(config={"mode": "bulk", "parquet_url": parquet_url})

        staging_counts = write_staging_entities(conn, records)
        promoted = promote_staging(conn)

        run_log.complete(
            records_fetched=len(rows),
            records_upserted=promoted,
            counts={
                "entities": staging_counts["entities"],
                "identifiers": staging_counts["identifiers"],
                "aliases": staging_counts["aliases"],
                "promoted": promoted,
            },
        )

        elapsed = time.monotonic() - t0
        logger.info("Bulk harvest complete: %d entities in %.1fs", promoted, elapsed)
        return promoted

    except Exception as exc:
        try:
            run_log.fail(str(exc))
        except Exception:
            logger.warning("Failed to update harvest_run status to 'failed'")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Seed mode: SsODNet API
# ---------------------------------------------------------------------------


def fetch_ssocard(
    client: ResilientClient,
    name: str,
) -> dict[str, Any] | None:
    """Fetch an SsODNet ssocard for a named object.

    Args:
        client: ResilientClient instance.
        name: Object name (e.g. 'Ceres').

    Returns:
        Parsed JSON dict, or None on failure.
    """
    url = f"{API_BASE}/{name}"
    try:
        response = client.get(url, headers={"Accept": "application/json"})
        data = response.json()
        logger.debug("Fetched ssocard for %s", name)
        return data
    except Exception as exc:
        logger.warning("Failed to fetch ssocard for %s: %s", name, exc)
        return None


def parse_ssocard(data: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Parse an SsODNet ssocard response into a harvester record.

    Args:
        data: JSON response from the ssocard API.
        name: The queried object name.

    Returns:
        Parsed record dict or None if data is invalid.
    """
    # The ssocard response structure wraps data under the object name
    sso_data = data.get(name) or data.get(name.lower()) or data or {}

    # Try to extract canonical name from IAU preferred designation
    canonical_name = name
    parameters = sso_data.get("parameters", {})
    physical = parameters.get("physical", {})
    dynamical = parameters.get("dynamical", {})

    properties: dict[str, Any] = {}

    # Extract physical properties
    diameter_data = physical.get("diameter", {})
    if isinstance(diameter_data, dict) and "value" in diameter_data:
        properties["diameter"] = diameter_data["value"]

    albedo_data = physical.get("albedo", {})
    if isinstance(albedo_data, dict) and "value" in albedo_data:
        properties["albedo"] = albedo_data["value"]

    taxonomy_data = physical.get("taxonomy", {})
    if isinstance(taxonomy_data, dict):
        tax_class = taxonomy_data.get("class") or taxonomy_data.get("value")
        if tax_class:
            properties["taxonomy"] = tax_class

    # Identifiers
    identifiers: list[dict[str, Any]] = [
        {"id_scheme": "ssodnet", "external_id": canonical_name, "is_primary": True},
    ]

    # Try to get SPK-ID from the data
    spkid = sso_data.get("spkid") or parameters.get("spkid")
    if spkid:
        identifiers.append(
            {"id_scheme": "sbdb_spkid", "external_id": str(spkid), "is_primary": False}
        )

    # Aliases from other names
    aliases: list[str] = []
    other_names = sso_data.get("other_names", [])
    if isinstance(other_names, list):
        for alias_name in other_names:
            if isinstance(alias_name, str) and alias_name.strip() and alias_name != canonical_name:
                aliases.append(alias_name.strip())

    return {
        "canonical_name": canonical_name,
        "entity_type": ENTITY_TYPE,
        "source": SOURCE,
        "discipline": DISCIPLINE,
        "properties": properties,
        "identifiers": identifiers,
        "aliases": aliases,
    }


def run_seed_harvest(
    dsn: str | None = None,
    objects: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Run the seed harvest pipeline: fetch well-known objects from API.

    Args:
        dsn: Database connection string.
        objects: List of object names. Defaults to WELL_KNOWN_OBJECTS.
        dry_run: If True, fetch and parse but skip DB writes.

    Returns:
        Number of entities loaded.
    """
    t0 = time.monotonic()
    target_objects = objects if objects is not None else WELL_KNOWN_OBJECTS

    client = _get_client()
    records: list[dict[str, Any]] = []

    for name in target_objects:
        data = fetch_ssocard(client, name)
        if data is None:
            continue
        rec = parse_ssocard(data, name)
        if rec is not None:
            records.append(rec)

    logger.info("Fetched %d objects from SsODNet API", len(records))

    if dry_run:
        logger.info("Dry run — would load %d entities", len(records))
        return len(records)

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, SOURCE)
    try:
        run_log.start(config={"mode": "seed", "objects": target_objects})

        count = 0
        for rec in records:
            entity_id = upsert_entity(
                conn,
                canonical_name=rec["canonical_name"],
                entity_type=rec["entity_type"],
                source=rec["source"],
                discipline=rec["discipline"],
                harvest_run_id=run_log.run_id,
                properties=rec["properties"],
            )

            for ident in rec.get("identifiers", []):
                upsert_entity_identifier(
                    conn,
                    entity_id=entity_id,
                    id_scheme=ident["id_scheme"],
                    external_id=ident["external_id"],
                    is_primary=ident.get("is_primary", False),
                )

            for alias in rec.get("aliases", []):
                upsert_entity_alias(
                    conn,
                    entity_id=entity_id,
                    alias=alias,
                    alias_source=SOURCE,
                )

            count += 1

        conn.commit()

        run_log.complete(
            records_fetched=len(target_objects),
            records_upserted=count,
            counts={"entities": count},
        )

        elapsed = time.monotonic() - t0
        logger.info("Seed harvest complete: %d entities in %.1fs", count, elapsed)
        return count

    except Exception as exc:
        try:
            run_log.fail(str(exc))
        except Exception:
            logger.warning("Failed to update harvest_run status to 'failed'")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def run_harvest(
    dsn: str | None = None,
    mode: str = "seed",
    dry_run: bool = False,
) -> int:
    """Run the SsODNet harvest pipeline.

    Args:
        dsn: Database connection string.
        mode: 'bulk' for ssoBFT Parquet, 'seed' for API-based.
        dry_run: If True, fetch/parse without DB writes.

    Returns:
        Number of entities loaded.
    """
    if mode == "bulk":
        return run_bulk_harvest(dsn=dsn, dry_run=dry_run)
    elif mode == "seed":
        return run_seed_harvest(dsn=dsn, dry_run=dry_run)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'bulk' or 'seed'.")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the SsODNet harvest pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest SsODNet solar system objects into entity graph",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "--mode",
        choices=["bulk", "seed"],
        default="seed",
        help="Harvest mode: 'bulk' downloads ssoBFT Parquet, 'seed' queries API (default: seed)",
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

    count = run_harvest(dsn=args.dsn, mode=args.mode, dry_run=args.dry_run)

    if args.dry_run:
        print(f"Dry run — {count} SsODNet entities would be loaded ({args.mode} mode)")
    else:
        print(f"SsODNet harvest complete: {count} entities ({args.mode} mode)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
