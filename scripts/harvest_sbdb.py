#!/usr/bin/env python3
"""Enrich SsODNet entities with JPL SBDB (Small-Body Database) data.

Queries the JPL SBDB API for each entity already in the entities table
(source='ssodnet') and merges orbital/discovery metadata into the entity's
properties JSONB column:

  - orbital_class (e.g. "Main-belt Asteroid", "Near-Earth Asteroid")
  - neo (bool — Near-Earth Object flag)
  - pha (bool — Potentially Hazardous Asteroid flag)
  - discovery_date, discovery_site, discoverer

Supports cursor-based resumption via harvest_runs.cursor_state so that
interrupted runs can continue where they left off.

Usage:
    python scripts/harvest_sbdb.py --help
    python scripts/harvest_sbdb.py --dry-run
    python scripts/harvest_sbdb.py -v --limit 100
    python scripts/harvest_sbdb.py --no-resume
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
from scix.harvest_utils import HarvestRunLog
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SBDB_API_BASE = "https://ssd-api.jpl.nasa.gov/sbdb.api"
SOURCE = "sbdb"
ENRICHES_SOURCE = "ssodnet"

# ---------------------------------------------------------------------------
# Module-level client (lazy init)
# ---------------------------------------------------------------------------

_client: ResilientClient | None = None


def _get_client() -> ResilientClient:
    """Return a shared ResilientClient with strict 1 req/s rate limit."""
    global _client
    if _client is None:
        _client = ResilientClient(
            user_agent="scix-experiments/1.0",
            max_retries=3,
            backoff_base=2.0,
            rate_limit=1.0,
            cache_dir=Path(".cache/sbdb"),
            cache_ttl=86400.0,
            timeout=30.0,
        )
    return _client


# ---------------------------------------------------------------------------
# SBDB API interaction
# ---------------------------------------------------------------------------


def parse_sbdb_response(data: dict[str, Any]) -> dict[str, Any]:
    """Extract enrichment fields from a SBDB API response.

    Args:
        data: Parsed JSON response from the SBDB API.

    Returns:
        Dict of enrichment fields to merge into entity properties.
    """
    enrichment: dict[str, Any] = {}

    # Orbital class from object.orbit_class
    obj = data.get("object", {})
    orbit_class = obj.get("orbit_class", {})
    if isinstance(orbit_class, dict) and orbit_class.get("name"):
        enrichment["orbital_class"] = orbit_class["name"]

    # NEO and PHA flags
    neo_val = obj.get("neo")
    if neo_val is not None:
        enrichment["neo"] = bool(neo_val)

    pha_val = obj.get("pha")
    if pha_val is not None:
        enrichment["pha"] = bool(pha_val)

    # Discovery information
    discovery = data.get("discovery", {})
    if isinstance(discovery, dict):
        if discovery.get("date"):
            enrichment["discovery_date"] = discovery["date"]
        if discovery.get("site"):
            enrichment["discovery_site"] = discovery["site"]
        if discovery.get("name"):
            enrichment["discoverer"] = discovery["name"]

    return enrichment


def fetch_sbdb_record(
    client: ResilientClient,
    designation: str,
    spk_id: str | None = None,
) -> dict[str, Any] | None:
    """Fetch SBDB data for a single small body by SPK-ID or designation.

    Args:
        client: ResilientClient instance.
        designation: Object name/designation (e.g. 'Ceres', '433').
        spk_id: Optional SPK-ID to use instead of designation.

    Returns:
        Parsed enrichment dict, or None on failure.
    """
    params: dict[str, str] = {
        "phys-par": "true",
        "discovery": "true",
        "ca-data": "false",
    }
    # Use sstr (search string) instead of des — SBDB rejects name-only
    # designations like 'Bennu' via des= but accepts them via sstr=
    if spk_id:
        params["spk"] = spk_id
    else:
        params["sstr"] = designation
    try:
        response = client.get(SBDB_API_BASE, params=params)
        data = response.json()

        # SBDB returns a code field on error (e.g. code=200 is not-found)
        if "code" in data and str(data["code"]) != "200":
            logger.debug(
                "SBDB returned code %s for %s: %s",
                data.get("code"),
                designation,
                data.get("message", ""),
            )
            return None

        enrichment = parse_sbdb_response(data)
        if enrichment:
            logger.debug("SBDB enrichment for %s: %s", designation, enrichment)
            return enrichment

        logger.debug("No enrichment fields found for %s", designation)
        return None

    except Exception as exc:
        logger.warning("Failed to fetch SBDB data for %s: %s", designation, exc)
        return None


# ---------------------------------------------------------------------------
# Cursor-based resumption
# ---------------------------------------------------------------------------


def get_last_cursor(conn: Any) -> int | None:
    """Get the last processed entity_id from the most recent completed SBDB run.

    Args:
        conn: Database connection.

    Returns:
        Last entity_id, or None if no prior run exists.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT cursor_state
            FROM harvest_runs
            WHERE source = %s AND status = 'completed'
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            (SOURCE,),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None

        cursor_state = row[0]
        if isinstance(cursor_state, str):
            cursor_state = json.loads(cursor_state)

        return cursor_state.get("last_entity_id")


def save_cursor(conn: Any, run_id: int, entity_id: int) -> None:
    """Persist the current cursor position to the harvest_runs row.

    Args:
        conn: Database connection.
        run_id: The harvest run id.
        entity_id: The last successfully processed entity id.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE harvest_runs
            SET cursor_state = %s
            WHERE id = %s
            """,
            (json.dumps({"last_entity_id": entity_id}), run_id),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Entity queries
# ---------------------------------------------------------------------------


def fetch_ssodnet_entities(
    conn: Any,
    after_id: int | None = None,
) -> list[tuple[int, str, str | None]]:
    """Fetch ssodnet entities to enrich, ordered by id.

    Also fetches SPK-ID from entity_identifiers so SBDB can be queried
    by SPK-ID (SBDB rejects name-only designations like 'Bennu').

    Args:
        conn: Database connection.
        after_id: If set, only return entities with id > after_id (for resumption).

    Returns:
        List of (entity_id, canonical_name, spk_id_or_none) tuples.
    """
    if after_id is not None:
        query = """
            SELECT e.id, e.canonical_name, ei.external_id AS spk_id
            FROM entities e
            LEFT JOIN entity_identifiers ei
                ON ei.entity_id = e.id AND ei.id_scheme = 'sbdb_spkid'
            WHERE e.source = %s AND e.id > %s
            ORDER BY e.id
        """
        params: tuple[Any, ...] = (ENRICHES_SOURCE, after_id)
    else:
        query = """
            SELECT e.id, e.canonical_name, ei.external_id AS spk_id
            FROM entities e
            LEFT JOIN entity_identifiers ei
                ON ei.entity_id = e.id AND ei.id_scheme = 'sbdb_spkid'
            WHERE e.source = %s
            ORDER BY e.id
        """
        params = (ENRICHES_SOURCE,)

    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def update_entity_properties(
    conn: Any,
    entity_id: int,
    enrichment: dict[str, Any],
) -> None:
    """Merge enrichment properties into an entity's existing properties JSONB.

    Uses PostgreSQL jsonb concatenation (||) to merge without overwriting
    fields not present in the enrichment dict.

    Args:
        conn: Database connection.
        entity_id: The entity to update.
        enrichment: Dict of new properties to merge.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE entities
            SET properties = properties || %s::jsonb,
                updated_at = NOW()
            WHERE id = %s
            """,
            (json.dumps(enrichment), entity_id),
        )


# ---------------------------------------------------------------------------
# Main harvest pipeline
# ---------------------------------------------------------------------------


def run_harvest(
    dsn: str | None = None,
    dry_run: bool = False,
    resume: bool = True,
    limit: int | None = None,
) -> int:
    """Run the SBDB enrichment pipeline.

    Fetches ssodnet entities, queries SBDB API for each, and merges
    orbital/discovery data into the entity properties.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        dry_run: If True, fetch from API but skip DB writes.
        resume: If True, resume from last cursor position.
        limit: Maximum number of entities to process (for testing).

    Returns:
        Number of entities enriched.
    """
    t0 = time.monotonic()
    conn = get_connection(dsn)
    client = _get_client()

    # Determine cursor position for resumption
    after_id: int | None = None
    if resume:
        after_id = get_last_cursor(conn)
        if after_id is not None:
            logger.info("Resuming from entity_id > %d", after_id)

    # Fetch target entities
    entities = fetch_ssodnet_entities(conn, after_id=after_id)
    if limit is not None:
        entities = entities[:limit]

    logger.info(
        "Found %d ssodnet entities to enrich%s",
        len(entities),
        f" (after id {after_id})" if after_id else "",
    )

    if not entities:
        logger.info("No entities to enrich — exiting")
        conn.close()
        return 0

    if dry_run:
        # In dry-run mode, still query the API but skip DB writes
        enriched = 0
        for entity_id, canonical_name, spk_id in entities:
            result = fetch_sbdb_record(client, canonical_name, spk_id=spk_id)
            if result is not None:
                enriched += 1
                logger.info(
                    "Would enrich %s (id=%d): %s",
                    canonical_name,
                    entity_id,
                    result,
                )
        logger.info("Dry run — would enrich %d / %d entities", enriched, len(entities))
        conn.close()
        return enriched

    run_log = HarvestRunLog(conn, SOURCE)
    try:
        run_log.start(
            config={
                "enriches_source": ENRICHES_SOURCE,
                "resume_after_id": after_id,
                "limit": limit,
            }
        )

        enriched = 0
        fetched = 0
        errors = 0

        for entity_id, canonical_name, spk_id in entities:
            fetched += 1
            result = fetch_sbdb_record(client, canonical_name, spk_id=spk_id)

            if result is not None:
                update_entity_properties(conn, entity_id, result)
                enriched += 1

            # Save cursor after each entity (whether enriched or not)
            save_cursor(conn, run_log.run_id, entity_id)

            if fetched % 100 == 0:
                logger.info(
                    "Progress: %d/%d fetched, %d enriched",
                    fetched,
                    len(entities),
                    enriched,
                )

        run_log.complete(
            records_fetched=fetched,
            records_upserted=enriched,
            counts={
                "entities_checked": fetched,
                "entities_enriched": enriched,
            },
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "SBDB enrichment complete: %d/%d entities enriched in %.1fs",
            enriched,
            fetched,
            elapsed,
        )
        return enriched

    except Exception as exc:
        try:
            run_log.fail(str(exc))
        except Exception:
            logger.warning("Failed to update harvest_run status to 'failed'")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the SBDB enrichment pipeline."""
    parser = argparse.ArgumentParser(
        description="Enrich SsODNet entities with JPL SBDB orbital/discovery data",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query API but skip database writes",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from the beginning instead of resuming from last cursor",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of entities to process",
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

    count = run_harvest(
        dsn=args.dsn,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        limit=args.limit,
    )

    if args.dry_run:
        print(f"Dry run — {count} entities would be enriched with SBDB data")
    else:
        print(f"SBDB enrichment complete: {count} entities enriched")

    return 0


if __name__ == "__main__":
    sys.exit(main())
