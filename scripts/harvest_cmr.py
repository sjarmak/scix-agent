#!/usr/bin/env python3
"""Harvest NASA CMR (Common Metadata Repository) collections into the datasets table.

Downloads collection metadata from the CMR Search API using UMM-JSON format,
stores them in the datasets table (source='cmr', canonical_id=concept-id),
and cross-references GCMD entities via entity_identifiers(id_scheme='gcmd_uuid').

Extracts from each collection:
  - Instruments (nested under Platforms in UMM-JSON)
  - Platforms
  - Science Keywords (Category/Topic/Term hierarchy)

Uses Search-After header pagination (not offset-based) per CMR API best practices.

Usage:
    python scripts/harvest_cmr.py --help
    python scripts/harvest_cmr.py --dry-run
    python scripts/harvest_cmr.py --verbose
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

CMR_BASE_URL = "https://cmr.earthdata.nasa.gov/search/collections"
SOURCE = "cmr"
DISCIPLINE = "earth_science"
UMM_JSON_ACCEPT = "application/vnd.nasa.cmr.umm_results+json"
DEFAULT_PAGE_SIZE = 2000


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _make_client() -> ResilientClient:
    """Create a ResilientClient configured for CMR."""
    return ResilientClient(
        max_retries=3,
        backoff_base=2.0,
        rate_limit=5.0,
        cache_dir=Path(".cache/cmr"),
        cache_ttl=86400.0,
        user_agent="scix-harvester/1.0",
        timeout=120.0,
    )


# ---------------------------------------------------------------------------
# Fetch — Search-After pagination
# ---------------------------------------------------------------------------


def fetch_collections(
    client: ResilientClient,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch all CMR collections using Search-After header pagination.

    Args:
        client: ResilientClient instance.
        page_size: Number of collections per page.

    Returns:
        Tuple of (list of collection items, number of pages fetched).
    """
    all_items: list[dict[str, Any]] = []
    search_after: str | None = None
    page_count = 0
    total_hits = 0

    while True:
        headers: dict[str, str] = {"Accept": UMM_JSON_ACCEPT}
        if search_after is not None:
            # CMR's request header for paging is CMR-Search-After (mirrors
            # the response header). Plain "Search-After" is silently ignored
            # and yields the first page on every request.
            headers["CMR-Search-After"] = search_after

        resp = client.get(
            CMR_BASE_URL,
            params={"page_size": str(page_size)},
            headers=headers,
        )
        data = resp.json()
        page_count += 1

        if page_count == 1:
            total_hits = data.get("hits", 0)
            logger.info("CMR reports %d total collections", total_hits)

        items = data.get("items", [])
        all_items.extend(items)

        logger.info(
            "CMR page %d: fetched %d items (%d/%d total)",
            page_count,
            len(items),
            len(all_items),
            total_hits,
        )

        if not items or len(all_items) >= total_hits:
            break

        # Read Search-After from response headers for next page
        resp_headers = getattr(resp, "headers", {})
        cmr_search_after = resp_headers.get("CMR-Search-After")
        if cmr_search_after is None:
            logger.warning("No CMR-Search-After header in response, stopping pagination")
            break
        search_after = cmr_search_after

    logger.info("Fetched %d collections in %d pages", len(all_items), page_count)
    return all_items, page_count


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


def _extract_platforms(umm: dict[str, Any]) -> list[str]:
    """Extract unique platform ShortNames from UMM-JSON.

    Args:
        umm: The 'umm' portion of a CMR collection item.

    Returns:
        List of unique platform short names.
    """
    platforms: list[str] = []
    seen: set[str] = set()
    for platform in umm.get("Platforms", []):
        name = platform.get("ShortName", "").strip()
        if name and name not in seen:
            platforms.append(name)
            seen.add(name)
    return platforms


def _extract_instruments(umm: dict[str, Any]) -> list[str]:
    """Extract unique instrument ShortNames from UMM-JSON.

    Instruments are nested under Platforms in CMR UMM-JSON.

    Args:
        umm: The 'umm' portion of a CMR collection item.

    Returns:
        List of unique instrument short names.
    """
    instruments: list[str] = []
    seen: set[str] = set()
    for platform in umm.get("Platforms", []):
        for instrument in platform.get("Instruments", []):
            name = instrument.get("ShortName", "").strip()
            if name and name not in seen:
                instruments.append(name)
                seen.add(name)
    return instruments


def _extract_science_keywords(umm: dict[str, Any]) -> list[dict[str, str]]:
    """Extract science keywords from UMM-JSON.

    Args:
        umm: The 'umm' portion of a CMR collection item.

    Returns:
        List of keyword dicts with Category, Topic, Term fields.
    """
    keywords: list[dict[str, str]] = []
    for kw in umm.get("ScienceKeywords", []):
        entry: dict[str, str] = {}
        for field in ("Category", "Topic", "Term", "VariableLevel1", "VariableLevel2"):
            val = kw.get(field, "").strip()
            if val:
                entry[field] = val
        if entry:
            keywords.append(entry)
    return keywords


def parse_collection(item: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a single CMR collection item into a structured record.

    Args:
        item: A single item from the CMR search response.

    Returns:
        Parsed collection dict, or None if concept-id is missing.
    """
    meta = item.get("meta", {})
    umm = item.get("umm", {})

    concept_id = meta.get("concept-id", "").strip()
    if not concept_id:
        return None

    entry_title = umm.get("EntryTitle", "").strip()
    short_name = umm.get("ShortName", "").strip()
    name = entry_title or short_name or concept_id

    record: dict[str, Any] = {
        "concept_id": concept_id,
        "name": name,
        "short_name": short_name,
        "platforms": _extract_platforms(umm),
        "instruments": _extract_instruments(umm),
        "science_keywords": _extract_science_keywords(umm),
    }

    abstract = umm.get("Abstract", "").strip()
    if abstract:
        record["abstract"] = abstract

    # Temporal extent
    temporal = umm.get("TemporalExtents", [])
    if temporal:
        range_dts = temporal[0].get("RangeDateTimes", [])
        if range_dts:
            begin = range_dts[0].get("BeginningDateTime", "")
            if begin:
                record["temporal_start"] = begin[:10]
            end = range_dts[0].get("EndingDateTime", "")
            if end:
                record["temporal_end"] = end[:10]

    return record


def parse_collections(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse all CMR collection items.

    Args:
        items: Raw items from the CMR search response.

    Returns:
        List of parsed collection dicts.
    """
    collections: list[dict[str, Any]] = []
    skipped = 0
    for item in items:
        parsed = parse_collection(item)
        if parsed is not None:
            collections.append(parsed)
        else:
            skipped += 1

    if skipped:
        logger.warning("Skipped %d items missing concept-id", skipped)
    logger.info("Parsed %d collections", len(collections))
    return collections


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------


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

    Deduplicates on (source, canonical_id) via ON CONFLICT.

    Args:
        conn: Database connection.
        name: Dataset display name.
        discipline: Discipline tag.
        source: Source identifier.
        canonical_id: Canonical dataset identifier (concept-id for CMR).
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


def _lookup_gcmd_entities_by_name(
    conn: Any,
    names: list[str],
    entity_type: str,
) -> dict[str, int]:
    """Look up GCMD entities by canonical_name, returning name->entity_id map.

    Only returns entities that have a gcmd_uuid identifier, confirming
    they are genuine GCMD entities.

    Args:
        conn: Database connection.
        names: List of entity names to look up.
        entity_type: The entity_type to filter on (e.g. 'instrument').

    Returns:
        Dict mapping canonical_name -> entity_id for found entities.
    """
    if not names:
        return {}

    result: dict[str, int] = {}
    with conn.cursor() as cur:
        # Use ANY() for batch lookup
        cur.execute(
            """
            SELECT e.canonical_name, e.id
            FROM entities e
            JOIN entity_identifiers ei ON ei.entity_id = e.id
            WHERE ei.id_scheme = 'gcmd_uuid'
              AND e.source = 'gcmd'
              AND e.entity_type = %s
              AND e.canonical_name = ANY(%s)
            """,
            (entity_type, names),
        )
        for row in cur.fetchall():
            result[row[0]] = row[1]

    return result


def store_collections(
    conn: Any,
    collections: list[dict[str, Any]],
    run_id: int,
) -> dict[str, int]:
    """Store parsed CMR collections into the database.

    For each collection:
    1. Upsert into datasets table (source='cmr', canonical_id=concept-id)
    2. Store science_keywords in properties JSONB
    3. Look up GCMD instrument/platform entities and link via dataset_entities

    Args:
        conn: Database connection.
        collections: Parsed collection records.
        run_id: Harvest run ID.

    Returns:
        Dict of counts by category.
    """
    # Pre-collect all instrument and platform names for batch GCMD lookup
    all_instrument_names: set[str] = set()
    all_platform_names: set[str] = set()
    for coll in collections:
        all_instrument_names.update(coll.get("instruments", []))
        all_platform_names.update(coll.get("platforms", []))

    # Batch lookup GCMD entities
    gcmd_instruments = _lookup_gcmd_entities_by_name(conn, list(all_instrument_names), "instrument")
    gcmd_platforms = _lookup_gcmd_entities_by_name(
        conn,
        list(all_platform_names),
        "instrument",  # platforms are entity_type='instrument' in GCMD
    )

    logger.info(
        "GCMD cross-reference: %d/%d instruments, %d/%d platforms matched",
        len(gcmd_instruments),
        len(all_instrument_names),
        len(gcmd_platforms),
        len(all_platform_names),
    )

    dataset_count = 0
    instrument_links = 0
    platform_links = 0

    for coll in collections:
        properties: dict[str, Any] = {
            "short_name": coll.get("short_name", ""),
            "platforms": coll.get("platforms", []),
            "instruments": coll.get("instruments", []),
            "science_keywords": coll.get("science_keywords", []),
        }

        ds_id = _upsert_dataset(
            conn,
            name=coll["name"],
            discipline=DISCIPLINE,
            source=SOURCE,
            canonical_id=coll["concept_id"],
            description=coll.get("abstract"),
            temporal_start=coll.get("temporal_start"),
            temporal_end=coll.get("temporal_end"),
            properties=properties,
            harvest_run_id=run_id,
        )
        dataset_count += 1

        # Link to GCMD instrument entities
        for inst_name in coll.get("instruments", []):
            entity_id = gcmd_instruments.get(inst_name)
            if entity_id is not None:
                _upsert_dataset_entity(
                    conn,
                    dataset_id=ds_id,
                    entity_id=entity_id,
                    relationship="has_instrument",
                )
                instrument_links += 1

        # Link to GCMD platform entities
        for plat_name in coll.get("platforms", []):
            entity_id = gcmd_platforms.get(plat_name)
            if entity_id is not None:
                _upsert_dataset_entity(
                    conn,
                    dataset_id=ds_id,
                    entity_id=entity_id,
                    relationship="on_platform",
                )
                platform_links += 1

    conn.commit()

    counts = {
        "datasets": dataset_count,
        "instrument_links": instrument_links,
        "platform_links": platform_links,
        "gcmd_instruments_matched": len(gcmd_instruments),
        "gcmd_platforms_matched": len(gcmd_platforms),
    }
    logger.info(
        "Stored %d datasets with %d instrument + %d platform links",
        dataset_count,
        instrument_links,
        platform_links,
    )
    return counts


# ---------------------------------------------------------------------------
# Harvest orchestration
# ---------------------------------------------------------------------------


def run_harvest(
    dsn: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Run the full CMR harvest pipeline.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        dry_run: If True, fetch and parse without writing to DB.

    Returns:
        Dict of counts by category.
    """
    t0 = time.monotonic()

    client = _make_client()
    items, page_count = fetch_collections(client)
    collections = parse_collections(items)

    if dry_run:
        counts = {
            "collections": len(collections),
            "pages": page_count,
        }
        elapsed = time.monotonic() - t0
        logger.info("Dry run complete in %.1fs: %s", elapsed, counts)
        return counts

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, SOURCE)
    try:
        run_log.start(
            config={
                "source_url": CMR_BASE_URL,
                "page_size": DEFAULT_PAGE_SIZE,
            }
        )

        counts = store_collections(conn, collections, run_log.run_id)
        counts["pages"] = page_count

        run_log.complete(
            records_fetched=len(items),
            records_upserted=counts["datasets"],
            counts=counts,
        )

        return counts

    except Exception:
        run_log.fail(str(sys.exc_info()[1]))
        raise
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the CMR harvest pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest NASA CMR collections into datasets table",
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
        print(f"Dry run — {counts.get('collections', 0)} collections parsed")
    else:
        print(f"CMR harvest complete: {counts}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
