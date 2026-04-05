#!/usr/bin/env python3
"""Harvest PDS4 context products into entity_dictionary and entity graph.

Downloads Investigation, Instrument, and Target context products from the
NASA PDS Registry API and bulk-loads them via scix.dictionary.bulk_load()
with discipline='planetary_science' and source='pds4'.  Also writes to the
entity graph tables (entities, entity_identifiers, entity_aliases,
entity_relationships) and logs a harvest_run.

Usage:
    python scripts/harvest_pds4.py --help
    python scripts/harvest_pds4.py --dry-run
    python scripts/harvest_pds4.py --product-type investigation instrument
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

PDS_API_BASE = "https://pds.nasa.gov/api/search/1/products"

# Maps PDS context type -> (URN segment, entity_type for dictionary)
PRODUCT_TYPE_MAP: dict[str, tuple[str, str]] = {
    "investigation": ("investigation", "mission"),
    "instrument": ("instrument", "instrument"),
    "target": ("target", "target"),
}

ALL_PRODUCT_TYPES: tuple[str, ...] = tuple(PRODUCT_TYPE_MAP.keys())

# Fields to request from the API
REQUEST_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "pds:Identification_Area.pds:logical_identifier",
    "pds:Identification_Area.pds:title",
    "pds:Identification_Area.pds:alternate_title",
    "pds:Alias_List.pds:Alias.pds:alternate_title",
    "pds:Investigation.pds:type",
    "pds:Investigation.pds:description",
    "pds:Instrument.pds:type",
    "pds:Instrument.pds:description",
    "pds:Target.pds:type",
    "pds:Target.pds:description",
    "ref_lid_instrument",
    "ref_lid_investigation",
    "ref_lid_target",
)

PAGE_SIZE = 2000

# Regex to extract parenthesised abbreviations from titles
_ABBREV_RE = re.compile(r"\(([A-Z][A-Z0-9/.\-]{0,20})\)")

# Regex to extract investigation name from instrument URN
# e.g. urn:nasa:pds:context:instrument:spacecraft.cassini-huygens.cirs
#   -> "cassini-huygens" from the parent segment
_INSTRUMENT_PARENT_RE = re.compile(r"^urn:nasa:pds:context:instrument:[^.]+\.([^.]+)\.")


def _get_prop(properties: dict[str, Any], key: str) -> str | None:
    """Extract a single string value from a PDS properties dict.

    PDS API returns property values as arrays. Values of ``["null"]`` or
    empty arrays are treated as absent.

    Args:
        properties: The product's properties dict.
        key: The dot-notation property key.

    Returns:
        The first non-null string value, or None.
    """
    vals = properties.get(key, [])
    if not vals:
        return None
    val = vals[0]
    if val is None or val == "null":
        return None
    return str(val).strip() or None


def _get_prop_list(properties: dict[str, Any], key: str) -> list[str]:
    """Extract a list of string values from a PDS properties dict.

    Filters out ``"null"`` and empty strings.

    Args:
        properties: The product's properties dict.
        key: The dot-notation property key.

    Returns:
        List of non-null string values.
    """
    vals = properties.get(key, [])
    if not vals:
        return []
    return [str(v).strip() for v in vals if v is not None and v != "null" and str(v).strip()]


def extract_aliases(title: str, properties: dict[str, Any]) -> list[str]:
    """Extract alternate names and abbreviations from a PDS product.

    Sources:
    1. Parenthesised abbreviations in the title (e.g. "Cassini-Huygens (CASSINI)")
    2. ``pds:Identification_Area.pds:alternate_title``
    3. ``pds:Alias_List.pds:Alias.pds:alternate_title``

    Args:
        title: The product's canonical title.
        properties: The product's properties dict.

    Returns:
        Deduplicated list of aliases (excluding the title itself).
    """
    seen: set[str] = set()
    aliases: list[str] = []

    def _add(name: str) -> None:
        stripped = name.strip()
        if stripped and stripped != title and stripped.lower() not in seen:
            seen.add(stripped.lower())
            aliases.append(stripped)

    # Abbreviations from title parentheses
    for match in _ABBREV_RE.finditer(title):
        _add(match.group(1))

    # Alternate title from Identification_Area
    alt_title = _get_prop(properties, "pds:Identification_Area.pds:alternate_title")
    if alt_title:
        _add(alt_title)

    # Alias list entries
    for alt in _get_prop_list(properties, "pds:Alias_List.pds:Alias.pds:alternate_title"):
        _add(alt)

    return aliases


def fetch_pds4_page(
    context_type: str,
    *,
    client: ResilientClient,
    limit: int = PAGE_SIZE,
    search_after: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch a single page of PDS4 context products from the Registry API.

    Uses cursor-based pagination via the ``search-after`` query parameter.

    Args:
        context_type: One of ``PRODUCT_TYPE_MAP`` keys (investigation,
            instrument, target).
        client: ResilientClient instance for HTTP requests.
        limit: Page size (default 2000, large enough for most collections).
        search_after: Sort values from the last product of the previous page,
            used for cursor-based pagination. None for the first page.

    Returns:
        Parsed JSON response dict with ``summary`` and ``data`` keys.

    Raises:
        requests.RequestException: If the request fails after retries.
    """
    urn_segment = PRODUCT_TYPE_MAP[context_type][0]
    q = (
        f"pds:Identification_Area.pds:logical_identifier like "
        f'"urn:nasa:pds:context:{urn_segment}:*"'
    )
    params: dict[str, str] = {
        "q": q,
        "limit": str(limit),
        "fields": ",".join(REQUEST_FIELDS),
    }
    if search_after is not None:
        params["search-after"] = ",".join(search_after)

    response = client.get(PDS_API_BASE, params=params)
    return response.json()


def download_pds4_context(
    product_types: tuple[str, ...] = ALL_PRODUCT_TYPES,
    *,
    client: ResilientClient | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Download all PDS4 context products for the given types.

    Handles pagination automatically.

    Args:
        product_types: Tuple of context types to download.
        client: Optional ResilientClient instance. Creates a default if None.

    Returns:
        Dict mapping context type to list of raw product dicts from the API.
    """
    if client is None:
        client = ResilientClient(
            user_agent="scix-experiments/1.0",
            timeout=120.0,
        )

    results: dict[str, list[dict[str, Any]]] = {}

    for ptype in product_types:
        if ptype not in PRODUCT_TYPE_MAP:
            raise ValueError(
                f"Unknown product type {ptype!r}. "
                f"Must be one of: {', '.join(ALL_PRODUCT_TYPES)}"
            )

        products: list[dict[str, Any]] = []
        search_after: list[str] | None = None

        # Fetch pages using cursor-based pagination
        while True:
            page = fetch_pds4_page(ptype, client=client, limit=PAGE_SIZE, search_after=search_after)
            total_hits = page.get("summary", {}).get("hits", 0)
            batch = page.get("data", [])

            if not batch:
                break

            products.extend(batch)
            logger.info(
                "PDS4 %s: fetched %d / %d",
                ptype,
                len(products),
                total_hits,
            )

            # If we got all results, we are done
            if len(products) >= total_hits:
                break

            # Extract sort values from the last product for cursor pagination
            last_product = batch[-1]
            sort_values = last_product.get("sort", None)
            if sort_values is None:
                # Fallback: use the product id as sort cursor
                last_id = last_product.get("id", "")
                if last_id:
                    search_after = [last_id]
                else:
                    logger.warning(
                        "PDS4 %s: no sort values in response, stopping pagination",
                        ptype,
                    )
                    break
            else:
                search_after = [str(v) for v in sort_values]

        results[ptype] = products
        logger.info("PDS4 %s: downloaded %d products total", ptype, len(products))

    return results


def parse_pds4_products(
    raw_products: list[dict[str, Any]],
    context_type: str,
) -> list[dict[str, Any]]:
    """Parse raw PDS4 API products into entity_dictionary records.

    Args:
        raw_products: List of product dicts from the PDS API.
        context_type: The context type key (investigation, instrument, target).

    Returns:
        List of entity dictionary entry dicts compatible with bulk_load().
    """
    _, entity_type = PRODUCT_TYPE_MAP[context_type]
    entries: list[dict[str, Any]] = []
    skipped = 0

    # Description field varies by product type
    desc_keys = {
        "investigation": "pds:Investigation.pds:description",
        "instrument": "pds:Instrument.pds:description",
        "target": "pds:Target.pds:description",
    }
    type_keys = {
        "investigation": "pds:Investigation.pds:type",
        "instrument": "pds:Instrument.pds:type",
        "target": "pds:Target.pds:type",
    }

    for product in raw_products:
        properties = product.get("properties", {})
        title = product.get("title", "").strip()
        lid = _get_prop(properties, "pds:Identification_Area.pds:logical_identifier")

        if not title or not lid:
            skipped += 1
            continue

        # Use the logical_identifier (without version) as external_id
        external_id = lid

        aliases = extract_aliases(title, properties)

        # Build metadata
        metadata: dict[str, Any] = {}
        pds_type = _get_prop(properties, type_keys.get(context_type, ""))
        if pds_type:
            metadata["pds_type"] = pds_type

        description = _get_prop(properties, desc_keys.get(context_type, ""))
        if description and description != "none":
            metadata["description"] = description

        # Store the versioned id for provenance
        metadata["pds_versioned_id"] = product.get("id", "")

        # Store reference LIDs for relationship extraction
        ref_investigations = _get_prop_list(properties, "ref_lid_investigation")
        if ref_investigations:
            metadata["ref_lid_investigation"] = ref_investigations

        ref_targets = _get_prop_list(properties, "ref_lid_target")
        if ref_targets:
            metadata["ref_lid_target"] = ref_targets

        entries.append(
            {
                "canonical_name": title,
                "entity_type": entity_type,
                "source": "pds4",
                "external_id": external_id,
                "aliases": aliases,
                "metadata": metadata,
            }
        )

    if skipped:
        logger.warning(
            "Skipped %d %s products missing title or logical_identifier",
            skipped,
            context_type,
        )

    logger.info(
        "Parsed %d PDS4 %s products into %s dictionary records",
        len(entries),
        context_type,
        entity_type,
    )
    return entries


def _start_harvest_run(
    conn: "psycopg.Connection",
    product_types: tuple[str, ...],
) -> int:
    """Insert a harvest_runs row with status='running'. Returns the run id."""
    import psycopg

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO harvest_runs (source, status, config)
            VALUES ('pds4', 'running', %(config)s)
            RETURNING id
            """,
            {"config": json.dumps({"product_types": list(product_types)})},
        )
        run_id: int = cur.fetchone()[0]
    conn.commit()
    return run_id


def _finish_harvest_run(
    conn: "psycopg.Connection",
    run_id: int,
    *,
    records_fetched: int,
    records_upserted: int,
    counts: dict[str, int],
    status: str = "completed",
    error_message: str | None = None,
) -> None:
    """Update a harvest_runs row with final status."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE harvest_runs
            SET finished_at = now(),
                status = %(status)s,
                records_fetched = %(records_fetched)s,
                records_upserted = %(records_upserted)s,
                counts = %(counts)s,
                error_message = %(error_message)s
            WHERE id = %(run_id)s
            """,
            {
                "run_id": run_id,
                "status": status,
                "records_fetched": records_fetched,
                "records_upserted": records_upserted,
                "counts": json.dumps(counts),
                "error_message": error_message,
            },
        )
    conn.commit()


def _write_entity_graph(
    conn: "psycopg.Connection",
    entries: list[dict[str, Any]],
    harvest_run_id: int,
) -> int:
    """Write parsed entries to entities, entity_identifiers, and entity_aliases.

    Returns the number of entities upserted.
    """
    count = 0
    for entry in entries:
        with conn.cursor() as cur:
            # Upsert entity
            cur.execute(
                """
                INSERT INTO entities
                    (canonical_name, entity_type, discipline, source,
                     harvest_run_id, properties)
                VALUES (%(canonical_name)s, %(entity_type)s, 'planetary_science',
                        'pds4', %(harvest_run_id)s, %(properties)s)
                ON CONFLICT (canonical_name, entity_type, source) DO UPDATE SET
                    discipline = EXCLUDED.discipline,
                    harvest_run_id = EXCLUDED.harvest_run_id,
                    properties = EXCLUDED.properties,
                    updated_at = NOW()
                RETURNING id
                """,
                {
                    "canonical_name": entry["canonical_name"],
                    "entity_type": entry["entity_type"],
                    "harvest_run_id": harvest_run_id,
                    "properties": json.dumps(entry.get("metadata", {})),
                },
            )
            entity_id: int = cur.fetchone()[0]

            # Upsert PDS URN identifier
            if entry.get("external_id"):
                cur.execute(
                    """
                    INSERT INTO entity_identifiers
                        (entity_id, id_scheme, external_id, is_primary)
                    VALUES (%(entity_id)s, 'pds_urn', %(external_id)s, true)
                    ON CONFLICT (id_scheme, external_id) DO UPDATE SET
                        entity_id = EXCLUDED.entity_id,
                        is_primary = EXCLUDED.is_primary
                    """,
                    {
                        "entity_id": entity_id,
                        "external_id": entry["external_id"],
                    },
                )

            # Upsert aliases
            for alias in entry.get("aliases", []):
                cur.execute(
                    """
                    INSERT INTO entity_aliases (entity_id, alias, alias_source)
                    VALUES (%(entity_id)s, %(alias)s, 'pds4')
                    ON CONFLICT (entity_id, alias) DO NOTHING
                    """,
                    {"entity_id": entity_id, "alias": alias},
                )

        count += 1

    conn.commit()
    logger.info("Wrote %d entities to entity graph", count)
    return count


def _write_relationships(
    conn: "psycopg.Connection",
    entries: list[dict[str, Any]],
    harvest_run_id: int,
) -> int:
    """Create entity_relationships from PDS URN structure and reference LIDs.

    Relationships created:
    - instrument part_of_mission: from instrument URN parent segment or ref_lid_investigation
    - mission observes_target: from ref_lid_target on investigation products

    Returns the number of relationships created.
    """
    count = 0

    # Build lookup: URN -> entity_id
    urn_to_entity: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ei.external_id, ei.entity_id
            FROM entity_identifiers ei
            WHERE ei.id_scheme = 'pds_urn'
            """)
        for row in cur.fetchall():
            urn_to_entity[row[0]] = row[1]

    # Build lookup: investigation slug -> entity_id (for URN-based matching)
    # e.g. "cassini-huygens" -> entity_id for urn:...:investigation:mission.cassini-huygens
    slug_to_mission: dict[str, int] = {}
    for urn, eid in urn_to_entity.items():
        if ":investigation:" in urn:
            # Extract slug: last segment after the last dot-separated prefix
            # urn:nasa:pds:context:investigation:mission.cassini-huygens -> cassini-huygens
            parts = urn.split(":")[-1]  # "mission.cassini-huygens"
            slug = parts.split(".", 1)[-1] if "." in parts else parts
            slug_to_mission[slug] = eid

    with conn.cursor() as cur:
        for entry in entries:
            external_id = entry.get("external_id", "")
            entity_id = urn_to_entity.get(external_id)
            if entity_id is None:
                continue

            metadata = entry.get("metadata", {})

            # instrument part_of_mission from URN structure
            if entry["entity_type"] == "instrument":
                match = _INSTRUMENT_PARENT_RE.match(external_id)
                if match:
                    parent_slug = match.group(1)
                    mission_id = slug_to_mission.get(parent_slug)
                    if mission_id is not None:
                        cur.execute(
                            """
                            INSERT INTO entity_relationships
                                (subject_entity_id, predicate, object_entity_id,
                                 source, harvest_run_id)
                            VALUES (%(subject)s, 'part_of_mission', %(object)s,
                                    'pds4', %(run_id)s)
                            ON CONFLICT (subject_entity_id, predicate, object_entity_id)
                            DO NOTHING
                            """,
                            {
                                "subject": entity_id,
                                "object": mission_id,
                                "run_id": harvest_run_id,
                            },
                        )
                        count += 1

                # Also check ref_lid_investigation
                for ref_lid in metadata.get("ref_lid_investigation", []):
                    mission_id = urn_to_entity.get(ref_lid)
                    if mission_id is not None:
                        cur.execute(
                            """
                            INSERT INTO entity_relationships
                                (subject_entity_id, predicate, object_entity_id,
                                 source, harvest_run_id)
                            VALUES (%(subject)s, 'part_of_mission', %(object)s,
                                    'pds4', %(run_id)s)
                            ON CONFLICT (subject_entity_id, predicate, object_entity_id)
                            DO NOTHING
                            """,
                            {
                                "subject": entity_id,
                                "object": mission_id,
                                "run_id": harvest_run_id,
                            },
                        )
                        count += 1

            # mission observes_target from ref_lid_target
            if entry["entity_type"] == "mission":
                for ref_lid in metadata.get("ref_lid_target", []):
                    target_id = urn_to_entity.get(ref_lid)
                    if target_id is not None:
                        cur.execute(
                            """
                            INSERT INTO entity_relationships
                                (subject_entity_id, predicate, object_entity_id,
                                 source, harvest_run_id)
                            VALUES (%(subject)s, 'observes_target', %(object)s,
                                    'pds4', %(run_id)s)
                            ON CONFLICT (subject_entity_id, predicate, object_entity_id)
                            DO NOTHING
                            """,
                            {
                                "subject": entity_id,
                                "object": target_id,
                                "run_id": harvest_run_id,
                            },
                        )
                        count += 1

    conn.commit()
    logger.info("Created %d entity relationships", count)
    return count


def run_harvest(
    dsn: str | None = None,
    product_types: tuple[str, ...] = ALL_PRODUCT_TYPES,
    dry_run: bool = False,
    client: ResilientClient | None = None,
) -> dict[str, int]:
    """Run the full PDS4 harvest pipeline.

    Downloads context products, parses them, and loads into entity_dictionary
    and entity graph tables. Logs a harvest_run.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        product_types: Which context types to harvest.
        dry_run: If True, parse and report stats without loading to DB.
        client: Optional ResilientClient instance. Creates a default if None.

    Returns:
        Dict mapping entity_type to count of entries loaded/parsed.
    """
    t0 = time.monotonic()

    if client is None:
        client = ResilientClient(
            user_agent="scix-experiments/1.0",
            timeout=120.0,
        )

    raw_by_type = download_pds4_context(product_types, client=client)

    all_entries: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for ptype, raw_products in raw_by_type.items():
        entries = parse_pds4_products(raw_products, ptype)
        entity_type = PRODUCT_TYPE_MAP[ptype][1]
        counts[entity_type] = len(entries)
        all_entries.extend(entries)

    records_fetched = sum(len(prods) for prods in raw_by_type.values())

    if dry_run:
        elapsed = time.monotonic() - t0
        logger.info(
            "PDS4 harvest dry run: %d total entries in %.1fs — %s",
            len(all_entries),
            elapsed,
            counts,
        )
        return counts

    conn = get_connection(dsn)
    run_id: int | None = None
    try:
        # Log harvest run start
        run_id = _start_harvest_run(conn, product_types)

        # Backward-compatible: write to entity_dictionary
        records_upserted = bulk_load(conn, all_entries, discipline="planetary_science")

        # Write to entity graph tables
        _write_entity_graph(conn, all_entries, run_id)
        rel_count = _write_relationships(conn, all_entries, run_id)

        # Finalize harvest run
        _finish_harvest_run(
            conn,
            run_id,
            records_fetched=records_fetched,
            records_upserted=records_upserted,
            counts=counts,
        )
    except Exception as exc:
        if run_id is not None:
            try:
                _finish_harvest_run(
                    conn,
                    run_id,
                    records_fetched=records_fetched,
                    records_upserted=0,
                    counts=counts,
                    status="failed",
                    error_message=str(exc),
                )
            except Exception:
                logger.exception("Failed to update harvest_run on error")
        raise
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "PDS4 harvest complete: %d entries loaded, %d relationships in %.1fs — %s",
        len(all_entries),
        rel_count,
        elapsed,
        counts,
    )
    return counts


def main() -> None:
    """Parse arguments and run the PDS4 harvest pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest PDS4 context products into entity_dictionary",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print stats without loading to database",
    )
    parser.add_argument(
        "--product-type",
        nargs="+",
        choices=ALL_PRODUCT_TYPES,
        default=list(ALL_PRODUCT_TYPES),
        help="Product types to harvest (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    product_types = tuple(args.product_type)
    counts = run_harvest(
        dsn=args.dsn,
        product_types=product_types,
        dry_run=args.dry_run,
    )

    total = sum(counts.values())
    print(f"PDS4 harvest: {total} entries — {counts}")


if __name__ == "__main__":
    main()
