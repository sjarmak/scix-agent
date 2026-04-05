#!/usr/bin/env python3
"""Harvest PDS4 context products into entity_dictionary.

Downloads Investigation, Instrument, and Target context products from the
NASA PDS Registry API and bulk-loads them via scix.dictionary.bulk_load()
with discipline='planetary_science' and source='pds4'.

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
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load

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
)

PAGE_SIZE = 2000
MAX_RETRIES = 3

# Regex to extract parenthesised abbreviations from titles
_ABBREV_RE = re.compile(r"\(([A-Z][A-Z0-9/.\-]{0,20})\)")


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
    limit: int = PAGE_SIZE,
    search_after: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch a single page of PDS4 context products from the Registry API.

    Uses cursor-based pagination via the ``search-after`` query parameter.
    The PDS Registry API does not support offset-based ``start`` parameters.

    Args:
        context_type: One of ``PRODUCT_TYPE_MAP`` keys (investigation,
            instrument, target).
        limit: Page size (default 2000, large enough for most collections).
        search_after: Sort values from the last product of the previous page,
            used for cursor-based pagination. None for the first page.

    Returns:
        Parsed JSON response dict with ``summary`` and ``data`` keys.

    Raises:
        urllib.error.URLError: If the request fails after retries.
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
    url = f"{PDS_API_BASE}?{urllib.parse.urlencode(params, safe=':*,.')}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "scix-experiments/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            return json.loads(data)
        except (urllib.error.URLError, OSError) as exc:
            if attempt < MAX_RETRIES:
                wait = 2**attempt
                logger.warning(
                    "PDS API attempt %d/%d failed: %s — retrying in %ds",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Failed to fetch PDS4 %s products after %d attempts: %s",
                    context_type,
                    MAX_RETRIES,
                    exc,
                )
                raise

    raise RuntimeError("fetch_pds4_page: unexpected exit from retry loop")


def download_pds4_context(
    product_types: tuple[str, ...] = ALL_PRODUCT_TYPES,
) -> dict[str, list[dict[str, Any]]]:
    """Download all PDS4 context products for the given types.

    Handles pagination automatically.

    Args:
        product_types: Tuple of context types to download.

    Returns:
        Dict mapping context type to list of raw product dicts from the API.
    """
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
            page = fetch_pds4_page(ptype, limit=PAGE_SIZE, search_after=search_after)
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


def run_harvest(
    dsn: str | None = None,
    product_types: tuple[str, ...] = ALL_PRODUCT_TYPES,
    dry_run: bool = False,
) -> dict[str, int]:
    """Run the full PDS4 harvest pipeline.

    Downloads context products, parses them, and loads into entity_dictionary.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        product_types: Which context types to harvest.
        dry_run: If True, parse and report stats without loading to DB.

    Returns:
        Dict mapping entity_type to count of entries loaded/parsed.
    """
    t0 = time.monotonic()

    raw_by_type = download_pds4_context(product_types)

    all_entries: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for ptype, raw_products in raw_by_type.items():
        entries = parse_pds4_products(raw_products, ptype)
        entity_type = PRODUCT_TYPE_MAP[ptype][1]
        counts[entity_type] = len(entries)
        all_entries.extend(entries)

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
    try:
        bulk_load(conn, all_entries, discipline="planetary_science")
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "PDS4 harvest complete: %d entries loaded in %.1fs — %s",
        len(all_entries),
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
