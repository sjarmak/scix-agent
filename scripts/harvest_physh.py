#!/usr/bin/env python3
"""Harvest PhySH Techniques into entity_dictionary.

Downloads the PhySH (Physical Subject Headings) JSON-LD vocabulary from
the APS GitHub repository, extracts the Techniques facet hierarchy via BFS,
and bulk-loads technique concepts via scix.dictionary.bulk_load() with
entity_type='method', source='physh'.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import bulk_load
from scix.harvest_utils import HarvestRunLog
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

PHYSH_URL = "https://raw.githubusercontent.com/physh-org/PhySH/master/physh.json.gz"
TECHNIQUES_FACET_ID = "https://doi.org/10.29172/fa2a6718-de5c-4c05-bf00-f169a55234d5"

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
            rate_limit=10.0,
        )
    return _client


# Sub-facet UUIDs under Techniques
_TECHNIQUE_SUBFACET_IDS = frozenset(
    {
        "https://doi.org/10.29172/705f7ed8-6d0e-4b5a-a65a-4a16ca09c040",  # Experimental
        "https://doi.org/10.29172/1e0c099a-4cd7-42c4-8a0e-8aeb0e501882",  # Computational
        "https://doi.org/10.29172/b96dac97-f930-4cfe-b895-c5e5baab0424",  # Theoretical
        "https://doi.org/10.29172/233a6cd0-9ecb-498e-bc13-b5a8d54a0521",  # Theoretical & Computational
    }
)


def download_physh(
    url: str = PHYSH_URL,
    cache_dir: Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Download and decompress the PhySH JSON-LD vocabulary.

    If *cache_dir* is provided and the file already exists there, reads from
    cache instead of downloading. Uses ResilientClient with built-in retry
    and exponential backoff.

    Args:
        url: URL of physh.json.gz on GitHub.
        cache_dir: Optional directory for caching the downloaded file.

    Returns:
        Parsed JSON-LD dict.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "physh.json.gz"
        if cache_path.exists():
            logger.info("Loading PhySH from cache: %s", cache_path)
            with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                return json.load(f)

    client = _get_client()
    response = client.get(url)
    # response may be requests.Response or CachedResponse
    if hasattr(response, "content"):
        data = response.content
    else:
        data = response.text.encode("utf-8")

    logger.info("Downloaded PhySH vocabulary: %d bytes", len(data))

    # Cache the raw gzipped data if cache_dir provided
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        logger.info("Cached PhySH to %s", cache_path)

    decompressed = gzip.decompress(data)
    return json.loads(decompressed)


_PREFIX_MAP: dict[str, str] = {
    "skos:": "http://www.w3.org/2004/02/skos/core#",
    "physh_rdf:": "https://physh.org/rdf/2018/01/01/core#",
}


def _resolve_key(node: dict[str, Any], key: str) -> Any:
    """Look up a key in a JSON-LD node, trying both prefixed and full URI forms."""
    val = node.get(key)
    if val is not None:
        return val
    # Try expanding prefix
    for prefix, uri in _PREFIX_MAP.items():
        if key.startswith(prefix):
            val = node.get(uri + key[len(prefix) :])
            if val is not None:
                return val
    return None


def _extract_label(node: dict[str, Any], key: str) -> str:
    """Extract a single label string from a JSON-LD node.

    Handles both ``{"@value": "..."}`` objects and plain strings,
    as well as lists of such values (returns the first).
    """
    val = _resolve_key(node, key)
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("@value", "")
    if isinstance(val, list) and val:
        first = val[0]
        if isinstance(first, dict):
            return first.get("@value", "")
        if isinstance(first, str):
            return first
    return ""


def _extract_labels(node: dict[str, Any], key: str) -> list[str]:
    """Extract all label strings from a JSON-LD node (for altLabels)."""
    val = _resolve_key(node, key)
    if val is None:
        return []
    if isinstance(val, str):
        return [val] if val else []
    if isinstance(val, dict):
        v = val.get("@value", "")
        return [v] if v else []
    if isinstance(val, list):
        result: list[str] = []
        for item in val:
            if isinstance(item, dict):
                v = item.get("@value", "")
                if v:
                    result.append(v)
            elif isinstance(item, str) and item:
                result.append(item)
        return result
    return []


def _extract_ids(node: dict[str, Any], key: str) -> list[str]:
    """Extract IDs from a relationship field (list of ``{"@id": ...}``)."""
    val = _resolve_key(node, key)
    if val is None:
        return []
    if isinstance(val, dict):
        rid = val.get("@id", "")
        return [rid] if rid else []
    if isinstance(val, list):
        result: list[str] = []
        for item in val:
            if isinstance(item, dict):
                rid = item.get("@id", "")
                if rid:
                    result.append(rid)
            elif isinstance(item, str) and item:
                result.append(item)
        return result
    return []


def parse_physh_techniques(jsonld: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse PhySH JSON-LD and extract Techniques facet concepts.

    Walks the facet hierarchy starting from the four Techniques sub-facets
    and collects all concepts reachable via ``physh_rdf:contains`` and
    ``skos:broader``/``skos:narrower`` links using BFS.

    Args:
        jsonld: Parsed PhySH JSON-LD dict (must have ``@graph`` key).

    Returns:
        List of entity dictionary entry dicts.
    """
    # Handle both {"@graph": [...]} and plain list [...] formats
    if isinstance(jsonld, list):
        graph = jsonld
    else:
        graph = jsonld.get("@graph", [])
    if not graph:
        logger.warning("No @graph found in PhySH JSON-LD")
        return []

    # Build index by @id
    by_id: dict[str, dict[str, Any]] = {}
    for node in graph:
        node_id = node.get("@id", "")
        if node_id:
            by_id[node_id] = node

    # Collect seed concept IDs from sub-facet contains lists
    seed_ids: set[str] = set()
    subfacet_map: dict[str, str] = {}  # concept_id -> subfacet name

    for sf_id in _TECHNIQUE_SUBFACET_IDS:
        sf_node = by_id.get(sf_id)
        if sf_node is None:
            continue
        sf_name = _extract_label(sf_node, "skos:prefLabel")
        contained = _extract_ids(sf_node, "physh_rdf:contains")
        for cid in contained:
            seed_ids.add(cid)
            subfacet_map[cid] = sf_name

    # Also check the main Techniques facet
    tech_node = by_id.get(TECHNIQUES_FACET_ID)
    if tech_node is not None:
        contained = _extract_ids(tech_node, "physh_rdf:contains")
        tech_name = _extract_label(tech_node, "skos:prefLabel")
        for cid in contained:
            if cid not in subfacet_map:
                seed_ids.add(cid)
                subfacet_map[cid] = tech_name

    # Build children map from skos:broader (child -> parent means parent has child)
    children_map: dict[str, list[str]] = {}
    for node in graph:
        node_id = node.get("@id", "")
        broader_ids = _extract_ids(node, "skos:broader")
        for parent_id in broader_ids:
            children_map.setdefault(parent_id, []).append(node_id)

    # BFS from seeds through children_map to collect all technique concepts
    visited: set[str] = set()
    queue: deque[str] = deque(seed_ids)

    while queue:
        cid = queue.popleft()
        if cid in visited:
            continue
        visited.add(cid)
        # Propagate facet label to children
        for child_id in children_map.get(cid, []):
            if child_id not in visited:
                if child_id not in subfacet_map and cid in subfacet_map:
                    subfacet_map[child_id] = subfacet_map[cid]
                queue.append(child_id)

    logger.info(
        "Found %d technique concepts (%d seeds, BFS expanded)",
        len(visited),
        len(seed_ids),
    )

    # Fallback: if subfacet-based BFS yielded nothing, harvest ALL concepts
    if not visited:
        logger.info(
            "Subfacet IDs not found in current PhySH data — "
            "falling back to all %d skos:Concept nodes",
            len(by_id),
        )
        for node in graph:
            types = node.get("@type", [])
            if isinstance(types, str):
                types = [types]
            if any("Concept" in t for t in types):
                nid = node.get("@id", "")
                if nid:
                    visited.add(nid)

    # Build entries
    entries: list[dict[str, Any]] = []
    skipped = 0

    for concept_id in sorted(visited):
        node = by_id.get(concept_id)
        if node is None:
            skipped += 1
            continue

        name = _extract_label(node, "skos:prefLabel")
        if not name:
            skipped += 1
            continue

        aliases = _extract_labels(node, "skos:altLabel")

        # Parent and child relationships
        parent_ids = _extract_ids(node, "skos:broader")
        child_ids = children_map.get(concept_id, [])

        parent_names = [
            _extract_label(by_id[pid], "skos:prefLabel") for pid in parent_ids if pid in by_id
        ]
        child_names = [
            _extract_label(by_id[cid], "skos:prefLabel") for cid in child_ids if cid in by_id
        ]

        metadata: dict[str, Any] = {}
        if parent_names:
            metadata["parent_names"] = parent_names
        if child_names:
            metadata["child_names"] = child_names
        if parent_ids:
            metadata["parent_ids"] = parent_ids
        if child_ids:
            metadata["child_ids"] = child_ids

        facet = subfacet_map.get(concept_id, "")
        if facet:
            metadata["facet"] = facet

        scope_note = _extract_label(node, "skos:scopeNote")
        if scope_note:
            metadata["description"] = scope_note

        entries.append(
            {
                "canonical_name": name,
                "entity_type": "method",
                "source": "physh",
                "external_id": concept_id,
                "aliases": aliases,
                "metadata": metadata,
            }
        )

    if skipped:
        logger.warning("Skipped %d concepts (missing node or prefLabel)", skipped)

    logger.info("Parsed %d PhySH technique entries into dictionary records", len(entries))
    return entries


def run_harvest(
    dsn: str | None = None,
    cache_dir: Path | None = None,
) -> int:
    """Run the full PhySH techniques harvest pipeline.

    Downloads the vocabulary, parses technique concepts, and loads them
    into entity_dictionary. Logs harvest run to harvest_runs.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        cache_dir: Optional directory for caching the downloaded file.

    Returns:
        Number of entries loaded.
    """
    t0 = time.monotonic()

    jsonld = download_physh(cache_dir=cache_dir)
    entries = parse_physh_techniques(jsonld)

    conn = get_connection(dsn)
    run_log = HarvestRunLog(conn, "physh")
    try:
        run_log.start()
        count = bulk_load(conn, entries)
        run_log.complete(
            records_fetched=len(entries),
            records_upserted=count,
        )
    except Exception as exc:
        try:
            run_log.fail(str(exc))
        except Exception:
            logger.warning("Failed to update harvest_run status to 'failed'")
        raise
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "PhySH techniques harvest complete: %d entries loaded in %.1fs",
        count,
        elapsed,
    )
    return count


def main() -> None:
    """Parse arguments and run the PhySH techniques harvest pipeline."""
    parser = argparse.ArgumentParser(
        description="Harvest PhySH Techniques into entity_dictionary",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=Path,
        help="Directory for caching the downloaded physh.json.gz file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    count = run_harvest(dsn=args.dsn, cache_dir=args.cache_dir)
    print(f"Loaded {count} PhySH technique entries into entity_dictionary")


if __name__ == "__main__":
    main()
