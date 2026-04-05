#!/usr/bin/env python3
"""Enrich multi-discipline entity_dictionary entries with Wikidata aliases and QIDs.

Queries the Wikidata SPARQL endpoint in batches (up to 50 names per request)
for GCMD instruments, PDS4 missions, and PDS4 targets.  For each match the
script updates aliases[] and metadata.wikidata_qid in entity_dictionary.

SPARQL results are cached to disk so re-runs skip already-fetched batches.

Usage:
    python scripts/enrich_wikidata_multi.py --help
    python scripts/enrich_wikidata_multi.py --dry-run
    python scripts/enrich_wikidata_multi.py --source pds4 --entity-type mission
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scix.db import get_connection
from scix.dictionary import upsert_entry

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "scix-experiments/1.0 (https://github.com/scix; mailto:scix@example.org)"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "wikidata_cache"
MAX_BATCH_SIZE = 50
MIN_DELAY = 2.0


# ---------------------------------------------------------------------------
# Entity type configuration
# ---------------------------------------------------------------------------


class EntityTypeConfig(NamedTuple):
    """Configuration for an entity type to enrich."""

    source: str
    entity_type: str
    label: str


ENTITY_TYPE_CONFIGS: tuple[EntityTypeConfig, ...] = (
    EntityTypeConfig(source="gcmd", entity_type="instrument", label="GCMD instruments"),
    EntityTypeConfig(source="pds4", entity_type="mission", label="PDS4 missions"),
    EntityTypeConfig(source="pds4", entity_type="target", label="PDS4 targets"),
)


# ---------------------------------------------------------------------------
# SPARQL query construction
# ---------------------------------------------------------------------------


def build_batch_sparql_query(names: list[str]) -> str:
    """Build a SPARQL query that searches Wikidata for multiple names at once.

    Uses a VALUES clause to batch up to *MAX_BATCH_SIZE* names in a single
    request.  Returns QID, rdfs:label match, and English altLabels.

    Args:
        names: List of canonical names to search (max 50).

    Returns:
        SPARQL query string.

    Raises:
        ValueError: If *names* is empty or exceeds MAX_BATCH_SIZE.
    """
    if not names:
        raise ValueError("names must not be empty")
    if len(names) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size {len(names)} exceeds maximum {MAX_BATCH_SIZE}")

    escaped_values = []
    for name in names:
        escaped = name.replace("\\", "\\\\").replace('"', '\\"')
        escaped_values.append(f'"{escaped}"@en')

    values_clause = " ".join(escaped_values)

    return f"""\
SELECT ?item ?name ?altLabel WHERE {{
  VALUES ?name {{ {values_clause} }}
  ?item rdfs:label ?name .
  OPTIONAL {{ ?item skos:altLabel ?altLabel . FILTER(LANG(?altLabel) = "en") }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}"""


# ---------------------------------------------------------------------------
# SPARQL execution
# ---------------------------------------------------------------------------


def execute_sparql(query: str, endpoint: str = WIKIDATA_SPARQL_URL) -> dict[str, Any]:
    """Execute a SPARQL query against the Wikidata endpoint with retries.

    Args:
        query: SPARQL query string.
        endpoint: SPARQL endpoint URL.

    Returns:
        Parsed JSON response dict.

    Raises:
        urllib.error.URLError: On network failure after retries.
        ValueError: On malformed JSON response.
    """
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    url = f"{endpoint}?{params}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
        },
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            return json.loads(data.decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            if attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    "SPARQL attempt %d/%d failed: %s -- retrying in %ds",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "SPARQL query failed after %d attempts: %s",
                    max_retries,
                    exc,
                )
                raise
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Malformed SPARQL response: {exc}") from exc

    raise RuntimeError("execute_sparql: unexpected exit from retry loop")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_batch_results(
    sparql_json: dict[str, Any],
) -> dict[str, tuple[str, list[str]]]:
    """Parse batched SPARQL results into a mapping of name -> (qid, aliases).

    Args:
        sparql_json: Parsed Wikidata SPARQL JSON response.

    Returns:
        Dict mapping each matched canonical name (str) to a tuple of
        (wikidata_qid, sorted list of English altLabels).
    """
    bindings = sparql_json.get("results", {}).get("bindings", [])
    if not bindings:
        return {}

    # Collect per-name data
    name_data: dict[str, dict[str, Any]] = {}

    for binding in bindings:
        name_val = binding.get("name", {}).get("value", "")
        if not name_val:
            continue

        if name_val not in name_data:
            name_data[name_val] = {"qid": None, "aliases": set()}

        # Extract QID from item URI
        item_uri = binding.get("item", {}).get("value", "")
        if "/entity/" in item_uri and name_data[name_val]["qid"] is None:
            name_data[name_val]["qid"] = item_uri.rsplit("/", 1)[-1]

        alt = binding.get("altLabel", {}).get("value", "")
        if alt:
            name_data[name_val]["aliases"].add(alt)

    result: dict[str, tuple[str, list[str]]] = {}
    for name, data in name_data.items():
        qid = data["qid"]
        if qid is not None:
            result[name] = (qid, sorted(data["aliases"]))

    return result


# ---------------------------------------------------------------------------
# Alias merging
# ---------------------------------------------------------------------------


def merge_aliases(existing: list[str], new: list[str]) -> list[str]:
    """Merge new aliases into an existing list, deduplicating case-insensitively.

    Preserves the original casing of existing entries.  New entries are added
    only if no case-insensitive match exists.

    Args:
        existing: Current alias list.
        new: Aliases to merge in.

    Returns:
        Deduplicated merged alias list.
    """
    lower_set: set[str] = {a.lower() for a in existing}
    merged = list(existing)
    for alias in new:
        if alias.lower() not in lower_set:
            merged.append(alias)
            lower_set.add(alias.lower())
    return merged


# ---------------------------------------------------------------------------
# Disk caching
# ---------------------------------------------------------------------------


def cache_path(cache_dir: Path, source: str, entity_type: str, batch_idx: int) -> Path:
    """Return the cache file path for a given batch.

    Args:
        cache_dir: Root cache directory.
        source: Entity source (e.g. 'gcmd', 'pds4').
        entity_type: Entity type (e.g. 'instrument', 'mission').
        batch_idx: Zero-based batch index.

    Returns:
        Path to the cache JSON file.
    """
    return cache_dir / f"{source}_{entity_type}_{batch_idx}.json"


def load_cache(path: Path) -> dict[str, Any] | None:
    """Load cached SPARQL JSON from disk.

    Args:
        path: Path to the cache file.

    Returns:
        Parsed JSON dict, or None if the file does not exist or is invalid.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cache read failed for %s: %s", path, exc)
        return None


def save_cache(path: Path, data: dict[str, Any]) -> None:
    """Save SPARQL JSON response to disk cache.

    Args:
        path: Path to write the cache file.
        data: SPARQL JSON response dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.debug("Cached SPARQL response to %s", path)


# ---------------------------------------------------------------------------
# DB operations
# ---------------------------------------------------------------------------


def fetch_entries(
    conn: Any,
    source: str,
    entity_type: str,
) -> list[dict[str, Any]]:
    """Fetch entity_dictionary entries for a given source and entity_type.

    Args:
        conn: Database connection (psycopg).
        source: Source filter (e.g. 'gcmd', 'pds4').
        entity_type: Entity type filter.

    Returns:
        List of row dicts with id, canonical_name, aliases, metadata, etc.
    """
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, canonical_name, entity_type, source,
                   external_id, aliases, metadata
            FROM entity_dictionary
            WHERE source = %(source)s AND entity_type = %(entity_type)s
            ORDER BY canonical_name
            """,
            {"source": source, "entity_type": entity_type},
        )
        return [dict(row) for row in cur.fetchall()]


def apply_enrichments(
    conn: Any,
    entries: list[dict[str, Any]],
    matches: dict[str, tuple[str, list[str]]],
    *,
    dry_run: bool = False,
) -> int:
    """Apply Wikidata enrichments to matching entity_dictionary entries.

    Args:
        conn: Database connection.
        entries: List of entity_dictionary row dicts.
        matches: Mapping of canonical_name -> (qid, wikidata_aliases).
        dry_run: If True, log changes but do not write to DB.

    Returns:
        Number of entries enriched.
    """
    enriched = 0

    for entry in entries:
        name = entry["canonical_name"]
        if name not in matches:
            continue

        qid, new_aliases = matches[name]

        existing_aliases = entry.get("aliases") or []
        merged_aliases = merge_aliases(existing_aliases, new_aliases)

        existing_metadata = entry.get("metadata") or {}
        updated_metadata = {**existing_metadata, "wikidata_qid": qid}

        if dry_run:
            logger.info(
                "[DRY RUN] Would enrich '%s' with QID=%s, %d new aliases",
                name,
                qid,
                len(merged_aliases) - len(existing_aliases),
            )
        else:
            upsert_entry(
                conn,
                canonical_name=name,
                entity_type=entry["entity_type"],
                source=entry["source"],
                external_id=entry.get("external_id"),
                aliases=merged_aliases,
                metadata=updated_metadata,
            )
            logger.info(
                "Enriched '%s' with QID=%s, %d new aliases",
                name,
                qid,
                len(merged_aliases) - len(existing_aliases),
            )

        enriched += 1

    return enriched


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------


def chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into chunks of at most *size* elements.

    Args:
        items: The list to split.
        size: Maximum chunk size.

    Returns:
        List of sub-lists.
    """
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_enrich(
    dsn: str | None = None,
    *,
    source_filter: str | None = None,
    entity_type_filter: str | None = None,
    batch_size: int = MAX_BATCH_SIZE,
    delay: float = MIN_DELAY,
    endpoint: str = WIKIDATA_SPARQL_URL,
    dry_run: bool = False,
    use_cache: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[int, int]:
    """Run the Wikidata enrichment pipeline for multi-discipline entities.

    Args:
        dsn: Database connection string. Uses SCIX_DSN or default if None.
        source_filter: If set, only enrich this source.
        entity_type_filter: If set, only enrich this entity type.
        batch_size: Max names per SPARQL VALUES clause (capped at 50).
        delay: Seconds to wait between batch SPARQL requests (min 2.0).
        endpoint: SPARQL endpoint URL.
        dry_run: If True, skip DB writes.
        use_cache: If True, read/write SPARQL cache files.
        cache_dir: Directory for cache files.

    Returns:
        Tuple of (total_processed, total_enriched).
    """
    # Enforce constraints
    batch_size = min(batch_size, MAX_BATCH_SIZE)
    delay = max(delay, MIN_DELAY)

    t0 = time.monotonic()

    # Select which configs to process
    configs = ENTITY_TYPE_CONFIGS
    if source_filter:
        configs = tuple(c for c in configs if c.source == source_filter)
    if entity_type_filter:
        configs = tuple(c for c in configs if c.entity_type == entity_type_filter)

    if not configs:
        logger.warning("No matching entity type configs after filtering")
        return 0, 0

    conn = get_connection(dsn)
    total_processed = 0
    total_enriched = 0

    try:
        for config in configs:
            logger.info("Processing %s (%s/%s)", config.label, config.source, config.entity_type)

            entries = fetch_entries(conn, config.source, config.entity_type)
            if not entries:
                logger.info("No entries found for %s/%s", config.source, config.entity_type)
                continue

            names = [e["canonical_name"] for e in entries]
            batches = chunk_list(names, batch_size)

            logger.info(
                "Found %d entries, split into %d batches of up to %d",
                len(entries),
                len(batches),
                batch_size,
            )

            all_matches: dict[str, tuple[str, list[str]]] = {}

            for batch_idx, batch_names in enumerate(batches):
                if batch_idx > 0:
                    time.sleep(delay)

                # Try cache first
                cp = cache_path(cache_dir, config.source, config.entity_type, batch_idx)
                sparql_json = None

                if use_cache:
                    sparql_json = load_cache(cp)
                    if sparql_json is not None:
                        logger.debug("Cache hit for batch %d", batch_idx)

                if sparql_json is None:
                    query = build_batch_sparql_query(batch_names)
                    try:
                        sparql_json = execute_sparql(query, endpoint=endpoint)
                    except (urllib.error.URLError, ValueError) as exc:
                        logger.warning(
                            "Skipping batch %d: SPARQL query failed: %s",
                            batch_idx,
                            exc,
                        )
                        continue

                    if use_cache:
                        save_cache(cp, sparql_json)

                batch_matches = parse_batch_results(sparql_json)
                all_matches.update(batch_matches)

                logger.info(
                    "Batch %d/%d: %d/%d names matched",
                    batch_idx + 1,
                    len(batches),
                    len(batch_matches),
                    len(batch_names),
                )

            enriched = apply_enrichments(conn, entries, all_matches, dry_run=dry_run)
            total_processed += len(entries)
            total_enriched += enriched

            logger.info(
                "%s: enriched %d/%d entries",
                config.label,
                enriched,
                len(entries),
            )
    finally:
        conn.close()

    elapsed = time.monotonic() - t0
    logger.info(
        "Wikidata multi-enrichment complete: %d/%d entries enriched in %.1fs",
        total_enriched,
        total_processed,
        elapsed,
    )
    return total_processed, total_enriched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the Wikidata multi-discipline enrichment pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Enrich multi-discipline entity_dictionary entries with "
            "Wikidata aliases and QIDs (GCMD instruments, PDS4 missions, "
            "PDS4 targets)"
        ),
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database connection string (uses SCIX_DSN env var if not provided)",
    )
    parser.add_argument(
        "--source",
        default=None,
        choices=["gcmd", "pds4"],
        help="Only enrich entries from this source",
    )
    parser.add_argument(
        "--entity-type",
        default=None,
        choices=["instrument", "mission", "target"],
        help="Only enrich entries of this entity type",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=MAX_BATCH_SIZE,
        help=f"Max names per SPARQL query (default/max: {MAX_BATCH_SIZE})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=MIN_DELAY,
        help=f"Seconds between batch requests (default/min: {MIN_DELAY})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log enrichments without writing to the database",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip reading/writing SPARQL response cache",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
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

    total, enriched = run_enrich(
        dsn=args.dsn,
        source_filter=args.source,
        entity_type_filter=args.entity_type,
        batch_size=args.batch_size,
        delay=args.delay,
        dry_run=args.dry_run,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )
    print(f"Enriched {enriched}/{total} multi-discipline entries with Wikidata data")


if __name__ == "__main__":
    main()
