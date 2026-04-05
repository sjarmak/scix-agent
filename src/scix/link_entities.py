"""Document-entity linking pipeline.

Reads extracted mentions from the extractions table, resolves them against
the entities and entity_aliases tables, and writes results to document_entities.
Processes in configurable chunks with per-batch commits for resumability.

Uses a caching resolver that bulk-loads all entities and aliases once for O(1)
per-mention lookups. For interactive/per-query resolution with fuzzy and
discipline support, see entity_resolver.EntityResolver.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import psycopg

logger = logging.getLogger(__name__)

# Payload keys that map to entity types in the extractions table.
# Covers both entity_extraction_v3 combined payloads and per-type rows.
_PAYLOAD_KEYS = frozenset(
    {"instruments", "datasets", "methods", "observables", "materials", "software"}
)


@dataclass(frozen=True)
class ResolverMatch:
    """A single entity resolution result."""

    entity_id: int
    confidence: float
    match_method: str


class _CachingResolver:
    """Resolve mention strings to entity IDs via bulk-loaded cache.

    Loads the full entity+alias mapping on first use so subsequent lookups
    are pure dict hits (no per-mention queries). Used by the batch linking
    pipeline for performance.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn
        self._cache: dict[str, ResolverMatch] | None = None

    def _build_cache(self) -> dict[str, ResolverMatch]:
        cache: dict[str, ResolverMatch] = {}
        with self._conn.cursor() as cur:
            cur.execute("SELECT id, canonical_name FROM entities")
            for row in cur.fetchall():
                entity_id, name = row
                key = name.strip().lower()
                if key and key not in cache:
                    cache[key] = ResolverMatch(
                        entity_id=entity_id,
                        confidence=1.0,
                        match_method="canonical_exact",
                    )
            cur.execute("SELECT entity_id, alias FROM entity_aliases")
            for row in cur.fetchall():
                entity_id, alias = row
                key = alias.strip().lower()
                if key and key not in cache:
                    cache[key] = ResolverMatch(
                        entity_id=entity_id,
                        confidence=0.9,
                        match_method="alias_exact",
                    )
        return cache

    def resolve(self, mention: str) -> ResolverMatch | None:
        if self._cache is None:
            self._cache = self._build_cache()
        key = mention.strip().lower()
        return self._cache.get(key)


# Backward-compatible alias used by tests
EntityResolver = _CachingResolver


def _extract_mentions_from_payload(
    payload: dict[str, Any],
    extraction_type: str,
) -> list[tuple[str, str]]:
    """Extract (mention, payload_key) pairs from an extraction payload.

    Handles two payload shapes:
    - Combined: {"instruments": [...], "datasets": [...], ...}
    - Per-type:  {"entities": [...]}  (payload_key derived from extraction_type)
    """
    mentions: list[tuple[str, str]] = []

    # Combined payload (entity_extraction_v3 style)
    for key in _PAYLOAD_KEYS:
        items = payload.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str) and item.strip():
                    mentions.append((item.strip(), key))

    # Per-type payload (legacy extract.py style)
    entities = payload.get("entities")
    if isinstance(entities, list) and not mentions:
        for item in entities:
            if isinstance(item, str) and item.strip():
                mentions.append((item.strip(), extraction_type))

    return mentions


def link_entities_batch(
    conn: psycopg.Connection,
    *,
    batch_size: int = 1000,
    resume: bool = True,
    extraction_type: str = "entity_extraction_v3",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Link extracted mentions to entities in batches.

    Args:
        conn: Database connection (must NOT be in autocommit mode).
        batch_size: Number of bibcodes per commit chunk.
        resume: If True, skip bibcodes already in document_entities.
        extraction_type: Filter extractions by this type.
        dry_run: If True, count mentions without writing.

    Returns:
        Summary dict with bibcodes_processed, links_created, skipped_no_match.
    """
    resolver = _CachingResolver(conn)

    # Get all bibcodes with this extraction_type
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT bibcode FROM extractions WHERE extraction_type = %s",
            (extraction_type,),
        )
        all_bibcodes = [row[0] for row in cur.fetchall()]

    if not all_bibcodes:
        logger.info("No extractions found for type %s", extraction_type)
        return {"bibcodes_processed": 0, "links_created": 0, "skipped_no_match": 0}

    # Filter already-linked bibcodes if resuming
    if resume:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT bibcode FROM document_entities")
            linked = {row[0] for row in cur.fetchall()}
        bibcodes = [b for b in all_bibcodes if b not in linked]
        logger.info(
            "Resume: %d of %d bibcodes already linked, %d remaining",
            len(all_bibcodes) - len(bibcodes),
            len(all_bibcodes),
            len(bibcodes),
        )
    else:
        bibcodes = all_bibcodes

    total_links = 0
    total_skipped = 0
    total_processed = 0

    for batch_start in range(0, len(bibcodes), batch_size):
        batch = bibcodes[batch_start : batch_start + batch_size]

        # Load extraction payloads for this batch
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bibcode, extraction_type, payload
                FROM extractions
                WHERE extraction_type = %s AND bibcode = ANY(%s)
                """,
                (extraction_type, batch),
            )
            rows = cur.fetchall()

        batch_links = 0
        batch_skipped = 0

        if not dry_run:
            with conn.cursor() as cur:
                for bibcode, ext_type, payload_raw in rows:
                    payload = (
                        payload_raw if isinstance(payload_raw, dict) else json.loads(payload_raw)
                    )
                    mentions = _extract_mentions_from_payload(payload, ext_type)

                    for mention_text, payload_key in mentions:
                        match = resolver.resolve(mention_text)
                        if match is None:
                            batch_skipped += 1
                            continue

                        evidence = {
                            "mention": mention_text,
                            "extraction_type": extraction_type,
                            "payload_key": payload_key,
                        }
                        cur.execute(
                            """
                            INSERT INTO document_entities
                                (bibcode, entity_id, link_type, confidence,
                                 match_method, evidence)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (bibcode, entity_id, link_type) DO NOTHING
                            """,
                            (
                                bibcode,
                                match.entity_id,
                                "mentions",
                                match.confidence,
                                match.match_method,
                                json.dumps(evidence),
                            ),
                        )
                        batch_links += 1

            conn.commit()
        else:
            # Dry run: just count
            for bibcode, ext_type, payload_raw in rows:
                payload = payload_raw if isinstance(payload_raw, dict) else json.loads(payload_raw)
                mentions = _extract_mentions_from_payload(payload, ext_type)
                for mention_text, _payload_key in mentions:
                    match = resolver.resolve(mention_text)
                    if match is None:
                        batch_skipped += 1
                    else:
                        batch_links += 1

        total_links += batch_links
        total_skipped += batch_skipped
        total_processed += len(batch)

        logger.info(
            "Batch %d-%d: %d links, %d unresolved",
            batch_start,
            batch_start + len(batch),
            batch_links,
            batch_skipped,
        )

    summary = {
        "bibcodes_processed": total_processed,
        "links_created": total_links,
        "skipped_no_match": total_skipped,
    }
    logger.info("Linking complete: %s", summary)
    return summary


def get_linking_progress(conn: psycopg.Connection) -> dict[str, int]:
    """Return linking progress statistics.

    Returns:
        Dict with total_bibcodes, linked_bibcodes, pending_bibcodes.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(DISTINCT bibcode) FROM extractions")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT bibcode) FROM document_entities")
        linked = cur.fetchone()[0]

    return {
        "total_bibcodes": total,
        "linked_bibcodes": linked,
        "pending_bibcodes": total - linked,
    }
