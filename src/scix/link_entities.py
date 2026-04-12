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

from scix.harvest_utils import LINK_TYPE_MENTIONS

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

    Collision handling: multiple entities can share the same lowercase
    canonical name (e.g. SSODNet "Mercury" the planet vs. GCMD "Mercury"
    the chemical element). The cache stores ALL candidates per key as a
    list; `resolve_all()` exposes the full set so a disambiguator can see
    the ambiguity, while `resolve()` returns the highest-priority candidate
    (canonical matches first, then aliases) for back-compat with the
    single-match batch linker.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn
        self._cache: dict[str, list[ResolverMatch]] | None = None

    def _build_cache(self) -> dict[str, list[ResolverMatch]]:
        cache: dict[str, list[ResolverMatch]] = {}
        with self._conn.cursor() as cur:
            cur.execute("SELECT id, canonical_name FROM entities")
            for row in cur.fetchall():
                entity_id, name = row
                if name is None:
                    continue
                key = name.strip().lower()
                if not key:
                    continue
                cache.setdefault(key, []).append(
                    ResolverMatch(
                        entity_id=entity_id,
                        confidence=1.0,
                        match_method="canonical_exact",
                    )
                )
            cur.execute("SELECT entity_id, alias FROM entity_aliases")
            for row in cur.fetchall():
                entity_id, alias = row
                if alias is None:
                    continue
                key = alias.strip().lower()
                if not key:
                    continue
                cache.setdefault(key, []).append(
                    ResolverMatch(
                        entity_id=entity_id,
                        confidence=0.9,
                        match_method="alias_exact",
                    )
                )
        return cache

    def resolve_all(self, mention: str) -> list[ResolverMatch]:
        """Return ALL candidate matches for a mention.

        Surfaces ambiguity so callers/disambiguators can see collisions
        (e.g. "Mercury" the planet vs. the element). Candidates are
        ordered by insertion: canonical matches before alias matches.
        """
        if self._cache is None:
            self._cache = self._build_cache()
        key = mention.strip().lower()
        if not key:
            return []
        return list(self._cache.get(key, ()))

    def resolve(self, mention: str) -> ResolverMatch | None:
        """Return the highest-priority candidate, or None.

        Canonical matches outrank alias matches because they are inserted
        into the candidate list first. When there are multiple canonical
        collisions, the first-loaded one wins — use `resolve_all()` to see
        every candidate and route ambiguity through a disambiguator.
        """
        candidates = self.resolve_all(mention)
        return candidates[0] if candidates else None


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

    # Get bibcodes to process, excluding already-linked ones if resuming
    with conn.cursor() as cur:
        if resume:
            cur.execute(
                """
                SELECT DISTINCT e.bibcode
                FROM extractions e
                WHERE e.extraction_type = %s
                  AND NOT EXISTS (
                      SELECT 1 FROM document_entities de WHERE de.bibcode = e.bibcode
                  )
                """,
                (extraction_type,),
            )
        else:
            cur.execute(
                "SELECT DISTINCT bibcode FROM extractions WHERE extraction_type = %s",
                (extraction_type,),
            )
        bibcodes = [row[0] for row in cur.fetchall()]

    if not bibcodes:
        logger.info("No extractions to process for type %s", extraction_type)
        return {"bibcodes_processed": 0, "links_created": 0, "skipped_no_match": 0}

    if resume:
        logger.info("Resume: %d bibcodes remaining to link", len(bibcodes))

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

        # Resolve all mentions and collect insert params
        insert_params: list[tuple[str, int, str, float, str, str]] = []
        for bibcode, ext_type, payload_raw in rows:
            payload = payload_raw if isinstance(payload_raw, dict) else json.loads(payload_raw)
            mentions = _extract_mentions_from_payload(payload, ext_type)

            for mention_text, payload_key in mentions:
                match = resolver.resolve(mention_text)
                if match is None:
                    batch_skipped += 1
                    continue

                evidence = json.dumps(
                    {
                        "mention": mention_text,
                        "extraction_type": extraction_type,
                        "payload_key": payload_key,
                    }
                )
                insert_params.append(
                    (
                        bibcode,
                        match.entity_id,
                        LINK_TYPE_MENTIONS,
                        match.confidence,
                        match.match_method,
                        evidence,
                    )
                )

        batch_links = len(insert_params)

        if not dry_run and insert_params:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO document_entities
                        (bibcode, entity_id, link_type, confidence,
                         match_method, evidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (bibcode, entity_id, link_type) DO NOTHING
                    """,
                    insert_params,
                )
            conn.commit()

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
