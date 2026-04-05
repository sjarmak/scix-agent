"""Entity resolver: resolve text mentions to canonical entities.

Queries the normalized entity graph (migration 021) using a cascade of
match strategies: exact canonical name, alias, external identifier, and
optional fuzzy matching via pg_trgm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

MATCH_METHODS = frozenset({"exact_canonical", "alias", "identifier", "fuzzy"})


@dataclass(frozen=True)
class EntityCandidate:
    """A resolved entity candidate with confidence and match provenance."""

    entity_id: int
    canonical_name: str
    entity_type: str
    source: str
    discipline: str | None
    confidence: float
    match_method: str


class EntityResolver:
    """Resolve text mentions to canonical entities in the entity graph.

    Resolution cascade:
    1. Exact canonical name match (confidence 1.0)
    2. Alias lookup (confidence 0.9)
    3. External identifier lookup (confidence 0.85)
    4. Optional fuzzy match via pg_trgm (confidence = similarity score)
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def resolve(
        self,
        mention: str,
        *,
        discipline: str | None = None,
        fuzzy: bool = False,
        fuzzy_threshold: float = 0.3,
    ) -> list[EntityCandidate]:
        """Resolve a single mention to a list of entity candidates.

        Args:
            mention: Text mention to resolve.
            discipline: Optional discipline for ranking boost (+0.05).
            fuzzy: Enable pg_trgm fuzzy fallback.
            fuzzy_threshold: Minimum similarity for fuzzy matches.

        Returns:
            List of EntityCandidate sorted by confidence DESC,
            canonical_name ASC. Never limited to one result.
        """
        candidates: list[EntityCandidate] = []

        # Step 1: Exact canonical name match
        candidates.extend(self._match_canonical(mention))

        # Step 2: Alias match (only if no exact canonical match)
        if not candidates:
            candidates.extend(self._match_alias(mention))

        # Step 3: Identifier match (always — may add new entities)
        candidates.extend(self._match_identifier(mention))

        # Step 4: Fuzzy fallback (only if enabled and no results yet)
        if fuzzy and not candidates:
            candidates.extend(self._match_fuzzy(mention, fuzzy_threshold))

        # Apply discipline boost
        if discipline is not None:
            candidates = [
                EntityCandidate(
                    entity_id=c.entity_id,
                    canonical_name=c.canonical_name,
                    entity_type=c.entity_type,
                    source=c.source,
                    discipline=c.discipline,
                    confidence=(
                        min(c.confidence + 0.05, 1.0)
                        if c.discipline == discipline
                        else c.confidence
                    ),
                    match_method=c.match_method,
                )
                for c in candidates
            ]

        # Deduplicate by entity_id, keeping highest confidence
        candidates = _deduplicate(candidates)

        # Sort by confidence DESC, then canonical_name ASC
        candidates.sort(key=lambda c: (-c.confidence, c.canonical_name))

        return candidates

    def resolve_batch(
        self,
        mentions: list[str],
        *,
        discipline: str | None = None,
        fuzzy: bool = False,
        fuzzy_threshold: float = 0.3,
    ) -> dict[str, list[EntityCandidate]]:
        """Resolve multiple mentions. Returns dict mapping mention to candidates."""
        return {
            mention: self.resolve(
                mention,
                discipline=discipline,
                fuzzy=fuzzy,
                fuzzy_threshold=fuzzy_threshold,
            )
            for mention in mentions
        }

    # ------------------------------------------------------------------
    # Match strategies
    # ------------------------------------------------------------------

    def _match_canonical(self, mention: str) -> list[EntityCandidate]:
        """Exact case-insensitive match on entities.canonical_name."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, canonical_name, entity_type, source, discipline
                FROM entities
                WHERE lower(canonical_name) = lower(%(mention)s)
                """,
                {"mention": mention},
            )
            return [
                EntityCandidate(
                    entity_id=row["id"],
                    canonical_name=row["canonical_name"],
                    entity_type=row["entity_type"],
                    source=row["source"],
                    discipline=row["discipline"],
                    confidence=1.0,
                    match_method="exact_canonical",
                )
                for row in cur.fetchall()
            ]

    def _match_alias(self, mention: str) -> list[EntityCandidate]:
        """Case-insensitive alias lookup via entity_aliases JOIN entities."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT e.id, e.canonical_name, e.entity_type, e.source, e.discipline
                FROM entity_aliases ea
                JOIN entities e ON e.id = ea.entity_id
                WHERE lower(ea.alias) = lower(%(mention)s)
                """,
                {"mention": mention},
            )
            return [
                EntityCandidate(
                    entity_id=row["id"],
                    canonical_name=row["canonical_name"],
                    entity_type=row["entity_type"],
                    source=row["source"],
                    discipline=row["discipline"],
                    confidence=0.9,
                    match_method="alias",
                )
                for row in cur.fetchall()
            ]

    def _match_identifier(self, mention: str) -> list[EntityCandidate]:
        """Exact match on entity_identifiers.external_id JOIN entities."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT e.id, e.canonical_name, e.entity_type, e.source, e.discipline
                FROM entity_identifiers ei
                JOIN entities e ON e.id = ei.entity_id
                WHERE ei.external_id = %(mention)s
                """,
                {"mention": mention},
            )
            return [
                EntityCandidate(
                    entity_id=row["id"],
                    canonical_name=row["canonical_name"],
                    entity_type=row["entity_type"],
                    source=row["source"],
                    discipline=row["discipline"],
                    confidence=0.85,
                    match_method="identifier",
                )
                for row in cur.fetchall()
            ]

    def _match_fuzzy(self, mention: str, threshold: float) -> list[EntityCandidate]:
        """Fuzzy match via pg_trgm similarity on canonical_name."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, canonical_name, entity_type, source, discipline,
                       similarity(lower(canonical_name), lower(%(mention)s)) AS sim
                FROM entities
                WHERE similarity(lower(canonical_name), lower(%(mention)s)) > %(threshold)s
                """,
                {"mention": mention, "threshold": threshold},
            )
            return [
                EntityCandidate(
                    entity_id=row["id"],
                    canonical_name=row["canonical_name"],
                    entity_type=row["entity_type"],
                    source=row["source"],
                    discipline=row["discipline"],
                    confidence=float(row["sim"]),
                    match_method="fuzzy",
                )
                for row in cur.fetchall()
            ]


def _deduplicate(candidates: list[EntityCandidate]) -> list[EntityCandidate]:
    """Keep only the highest-confidence candidate per entity_id."""
    best: dict[int, EntityCandidate] = {}
    for c in candidates:
        existing = best.get(c.entity_id)
        if existing is None or c.confidence > existing.confidence:
            best[c.entity_id] = c
    return list(best.values())
