"""Entity resolver: resolve text mentions to canonical entities.

Queries the normalized entity graph (migration 021) using a documented
fallback cascade:

    1. exact_canonical    — entities.canonical_name (case-insensitive)
    2. alias              — entity_aliases.alias    (case-insensitive)
    3. identifier         — entity_identifiers.external_id (Q-IDs, DOIs, …)
    4. fuzzy              — pg_trgm similarity on canonical_name
                            (auto fallback; uses idx_entities_canonical_trgm
                             via the `%` operator — migration 063)

Steps 1-3 are tried in order; step 4 fires only when 1-3 produced no
candidates so the chain has clear precedence and never returns fuzzy
noise alongside an exact match. All steps return discipline metadata
on every candidate so callers can filter cross-discipline results
without a follow-up lookup.

Callers that want the legacy "exact-only" behaviour pass ``fuzzy=False``;
the default is now ``fuzzy=True`` because empty results were a silent
failure mode for non-astro queries (bead scix_experiments-dbl.8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

MATCH_METHODS = frozenset({"exact_canonical", "alias", "identifier", "fuzzy"})

# Cap the number of fuzzy candidates returned to the caller. The GIN trigram
# index can match thousands of rows for a short token; keeping the top-N by
# similarity is sufficient for disambiguation and avoids unbounded payloads.
DEFAULT_FUZZY_LIMIT = 20


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

    Resolution cascade (see module docstring for the full chain):
        1. Exact canonical name match     (confidence 1.0)
        2. Alias lookup                   (confidence 0.9)
        3. External identifier lookup     (confidence 0.85)
        4. Fuzzy match via pg_trgm        (confidence = similarity score)

    Steps 1-3 short-circuit on first match; step 4 is the final fallback
    and runs only when 1-3 returned nothing.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def resolve(
        self,
        mention: str,
        *,
        discipline: str | None = None,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.3,
        fuzzy_limit: int = DEFAULT_FUZZY_LIMIT,
    ) -> list[EntityCandidate]:
        """Resolve a single mention to a list of entity candidates.

        Args:
            mention: Text mention to resolve.
            discipline: Optional discipline for ranking boost (+0.05).
                Does NOT filter — discipline is informational on every
                candidate so callers can choose across disciplines.
            fuzzy: Enable pg_trgm fuzzy fallback. Default True; pass
                False to suppress fuzzy and get only exact matches.
            fuzzy_threshold: Minimum similarity for fuzzy matches.
            fuzzy_limit: Maximum number of fuzzy candidates to return.

        Returns:
            List of EntityCandidate sorted by confidence DESC,
            canonical_name ASC. Never limited to a single result.
            Empty list if nothing matches in any stage of the cascade —
            callers should treat empty as "register this entity" rather
            than "this entity does not exist".
        """
        candidates: list[EntityCandidate] = []

        # Step 1: Exact canonical name match.
        candidates.extend(self._match_canonical(mention))

        # Step 2: Alias match (only if no exact canonical match).
        if not candidates:
            candidates.extend(self._match_alias(mention))

        # Step 3: Identifier match (always — may add new entities, e.g.
        # when a Wikidata Q-ID also happens to be an alias).
        candidates.extend(self._match_identifier(mention))

        # Step 4: Fuzzy fallback. Auto-fires when nothing else matched so
        # the resolver doesn't silently return empty for capitalisation /
        # punctuation variants ('alpha-fold' vs 'AlphaFold').
        if fuzzy and not candidates:
            candidates.extend(
                self._match_fuzzy(mention, fuzzy_threshold, fuzzy_limit)
            )

        # Apply discipline boost — purely informational on candidates
        # whose discipline matches; never used to filter.
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

        # Deduplicate by entity_id, keeping highest confidence.
        candidates = _deduplicate(candidates)

        # Sort by confidence DESC, then canonical_name ASC.
        candidates.sort(key=lambda c: (-c.confidence, c.canonical_name))

        return candidates

    def resolve_batch(
        self,
        mentions: list[str],
        *,
        discipline: str | None = None,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.3,
        fuzzy_limit: int = DEFAULT_FUZZY_LIMIT,
    ) -> dict[str, list[EntityCandidate]]:
        """Resolve multiple mentions. Returns dict mapping mention to candidates."""
        return {
            mention: self.resolve(
                mention,
                discipline=discipline,
                fuzzy=fuzzy,
                fuzzy_threshold=fuzzy_threshold,
                fuzzy_limit=fuzzy_limit,
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

    def _match_fuzzy(
        self,
        mention: str,
        threshold: float,
        limit: int = DEFAULT_FUZZY_LIMIT,
    ) -> list[EntityCandidate]:
        """Fuzzy match via pg_trgm similarity on canonical_name.

        The ``%`` operator is the index-able predicate: it triggers a
        Bitmap Index Scan on idx_entities_canonical_trgm (migration 063)
        using the pg_trgm default similarity GUC (0.3). The
        ``similarity() > threshold`` clause re-filters those candidates
        when the caller wants a stricter threshold than the GUC. With
        the index in place a 19M-row resolve drops from ~40s p50 to
        sub-second; without the index the `%` operator falls back to a
        Seq Scan with the same semantics, so this code path stays
        functional during a partially-applied migration.
        """
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, canonical_name, entity_type, source, discipline,
                       similarity(lower(canonical_name), lower(%(mention)s)) AS sim
                FROM entities
                WHERE lower(canonical_name) %% lower(%(mention)s)
                  AND similarity(lower(canonical_name), lower(%(mention)s)) > %(threshold)s
                ORDER BY sim DESC, canonical_name ASC
                LIMIT %(limit)s
                """,
                {"mention": mention, "threshold": threshold, "limit": limit},
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
