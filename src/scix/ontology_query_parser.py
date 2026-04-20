"""Ontology-aware query parser.

Lifts ontology-shaped fragments (entity types, known mission names, asteroid
taxonomies) out of a free-text query and returns structured filters that a
downstream hybrid search caller can apply.

This module is pure mechanical orchestration code per the ZFC rule in
``CLAUDE.md`` — a frozen vocabulary plus a hand-rolled tokenizer. No LLM
calls, no DB access, no semantic classification. The parser is deterministic
and side-effect free; ``parse_query`` returns a fresh ``ParsedQuery`` per
invocation. ``residual_query`` always equals the original query: filters
AUGMENT lexical/vector search, they do not replace text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary (frozen at module load).
# ---------------------------------------------------------------------------

# Plural-or-singular surface form -> canonical entity_type.
_ENTITY_TYPE_TERMS_RAW: dict[str, str] = {
    "instruments": "instrument",
    "instrument": "instrument",
    "missions": "mission",
    "mission": "mission",
    "asteroids": "asteroid",
    "asteroid": "asteroid",
    "telescopes": "telescope",
    "telescope": "telescope",
    "spacecraft": "spacecraft",
    "comets": "comet",
    "comet": "comet",
    "exoplanets": "exoplanet",
    "exoplanet": "exoplanet",
}
ENTITY_TYPE_TERMS: Mapping[str, str] = MappingProxyType(_ENTITY_TYPE_TERMS_RAW)

# Surface form -> canonical mission name. Stored in a case-insensitive lookup
# table; the canonical form (preserving case) is the value we emit into the
# JSONB containment payload.
_KNOWN_MISSIONS_CANONICAL: tuple[str, ...] = (
    "JWST",
    "Hubble",
    "HST",
    "Chandra",
    "Spitzer",
    "Cassini",
    "Voyager",
    "Kepler",
    "TESS",
    "Galileo",
    "Juno",
    "MAVEN",
    "Perseverance",
)
KNOWN_MISSIONS: frozenset[str] = frozenset(_KNOWN_MISSIONS_CANONICAL)
_MISSION_LOOKUP: Mapping[str, str] = MappingProxyType(
    {m.lower(): m for m in _KNOWN_MISSIONS_CANONICAL}
)

# Asteroid taxonomy letters (Tholen / Bus-DeMeo principal classes).
ASTEROID_TAXONOMY_LETTERS: frozenset[str] = frozenset({"M", "S", "C", "V", "X", "B", "K", "L", "P"})

# Token boundary: split on any run of whitespace or punctuation that is not
# part of an asteroid-taxonomy hyphen (the asteroid pattern is matched
# separately via regex over the raw query).
_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_ASTEROID_RE: re.Pattern[str] = re.compile(r"\b([A-Za-z])-type\b")


# ---------------------------------------------------------------------------
# Public dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OntologyClause:
    """One lifted ontology fragment."""

    entity_type: str | None
    properties_filter: dict[str, str] | None
    surface: str
    span: tuple[int, int]


@dataclass(frozen=True)
class ParsedQuery:
    """Result of parsing a free-text query.

    ``residual_query`` is intentionally equal to ``original_query``: the
    lifted clauses AUGMENT lexical/vector retrieval rather than replacing
    text in the query. The caller decides how to combine clauses with the
    underlying hybrid search.
    """

    original_query: str
    residual_query: str
    clauses: tuple[OntologyClause, ...]
    entity_types: tuple[str, ...]
    properties_filters: tuple[dict[str, str], ...]


# ---------------------------------------------------------------------------
# Vocabulary accessor.
# ---------------------------------------------------------------------------


def default_vocabulary() -> Mapping[str, object]:
    """Return the built-in vocabulary as a read-only mapping."""
    return MappingProxyType(
        {
            "entity_type_terms": ENTITY_TYPE_TERMS,
            "known_missions": KNOWN_MISSIONS,
            "asteroid_taxonomy_letters": ASTEROID_TAXONOMY_LETTERS,
        }
    )


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _resolve_vocabulary(
    vocabulary: Mapping[str, object] | None,
) -> tuple[Mapping[str, str], frozenset[str], Mapping[str, str], frozenset[str]]:
    """Return (entity_terms, known_missions, mission_lookup, taxonomy_letters)."""
    if vocabulary is None:
        return ENTITY_TYPE_TERMS, KNOWN_MISSIONS, _MISSION_LOOKUP, ASTEROID_TAXONOMY_LETTERS

    entity_terms = vocabulary.get("entity_type_terms", ENTITY_TYPE_TERMS)
    known_missions = vocabulary.get("known_missions", KNOWN_MISSIONS)
    taxonomy_letters = vocabulary.get("asteroid_taxonomy_letters", ASTEROID_TAXONOMY_LETTERS)

    if not isinstance(entity_terms, Mapping):
        raise TypeError("vocabulary['entity_type_terms'] must be a Mapping")
    if not isinstance(known_missions, (frozenset, set, tuple, list)):
        raise TypeError("vocabulary['known_missions'] must be an iterable of strings")
    if not isinstance(taxonomy_letters, (frozenset, set, tuple, list)):
        raise TypeError("vocabulary['asteroid_taxonomy_letters'] must be an iterable of strings")

    missions_set: frozenset[str] = frozenset(str(m) for m in known_missions)
    mission_lookup: Mapping[str, str] = MappingProxyType({m.lower(): m for m in missions_set})
    letters_set: frozenset[str] = frozenset(str(letter) for letter in taxonomy_letters)
    return entity_terms, missions_set, mission_lookup, letters_set


def _scan_entity_type_clauses(
    query: str,
    entity_terms: Mapping[str, str],
) -> list[OntologyClause]:
    """Emit a clause per token whose lowercased form is an entity-type term."""
    clauses: list[OntologyClause] = []
    for match in _TOKEN_RE.finditer(query):
        token = match.group(0)
        entity_type = entity_terms.get(token.lower())
        if entity_type is None:
            continue
        clauses.append(
            OntologyClause(
                entity_type=entity_type,
                properties_filter=None,
                surface=token,
                span=(match.start(), match.end()),
            )
        )
    return clauses


def _scan_mission_clauses(
    query: str,
    mission_lookup: Mapping[str, str],
    entity_type_present: bool,
    fallback_entity_type: str | None,
) -> list[OntologyClause]:
    """Emit a properties-filter clause for each KNOWN_MISSIONS hit.

    Mission tokens only lift to a properties filter when the query also
    contains at least one entity-type term (the "JWST instruments" pattern).
    The ``fallback_entity_type`` is the entity_type carried on the lifted
    clause; when multiple entity types are present, the caller's downstream
    UNION over entity_types still works because the JSONB filter is
    paired with each.
    """
    if not entity_type_present:
        return []
    clauses: list[OntologyClause] = []
    for match in _TOKEN_RE.finditer(query):
        token = match.group(0)
        canonical = mission_lookup.get(token.lower())
        if canonical is None:
            continue
        clauses.append(
            OntologyClause(
                entity_type=fallback_entity_type,
                properties_filter={"mission": canonical},
                surface=token,
                span=(match.start(), match.end()),
            )
        )
    return clauses


def _scan_asteroid_taxonomy_clauses(
    query: str,
    taxonomy_letters: frozenset[str],
) -> list[OntologyClause]:
    """Emit an asteroid-taxonomy clause for each ``[A-Z]-type`` hit in the allowlist."""
    clauses: list[OntologyClause] = []
    for match in _ASTEROID_RE.finditer(query):
        letter = match.group(1).upper()
        if letter not in taxonomy_letters:
            continue
        clauses.append(
            OntologyClause(
                entity_type="asteroid",
                properties_filter={"taxonomy": letter},
                surface=match.group(0),
                span=(match.start(), match.end()),
            )
        )
    return clauses


def _dedupe_clauses(clauses: list[OntologyClause]) -> tuple[OntologyClause, ...]:
    """Drop later clauses that share ``(entity_type, properties_filter)``."""
    seen: set[tuple[str | None, tuple[tuple[str, str], ...] | None]] = set()
    deduped: list[OntologyClause] = []
    for clause in clauses:
        key_props: tuple[tuple[str, str], ...] | None
        if clause.properties_filter is None:
            key_props = None
        else:
            key_props = tuple(sorted(clause.properties_filter.items()))
        key = (clause.entity_type, key_props)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clause)
    return tuple(deduped)


def _project_entity_types(clauses: tuple[OntologyClause, ...]) -> tuple[str, ...]:
    """De-duped, insertion-ordered tuple of entity types across clauses."""
    seen: set[str] = set()
    out: list[str] = []
    for clause in clauses:
        if clause.entity_type is None or clause.entity_type in seen:
            continue
        seen.add(clause.entity_type)
        out.append(clause.entity_type)
    return tuple(out)


def _project_properties_filters(
    clauses: tuple[OntologyClause, ...],
) -> tuple[dict[str, str], ...]:
    """De-duped, insertion-ordered tuple of properties payloads across clauses."""
    seen: set[tuple[tuple[str, str], ...]] = set()
    out: list[dict[str, str]] = []
    for clause in clauses:
        if clause.properties_filter is None:
            continue
        key = tuple(sorted(clause.properties_filter.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(clause.properties_filter))
    return tuple(out)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def parse_query(
    query: str,
    *,
    vocabulary: Mapping[str, object] | None = None,
) -> ParsedQuery:
    """Parse ``query`` and return its lifted ontology clauses.

    The parser is a hand-rolled tokenizer: it scans the query for entity-type
    terms, known-mission tokens, and asteroid-taxonomy patterns. It does not
    rewrite or strip the query — ``residual_query`` always equals
    ``original_query``. Filters AUGMENT lexical/vector retrieval; they do
    not replace text.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")

    entity_terms, _missions_set, mission_lookup, taxonomy_letters = _resolve_vocabulary(vocabulary)

    if query == "":
        return ParsedQuery(
            original_query="",
            residual_query="",
            clauses=(),
            entity_types=(),
            properties_filters=(),
        )

    entity_clauses = _scan_entity_type_clauses(query, entity_terms)
    fallback_entity_type = entity_clauses[0].entity_type if entity_clauses else None
    mission_clauses = _scan_mission_clauses(
        query,
        mission_lookup,
        entity_type_present=bool(entity_clauses),
        fallback_entity_type=fallback_entity_type,
    )
    asteroid_clauses = _scan_asteroid_taxonomy_clauses(query, taxonomy_letters)

    all_clauses = entity_clauses + mission_clauses + asteroid_clauses
    deduped = _dedupe_clauses(all_clauses)

    parsed = ParsedQuery(
        original_query=query,
        residual_query=query,
        clauses=deduped,
        entity_types=_project_entity_types(deduped),
        properties_filters=_project_properties_filters(deduped),
    )
    logger.debug(
        "parse_query: query=%r clauses=%d entity_types=%s properties_filters=%s",
        query,
        len(parsed.clauses),
        parsed.entity_types,
        parsed.properties_filters,
    )
    return parsed
