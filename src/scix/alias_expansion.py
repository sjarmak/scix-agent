"""Query-time alias expansion (xz4.1.24).

Given a free-text user query, scan it for entity surface forms and expand
each match to the entity's canonical name plus its other known aliases. The
expanded term set is intended to be OR-joined into a downstream tsvector
query and the entity ids fed to ``SearchFilters.entity_ids`` to boost
recall in :func:`scix.search.hybrid_search`.

The module is a pure library — no logging side effects, no LLM calls. It
takes a :class:`psycopg.Connection` only to load entity rows; the actual
matching uses :mod:`pyahocorasick` (already a hard dep via
:mod:`scix.aho_corasick`).

Disambiguation
--------------

Many aliases are homographs — e.g. ``HST`` could be ``Hubble Space
Telescope``, ``High Speed Train`` or ``Honolulu Standard Time``. We reuse
the long-form rule from :mod:`scix.aho_corasick`: an alias counts as a
disambiguator if it has at least ``DISAMBIGUATOR_MIN_CHARS`` characters or
``DISAMBIGUATOR_MIN_TOKENS`` whitespace tokens.

When ``require_long_form_disambiguator=True`` (default), an ambiguous
match (a surface form bound to more than one entity in the loaded
automaton) fires only if a long-form alias of the same entity is also
present in the query. User queries are usually too short to satisfy this
co-occurrence rule, so we ALSO relax it: if the matched surface is the
only candidate for that surface form in the loaded automaton (i.e. the
surface is NOT a homograph in the loaded set), the match always fires.
This is the key relaxation versus :func:`scix.aho_corasick.link_abstract`,
which assumes abstract-length context.

Caching
-------

Building an automaton from the full ``entity_aliases`` table is expensive
(1.5M rows). The automaton is cached in module state keyed by
``(id(conn), entity_types)``. The cache is a small insertion-ordered LRU
because the typical caller uses very few distinct filter combinations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

import ahocorasick

from scix.aho_corasick import (
    DISAMBIGUATOR_MIN_CHARS,
    DISAMBIGUATOR_MIN_TOKENS,
    AhocorasickAutomaton,
    EntityRow,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_MAX_ALIASES_PER_ENTITY: int = 8

# Capacity of the module-level automaton cache. Most callers use a single
# filter combination; 8 leaves headroom without risking memory pressure on
# the 1.5M-row full set.
_AUTOMATON_CACHE_SIZE: int = 8

_WORD_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasExpansion:
    """One entity match found in the original query."""

    entity_id: int
    canonical_name: str
    entity_type: str
    matched_surface: str
    span: tuple[int, int]
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ExpansionResult:
    """Output of :func:`expand_query`."""

    original_query: str
    matches: tuple[AliasExpansion, ...]
    expanded_terms: tuple[str, ...]
    entity_ids: tuple[int, ...]


# ---------------------------------------------------------------------------
# Internal automaton payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SurfacePayload:
    """One entity's view of a single surface form.

    Multiple ``_SurfacePayload`` records share the same lower-cased
    surface form when that surface is a homograph (e.g. ``HST``).
    """

    entity_id: int
    canonical_name: str
    entity_type: str
    surface: str  # original casing as stored
    is_long_form: bool


@dataclass(frozen=True)
class AliasAutomaton:
    """Pre-built automaton plus per-entity metadata.

    Returned by :func:`build_alias_automaton` /
    :func:`build_alias_automaton_from_rows` and consumed by
    :func:`expand_query`.
    """

    automaton: AhocorasickAutomaton
    aliases_by_entity: dict[int, tuple[str, ...]] = field(default_factory=dict)
    canonical_by_entity: dict[int, str] = field(default_factory=dict)
    type_by_entity: dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_long_form(surface: str) -> bool:
    """Return True if ``surface`` is long enough to act as a disambiguator."""
    if len(surface) >= DISAMBIGUATOR_MIN_CHARS:
        return True
    return len(_WORD_RE.findall(surface)) >= DISAMBIGUATOR_MIN_TOKENS


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """Return True if ``text[start:end]`` sits on word boundaries."""
    left_ok = start == 0 or not text[start - 1].isalnum()
    right_ok = end == len(text) or not text[end].isalnum()
    return left_ok and right_ok


# ---------------------------------------------------------------------------
# In-memory automaton construction
# ---------------------------------------------------------------------------


def build_alias_automaton_from_rows(
    rows: Iterable[EntityRow],
    *,
    entity_type_by_id: dict[int, str] | None = None,
) -> AliasAutomaton:
    """Build an :class:`AliasAutomaton` from in-memory ``EntityRow`` records.

    Parameters
    ----------
    rows
        One :class:`EntityRow` per ``(entity_id, surface)`` pair. The
        canonical name should also appear as a row (with
        ``is_alias=False``).
    entity_type_by_id
        Map from ``entity_id`` to ``entity_type``. ``EntityRow`` itself
        does not carry the type, so callers supply it explicitly. Missing
        ids default to the empty string.
    """
    type_lookup = entity_type_by_id or {}
    automaton = ahocorasick.Automaton()
    payloads_by_key: dict[str, list[_SurfacePayload]] = {}
    aliases_acc: dict[int, list[str]] = {}
    canonical: dict[int, str] = {}
    type_map: dict[int, str] = {}

    for row in rows:
        if row.ambiguity_class == "banned":
            continue
        surface = row.surface.strip()
        if not surface:
            continue
        eid = int(row.entity_id)
        canonical.setdefault(eid, row.canonical_name)
        type_map.setdefault(eid, type_lookup.get(eid, ""))
        key = surface.lower()
        payload = _SurfacePayload(
            entity_id=eid,
            canonical_name=row.canonical_name,
            entity_type=type_map[eid],
            surface=surface,
            is_long_form=_is_long_form(surface),
        )
        bucket = payloads_by_key.setdefault(key, [])
        if any(p.entity_id == eid and p.surface == surface for p in bucket):
            continue
        bucket.append(payload)
        if row.is_alias:
            aliases_acc.setdefault(eid, []).append(surface)

    for key, bucket in payloads_by_key.items():
        automaton.add_word(key, tuple(bucket))
    if payloads_by_key:
        automaton.make_automaton()

    aliases_by_entity = {eid: tuple(dict.fromkeys(a)) for eid, a in aliases_acc.items()}
    return AliasAutomaton(
        automaton=automaton,
        aliases_by_entity=aliases_by_entity,
        canonical_by_entity=canonical,
        type_by_entity=type_map,
    )


# ---------------------------------------------------------------------------
# DB-backed automaton construction (cached)
# ---------------------------------------------------------------------------


def _fetch_entity_rows(
    conn,
    entity_types: tuple[str, ...] | None,
) -> tuple[list[EntityRow], dict[int, str]]:
    """Pull entity surface rows from Postgres.

    Returns the rows plus an ``entity_id -> entity_type`` map. Surfaces
    include both canonical names and registered aliases.
    """
    rows: list[EntityRow] = []
    type_map: dict[int, str] = {}

    type_filter_sql = ""
    type_params: list[object] = []
    if entity_types:
        type_filter_sql = "WHERE e.entity_type = ANY(%s)"
        type_params = [list(entity_types)]

    canonical_sql = (
        f"SELECT e.id, e.canonical_name, e.entity_type FROM entities e {type_filter_sql}"
    )
    alias_sql = (
        "SELECT ea.entity_id, ea.alias, e.canonical_name, e.entity_type "
        "FROM entity_aliases ea "
        f"JOIN entities e ON e.id = ea.entity_id {type_filter_sql}"
    )

    with conn.cursor() as cur:
        cur.execute(canonical_sql, type_params)
        for eid, canonical_name, entity_type in cur.fetchall():
            type_map[int(eid)] = entity_type or ""
            rows.append(
                EntityRow(
                    entity_id=int(eid),
                    surface=canonical_name,
                    canonical_name=canonical_name,
                    ambiguity_class="unique",
                    is_alias=False,
                )
            )
        cur.execute(alias_sql, type_params)
        for eid, alias, canonical_name, entity_type in cur.fetchall():
            type_map.setdefault(int(eid), entity_type or "")
            rows.append(
                EntityRow(
                    entity_id=int(eid),
                    surface=alias,
                    canonical_name=canonical_name,
                    ambiguity_class="unique",
                    is_alias=True,
                )
            )

    return rows, type_map


# Manual LRU because :func:`functools.lru_cache` cannot key on a live
# connection object (psycopg connections are unhashable across pool
# re-issuance) and we want explicit eviction order. Insertion-ordered
# dict + length cap gives us LRU semantics.
_BUNDLE_CACHE: dict[tuple[int, tuple[str, ...] | None], AliasAutomaton] = {}


def _store_bundle(
    key: tuple[int, tuple[str, ...] | None],
    bundle: AliasAutomaton,
) -> None:
    """Insert ``bundle`` and evict oldest if over capacity."""
    _BUNDLE_CACHE[key] = bundle
    while len(_BUNDLE_CACHE) > _AUTOMATON_CACHE_SIZE:
        oldest = next(iter(_BUNDLE_CACHE))
        del _BUNDLE_CACHE[oldest]


def clear_automaton_cache() -> None:
    """Evict every cached automaton. Intended for tests."""
    _BUNDLE_CACHE.clear()


def build_alias_automaton(
    conn,
    *,
    entity_types: tuple[str, ...] | None = None,
) -> AliasAutomaton:
    """Build (and cache) an automaton from ``entities`` + ``entity_aliases``.

    Parameters
    ----------
    conn
        Read-only :class:`psycopg.Connection`.
    entity_types
        Optional filter — restrict to these ``entity_type`` values.
        ``None`` loads every entity (expensive on the production corpus).

    Returns
    -------
    AliasAutomaton
        Pre-built automaton plus per-entity canonical/alias/type metadata.
        The same bundle is returned on subsequent calls with the same
        ``(conn, entity_types)`` pair.
    """
    key = (id(conn), tuple(entity_types) if entity_types else None)
    cached = _BUNDLE_CACHE.get(key)
    if cached is not None:
        return cached

    rows, type_map = _fetch_entity_rows(conn, entity_types)
    bundle = build_alias_automaton_from_rows(rows, entity_type_by_id=type_map)
    _store_bundle(key, bundle)
    return bundle


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------


def _scan_query(
    query: str,
    bundle: AliasAutomaton,
) -> tuple[list[tuple[int, int, _SurfacePayload]], dict[int, set[str]]]:
    """Return ``(raw_hits, long_form_present_by_entity)``.

    Each homograph surface yields one entry per entity sharing it, all at
    the same span. ``long_form_present_by_entity`` records which entity
    ids have a long-form surface co-present somewhere in the query.
    """
    if not query or len(bundle.automaton) == 0:
        return [], {}

    query_lower = query.lower()
    raw_hits: list[tuple[int, int, _SurfacePayload]] = []
    long_form_present: dict[int, set[str]] = {}

    for end_idx_inclusive, payloads in bundle.automaton.iter(query_lower):
        first = payloads[0]
        # The automaton key is the lower-cased surface; its length defines
        # the match span. Original-cased surfaces may differ in length for
        # exotic Unicode but we always use the key length here.
        key_len = len(first.surface.lower())
        start = end_idx_inclusive - key_len + 1
        end = end_idx_inclusive + 1
        if start < 0 or end > len(query_lower):
            continue
        if not _boundary_ok(query_lower, start, end):
            continue
        for payload in payloads:
            raw_hits.append((start, end, payload))
            if payload.is_long_form:
                long_form_present.setdefault(payload.entity_id, set()).add(payload.surface.lower())

    return raw_hits, long_form_present


def _is_homograph_surface(bundle: AliasAutomaton, surface_lower: str) -> bool:
    """Return True if ``surface_lower`` is bound to more than one entity."""
    payloads = bundle.automaton.get(surface_lower, None)
    if payloads is None:
        return False
    return len({p.entity_id for p in payloads}) > 1


def expand_query(
    conn,
    query: str,
    *,
    automaton: AliasAutomaton | None = None,
    max_aliases_per_entity: int = DEFAULT_MAX_ALIASES_PER_ENTITY,
    require_long_form_disambiguator: bool = True,
) -> ExpansionResult:
    """Expand ``query`` to canonical names + aliases of matched entities.

    Parameters
    ----------
    conn
        Read-only :class:`psycopg.Connection`. Used only when
        ``automaton`` is ``None``, in which case the full automaton is
        built and cached.
    query
        Free-text user query.
    automaton
        Optional pre-built bundle from :func:`build_alias_automaton`.
        Pass one explicitly to scope the expansion (e.g. only
        ``mission`` and ``instrument`` entities) and to control caching.
    max_aliases_per_entity
        Cap on the alias list returned per match. Defends against
        pathological entities with hundreds of aliases blowing up the
        downstream tsvector OR-clause.
    require_long_form_disambiguator
        When True, ambiguous matches (a surface bound to more than one
        entity in the loaded automaton) fire only if a long-form alias
        of the same entity is also present in the query. Unambiguous
        matches always fire — this is the relaxation versus
        :func:`scix.aho_corasick.link_abstract`, which assumes
        abstract-length context. Pass False to disable the long-form
        gate entirely.
    """
    if not query:
        return ExpansionResult(
            original_query=query,
            matches=(),
            expanded_terms=(),
            entity_ids=(),
        )

    bundle = automaton if automaton is not None else build_alias_automaton(conn)
    raw_hits, long_form_present = _scan_query(query, bundle)

    matches: list[AliasExpansion] = []
    seen_match_keys: set[tuple[int, int, int]] = set()

    for start, end, payload in raw_hits:
        eid = payload.entity_id
        match_key = (eid, start, end)
        if match_key in seen_match_keys:
            continue

        surface_lower = payload.surface.lower()
        is_homograph = _is_homograph_surface(bundle, surface_lower)

        if require_long_form_disambiguator and is_homograph and not payload.is_long_form:
            if eid not in long_form_present:
                continue

        seen_match_keys.add(match_key)
        other_aliases = tuple(
            a
            for a in bundle.aliases_by_entity.get(eid, ())
            if a != payload.surface and a != payload.canonical_name
        )[:max_aliases_per_entity]
        matches.append(
            AliasExpansion(
                entity_id=eid,
                canonical_name=payload.canonical_name,
                entity_type=bundle.type_by_entity.get(eid, ""),
                matched_surface=payload.surface,
                span=(start, end),
                aliases=other_aliases,
            )
        )

    expanded_seen: dict[str, None] = {}
    entity_id_seen: dict[int, None] = {}
    for m in matches:
        entity_id_seen.setdefault(m.entity_id, None)
        expanded_seen.setdefault(m.canonical_name, None)
        for a in m.aliases:
            expanded_seen.setdefault(a, None)

    return ExpansionResult(
        original_query=query,
        matches=tuple(matches),
        expanded_terms=tuple(expanded_seen),
        entity_ids=tuple(entity_id_seen),
    )


__all__ = [
    "AliasAutomaton",
    "AliasExpansion",
    "ExpansionResult",
    "build_alias_automaton",
    "build_alias_automaton_from_rows",
    "clear_automaton_cache",
    "expand_query",
]
