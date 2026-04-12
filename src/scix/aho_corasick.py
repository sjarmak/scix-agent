"""Tier-2 Aho-Corasick linker core — PRD §M6 / §S2 / work unit u09.

This module is a pure library: no DB access, no logging of side effects. It
takes pre-fetched :class:`EntityRow` records, builds a pickleable
:class:`ahocorasick.Automaton`, and exposes :func:`link_abstract` which scans
a single abstract string and returns :class:`LinkCandidate` records.

Ambiguity handling
------------------

The automaton stores every surface form (canonical name + aliases) for every
entity in the curated core that is not ``banned``. At match time, homograph
entities are emitted only when at least one "long-form" disambiguating alias
of that entity is co-present in the same abstract — this implements the
"ambiguity-aware firing" rule from §M6. The long-form rule is conservative:
an alias counts as a disambiguator if it is ≥ ``DISAMBIGUATOR_MIN_CHARS``
characters OR has ≥ ``DISAMBIGUATOR_MIN_TOKENS`` whitespace-separated tokens.

Confidence calibration
----------------------

Every candidate carries ``FIXED_CONFIDENCE`` (0.85). The real 7-feature
calibration logistic regression is tracked under work unit u11 / milestone
M9. Until u11 lands, downstream fusion must treat Tier 2 confidences as
uniform and ignore ordering by confidence alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

import ahocorasick

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Minimum character length for an alias to act as a disambiguator for a
# homograph canonical. "HST" < 10 chars → not a disambiguator. "Hubble Space
# Telescope" ≥ 10 chars → a disambiguator.
DISAMBIGUATOR_MIN_CHARS: int = 10

# Alternative trigger: any alias with at least this many whitespace tokens
# counts as a disambiguating long-form.
DISAMBIGUATOR_MIN_TOKENS: int = 2

# Fixed confidence emitted by every Tier-2 candidate. Calibration is owned
# by u11 / M9; see module docstring. stub-ok
FIXED_CONFIDENCE: float = 0.85

# Public type alias so callers can hint against a real symbol rather than
# reaching into the ahocorasick module directly.
AhocorasickAutomaton = ahocorasick.Automaton


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityRow:
    """Flattened row passed into :func:`build_automaton`.

    One :class:`EntityRow` represents a single surface form (the canonical
    name or one alias) of a single entity. Callers should emit one row per
    ``(entity_id, surface_form)`` pair.

    Attributes
    ----------
    entity_id
        The ``entities.id`` primary key.
    surface
        The exact surface form to match (canonical name or alias).
    canonical_name
        The entity's canonical name. Carried so :class:`LinkCandidate`
        reports the entity without an extra DB lookup.
    ambiguity_class
        One of ``unique`` / ``domain_safe`` / ``homograph``. ``banned`` is
        filtered out inside :func:`build_automaton` as a safety net.
    is_alias
        True if ``surface`` came from ``entity_aliases``; False if it is
        the canonical name itself.
    """

    entity_id: int
    surface: str
    canonical_name: str
    ambiguity_class: str
    is_alias: bool = False


@dataclass(frozen=True)
class LinkCandidate:
    """One candidate entity link produced by :func:`link_abstract`.

    Tier 2 candidates are unresolved links: the writer
    (``scripts/link_tier2.py``) routes them through M13
    ``resolve_entities`` or annotates the SQL with ``# noqa:
    resolver-lint`` as a transitional exemption.
    """

    entity_id: int
    canonical_name: str
    matched_surface: str
    start: int
    end: int
    confidence: float
    ambiguity_class: str
    match_method: str = "aho_corasick_abstract"
    is_alias: bool = False


class Disambiguator(Protocol):
    """Optional callable injected into :func:`link_abstract`.

    Given a homograph ``entity_id``, the matched surface form, and the full
    abstract, return True if the candidate survives, False otherwise.

    Real Adeft classifiers ship in :mod:`scix.adeft_disambig`; this
    Protocol keeps :func:`link_abstract` decoupled from sklearn.
    """

    def __call__(self, entity_id: int, surface: str, abstract: str) -> bool: ...


# ---------------------------------------------------------------------------
# Internal payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Payload:
    """Automaton payload. Stored verbatim on every key in the trie.

    ``surface_len`` lets :func:`link_abstract` recover the match span
    without an extra lookup: ``ahocorasick.Automaton.iter`` only yields the
    inclusive end index of each match.
    """

    entity_id: int
    canonical_name: str
    ambiguity_class: str
    is_alias: bool
    is_long_form: bool
    surface_len: int
    surface: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _is_long_form(surface: str) -> bool:
    """Return True if ``surface`` is long enough to be a disambiguator."""
    if len(surface) >= DISAMBIGUATOR_MIN_CHARS:
        return True
    tokens = _WORD_RE.findall(surface)
    return len(tokens) >= DISAMBIGUATOR_MIN_TOKENS


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """Return True if ``text[start:end]`` is a whole-word match.

    ``pyahocorasick`` matches on raw character offsets, so "ACT" would hit
    inside "ACTION". We require non-alphanumeric characters (or string
    boundaries) immediately outside the match.
    """
    left_ok = start == 0 or not text[start - 1].isalnum()
    right_ok = end == len(text) or not text[end].isalnum()
    return left_ok and right_ok


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_automaton(entities: Iterable[EntityRow]) -> AhocorasickAutomaton:
    """Build a pickleable :class:`ahocorasick.Automaton` from ``entities``.

    Each surface form is added case-insensitively (we lowercase both the
    surface and — at match time — the abstract). Duplicate surfaces
    collapse to the first row seen.

    The resulting automaton is pickle-safe so the caller can fan it out
    to worker processes via ``multiprocessing.Pool``.
    """
    automaton = ahocorasick.Automaton()
    seen: set[str] = set()

    for row in entities:
        if row.ambiguity_class == "banned":
            # Defensive net — callers are expected to filter, but a banned
            # row must never land in the Tier 2 index.
            continue
        surface = row.surface.strip()
        if not surface:
            continue
        key = surface.lower()
        if key in seen:
            continue
        seen.add(key)
        automaton.add_word(
            key,
            _Payload(
                entity_id=int(row.entity_id),
                canonical_name=row.canonical_name,
                ambiguity_class=row.ambiguity_class,
                is_alias=bool(row.is_alias),
                is_long_form=_is_long_form(surface),
                surface_len=len(key),
                surface=surface,
            ),
        )

    if seen:
        automaton.make_automaton()
    return automaton


def link_abstract(
    abstract: str,
    automaton: AhocorasickAutomaton,
    disambiguator: Optional[Disambiguator] = None,
) -> list[LinkCandidate]:
    """Return every entity mention in ``abstract`` honoring ambiguity rules.

    Parameters
    ----------
    abstract
        Abstract text to scan. Empty / ``None``-safe.
    automaton
        Built via :func:`build_automaton`.
    disambiguator
        Optional Adeft-style classifier. Invoked only for homograph
        candidates whose matched surface is NOT itself a long-form. If
        omitted, the co-presence rule is the only gate on homographs.

    Ambiguity rules (§M6 / §S2)
    ---------------------------

    * ``unique`` → always fire.
    * ``domain_safe`` → always fire.
    * ``homograph``:
        - If the matched surface is itself a long-form, fire immediately
          (the long-form IS the disambiguator).
        - Otherwise, fire only when (a) a long-form alias of the same
          entity is co-present in the abstract, OR (b) ``disambiguator
          (entity_id, surface, abstract)`` returns True.
    """
    if not abstract:
        return []
    if len(automaton) == 0:
        return []

    abstract_lower = abstract.lower()

    # Single automaton pass: collect every match AND note which entities
    # have a long-form surface co-present. O(n + matches).
    raw_hits: list[tuple[int, int, _Payload]] = []
    long_form_present: set[int] = set()
    for end_idx_inclusive, payload in automaton.iter(abstract_lower):
        start = end_idx_inclusive - payload.surface_len + 1
        end = end_idx_inclusive + 1  # half-open, Python-slicing shape
        if not _boundary_ok(abstract_lower, start, end):
            continue
        raw_hits.append((start, end, payload))
        if payload.is_long_form:
            long_form_present.add(payload.entity_id)

    candidates: list[LinkCandidate] = []
    for start, end, payload in raw_hits:
        ambiguity = payload.ambiguity_class
        entity_id = payload.entity_id

        if ambiguity == "homograph" and not payload.is_long_form:
            if entity_id not in long_form_present:
                if disambiguator is None:
                    continue
                try:
                    if not disambiguator(entity_id, payload.surface, abstract):
                        continue
                except Exception:
                    # A broken classifier must never take out the pipeline:
                    # fail closed (drop the candidate) and keep scanning.
                    continue

        candidates.append(
            LinkCandidate(
                entity_id=entity_id,
                canonical_name=payload.canonical_name,
                matched_surface=payload.surface,
                start=start,
                end=end,
                confidence=FIXED_CONFIDENCE,
                ambiguity_class=ambiguity,
                is_alias=payload.is_alias,
            )
        )

    return candidates


__all__ = [
    "AhocorasickAutomaton",
    "DISAMBIGUATOR_MIN_CHARS",
    "DISAMBIGUATOR_MIN_TOKENS",
    "Disambiguator",
    "EntityRow",
    "FIXED_CONFIDENCE",
    "LinkCandidate",
    "build_automaton",
    "link_abstract",
]
