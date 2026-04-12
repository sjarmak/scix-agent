"""Canonical result type returned by :func:`scix.resolve_entities.resolve_entities`.

``EntityLinkSet`` is the M13 single-entry-point value object. It intentionally
cannot be constructed from outside :mod:`scix.resolve_entities`: the
constructor requires a module-private ``_ResolverToken`` sentinel, and any
call site that omits or forges the token receives :class:`TypeError` at
runtime.

Design note: the spec requires ``@dataclass(frozen=True)`` *and* a
token-gated ``__init__``. We get both by (a) making ``EntityLinkSet`` a
frozen dataclass whose first field is the token, and (b) raising
``TypeError`` in ``__post_init__`` when the token is not the real sentinel.
External callers that try to build one either miss the positional argument
(``TypeError: __init__() missing 1 required positional argument``) or supply
a non-sentinel value (``TypeError`` from ``__post_init__``) — both satisfy
the acceptance criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from scix._resolver_token import _ResolverToken


@dataclass(frozen=True)
class EntityLink:
    """A single (document, entity) resolution result.

    ``lane`` records which of the four M13 lanes produced this link so the
    downstream fusion layer (u08) can audit provenance.
    """

    entity_id: int
    confidence: float
    link_type: str
    tier: int
    lane: str


@dataclass(frozen=True)
class EntityLinkSet:
    """Immutable bundle of entity links for a single document.

    Construction is gated on a module-private ``_ResolverToken`` sentinel —
    callers outside :mod:`scix.resolve_entities` will raise ``TypeError``.
    """

    _token: _ResolverToken
    bibcode: str
    entities: frozenset[EntityLink]
    lane: str
    model_version: str
    candidate_set_hash: int = 0
    extras: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # The sentinel check is the runtime teeth behind the single-entry-point
        # invariant. Any caller that slipped past mypy / import conventions
        # still gets stopped here.
        if not isinstance(self._token, _ResolverToken):
            raise TypeError(
                "EntityLinkSet can only be constructed by scix.resolve_entities; "
                "direct construction is not permitted."
            )
        if not isinstance(self.entities, frozenset):
            raise TypeError("EntityLinkSet.entities must be a frozenset[EntityLink]")

    # ------------------------------------------------------------------
    # Convenience accessors for consumers (tests, fusion MV, MCP layer).
    # ------------------------------------------------------------------

    def entity_ids(self) -> frozenset[int]:
        """Return the set of entity IDs — the invariant-test primary key."""
        return frozenset(link.entity_id for link in self.entities)

    def confidence_for(self, entity_id: int) -> float:
        """Return the confidence assigned to ``entity_id`` in this lane.

        Raises ``KeyError`` if the entity is not present. Used by the
        Hypothesis invariant test to verify cross-lane drift stays ≤ 0.01.
        """
        for link in self.entities:
            if link.entity_id == entity_id:
                return link.confidence
        raise KeyError(entity_id)

    def as_iterable(self) -> Iterable[EntityLink]:
        return iter(self.entities)
