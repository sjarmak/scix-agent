"""M13 single entry point for document-entity resolution.

This module is the ONLY place in the codebase that is allowed to write to
``document_entities`` / ``document_entities_jit_cache`` or read from
``document_entities_canonical``. That invariant is enforced statically by
``scripts/ast_lint_resolver.py``, which walks ``src/`` with libcst and fails
CI on any violation originating outside this file.

At this stage (u03) all four lanes are mock/stub implementations:

* **static-core** — will eventually read the ``document_entities_canonical``
  materialized view built by u08. For now the lane uses an in-module dict.
* **jit_cache_hit** — will eventually read the partitioned
  ``document_entities_jit_cache`` table built by u10. For now the lane uses
  an in-module dict keyed on ``(bibcode, candidate_set_hash, model_version)``.
* **live_jit** — will eventually call Anthropic Haiku. For now the lane uses
  a deterministic in-module stub with tunable latency.
* **local_ner** — will eventually run SciBERT / INDUS-NER inference. For now
  the lane uses a deterministic in-module stub with tunable latency.

Invariants (spec §M13, acceptance criteria 2 / 5):

* ``resolve_entities(bibcode, context) -> EntityLinkSet`` is the sole public
  entry point.
* ``EntityLinkSet`` can only be built via the private ``_RESOLVER_INTERNAL``
  sentinel — callers outside this module get ``TypeError``.
* For a fixed ``(bibcode, candidate_set_hash, model_version)``, every lane
  returns the same set of entity IDs; confidence may differ by ≤ 0.01.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from scix._resolver_token import _RESOLVER_INTERNAL
from scix.entity_link_set import EntityLink, EntityLinkSet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public context type
# ---------------------------------------------------------------------------

ResolveMode = Literal["static", "jit", "auto", "live_jit", "local_ner"]


@dataclass(frozen=True)
class EntityResolveContext:
    """Caller-supplied context for a single :func:`resolve_entities` call.

    Attributes
    ----------
    candidate_set
        Frozen set of candidate entity IDs. Used by the JIT/NER lanes to
        scope the resolution space and by :func:`candidate_set_hash` to
        produce a stable cache key.
    mode
        Which lane(s) to try. ``"auto"`` walks the lanes in priority order
        static -> jit_cache -> live_jit -> local_ner and returns the first
        one that has a seeded hit. The explicit lane modes force a specific
        backend — used by the invariant test to compare lane outputs.
    ttl_max
        Maximum acceptable age (seconds) of a cached JIT result. Only the
        real u10 backend will honor this; the mock treats it as metadata.
    budget_remaining
        Remaining per-query dollar budget (0..1 normalized). The real
        live_jit backend will refuse the call if budget is depleted; the
        mock records the value for observability.
    model_version
        Model version string threaded into the cache key and attached to
        the resulting :class:`EntityLinkSet`.
    """

    candidate_set: frozenset[int]
    mode: ResolveMode = "auto"
    ttl_max: int = 3600
    budget_remaining: float = 1.0
    model_version: str = "v1"


# ---------------------------------------------------------------------------
# Mock backends — all four lanes use in-module state at u03.
# u08 replaces the static lane, u10 replaces the jit_cache / live_jit lanes.
# ---------------------------------------------------------------------------

# Maps bibcode -> frozenset[int] entity IDs.
_STATIC_MOCK: dict[str, frozenset[int]] = {}

# Maps (bibcode, candidate_set_hash, model_version) -> frozenset[int].
_JIT_CACHE_MOCK: dict[tuple[str, int, str], frozenset[int]] = {}

# Maps (bibcode, candidate_set_hash) -> frozenset[int].
_LIVE_JIT_MOCK: dict[tuple[str, int], frozenset[int]] = {}

# Maps bibcode -> frozenset[int].
_LOCAL_NER_MOCK: dict[str, frozenset[int]] = {}

# Injected per-lane mock latency in seconds. Benchmarks override this.
_LANE_LATENCIES: dict[str, float] = {
    "static": 0.0,
    "jit_cache_hit": 0.0,
    "live_jit": 0.0,
    "local_ner": 0.0,
}

# Per-lane deterministic confidence offsets. Kept ≤ 0.01 so the invariant
# test can verify the "sets equal, confidence may differ by ≤ 0.01" rule.
_LANE_CONFIDENCE_BASE: dict[str, float] = {
    "static": 0.990,
    "jit_cache_hit": 0.985,
    "live_jit": 0.982,
    "local_ner": 0.988,
}


def _reset_mocks() -> None:
    """Clear all in-module mock state. Used by tests."""
    _STATIC_MOCK.clear()
    _JIT_CACHE_MOCK.clear()
    _LIVE_JIT_MOCK.clear()
    _LOCAL_NER_MOCK.clear()


def _seed_static(bibcode: str, ids: frozenset[int]) -> None:
    _STATIC_MOCK[bibcode] = ids


def _seed_jit_cache(bibcode: str, cset_hash: int, model_version: str, ids: frozenset[int]) -> None:
    _JIT_CACHE_MOCK[(bibcode, cset_hash, model_version)] = ids


def _seed_live_jit(bibcode: str, cset_hash: int, ids: frozenset[int]) -> None:
    _LIVE_JIT_MOCK[(bibcode, cset_hash)] = ids


def _seed_local_ner(bibcode: str, ids: frozenset[int]) -> None:
    _LOCAL_NER_MOCK[bibcode] = ids


# ---------------------------------------------------------------------------
# Hashing helper
# ---------------------------------------------------------------------------


def candidate_set_hash(context: EntityResolveContext) -> int:
    """Stable hash of the candidate set (and model version).

    We use a sorted tuple so the hash is independent of iteration order. We
    intentionally do NOT use Python's ``hash()`` on the frozenset directly
    because its seed varies per process — and the real u10 backend will key
    on a byte-stable hash. Here we use ``hash(tuple)`` which is deterministic
    within a process, sufficient for u03 mock behavior.
    """
    sorted_ids = tuple(sorted(context.candidate_set))
    return hash((sorted_ids, context.model_version))


# ---------------------------------------------------------------------------
# Lane implementations
# ---------------------------------------------------------------------------


def _links_from_ids(ids: frozenset[int], lane: str, model_version: str) -> frozenset[EntityLink]:
    """Build an EntityLink frozenset from a raw id set for a given lane."""
    base = _LANE_CONFIDENCE_BASE[lane]
    return frozenset(
        EntityLink(
            entity_id=eid,
            confidence=base,
            link_type="mention",
            tier=0,
            lane=lane,
        )
        for eid in ids
    )


def _make_link_set(
    *,
    bibcode: str,
    ids: frozenset[int],
    lane: str,
    context: EntityResolveContext,
    cset_hash: int,
) -> EntityLinkSet:
    return EntityLinkSet(
        _RESOLVER_INTERNAL,
        bibcode=bibcode,
        entities=_links_from_ids(ids, lane, context.model_version),
        lane=lane,
        model_version=context.model_version,
        candidate_set_hash=cset_hash,
    )


def _maybe_sleep(lane: str) -> None:
    delay = _LANE_LATENCIES.get(lane, 0.0)
    if delay > 0:
        time.sleep(delay)


def _lane_static(
    bibcode: str, context: EntityResolveContext, cset_hash: int
) -> EntityLinkSet | None:
    """Static-core lane. Will eventually read document_entities_canonical."""
    _maybe_sleep("static")
    ids = _STATIC_MOCK.get(bibcode)
    if ids is None:
        return None
    return _make_link_set(
        bibcode=bibcode,
        ids=ids,
        lane="static",
        context=context,
        cset_hash=cset_hash,
    )


def _lane_jit_cache(
    bibcode: str, context: EntityResolveContext, cset_hash: int
) -> EntityLinkSet | None:
    """JIT cache hit lane. Will eventually read document_entities_jit_cache."""
    _maybe_sleep("jit_cache_hit")
    key = (bibcode, cset_hash, context.model_version)
    ids = _JIT_CACHE_MOCK.get(key)
    if ids is None:
        return None
    return _make_link_set(
        bibcode=bibcode,
        ids=ids,
        lane="jit_cache_hit",
        context=context,
        cset_hash=cset_hash,
    )


def _lane_live_jit(
    bibcode: str, context: EntityResolveContext, cset_hash: int
) -> EntityLinkSet | None:
    """Live JIT lane. Will eventually call Anthropic Haiku."""
    _maybe_sleep("live_jit")
    key = (bibcode, cset_hash)
    ids = _LIVE_JIT_MOCK.get(key)
    if ids is None:
        return None
    return _make_link_set(
        bibcode=bibcode,
        ids=ids,
        lane="live_jit",
        context=context,
        cset_hash=cset_hash,
    )


def _lane_local_ner(
    bibcode: str, context: EntityResolveContext, cset_hash: int
) -> EntityLinkSet | None:
    """Local NER lane. Will eventually run SciBERT / INDUS-NER inference."""
    _maybe_sleep("local_ner")
    ids = _LOCAL_NER_MOCK.get(bibcode)
    if ids is None:
        return None
    return _make_link_set(
        bibcode=bibcode,
        ids=ids,
        lane="local_ner",
        context=context,
        cset_hash=cset_hash,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ResolutionFailed(RuntimeError):
    """Raised when no lane can produce a result."""


def resolve_entities(bibcode: str, context: EntityResolveContext) -> EntityLinkSet:
    """Resolve ``bibcode`` to a set of entity links.

    This is the single canonical entry point for entity resolution in the
    SciX codebase (M13 in the entity-enrichment PRD). All four internal
    lanes are routed through here so the fusion layer and downstream MCP
    tools only ever see a uniform :class:`EntityLinkSet`.

    Parameters
    ----------
    bibcode
        ADS bibcode of the target document.
    context
        Caller-supplied :class:`EntityResolveContext`.

    Returns
    -------
    EntityLinkSet
        Frozen bundle of entity links, tagged with the lane that produced
        them.

    Raises
    ------
    ResolutionFailed
        If the requested lane (or all lanes in ``auto`` mode) has no data.
    """
    if not isinstance(bibcode, str) or not bibcode:
        raise ValueError("bibcode must be a non-empty string")
    if not isinstance(context, EntityResolveContext):
        raise TypeError("context must be an EntityResolveContext")

    cset_hash = candidate_set_hash(context)
    mode = context.mode

    if mode == "static":
        result = _lane_static(bibcode, context, cset_hash)
    elif mode == "jit":
        result = _lane_jit_cache(bibcode, context, cset_hash)
    elif mode == "live_jit":
        result = _lane_live_jit(bibcode, context, cset_hash)
    elif mode == "local_ner":
        result = _lane_local_ner(bibcode, context, cset_hash)
    elif mode == "auto":
        result = (
            _lane_static(bibcode, context, cset_hash)
            or _lane_jit_cache(bibcode, context, cset_hash)
            or _lane_live_jit(bibcode, context, cset_hash)
            or _lane_local_ner(bibcode, context, cset_hash)
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown resolve mode: {mode!r}")

    if result is None:
        raise ResolutionFailed(f"no lane produced a result for bibcode={bibcode!r} mode={mode!r}")
    return result


__all__ = [
    "EntityResolveContext",
    "ResolutionFailed",
    "ResolveMode",
    "candidate_set_hash",
    "resolve_entities",
]
