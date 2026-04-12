"""Routing policy for the JIT lane (PRD §M11c).

Flow
----
Normal path::

    route_jit(...)  ->  cache hit?  ->  return cached
                    ->  5% canary?  ->  run_local_ner
                    ->  else       ->  bulkhead(call_live_jit)
                    ->  on degrade ->  run_local_ner
                    ->  on failure ->  static-core sentinel

The canary branch takes ``rng() < 0.05`` of live traffic and routes it to
the local NER model so we keep a warm signal on the local lane quality —
without it, the local lane would only run when live_jit is degraded, and
we'd have no way to detect regression before the bulkhead actually fires.

The spec's fallback order ("bulkhead-degrade -> local NER -> static-core
fallback") is explicit: on live_jit degrade we try local first, and only
if local *also* fails do we fall through to the static-core sentinel. The
static-core result is NOT produced here — the router returns a
:data:`STATIC_CORE_FALLBACK` sentinel that the resolver interprets as
"ask the static lane to serve this".
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Final, Optional, Union

from scix.jit.bulkhead import DEGRADED, JITBulkhead
from scix.jit.local_ner import LocalNERResult, run_local_ner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinels & result types
# ---------------------------------------------------------------------------


class _StaticCoreFallback:
    """Singleton sentinel: both JIT lanes failed, ask static-core."""

    _instance: "_StaticCoreFallback | None" = None

    def __new__(cls) -> "_StaticCoreFallback":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "<STATIC_CORE_FALLBACK>"


STATIC_CORE_FALLBACK: Final[_StaticCoreFallback] = _StaticCoreFallback()


#: 5% canary share — spec-mandated.
CANARY_SHARE: Final[float] = 0.05


@dataclass(frozen=True)
class LiveJITResult:
    """Result produced by a successful live_jit (Haiku) call."""

    bibcode: str
    entity_ids: frozenset[int]
    confidences: frozenset[tuple[int, float]]
    model_version: str
    lane: str = "live_jit"


RouterOutcome = Union[LiveJITResult, LocalNERResult, _StaticCoreFallback]


# ---------------------------------------------------------------------------
# live_jit shim (will call Haiku in production)
# ---------------------------------------------------------------------------


async def call_live_jit(
    bibcode: str,
    text: str,
    candidate_set: frozenset[int],
    *,
    model_version: str = "haiku-v1",
) -> LiveJITResult:
    """Stub live_jit call. Production swaps this for a real Haiku call.

    Tests monkey-patch the module-level reference to simulate outages.
    The stub echoes the candidate set so the lane is deterministic in
    unit tests that don't set up explicit mock behaviour.
    """
    confidences = frozenset((eid, 0.95) for eid in candidate_set)
    return LiveJITResult(
        bibcode=bibcode,
        entity_ids=frozenset(candidate_set),
        confidences=confidences,
        model_version=model_version,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


RngFn = Callable[[], float]
LiveJITFn = Callable[[str, str, frozenset[int]], Awaitable[LiveJITResult]]
LocalNERFn = Callable[[str, frozenset[int]], Awaitable[LocalNERResult]]


async def route_jit(
    bibcode: str,
    text: str,
    candidate_set: frozenset[int],
    *,
    bulkhead: Optional[JITBulkhead] = None,
    rng: RngFn = random.random,
    live_jit_fn: Optional[LiveJITFn] = None,
    local_ner_fn: Optional[LocalNERFn] = None,
) -> RouterOutcome:
    """Route a JIT resolution request.

    Parameters
    ----------
    bibcode, text, candidate_set
        Inputs to the lane.
    bulkhead
        Optional :class:`JITBulkhead`. Constructed with defaults if not
        supplied.
    rng
        Random source used for canary selection. Tests inject a
        deterministic callable (e.g. ``lambda: 0.01`` to force canary).
    live_jit_fn
        Override for ``call_live_jit``. Used by tests to simulate a
        vendor outage (raise an exception) or a slow upstream.
    local_ner_fn
        Override for ``run_local_ner``. Used by tests to simulate local
        failure.

    Returns
    -------
    RouterOutcome
        One of :class:`LiveJITResult`, :class:`LocalNERResult`, or the
        :data:`STATIC_CORE_FALLBACK` sentinel.
    """
    bulkhead = bulkhead or JITBulkhead()
    live_fn = live_jit_fn or call_live_jit
    local_fn = local_ner_fn or _local_ner_adapter

    # --- canary branch -------------------------------------------------
    try:
        canary_roll = rng()
    except Exception as exc:  # noqa: BLE001
        logger.warning("router rng failed: %s — skipping canary", exc)
        canary_roll = 1.0

    if canary_roll < CANARY_SHARE:
        logger.debug("jit-router canary branch for bibcode=%s", bibcode)
        canary_result = await _safe_local(local_fn, text, candidate_set, bibcode)
        if canary_result is not None:
            return canary_result
        # Canary failed -> fall through to live_jit, still try bulkhead.

    # --- primary: live_jit under bulkhead ------------------------------
    async def _inner() -> LiveJITResult:
        return await live_fn(bibcode, text, candidate_set)

    bh_result = await bulkhead.run(_inner())

    if bh_result is not DEGRADED and isinstance(bh_result, LiveJITResult):
        return bh_result

    # --- fallback: local NER -------------------------------------------
    logger.info(
        "jit-router bulkhead degraded for bibcode=%s -> local NER fallback",
        bibcode,
    )
    local_result = await _safe_local(local_fn, text, candidate_set, bibcode)
    if local_result is not None:
        return local_result

    # --- final: static-core sentinel -----------------------------------
    logger.warning(
        "jit-router both live_jit and local_ner failed for bibcode=%s — " "static-core fallback",
        bibcode,
    )
    return STATIC_CORE_FALLBACK


async def _safe_local(
    local_fn: LocalNERFn,
    text: str,
    candidate_set: frozenset[int],
    bibcode: str,
) -> Optional[LocalNERResult]:
    try:
        return await local_fn(text, candidate_set)
    except Exception as exc:  # noqa: BLE001
        logger.warning("jit-router local NER failed for bibcode=%s: %s", bibcode, exc)
        return None


async def _local_ner_adapter(text: str, candidate_set: frozenset[int]) -> LocalNERResult:
    """Default local-NER callable used by :func:`route_jit`.

    We wrap :func:`run_local_ner` so the two-arg ``LocalNERFn`` signature
    matches — ``run_local_ner`` takes an optional ``bibcode`` we don't
    know at router time in the canary branch.
    """
    return await run_local_ner(text, candidate_set)


__all__ = [
    "CANARY_SHARE",
    "LiveJITResult",
    "RouterOutcome",
    "STATIC_CORE_FALLBACK",
    "call_live_jit",
    "route_jit",
]
