"""Local NER fallback (PRD §M11c).

In production this module wraps a SciBERT / INDUS-NER checkpoint running on
the RTX 5090 box. For u10 we ship a **deterministic CPU stub** that:

* accepts a document text + frozenset candidate set,
* returns one :class:`LocalNERResult` containing every candidate entity
  with fixed confidence ``0.75`` (the calibrated fallback confidence for
  the local lane — matches u04 tier-5 policy),
* simulates inference latency via a monkey-patchable ``_latency_seconds``
  module attribute so tests can assert the p95-≤-275ms SLO without
  actually burning wall time.

Real-deployment notes (out of scope for this unit)
--------------------------------------------------
* The production module will load the model lazily on first call and pin
  it to a single GPU via ``torch.cuda.set_device``.
* INDUS-NER inference on the full abstract is ~120ms on the 5090. With a
  40-token truncation guard we stay under 200ms p99 comfortably.
* For this stub we return confidence 0.75 for every candidate — this is
  intentional. The fusion layer (u08) will apply tier-weight calibration
  downstream; the router / cache just records what the lane produced.

Public API
----------
* :func:`run_local_ner` — async entry point used by :mod:`scix.jit.router`.
* :class:`LocalNERResult` — small dataclass the resolver wraps into a real
  :class:`scix.entity_link_set.EntityLinkSet` in a follow-up PR.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Tunables (monkey-patchable by tests)
# ---------------------------------------------------------------------------

#: Simulated per-call inference latency in seconds. The real backend
#: replaces this with actual torch inference. Tests set this to 0 to
#: measure pure overhead.
_latency_seconds: float = 0.0

#: Fixed confidence the stub returns for every candidate. Calibrated to
#: match u04 tier-5 defaults; real checkpoint will produce per-entity
#: confidences.
LOCAL_NER_CONFIDENCE: Final[float] = 0.75

#: Model version tag emitted with every result. Bumped when the checkpoint
#: on disk changes.
LOCAL_NER_MODEL_VERSION: Final[str] = "scibert-stub-v1"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalNERResult:
    """Output of :func:`run_local_ner`.

    This is a domain-layer DTO, not a full :class:`EntityLinkSet`. The
    resolver module wraps it on its way out so only ``resolve_entities``
    can mint a real link set.
    """

    bibcode: str
    entity_ids: frozenset[int]
    confidences: frozenset[tuple[int, float]] = field(default_factory=frozenset)
    lane: str = "local_ner"
    model_version: str = LOCAL_NER_MODEL_VERSION


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_local_ner(
    text: str,
    candidate_set: frozenset[int],
    *,
    bibcode: str = "",
) -> LocalNERResult:
    """Run the local NER stub.

    Parameters
    ----------
    text
        Raw document text. The stub does not inspect it — a real
        implementation will tokenise + run the model here.
    candidate_set
        Frozenset of candidate entity IDs to score. The stub echoes the
        candidate set as the result — production will return the subset
        actually detected in ``text``.
    bibcode
        Optional bibcode for the result record. Defaults to ``""`` because
        the local NER lane is text-first and callers may not always have
        a bibcode in hand.

    Returns
    -------
    LocalNERResult
        Frozen result with the echoed ID set at confidence 0.75.
    """
    if not isinstance(candidate_set, frozenset):
        raise TypeError("candidate_set must be a frozenset[int]")
    if _latency_seconds > 0:
        await asyncio.sleep(_latency_seconds)

    confidences = frozenset((eid, LOCAL_NER_CONFIDENCE) for eid in candidate_set)
    return LocalNERResult(
        bibcode=bibcode,
        entity_ids=frozenset(candidate_set),
        confidences=confidences,
        lane="local_ner",
        model_version=LOCAL_NER_MODEL_VERSION,
    )


def set_latency_for_tests(seconds: float) -> None:
    """Test hook: override the simulated inference latency.

    Kept as an explicit function so tests don't have to reach into the
    module's private attribute.
    """
    global _latency_seconds
    if seconds < 0:
        raise ValueError("latency must be >= 0")
    _latency_seconds = seconds


__all__ = [
    "LOCAL_NER_CONFIDENCE",
    "LOCAL_NER_MODEL_VERSION",
    "LocalNERResult",
    "run_local_ner",
    "set_latency_for_tests",
]
