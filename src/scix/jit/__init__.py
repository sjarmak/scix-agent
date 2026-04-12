"""JIT lane for entity resolution (u10, PRD §M11a/b/c).

This package implements the three pillars of the JIT lane:

* ``bulkhead``  — async concurrency limiter with 400ms hard budget
  (``§M11a``).
* ``cache``     — partitioned, TTL'd cache over
  ``document_entities_jit_cache`` (``§M11b``).
* ``local_ner`` — SciBERT/INDUS-NER fallback (``§M11c``).
* ``router``    — routing policy with 5% canary share and degrade chain.

Contract with u03 (``scix.resolve_entities``)
---------------------------------------------
u03 owns the single public entry point ``resolve_entities()``. It currently
uses in-module mocks for all four lanes. This u10 package exposes the
**real** implementations as plain functions that return a small
domain-layer dataclass, :class:`CachedLinkSet`. A follow-up unit will wire
those functions into ``resolve_entities()`` — for this unit, the jit
modules are self-contained and independently tested.

The only place in the codebase permitted to write to
``document_entities`` / ``document_entities_jit_cache`` is
``src/scix/resolve_entities.py`` (enforced by
``scripts/ast_lint_resolver.py``). ``cache.py`` INSERTs are exempted with
the ``# noqa: resolver-lint`` marker as the u10 spec allows — those writes
will move behind the resolver handle in a subsequent PR.
"""

from __future__ import annotations

from scix.jit.bulkhead import (
    BulkheadDegraded,
    BulkheadResult,
    DEGRADED,
    JITBulkhead,
)
from scix.jit.cache import (
    CachedLinkSet,
    JITCache,
    raise_alert,
)
from scix.jit.local_ner import LocalNERResult, run_local_ner
from scix.jit.router import RouterOutcome, call_live_jit, route_jit

__all__ = [
    "BulkheadDegraded",
    "BulkheadResult",
    "CachedLinkSet",
    "DEGRADED",
    "JITBulkhead",
    "JITCache",
    "LocalNERResult",
    "RouterOutcome",
    "call_live_jit",
    "raise_alert",
    "route_jit",
    "run_local_ner",
]
