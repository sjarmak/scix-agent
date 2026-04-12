"""Tests for :mod:`scix.jit.router` (PRD §M11c routing policy).

Asserts:

* Forced Haiku outage -> router falls back to local NER, NOT static-core.
* Haiku AND local NER both down -> router returns the static-core
  sentinel.
* Canary branch: rng()<0.05 forces the canary path even when Haiku is
  healthy, and the canary result is a :class:`LocalNERResult`.
* Happy path: Haiku up, canary roll >=0.05 -> router returns
  :class:`LiveJITResult`.
"""

from __future__ import annotations

import asyncio

import pytest

from scix.jit import local_ner as local_ner_mod
from scix.jit.bulkhead import JITBulkhead
from scix.jit.local_ner import LocalNERResult
from scix.jit.router import (
    CANARY_SHARE,
    LiveJITResult,
    STATIC_CORE_FALLBACK,
    route_jit,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _healthy_live(bibcode, text, cset):
    return LiveJITResult(
        bibcode=bibcode,
        entity_ids=frozenset(cset),
        confidences=frozenset((eid, 0.9) for eid in cset),
        model_version="haiku-v1",
    )


async def _outage_live(bibcode, text, cset):
    raise RuntimeError("simulated haiku outage")


async def _healthy_local(text, cset):
    return LocalNERResult(
        bibcode="",
        entity_ids=frozenset(cset),
        confidences=frozenset((eid, 0.75) for eid in cset),
    )


async def _broken_local(text, cset):
    raise RuntimeError("simulated local NER failure")


@pytest.fixture(autouse=True)
def _no_latency():
    local_ner_mod.set_latency_for_tests(0.0)
    yield
    local_ner_mod.set_latency_for_tests(0.0)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_router_happy_path_returns_live_jit():
    # rng=0.99 keeps us out of the canary branch.
    result = _run(
        route_jit(
            "2024ApJ",
            "abstract",
            frozenset({1, 2, 3}),
            rng=lambda: 0.99,
            live_jit_fn=_healthy_live,
            local_ner_fn=_healthy_local,
        )
    )
    assert isinstance(result, LiveJITResult)
    assert result.entity_ids == frozenset({1, 2, 3})


# ---------------------------------------------------------------------------
# Fallback on Haiku outage
# ---------------------------------------------------------------------------


def test_router_falls_back_to_local_on_haiku_outage():
    result = _run(
        route_jit(
            "2024ApJ",
            "abstract",
            frozenset({10, 20}),
            rng=lambda: 0.99,
            live_jit_fn=_outage_live,
            local_ner_fn=_healthy_local,
        )
    )
    assert isinstance(result, LocalNERResult)
    assert result.entity_ids == frozenset({10, 20})
    # Explicit spec assertion: NOT static-core.
    assert result is not STATIC_CORE_FALLBACK


# ---------------------------------------------------------------------------
# Both lanes down -> static-core
# ---------------------------------------------------------------------------


def test_router_returns_static_core_when_both_down():
    result = _run(
        route_jit(
            "2024ApJ",
            "abstract",
            frozenset({7}),
            rng=lambda: 0.99,
            live_jit_fn=_outage_live,
            local_ner_fn=_broken_local,
        )
    )
    assert result is STATIC_CORE_FALLBACK


# ---------------------------------------------------------------------------
# Canary branch
# ---------------------------------------------------------------------------


def test_router_canary_selects_local_on_low_roll():
    # rng < 0.05 forces canary -> local NER even with healthy Haiku.
    assert CANARY_SHARE == 0.05
    result = _run(
        route_jit(
            "2024ApJ",
            "abstract",
            frozenset({1}),
            rng=lambda: 0.01,
            live_jit_fn=_healthy_live,
            local_ner_fn=_healthy_local,
        )
    )
    assert isinstance(result, LocalNERResult)


def test_router_canary_falls_through_to_live_if_local_fails():
    # Canary fires, local fails -> router must still try live_jit.
    result = _run(
        route_jit(
            "2024ApJ",
            "abstract",
            frozenset({1}),
            rng=lambda: 0.01,
            live_jit_fn=_healthy_live,
            local_ner_fn=_broken_local,
        )
    )
    assert isinstance(result, LiveJITResult)
