"""Tests for :mod:`scix.jit.local_ner` (PRD §M11c).

Acceptance criterion 4: ``p95 latency <= 275ms on stub``. We run 20 stub
invocations with a configurable simulated latency (<= 275ms) and verify
the sorted 95th percentile stays under the SLO.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from scix.jit import local_ner
from scix.jit.local_ner import (
    LOCAL_NER_CONFIDENCE,
    LOCAL_NER_MODEL_VERSION,
    LocalNERResult,
    run_local_ner,
    set_latency_for_tests,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_local_ner_echoes_candidate_set_at_fixed_confidence():
    candidates = frozenset({101, 202, 303})

    set_latency_for_tests(0.0)
    result = _run(run_local_ner("some text", candidates, bibcode="2024ApJ...1A"))

    assert isinstance(result, LocalNERResult)
    assert result.bibcode == "2024ApJ...1A"
    assert result.entity_ids == candidates
    assert result.model_version == LOCAL_NER_MODEL_VERSION
    assert result.lane == "local_ner"
    conf_map = dict(result.confidences)
    for eid in candidates:
        assert conf_map[eid] == LOCAL_NER_CONFIDENCE


def test_local_ner_rejects_non_frozenset():
    with pytest.raises(TypeError):
        _run(run_local_ner("txt", {1, 2, 3}))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# p95 latency SLO
# ---------------------------------------------------------------------------


def test_local_ner_p95_under_275ms():
    # Simulate a realistic inference latency that is still under SLO.
    set_latency_for_tests(0.050)  # 50 ms per call

    try:
        samples: list[float] = []

        async def _one():
            t0 = time.monotonic()
            await run_local_ner("text", frozenset({1, 2, 3}))
            samples.append(time.monotonic() - t0)

        async def _go():
            for _ in range(20):
                await _one()

        _run(_go())

        samples_sorted = sorted(samples)
        # 95th percentile on 20 samples -> index 18 (0-based -> 19th)
        p95 = samples_sorted[int(0.95 * len(samples_sorted)) - 1]
        assert p95 <= 0.275, f"p95 latency {p95*1000:.1f}ms exceeded 275ms SLO"
    finally:
        set_latency_for_tests(0.0)
