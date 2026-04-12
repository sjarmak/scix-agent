"""Benchmark suite for scix.resolve_entities per-lane latency.

Runs N calls per lane with mocked backend latencies, computes p95, writes a
markdown report to ``build-artifacts/m13_latency.md``, and asserts budgets.
All measurements are against the u03 in-module mocks — u08 / u10 will
replace the lanes with real backends at which point these budgets will be
re-tightened against production pgvector / Anthropic latencies.

Budgets (spec §M13 acceptance criterion 6):
    static       p95 ≤ 5ms      (real budget)
    jit_cache    p95 ≤ 25ms     (real budget)
    live_jit     p95 ≤ 80ms     (mock budget; real budget is ~300ms)
    local_ner    p95 ≤ 60ms     (mock budget; real budget is ~200ms)
"""

from __future__ import annotations

import statistics
import time
from dataclasses import replace
from pathlib import Path

import pytest

from scix import resolve_entities as re_mod
from scix.resolve_entities import (
    EntityResolveContext,
    candidate_set_hash,
    resolve_entities,
)

N_SAMPLES = 50

# Mock per-lane latencies (seconds). Deliberately small so the benchmark
# suite runs quickly in CI but still produces meaningful p95 signals.
MOCK_LATENCY_S = {
    "static": 0.0005,
    "jit_cache_hit": 0.002,
    "live_jit": 0.015,
    "local_ner": 0.010,
}

# p95 budgets (seconds). Static + jit_cache match the spec's real budgets;
# live_jit / local_ner use mock-level budgets.
P95_BUDGETS_S = {
    "static": 0.005,
    "jit_cache_hit": 0.025,
    "live_jit": 0.080,
    "local_ner": 0.060,
}

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "build-artifacts" / "m13_latency.md"


def _p95(samples: list[float]) -> float:
    """Return the 95th-percentile value from ``samples``."""
    if not samples:
        return 0.0
    if len(samples) < 20:
        # statistics.quantiles needs ≥2 data points but the resolution for
        # very small samples is coarse — fall back to a nearest-rank p95.
        ordered = sorted(samples)
        idx = max(0, int(round(0.95 * (len(ordered) - 1))))
        return ordered[idx]
    return statistics.quantiles(samples, n=20)[-1]


@pytest.fixture(autouse=True)
def _seed_and_reset():
    re_mod._reset_mocks()
    # Wire the deterministic mock latencies.
    for lane, delay in MOCK_LATENCY_S.items():
        re_mod._LANE_LATENCIES[lane] = delay
    yield
    # Clear latencies so other tests aren't slowed down.
    for lane in re_mod._LANE_LATENCIES:
        re_mod._LANE_LATENCIES[lane] = 0.0
    re_mod._reset_mocks()


def _measure(lane: str) -> list[float]:
    bibcode = f"bench_{lane}"
    ids = frozenset({1, 2, 3, 4})
    ctx_base = EntityResolveContext(
        candidate_set=ids,
        mode="static",  # overridden per lane below
        model_version="v1",
    )
    cset_hash = candidate_set_hash(ctx_base)

    # Seed the right mock for this lane
    seed_ids = frozenset({10, 20, 30, 40})
    if lane == "static":
        re_mod._seed_static(bibcode, seed_ids)
        mode = "static"
    elif lane == "jit_cache_hit":
        re_mod._seed_jit_cache(bibcode, cset_hash, "v1", seed_ids)
        mode = "jit"
    elif lane == "live_jit":
        re_mod._seed_live_jit(bibcode, cset_hash, seed_ids)
        mode = "live_jit"
    elif lane == "local_ner":
        re_mod._seed_local_ner(bibcode, seed_ids)
        mode = "local_ner"
    else:  # pragma: no cover - defensive
        raise AssertionError(f"unknown lane {lane}")

    ctx = replace(ctx_base, mode=mode)  # type: ignore[arg-type]
    samples: list[float] = []
    for _ in range(N_SAMPLES):
        t0 = time.perf_counter()
        result = resolve_entities(bibcode, ctx)
        elapsed = time.perf_counter() - t0
        assert result.entity_ids() == seed_ids
        samples.append(elapsed)
    return samples


def _write_report(per_lane_samples: dict[str, list[float]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# M13 resolve_entities() per-lane latency (u03 mocks)",
        "",
        f"Samples per lane: **{N_SAMPLES}**. All backends are u03 in-module ",
        "mocks with deterministic injected latency — real backends arrive in ",
        "u08 (static) and u10 (jit_cache / live_jit).",
        "",
        "| lane | mock_latency_ms | p50_ms | p95_ms | p99_ms | budget_p95_ms | status |",
        "|------|-----------------|--------|--------|--------|---------------|--------|",
    ]
    for lane, samples in per_lane_samples.items():
        p50 = statistics.median(samples) * 1000
        p95 = _p95(samples) * 1000
        ordered = sorted(samples)
        p99 = ordered[min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))] * 1000
        budget_ms = P95_BUDGETS_S[lane] * 1000
        status = "PASS" if (p95 / 1000) <= P95_BUDGETS_S[lane] else "FAIL"
        mock_ms = MOCK_LATENCY_S[lane] * 1000
        lines.append(
            f"| {lane} | {mock_ms:.1f} | {p50:.2f} | {p95:.2f} | {p99:.2f} | {budget_ms:.1f} | {status} |"
        )
    lines.append("")
    lines.append(
        "These mock-level budgets will be retightened against real backends "
        "once u08 / u10 land. The 5ms / 25ms budgets for static / jit_cache "
        "match the PRD §M13 acceptance criteria against real pgvector."
    )
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def test_benchmark_per_lane_p95_within_budget():
    per_lane = {lane: _measure(lane) for lane in P95_BUDGETS_S}
    _write_report(per_lane)
    for lane, samples in per_lane.items():
        p95 = _p95(samples)
        budget = P95_BUDGETS_S[lane]
        assert p95 <= budget, (
            f"lane={lane} p95={p95 * 1000:.2f}ms exceeds budget "
            f"{budget * 1000:.2f}ms (samples: n={len(samples)})"
        )
