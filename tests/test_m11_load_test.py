"""Unit tests for M11 JIT load test script.

All tests mock DB access — no real database required.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import m11_load_test as lt

# ---------------------------------------------------------------------------
# 1. Lognormal calibration
# ---------------------------------------------------------------------------


class TestLognormalCalibration:
    """Verify the fault injection latency distribution matches spec."""

    def test_p50_within_tolerance(self):
        samples = sorted(lt.sample_latency() for _ in range(10_000))
        p50 = samples[len(samples) // 2]
        assert 0.200 < p50 < 0.450, f"p50={p50:.3f}s, expected ~0.300s"

    def test_p99_within_tolerance(self):
        samples = sorted(lt.sample_latency() for _ in range(10_000))
        p99 = samples[int(len(samples) * 0.99)]
        assert 1.5 < p99 < 4.0, f"p99={p99:.3f}s, expected ~2.500s"

    def test_all_positive(self):
        samples = [lt.sample_latency() for _ in range(1_000)]
        assert all(s > 0 for s in samples)


# ---------------------------------------------------------------------------
# 2. Error rate
# ---------------------------------------------------------------------------


class TestErrorRate:
    """Verify the fault injection error rate matches spec."""

    def test_error_rate_within_bounds(self):
        errors = sum(1 for _ in range(10_000) if lt.should_inject_error())
        rate = errors / 10_000
        assert 0.03 < rate < 0.07, f"error_rate={rate:.3f}, expected ~0.05"


# ---------------------------------------------------------------------------
# 3. Drift computation
# ---------------------------------------------------------------------------


class TestDriftComputation:
    """Verify p95 drift calculation and pass/fail classification."""

    def test_zero_drift(self):
        baseline = [10.0] * 100
        loaded = [10.0] * 100
        result = lt.compute_drift("test_tool", baseline, loaded, threshold=0.10)
        assert result.drift_pct == pytest.approx(0.0, abs=0.001)
        assert result.passed is True

    def test_just_under_threshold(self):
        baseline = list(range(1, 101))  # p95 = 95
        loaded = [x * 1.09 for x in range(1, 101)]  # p95 = 103.55, drift ~9%
        result = lt.compute_drift("test_tool", baseline, loaded, threshold=0.10)
        assert result.drift_pct < 0.10
        assert result.passed is True

    def test_over_threshold(self):
        baseline = list(range(1, 101))  # p95 = 95
        loaded = [x * 1.20 for x in range(1, 101)]  # p95 = 114
        result = lt.compute_drift("test_tool", baseline, loaded, threshold=0.10)
        assert result.drift_pct > 0.10
        assert result.passed is False

    def test_negative_drift_passes(self):
        """Loaded being faster than baseline is fine."""
        baseline = [10.0] * 100
        loaded = [8.0] * 100
        result = lt.compute_drift("test_tool", baseline, loaded, threshold=0.10)
        assert result.drift_pct < 0
        assert result.passed is True


# ---------------------------------------------------------------------------
# 4. Percentile
# ---------------------------------------------------------------------------


class TestPercentile:
    """Verify percentile calculation on small lists."""

    def test_p50_odd_list(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert lt.percentile(data, 0.50) == pytest.approx(3.0)

    def test_p95_100_elements(self):
        data = list(range(1, 101))  # 1..100
        assert lt.percentile(data, 0.95) == pytest.approx(95.0, abs=1.0)

    def test_p99_100_elements(self):
        data = list(range(1, 101))
        assert lt.percentile(data, 0.99) == pytest.approx(99.0, abs=1.0)

    def test_single_element(self):
        assert lt.percentile([42.0], 0.95) == 42.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            lt.percentile([], 0.50)


# ---------------------------------------------------------------------------
# 5. Report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    """Verify markdown report contains required sections and verdicts."""

    def _make_report(self, pass_all: bool = True) -> str:
        drift_mult = 1.05 if pass_all else 1.25
        baseline = {
            "bm25_search": [10.0] * 100,
            "vector_search": [20.0] * 100,
            "citation_chain": [15.0] * 100,
        }
        loaded = {tool: [v * drift_mult for v in lats] for tool, lats in baseline.items()}
        drifts = [lt.compute_drift(tool, baseline[tool], loaded[tool]) for tool in baseline]
        jit_stats = lt.JITStats(
            total_calls=500,
            bulkhead_degrades=340,
            errors_injected=25,
            local_ner_fallbacks=340,
            mean_latency_ms=455.0,
        )
        report = lt.LoadTestReport(
            baseline=baseline,
            loaded=loaded,
            drifts=drifts,
            jit_stats=jit_stats,
            asyncio_warnings=[],
            config=lt.LoadTestConfig(),
        )
        return lt.generate_report(report)

    def test_contains_required_sections(self):
        md = self._make_report()
        for section in [
            "# M11 JIT Load Test Report",
            "## Baseline",
            "## Loaded",
            "## P95 Drift Analysis",
            "## JIT Load Statistics",
            "## asyncio.debug Traces",
            "## Verdict",
        ]:
            assert section in md, f"Missing section: {section}"

    def test_pass_verdict(self):
        md = self._make_report(pass_all=True)
        assert "PASS" in md

    def test_fail_verdict(self):
        md = self._make_report(pass_all=False)
        assert "FAIL" in md

    def test_contains_tool_names(self):
        md = self._make_report()
        for tool in ["bm25_search", "vector_search", "citation_chain"]:
            assert tool in md


# ---------------------------------------------------------------------------
# 6. JIT load loop
# ---------------------------------------------------------------------------


class TestJITLoadLoop:
    """Verify the JIT load loop respects stop_event and increments counters."""

    def test_terminates_on_stop_event(self):
        stats = lt.JITStats()
        stop_event = asyncio.Event()

        async def _test():
            async def _stop_after_one():
                # Let the loop run one iteration before stopping
                await asyncio.sleep(0.01)
                stop_event.set()

            await asyncio.gather(
                lt.jit_load_loop(
                    stop_event=stop_event,
                    bulkhead=None,
                    stats=stats,
                    candidate_set=frozenset({1, 2, 3}),
                ),
                _stop_after_one(),
            )

        asyncio.run(_test())
        assert stats.total_calls >= 1

    def test_counts_degrades(self):
        """Run a few iterations with a bulkhead that always degrades."""
        from scix.jit.bulkhead import JITBulkhead

        stats = lt.JITStats()
        stop_event = asyncio.Event()

        async def _test():
            # Create a bulkhead with very short budget to force degrades
            bh = JITBulkhead(concurrency=1, budget_ms=1)

            async def _slow_jit(bibcode, text, candidate_set, **kw):
                await asyncio.sleep(0.1)  # 100ms >> 1ms budget
                return MagicMock()

            # Run for a short time then stop
            async def _stop_later():
                await asyncio.sleep(0.05)
                stop_event.set()

            await asyncio.gather(
                lt.jit_load_loop(
                    stop_event=stop_event,
                    bulkhead=bh,
                    stats=stats,
                    candidate_set=frozenset({1, 2}),
                    live_jit_fn=_slow_jit,
                ),
                _stop_later(),
            )

        asyncio.run(_test())
        assert stats.total_calls >= 1
        # route_jit catches bulkhead DEGRADED and falls through to local NER,
        # so we see local_ner_fallbacks, not bulkhead_degrades directly.
        assert stats.local_ner_fallbacks >= 1


# ---------------------------------------------------------------------------
# 7. Measure tool wrapper
# ---------------------------------------------------------------------------


class TestMeasureTool:
    """Verify the measurement wrapper correctly records timings."""

    def test_records_latencies(self):
        def fake_tool(conn, query, **kwargs):
            import time

            time.sleep(0.001)  # ~1ms
            return MagicMock()

        async def _test():
            latencies = await lt.measure_tool(
                name="test_tool",
                tool_fn=fake_tool,
                queries=["q1", "q2", "q3"],
                conn=MagicMock(),
                iterations=3,
                warmup=0,
            )
            return latencies

        lats = asyncio.run(_test())
        assert len(lats) == 3
        assert all(lat > 0 for lat in lats)

    def test_warmup_excluded(self):
        calls = []

        def fake_tool(conn, query, **kwargs):
            calls.append(query)
            return MagicMock()

        async def _test():
            return await lt.measure_tool(
                name="test_tool",
                tool_fn=fake_tool,
                queries=["q1", "q2"],
                conn=MagicMock(),
                iterations=3,
                warmup=2,
            )

        lats = asyncio.run(_test())
        # 2 warmup + 3 measured = 5 total calls
        assert len(calls) == 5
        # Only 3 latencies returned (warmup excluded)
        assert len(lats) == 3
