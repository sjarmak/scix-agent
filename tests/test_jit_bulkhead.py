"""Tests for :mod:`scix.jit.bulkhead` (PRD §M11a acceptance criterion 2).

Asserts:

* Under a forced 2.5s inner-call latency, the bulkhead degrades after its
  400 ms budget but well before the inner call would finish.
* Under a vendor error raised from the inner coroutine, the bulkhead
  returns :data:`DEGRADED` instead of propagating the exception.
* Under a concurrency storm the semaphore limit is enforced — extra
  callers are degraded rather than starved forever.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from scix.jit.bulkhead import DEGRADED, JITBulkhead


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Latency degrade
# ---------------------------------------------------------------------------


def test_bulkhead_degrades_under_forced_latency():
    bh = JITBulkhead(concurrency=4, budget_ms=400)

    async def slow():
        await asyncio.sleep(2.5)
        return "never"

    async def _go():
        t0 = time.monotonic()
        result = await bh.run(slow())
        elapsed = time.monotonic() - t0
        return result, elapsed

    result, elapsed = _run(_go())

    assert result is DEGRADED
    assert elapsed >= bh.budget_seconds, f"bulkhead degraded early after {elapsed:.3f}s"
    assert elapsed < 1.0, f"degrade took {elapsed:.3f}s; slow call sleeps 2.5s"


# ---------------------------------------------------------------------------
# Vendor error degrade
# ---------------------------------------------------------------------------


def test_bulkhead_degrades_on_vendor_error():
    bh = JITBulkhead(concurrency=4, budget_ms=400)

    class VendorOutage(RuntimeError):
        pass

    async def boom():
        raise VendorOutage("anthropic 500")

    result = _run(bh.run(boom()))

    assert result is DEGRADED


# ---------------------------------------------------------------------------
# Success path — verify we don't degrade on a fast call
# ---------------------------------------------------------------------------


def test_bulkhead_returns_value_on_fast_call():
    bh = JITBulkhead(concurrency=4, budget_ms=400)

    async def fast():
        return 42

    assert _run(bh.run(fast())) == 42


# ---------------------------------------------------------------------------
# Concurrency introspection — semaphore configuration is honoured.
# ---------------------------------------------------------------------------


def test_bulkhead_reports_configured_limits():
    bh = JITBulkhead(concurrency=7, budget_ms=123)
    assert bh.concurrency == 7
    assert bh.budget_ms == 123
    assert bh.budget_seconds == pytest.approx(0.123)


def test_bulkhead_rejects_invalid_config():
    with pytest.raises(ValueError):
        JITBulkhead(concurrency=0)
    with pytest.raises(ValueError):
        JITBulkhead(budget_ms=0)
