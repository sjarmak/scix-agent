"""Integration tests for scix.llm_cost_ceiling.

Require SCIX_TEST_DSN to be set to a non-production database.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from helpers import is_production_dsn  # noqa: E402

from scix import llm_cost_ceiling  # noqa: E402
from scix.llm_cost_ceiling import (  # noqa: E402
    check_and_reserve,
    ensure_ledger,
    estimate_cost_usd,
    get_daily_total_usd,
    record_actual,
)

TEST_DSN = os.environ.get("SCIX_TEST_DSN")

pytestmark = pytest.mark.skipif(
    TEST_DSN is None or (TEST_DSN is not None and is_production_dsn(TEST_DSN)),
    reason="LLM ledger tests require SCIX_TEST_DSN (non-production)",
)


@pytest.fixture
def conn():
    assert TEST_DSN is not None
    c = psycopg.connect(TEST_DSN)
    c.autocommit = False
    try:
        yield c
    finally:
        c.close()


@pytest.fixture(autouse=True)
def _reset_ledger(conn):
    ensure_ledger(conn)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM llm_cost_ledger WHERE day = CURRENT_DATE")
    conn.commit()
    # reset thread-local reservation state
    llm_cost_ceiling._state.last_reservation_usd = 0.0
    yield
    with conn.cursor() as cur:
        cur.execute("DELETE FROM llm_cost_ledger WHERE day = CURRENT_DATE")
    conn.commit()


class TestEstimateCost:
    def test_zero_tokens_zero_cost(self):
        assert estimate_cost_usd(0, 0) == 0.0

    def test_haiku_rates(self):
        # 1M input tokens at $0.25/1M = $0.25
        assert estimate_cost_usd(1_000_000, 0) == pytest.approx(0.25)
        # 1M output tokens at $1.25/1M = $1.25
        assert estimate_cost_usd(0, 1_000_000) == pytest.approx(1.25)

    def test_negative_tokens_raises(self):
        with pytest.raises(ValueError):
            estimate_cost_usd(-1, 0)


class TestPerQueryCap:
    def test_over_per_query_cap_returns_false(self):
        assert check_and_reserve(0.02, dsn=TEST_DSN) is False
        # Ledger should be untouched
        assert get_daily_total_usd(dsn=TEST_DSN) == 0.0

    def test_at_per_query_cap_passes(self):
        assert check_and_reserve(0.01, dsn=TEST_DSN) is True
        assert get_daily_total_usd(dsn=TEST_DSN) == pytest.approx(0.01)


class TestDailyCap:
    def test_daily_cap_blocks_overflow(self):
        # Pre-fill ledger to just under the cap.
        with psycopg.connect(TEST_DSN) as c:
            with c.cursor() as cur:
                cur.execute(
                    "INSERT INTO llm_cost_ledger (day, total_usd, call_count) "
                    "VALUES (CURRENT_DATE, 49.990, 9999) "
                    "ON CONFLICT (day) DO UPDATE SET total_usd = 49.990"
                )
            c.commit()

        # A 0.005 call fits (49.990 + 0.005 = 49.995 < 50.0).
        assert check_and_reserve(0.005, dsn=TEST_DSN) is True
        # Next 0.01 call would bring total to 50.005 > 50.0 cap.
        assert check_and_reserve(0.01, dsn=TEST_DSN) is False

    def test_under_cap_accumulates(self):
        assert check_and_reserve(0.005, dsn=TEST_DSN) is True
        assert check_and_reserve(0.005, dsn=TEST_DSN) is True
        assert get_daily_total_usd(dsn=TEST_DSN) == pytest.approx(0.010)


class TestRecordActual:
    def test_record_actual_adjusts_delta(self):
        # Reserve 0.008, record actual 0.006 → ledger should end at 0.006.
        assert check_and_reserve(0.008, dsn=TEST_DSN) is True
        assert get_daily_total_usd(dsn=TEST_DSN) == pytest.approx(0.008)
        record_actual(0.006, dsn=TEST_DSN)
        assert get_daily_total_usd(dsn=TEST_DSN) == pytest.approx(0.006)

    def test_record_actual_negative_raises(self):
        with pytest.raises(ValueError):
            record_actual(-0.01, dsn=TEST_DSN)
