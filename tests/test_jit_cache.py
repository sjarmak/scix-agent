"""Tests for :mod:`scix.jit.cache` (PRD §M11b acceptance criterion 3 + 6).

Asserts:

* ``put`` enqueues rows and the background writer drains them to the
  ``document_entities_jit_cache`` partitioned table.
* ``get`` returns a ``CachedLinkSet`` for a warm key and ``None`` for a
  cold one.
* Queue saturation (``maxsize=1``) drops new puts and two consecutive
  drops fires the pager alert hook (monkey-patched).
* A "bulkhead-degraded" write is **not** sent to ``put`` — criterion 6
  forces the degraded path and asserts the cache row count did not move.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from scix.jit import cache as cache_mod
from scix.jit.bulkhead import DEGRADED, JITBulkhead
from scix.jit.cache import CachedLinkSet, JITCache

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------


TEST_DSN = os.environ.get("SCIX_TEST_DSN", "dbname=scix_test")
MIGRATION_PATH = Path(__file__).parent.parent / "migrations" / "034_jit_cache.sql"


def _conn():
    return psycopg.connect(TEST_DSN)


def _apply_migration_if_needed() -> None:
    sql = MIGRATION_PATH.read_text()
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def _can_reach_test_db() -> bool:
    try:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _can_reach_test_db(),
    reason="scix_test database not reachable — set SCIX_TEST_DSN",
)


@pytest.fixture(scope="module", autouse=True)
def _migration():
    _apply_migration_if_needed()
    yield


@pytest.fixture(autouse=True)
def _reset_drop_state():
    cache_mod._reset_drop_state()
    yield
    cache_mod._reset_drop_state()


def _run(coro):
    return asyncio.run(coro)


def _count_rows(bibcode: str) -> int:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM document_entities_jit_cache WHERE bibcode = %s",
                (bibcode,),
            )
            return int(cur.fetchone()[0])


def _cleanup_bibcode(bibcode: str) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_entities_jit_cache_default WHERE bibcode = %s",
                (bibcode,),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Basic put/get
# ---------------------------------------------------------------------------


def test_put_then_get_round_trip():
    bibcode = f"test-jit-{uuid.uuid4().hex}"
    _cleanup_bibcode(bibcode)

    cache = JITCache(_conn)
    row = CachedLinkSet(
        bibcode=bibcode,
        candidate_set_hash="hash-a",
        model_version="haiku-v1",
        entity_ids=frozenset({1, 2, 3}),
        confidences=frozenset({(1, 0.9), (2, 0.8), (3, 0.7)}),
    )

    async def _go():
        assert cache.put(row) is True
        written = await cache.drain_once()
        return written

    written = _run(_go())
    assert written == 1

    got = cache.get(bibcode, "hash-a", "haiku-v1")
    assert got is not None
    assert got.entity_ids == frozenset({1, 2, 3})
    conf_map = dict(got.confidences)
    assert conf_map[1] == pytest.approx(0.9, abs=1e-4)

    # Cold key returns None.
    miss = cache.get(bibcode, "does-not-exist", "haiku-v1")
    assert miss is None

    _cleanup_bibcode(bibcode)


# ---------------------------------------------------------------------------
# Queue saturation + pager alert on 2 consecutive drops
# ---------------------------------------------------------------------------


def test_queue_saturation_drops_and_alerts(monkeypatch):
    alerts: list[str] = []

    def fake_alert(msg: str) -> None:
        alerts.append(msg)

    monkeypatch.setattr(cache_mod, "raise_alert", fake_alert)

    bibcode = f"test-jit-{uuid.uuid4().hex}"
    _cleanup_bibcode(bibcode)

    # Tiny queue so we can saturate with 1 row + 2 drops.
    cache = JITCache(_conn, queue_maxsize=1)
    row = CachedLinkSet(
        bibcode=bibcode,
        candidate_set_hash="h",
        model_version="v",
        entity_ids=frozenset({42}),
    )

    # First put: accepted.
    assert cache.put(row) is True
    # Second put: queue full -> drop #1.
    assert cache.put(row) is False
    # No alert yet — one drop is not enough.
    assert alerts == []
    # Third put: drop #2 -> pager alert fires.
    assert cache.put(row) is False
    assert len(alerts) == 1
    assert "consecutive drops" in alerts[0]

    _cleanup_bibcode(bibcode)


# ---------------------------------------------------------------------------
# Criterion 6: degraded writes are NOT cached
# ---------------------------------------------------------------------------


def test_bulkhead_degraded_is_not_written_to_cache():
    bibcode = f"test-jit-{uuid.uuid4().hex}"
    _cleanup_bibcode(bibcode)

    cache = JITCache(_conn)
    before = _count_rows(bibcode)
    assert before == 0

    bh = JITBulkhead(concurrency=2, budget_ms=100)

    async def slow():
        await asyncio.sleep(1.0)
        return CachedLinkSet(
            bibcode=bibcode,
            candidate_set_hash="h",
            model_version="v",
            entity_ids=frozenset({99}),
        )

    async def _go():
        result = await bh.run(slow())
        # Correct caller behaviour: never cache a DEGRADED response.
        if result is not DEGRADED:
            cache.put(result)
            await cache.drain_once()
        return result

    result = _run(_go())
    assert result is DEGRADED

    after = _count_rows(bibcode)
    assert after == before, "degraded response must not be written to the cache"

    _cleanup_bibcode(bibcode)


# ---------------------------------------------------------------------------
# Expired rows are hidden from get()
# ---------------------------------------------------------------------------


def test_get_ignores_expired_rows():
    bibcode = f"test-jit-{uuid.uuid4().hex}"
    _cleanup_bibcode(bibcode)

    cache = JITCache(_conn)
    expired_row = CachedLinkSet(
        bibcode=bibcode,
        candidate_set_hash="h",
        model_version="v",
        entity_ids=frozenset({1}),
        confidences=frozenset({(1, 0.5)}),
        expires_at=datetime.now(timezone.utc) - timedelta(days=1),
    )

    async def _go():
        cache.put(expired_row)
        return await cache.drain_once()

    _run(_go())

    got = cache.get(bibcode, "h", "v")
    assert got is None

    _cleanup_bibcode(bibcode)
