"""Unit tests for the demo_readiness_smoke fallback anchor query.

Pins the SQL shape of the fallback anchor lookup (bead scix_experiments-zwcq):
the old `title ILIKE '%...%' ORDER BY citation_count` fallback forced a full
seq scan over 32M papers — the same query family as the claim_blame seed that
OOM-killed postgres (bead scix_experiments-82j0). The fallback must use the
GIN-indexed tsvector lane and bound its session resources via SET LOCAL.

No live database: a fake connection records executed SQL.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from scripts.demo_readiness_smoke import (
    FALLBACK_ANCHOR_SESSION_CAPS,
    FALLBACK_ANCHOR_SQL,
    fallback_anchor,
)


class _FakeCursor:
    def __init__(self, row: tuple[Any, ...] | None = None) -> None:
        self._row = row
        self.executed: list[str] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append(sql)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor

    @contextmanager
    def transaction(self) -> Any:
        yield


def test_fallback_anchor_uses_tsvector_lane_not_ilike() -> None:
    """The fallback query must hit papers.tsv, never ILIKE over title."""
    assert "tsv @@ plainto_tsquery('scix_english'" in FALLBACK_ANCHOR_SQL
    assert "ilike" not in FALLBACK_ANCHOR_SQL.lower()

    cur = _FakeCursor()
    fallback_anchor(_FakeConnection(cur), "machine learning")
    main_queries = [s for s in cur.executed if not s.startswith("SET LOCAL")]
    assert main_queries == [FALLBACK_ANCHOR_SQL]


def test_fallback_anchor_sets_session_resource_caps() -> None:
    """SET LOCAL caps per the 2026-06-11 postmaster-OOM rule, before the query."""
    cur = _FakeCursor()
    fallback_anchor(_FakeConnection(cur), "machine learning")

    set_locals = [s for s in cur.executed if s.startswith("SET LOCAL")]
    assert set_locals == list(FALLBACK_ANCHOR_SESSION_CAPS)
    joined = " ".join(set_locals)
    assert "statement_timeout" in joined
    assert "max_parallel_workers_per_gather = 0" in joined
    assert "work_mem = '256MB'" in joined
    # Caps must run before the main query inside the same transaction.
    assert cur.executed[-1] == FALLBACK_ANCHOR_SQL


def test_fallback_anchor_returns_bibcode_or_none() -> None:
    hit = fallback_anchor(_FakeConnection(_FakeCursor(("2020ApJ...1B",))), "galaxies")
    assert hit == "2020ApJ...1B"

    miss = fallback_anchor(_FakeConnection(_FakeCursor(None)), "galaxies")
    assert miss is None
