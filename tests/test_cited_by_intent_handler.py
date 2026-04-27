"""Tests for the cited_by_intent MCP handler.

Covers two recently added behaviours on top of the existing intent filter:

1. Result deduplication — a single citing paper that has multiple matching
   citation_contexts rows must collapse to one ``papers`` entry, with a
   ``n_contexts`` field reporting the per-source-paper count.
2. Coverage block — every response carries the standard
   ``citation_contexts`` coverage block so agents can tell 'no events'
   apart from 'no coverage' (the same distinction surfaced by
   ``claim_blame`` and ``find_replications``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scix.citation_contexts_coverage import COVERAGE_DOC_PATH
from scix.mcp_server import _handle_cited_by_intent


# ---------------------------------------------------------------------------
# Minimal fake DB plumbing — we recognise two query shapes:
#   * coverage probe (COUNT(DISTINCT bib) ... v_claim_edges)
#   * the cited_by_intent SELECT (FROM citation_contexts cc)
# ---------------------------------------------------------------------------


_COLUMNS = (
    "source_bibcode",
    "intent",
    "context_excerpt",
    "title",
    "year",
    "first_author",
    "citation_count",
    "n_contexts",
)


class _Desc:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeCursor:
    def __init__(
        self,
        rows: list[tuple[Any, ...]] | None = None,
        coverage_bibcodes: set[str] | None = None,
    ) -> None:
        self._rows = rows or []
        self._coverage = coverage_bibcodes or set()
        self._last_rows: list[tuple[Any, ...]] = []
        self.description: list[_Desc] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        sql_lower = sql.lower()
        if "count(distinct" in sql_lower and "v_claim_edges" in sql_lower:
            seeds = list(params[0]) if params else []
            covered = sum(1 for s in seeds if s in self._coverage)
            self._last_rows = [(covered,)]
            self.description = [_Desc("count")]
            return
        # cited_by_intent SELECT
        self._last_rows = list(self._rows)
        self.description = [_Desc(c) for c in _COLUMNS]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._last_rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._last_rows[0] if self._last_rows else None


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor


def _row(
    source_bibcode: str,
    intent: str,
    *,
    title: str = "T",
    year: int = 2020,
    first_author: str = "Doe, J.",
    citation_count: int = 0,
    n_contexts: int = 1,
    excerpt: str = "ctx",
) -> tuple[Any, ...]:
    return (
        source_bibcode,
        intent,
        excerpt,
        title,
        year,
        first_author,
        citation_count,
        n_contexts,
    )


# ---------------------------------------------------------------------------
# Coverage block — present in every response shape
# ---------------------------------------------------------------------------


def _coverage_keys() -> set[str]:
    return {"covered_seeds", "total_seeds", "coverage_pct", "note"}


def test_coverage_block_when_target_uncovered() -> None:
    """No rows + target not in v_claim_edges → coverage block reflects it."""
    cur = _FakeCursor(rows=[], coverage_bibcodes=set())
    conn = _FakeConnection(cur)

    out = json.loads(
        _handle_cited_by_intent(conn, {"target_bibcode": "2099UNCOVERED.....A"})
    )

    assert "coverage" in out
    assert _coverage_keys() <= set(out["coverage"].keys())
    assert out["coverage"]["covered_seeds"] == 0
    assert out["coverage"]["total_seeds"] == 1
    assert out["coverage"]["coverage_pct"] == 0.0
    assert COVERAGE_DOC_PATH in out["coverage"]["note"]
    assert out["papers"] == []
    assert out["total"] == 0


def test_coverage_block_when_target_covered_but_no_events() -> None:
    """Target IS in v_claim_edges but happens to have zero matching rows
    for the requested intent — coverage still 1.0 so the agent can
    trust the empty result."""
    cur = _FakeCursor(rows=[], coverage_bibcodes={"2010COVERED.....A"})
    conn = _FakeConnection(cur)

    out = json.loads(
        _handle_cited_by_intent(
            conn,
            {"target_bibcode": "2010COVERED.....A", "intent": "result_comparison"},
        )
    )
    assert out["coverage"]["covered_seeds"] == 1
    assert out["coverage"]["total_seeds"] == 1
    assert out["coverage"]["coverage_pct"] == pytest.approx(1.0)
    assert out["papers"] == []


def test_coverage_block_when_target_covered_with_events() -> None:
    cur = _FakeCursor(
        rows=[_row("2024CITER.....A", "method", citation_count=42)],
        coverage_bibcodes={"2010ORIGIN.....A"},
    )
    conn = _FakeConnection(cur)
    out = json.loads(
        _handle_cited_by_intent(
            conn,
            {"target_bibcode": "2010ORIGIN.....A", "intent": "method"},
        )
    )
    assert out["coverage"]["coverage_pct"] == pytest.approx(1.0)
    assert len(out["papers"]) == 1


# ---------------------------------------------------------------------------
# Dedup — the SQL CTE collapses (source_bibcode) per target. The fake
# cursor doesn't run the CTE, but we exercise the contract: when the
# handler is called it asks the DB for already-deduped rows that include
# n_contexts, and surfaces them on the response.
# ---------------------------------------------------------------------------


def test_response_surfaces_n_contexts_field() -> None:
    """When the same citing paper has many matching contexts, the
    handler exposes that as a single row with n_contexts set."""
    cur = _FakeCursor(
        rows=[
            _row("2004CITERA....A", "method", citation_count=204, n_contexts=7),
            _row("2002CITERB....A", "method", citation_count=48, n_contexts=2),
        ],
        coverage_bibcodes={"1996RvMP...68.1259J"},
    )
    conn = _FakeConnection(cur)

    out = json.loads(
        _handle_cited_by_intent(
            conn, {"target_bibcode": "1996RvMP...68.1259J", "intent": "method"}
        )
    )
    assert out["total"] == 2
    bibs = {p["source_bibcode"]: p for p in out["papers"]}
    assert bibs["2004CITERA....A"]["n_contexts"] == 7
    assert bibs["2002CITERB....A"]["n_contexts"] == 2


def test_dedup_sql_uses_row_number_and_partition_by_source() -> None:
    """Pin the dedup contract to the SQL — protects against a future edit
    that drops the CTE and lets duplicate source_bibcode rows leak back
    into the response."""

    captured: dict[str, str] = {}

    class _CapturingCursor(_FakeCursor):
        def execute(self, sql: str, params: Any = None) -> None:
            sql_lower = sql.lower()
            if "from citation_contexts" in sql_lower:
                captured["sql"] = sql
            super().execute(sql, params)

    cur = _CapturingCursor(rows=[], coverage_bibcodes=set())
    conn = _FakeConnection(cur)
    _handle_cited_by_intent(conn, {"target_bibcode": "X"})

    sql = captured.get("sql", "").lower()
    assert "row_number()" in sql
    assert "partition by cc.source_bibcode" in sql
    assert "n_contexts" in sql


# ---------------------------------------------------------------------------
# Existing input-validation behaviours still hold
# ---------------------------------------------------------------------------


def test_rejects_empty_target_bibcode() -> None:
    cur = _FakeCursor()
    conn = _FakeConnection(cur)
    out = json.loads(_handle_cited_by_intent(conn, {"target_bibcode": ""}))
    assert "error" in out


def test_rejects_unknown_intent_value() -> None:
    cur = _FakeCursor()
    conn = _FakeConnection(cur)
    out = json.loads(
        _handle_cited_by_intent(
            conn, {"target_bibcode": "2010ORIGIN.....A", "intent": "introduction"}
        )
    )
    assert "error" in out
