"""Tests for :mod:`scix.find_replications` (PRD MH-4).

Acceptance coverage:

* (a) forward citations ranked by intent_weight
* (b) hedge_present True for hedged-language fixtures
* (c) relation filter narrows results
* (d) ResearchScope filters apply

All tests use a fake DB pool — no real model loads, no DB calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from scix.find_replications import (
    DEFAULT_INTENT_WEIGHT,
    INTENT_WEIGHTS,
    Citation,
    _detect_hedge,
    _infer_relation,
    find_replications,
)
from scix.research_scope import ResearchScope


# ---------------------------------------------------------------------------
# Fake DB plumbing
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = list(rows)
        self.last_sql: str | None = None
        self.last_params: list[Any] = []
        self._is_coverage_probe = False

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        # Don't overwrite the last_sql/last_params from the real query with
        # the follow-up coverage probe — tests assert on the original SQL.
        self._is_coverage_probe = "count(distinct" in sql.lower()
        if not self._is_coverage_probe:
            self.last_sql = sql
            self.last_params = list(params) if params is not None else []

    def fetchall(self) -> list[tuple[Any, ...]]:
        if self._is_coverage_probe:
            return [(0,)]
        return list(self._rows)

    def fetchone(self) -> tuple[Any, ...] | None:
        if self._is_coverage_probe:
            return (0,)
        return self._rows[0] if self._rows else None


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor


class _FakePool:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def connection(self) -> _PoolCM:
        return _PoolCM(_FakeConnection(self._cursor))


class _PoolCM:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    def __enter__(self) -> _FakeConnection:
        return self._conn

    def __exit__(self, *exc_info: Any) -> bool:
        return False


# Row shape returned by _query_citations:
#   (citing_bibcode, year, intent, context_snippet, section_name)


def _row(
    citing: str,
    year: int | None,
    intent: str | None,
    snippet: str = "",
    section: str | None = None,
) -> tuple[Any, ...]:
    return (citing, year, intent, snippet, section)


# ---------------------------------------------------------------------------
# AC (a): forward citations ranked by intent_weight DESC
# ---------------------------------------------------------------------------


def test_forward_citations_ranked_by_intent_weight() -> None:
    rows = [
        _row("2020A", 2020, "background", "consistent with [REF]"),
        _row("2021B", 2021, "result_comparison", "we confirm [REF]"),
        _row("2019C", 2019, "method", "following [REF]"),
    ]
    pool = _FakePool(_FakeCursor(rows))

    out = find_replications("2010ORIGIN", db_pool=pool)
    citations = out["citations"]

    weights = [c["intent_weight"] for c in citations]
    assert weights == sorted(weights, reverse=True)
    # Top hit must be the result_comparison row.
    assert citations[0]["citing_bibcode"] == "2021B"
    # background row sinks to bottom.
    assert citations[-1]["intent"] == "background"


def test_intent_weight_uses_documented_map() -> None:
    rows = [
        _row("A", 2020, "background", ""),
        _row("B", 2020, "method", ""),
        _row("C", 2020, "result_comparison", ""),
        _row("D", 2020, None, ""),  # NULL intent
    ]
    pool = _FakePool(_FakeCursor(rows))

    out = find_replications("X", db_pool=pool)
    by_bib = {c["citing_bibcode"]: c["intent_weight"] for c in out["citations"]}
    assert by_bib["A"] == INTENT_WEIGHTS["background"]
    assert by_bib["B"] == INTENT_WEIGHTS["method"]
    assert by_bib["C"] == INTENT_WEIGHTS["result_comparison"]
    assert by_bib["D"] == DEFAULT_INTENT_WEIGHT


# ---------------------------------------------------------------------------
# AC (b): hedge_present True for hedged language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "snippet",
    [
        "this may suggest a different value than [REF]",
        "appears to be in tension with [REF]",
        "perhaps consistent with [REF]",
        "results are tentative — see [REF]",
        "likely to be smaller than [REF]",
    ],
)
def test_hedge_present_true_for_hedged_snippets(snippet: str) -> None:
    rows = [_row("CITER", 2020, "result_comparison", snippet)]
    pool = _FakePool(_FakeCursor(rows))
    out = find_replications("X", db_pool=pool)
    assert out["citations"][0]["hedge_present"] is True


def test_hedge_present_false_for_clean_assertion() -> None:
    rows = [
        _row(
            "CITER",
            2020,
            "result_comparison",
            "We confirm the measurement of [REF].",
        )
    ]
    pool = _FakePool(_FakeCursor(rows))
    out = find_replications("X", db_pool=pool)
    assert out["citations"][0]["hedge_present"] is False


def test_detect_hedge_unit() -> None:
    assert _detect_hedge("This may indicate something") is True
    assert _detect_hedge("Results confirm the prior work") is False
    assert _detect_hedge("") is False


# ---------------------------------------------------------------------------
# Relation inference unit tests
# ---------------------------------------------------------------------------


def test_infer_relation_replicates() -> None:
    assert _infer_relation("we confirm [REF]", hedge_present=False) == "replicates"
    assert _infer_relation("agrees with [REF]", hedge_present=False) == "replicates"
    assert _infer_relation("In agreement with [REF]", hedge_present=False) == "replicates"


def test_infer_relation_refutes() -> None:
    assert _infer_relation("contradicts [REF]", hedge_present=False) == "refutes"
    assert _infer_relation("rules out [REF]", hedge_present=False) == "refutes"


def test_infer_relation_qualifies() -> None:
    assert _infer_relation("extends [REF]", hedge_present=False) == "qualifies"


def test_infer_relation_partial() -> None:
    assert _infer_relation("partially consistent with [REF]", hedge_present=True) == "partial"


def test_infer_relation_hedged_replication_downgraded_to_qualifies() -> None:
    # Hedged agreement isn't a clean replication — should slide to qualifies.
    assert _infer_relation("may be consistent with [REF]", hedge_present=True) == "qualifies"


def test_infer_relation_unknown_for_neutral_text() -> None:
    assert _infer_relation("see [REF] for details", hedge_present=False) == "unknown"


# ---------------------------------------------------------------------------
# AC (c): relation filter narrows results
# ---------------------------------------------------------------------------


def test_relation_filter_narrows_results() -> None:
    rows = [
        _row("R1", 2020, "result_comparison", "we confirm [REF]"),
        _row("R2", 2021, "result_comparison", "agrees with [REF]"),
        _row("F1", 2022, "result_comparison", "contradicts [REF]"),
    ]
    pool = _FakePool(_FakeCursor(rows))

    refutes_only = find_replications("X", relation="refutes", db_pool=pool)["citations"]
    assert len(refutes_only) == 1
    assert refutes_only[0]["citing_bibcode"] == "F1"

    replicates_only = find_replications("X", relation="replicates", db_pool=pool)["citations"]
    assert {c["citing_bibcode"] for c in replicates_only} == {"R1", "R2"}


def test_invalid_relation_raises() -> None:
    pool = _FakePool(_FakeCursor([]))
    with pytest.raises(ValueError, match="relation must be"):
        find_replications("X", relation="nonsense", db_pool=pool)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC (d): ResearchScope filters apply
# ---------------------------------------------------------------------------


def test_research_scope_year_window_threads_to_sql() -> None:
    cursor = _FakeCursor(rows=[])
    pool = _FakePool(cursor)

    find_replications(
        "2010ORIGIN",
        scope=ResearchScope(year_window=(2018, 2024)),
        db_pool=pool,
    )

    assert cursor.last_sql is not None
    assert "vce.source_year" in cursor.last_sql
    # The window bounds must appear in the parameter list (after the
    # target_bibcode and before the LIMIT).
    assert 2018 in cursor.last_params
    assert 2024 in cursor.last_params


def test_research_scope_other_fields_thread_to_join() -> None:
    cursor = _FakeCursor(rows=[])
    pool = _FakePool(cursor)

    # methodology_class triggers the papers JOIN; verify the SQL grew it.
    find_replications(
        "X",
        scope=ResearchScope(methodology_class="observational"),
        db_pool=pool,
    )

    assert cursor.last_sql is not None
    assert "JOIN papers p" in cursor.last_sql
    assert "observational" in cursor.last_params


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_target_returns_empty_citations() -> None:
    pool = _FakePool(_FakeCursor([]))
    assert find_replications("", db_pool=pool)["citations"] == []
    assert find_replications("   ", db_pool=pool)["citations"] == []


def test_dispatch_with_explicit_conn_does_not_call_pool() -> None:
    cursor = _FakeCursor(
        rows=[_row("CITER", 2020, "result_comparison", "we confirm [REF]")]
    )
    conn = _FakeConnection(cursor)

    out = find_replications("X", conn=conn)
    citations = out["citations"]
    assert len(citations) == 1
    assert citations[0]["citing_bibcode"] == "CITER"
