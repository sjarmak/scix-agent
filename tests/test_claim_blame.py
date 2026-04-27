"""Tests for :mod:`scix.claim_blame` (PRD MH-4).

Acceptance coverage:

* (a) chronological-earliest origin selected over more-cited later restatement
* (b) retraction warnings populated when a hop touches a retracted paper
* (c) ResearchScope year_window honored
* (d) intent and intent_weight surfaced on every Hop
* (e) confidence in [0, 1]

All tests use a fake DB pool (MagicMock cursor with canned fetchall results)
and a fake INDUS embedder so they run fast and offline.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from scix.claim_blame import (
    DEFAULT_INTENT_WEIGHT,
    INTENT_WEIGHTS,
    Hop,
    claim_blame,
)
from scix.research_scope import ResearchScope


# ---------------------------------------------------------------------------
# Fake DB plumbing
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Cursor that returns canned rows by query keyword.

    The fake distinguishes three query types by inspecting the SQL:

    * SELECT ... FROM v_claim_edges  -> hop rows
    * SELECT bibcode FROM papers WHERE ... 'retraction'  -> retracted bibcodes
    * SELECT p.bibcode, p.year FROM papers p  -> seed candidates (only used
      when the test relies on the default seeder, which we mostly bypass)
    """

    def __init__(
        self,
        hop_rows_by_source: dict[str, list[tuple[Any, ...]]] | None = None,
        retracted_bibcodes: set[str] | None = None,
        candidate_rows: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self._hop_rows_by_source = hop_rows_by_source or {}
        self._retracted = retracted_bibcodes or set()
        self._candidate_rows = candidate_rows or []
        self._last_rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        sql_lower = sql.lower()
        if "count(distinct" in sql_lower and "v_claim_edges" in sql_lower:
            # citation_contexts coverage probe — return zero coverage by
            # default; tests focused on coverage live in
            # tests/test_claim_coverage_block.py.
            self._last_rows = [(0,)]
        elif "v_claim_edges" in sql_lower and "source_bibcode = %s" in sql_lower:
            source_bib = params[0]
            self._last_rows = list(self._hop_rows_by_source.get(source_bib, []))
        elif "from papers" in sql_lower and "retraction" in sql_lower:
            # params is a tuple containing a list of bibcodes
            wanted = list(params[0]) if params else []
            self._last_rows = [(b,) for b in wanted if b in self._retracted]
        elif "from papers p" in sql_lower:
            # default seed query
            self._last_rows = list(self._candidate_rows)
        else:
            self._last_rows = []

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._last_rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._last_rows[0] if self._last_rows else None


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


# ---------------------------------------------------------------------------
# Helpers to build hop tuples in v_claim_edges row order
# ---------------------------------------------------------------------------


def _hop_row(
    source_bibcode: str,
    target_bibcode: str,
    *,
    context_snippet: str = "",
    intent: str | None = None,
    section_name: str | None = None,
    source_year: int | None = None,
    target_year: int | None = None,
    char_offset: int | None = 0,
) -> tuple[Any, ...]:
    return (
        source_bibcode,
        target_bibcode,
        context_snippet,
        intent,
        section_name,
        source_year,
        target_year,
        char_offset,
    )


def _no_embed(_text: str) -> None:
    return None


# ---------------------------------------------------------------------------
# AC (a): chronological-earliest selected over more-cited later restatement
# ---------------------------------------------------------------------------


def test_chronologically_earliest_selected_over_later_restatement() -> None:
    # Two candidates that both assert the claim:
    #   2010 paper (the true origin)
    #   2023 paper (heavily-cited later restatement)
    # Both walk back to a single 2005 paper that the 2010 cites and the
    # 2023 paper cites. The chronologically-earliest non-retracted target
    # must win.
    cand_2010 = "2010ApJ...700..123A"
    cand_2023 = "2023ApJ...900..456B"
    earlier_target = "2005ApJ...600..001Z"

    hops = {
        cand_2010: [
            _hop_row(
                cand_2010,
                earlier_target,
                context_snippet="Following the original measurement of [REF]",
                intent="result_comparison",
                section_name="introduction",
                source_year=2010,
                target_year=2005,
                char_offset=10,
            )
        ],
        cand_2023: [
            _hop_row(
                cand_2023,
                earlier_target,
                context_snippet="Building on prior work [REF]",
                intent="background",
                section_name="introduction",
                source_year=2023,
                target_year=2005,
                char_offset=11,
            )
        ],
    }

    cursor = _FakeCursor(hop_rows_by_source=hops)
    pool = _FakePool(cursor)

    def seed(_conn: Any, _claim: str, _scope: ResearchScope, _limit: int):
        return [
            (cand_2010, 2010, None),
            (cand_2023, 2023, None),
        ]

    out = claim_blame(
        "test claim",
        db_pool=pool,
        seed_candidates_fn=seed,
        embed_query_fn=_no_embed,
    )

    assert out["origin"] == earlier_target
    # AC (e) — confidence is in unit interval
    assert 0.0 <= out["confidence"] <= 1.0
    # Lineage starts at origin
    assert out["lineage"][0]["bibcode"] == earlier_target


# ---------------------------------------------------------------------------
# AC (b): retraction warnings populate when a hop touches a retracted paper
# ---------------------------------------------------------------------------


def test_retraction_warning_populated_when_hop_touches_retracted_paper() -> None:
    candidate = "2014ApJ...800..111X"
    retracted_target = "2014ApJ...700..001Y"  # the BICEP2-style retracted paper
    clean_target = "2015ApJ...710..002Q"

    hops = {
        candidate: [
            _hop_row(
                candidate,
                retracted_target,
                context_snippet="confirmed by [REF]",
                intent="result_comparison",
                source_year=2014,
                target_year=2014,
                char_offset=20,
            ),
            _hop_row(
                candidate,
                clean_target,
                context_snippet="see also [REF]",
                intent="background",
                source_year=2014,
                target_year=2015,
                char_offset=21,
            ),
        ]
    }

    cursor = _FakeCursor(
        hop_rows_by_source=hops,
        retracted_bibcodes={retracted_target},
    )
    pool = _FakePool(cursor)

    def seed(_conn: Any, _claim: str, _scope: ResearchScope, _limit: int):
        return [(candidate, 2014, None)]

    out = claim_blame(
        "primordial gravitational waves",
        db_pool=pool,
        seed_candidates_fn=seed,
        embed_query_fn=_no_embed,
    )

    assert retracted_target in out["retraction_warnings"]
    # origin must NOT be the retracted paper
    assert out["origin"] != retracted_target
    # The clean earlier alternative is the candidate (2014) since both
    # universe entries (candidate 2014 and clean_target 2015) make it in,
    # but the candidate predates clean_target.
    assert out["origin"] in {candidate, clean_target}


# ---------------------------------------------------------------------------
# AC (c): ResearchScope year_window honored
# ---------------------------------------------------------------------------


def test_research_scope_year_window_filters_universe() -> None:
    cand_pre_window = "2010ApJ...700..123A"
    cand_in_window = "2020ApJ...800..222C"
    earlier_target_pre_window = "2005ApJ...600..001Z"

    hops = {
        cand_pre_window: [
            _hop_row(
                cand_pre_window,
                earlier_target_pre_window,
                source_year=2010,
                target_year=2005,
                intent="result_comparison",
                char_offset=1,
            )
        ],
        cand_in_window: [
            _hop_row(
                cand_in_window,
                cand_in_window,  # cites itself for simplicity (loopback)
                source_year=2020,
                target_year=2020,
                intent="result_comparison",
                char_offset=2,
            )
        ],
    }

    cursor = _FakeCursor(hop_rows_by_source=hops)
    pool = _FakePool(cursor)

    # Seed returns both, but only cand_in_window should survive year_window.
    def seed(_conn: Any, _claim: str, scope: ResearchScope, _limit: int):
        # Mimic SQL filter: drop the pre-window candidate.
        out = []
        for bib, yr in [
            (cand_pre_window, 2010),
            (cand_in_window, 2020),
        ]:
            if scope.year_window is None:
                out.append((bib, yr, None))
            else:
                lo, hi = scope.year_window
                if lo <= yr <= hi:
                    out.append((bib, yr, None))
        return out

    out = claim_blame(
        "test claim",
        scope=ResearchScope(year_window=(2018, 2024)),
        db_pool=pool,
        seed_candidates_fn=seed,
        embed_query_fn=_no_embed,
    )

    # The pre-window 2005 target should not appear.
    lineage_bibs = {h["bibcode"] for h in out["lineage"]}
    assert earlier_target_pre_window not in lineage_bibs
    assert cand_pre_window not in lineage_bibs
    assert out["origin"] == cand_in_window


# ---------------------------------------------------------------------------
# AC (d): intent and intent_weight surfaced on every Hop
# ---------------------------------------------------------------------------


def test_intent_and_intent_weight_on_every_hop() -> None:
    cand = "2015ApJ...111..111A"
    target = "2010ApJ...000..001Z"

    hops = {
        cand: [
            _hop_row(
                cand,
                target,
                context_snippet="we follow [REF]",
                intent="method",
                source_year=2015,
                target_year=2010,
                char_offset=0,
            )
        ]
    }
    cursor = _FakeCursor(hop_rows_by_source=hops)
    pool = _FakePool(cursor)

    def seed(_conn: Any, _claim: str, _scope: ResearchScope, _limit: int):
        return [(cand, 2015, None)]

    out = claim_blame(
        "test",
        db_pool=pool,
        seed_candidates_fn=seed,
        embed_query_fn=_no_embed,
    )

    assert out["lineage"], "lineage should be non-empty"
    for hop in out["lineage"]:
        assert "intent" in hop
        assert "intent_weight" in hop
        # intent_weight is always a float in the documented range
        assert 0.0 <= hop["intent_weight"] <= 1.0
        # The mapping is consistent with INTENT_WEIGHTS / DEFAULT_INTENT_WEIGHT.
        if hop["intent"] in INTENT_WEIGHTS:
            assert hop["intent_weight"] == INTENT_WEIGHTS[hop["intent"]]
        else:
            assert hop["intent_weight"] == DEFAULT_INTENT_WEIGHT


# ---------------------------------------------------------------------------
# AC (e): confidence in [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent",
    ["result_comparison", "method", "background", None],
)
def test_confidence_in_unit_interval_for_each_intent(intent: str | None) -> None:
    cand = "2020ApJ...111..111A"
    target = "2018ApJ...000..001Z"

    hops = {
        cand: [
            _hop_row(
                cand,
                target,
                context_snippet="x",
                intent=intent,
                source_year=2020,
                target_year=2018,
                char_offset=0,
            )
        ]
    }
    cursor = _FakeCursor(hop_rows_by_source=hops)
    pool = _FakePool(cursor)

    def seed(_conn: Any, _claim: str, _scope: ResearchScope, _limit: int):
        return [(cand, 2020, None)]

    out = claim_blame(
        "test",
        db_pool=pool,
        seed_candidates_fn=seed,
        embed_query_fn=_no_embed,
    )

    assert isinstance(out["confidence"], float)
    assert 0.0 <= out["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_claim_returns_empty_origin() -> None:
    cursor = _FakeCursor()
    pool = _FakePool(cursor)
    out = claim_blame("   ", db_pool=pool, embed_query_fn=_no_embed)
    # The coverage block is emitted unconditionally so the response shape
    # stays uniform (bead scix_experiments-7avw); for empty claim text the
    # block has zeroes because no DB probe ran.
    assert out["origin"] == ""
    assert out["lineage"] == []
    assert out["confidence"] == 0.0
    assert out["retraction_warnings"] == []
    assert out["coverage"]["covered_seeds"] == 0
    assert out["coverage"]["total_seeds"] == 0


def test_no_candidates_returns_empty_origin() -> None:
    cursor = _FakeCursor()
    pool = _FakePool(cursor)

    def seed(_c: Any, _t: str, _s: ResearchScope, _l: int):
        return []

    out = claim_blame(
        "no matches",
        db_pool=pool,
        seed_candidates_fn=seed,
        embed_query_fn=_no_embed,
    )
    assert out["origin"] == ""
    assert out["lineage"] == []
    assert out["confidence"] == 0.0


def test_all_candidates_retracted_returns_empty_origin_with_warnings() -> None:
    cand = "2014ApJ...111..111A"
    cursor = _FakeCursor(
        hop_rows_by_source={cand: []},
        retracted_bibcodes={cand},
    )
    pool = _FakePool(cursor)

    def seed(_c: Any, _t: str, _s: ResearchScope, _l: int):
        return [(cand, 2014, None)]

    out = claim_blame(
        "test",
        db_pool=pool,
        seed_candidates_fn=seed,
        embed_query_fn=_no_embed,
    )

    assert out["origin"] == ""
    assert cand in out["retraction_warnings"]
