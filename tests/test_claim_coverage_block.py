"""Tests for the citation_contexts coverage block surfaced by claim_blame
and find_replications (bead scix_experiments-7avw).

Acceptance:
1. ``claim_blame`` response includes a ``coverage`` block with
   ``covered_seeds``, ``total_seeds``, ``coverage_pct``, ``note``.
2. ``find_replications`` response includes the same coverage block.
3. When ``coverage_pct == 0``, the response shape MUST still include the
   coverage block.
4. ``covered_seeds`` reflects the actual count when some seeds are covered.
5. The note string references the coverage doc path.

The coverage block lets agents distinguish 'no events' (target has citation
context coverage but no replication / blame events) from 'no coverage'
(target is not in citation_contexts at all, so an empty result is silent
about the underlying topic).
"""

from __future__ import annotations

from typing import Any

import pytest

from scix.citation_contexts_coverage import (
    COVERAGE_DOC_PATH,
    DEFAULT_COVERAGE_NOTE,
    compute_coverage,
)
from scix.claim_blame import claim_blame
from scix.find_replications import find_replications
from scix.research_scope import ResearchScope

# ---------------------------------------------------------------------------
# Fake DB plumbing — mirrors the helpers in the existing claim/replication
# tests, but extended to answer the coverage SELECT against v_claim_edges.
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Cursor that answers four query types by inspecting the SQL.

    * ``SELECT ... FROM v_claim_edges WHERE source_bibcode = %s`` ->
      ``hop_rows_by_source``
    * ``SELECT bibcode FROM papers WHERE ... 'retraction'`` -> retracted bibs
    * ``SELECT p.bibcode, p.year FROM papers p`` -> seed candidates (default
      seeder; tests usually override via ``seed_candidates_fn``)
    * ``SELECT ... FROM v_claim_edges`` (other shapes — i.e. coverage probe
      and find_replications query) -> tests configure two paths:
        - ``forward_rows`` for the find_replications WHERE target_bibcode = %s
        - ``coverage_bibcodes`` for the COUNT(DISTINCT) coverage probe; we
          return one row per bibcode in the seed list that is in
          ``coverage_bibcodes``.
    """

    def __init__(
        self,
        hop_rows_by_source: dict[str, list[tuple[Any, ...]]] | None = None,
        retracted_bibcodes: set[str] | None = None,
        candidate_rows: list[tuple[Any, ...]] | None = None,
        forward_rows: list[tuple[Any, ...]] | None = None,
        coverage_bibcodes: set[str] | None = None,
    ) -> None:
        self._hop_rows_by_source = hop_rows_by_source or {}
        self._retracted = retracted_bibcodes or set()
        self._candidate_rows = candidate_rows or []
        self._forward_rows = forward_rows or []
        self._coverage_bibcodes = coverage_bibcodes or set()
        self._last_rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        sql_lower = sql.lower()

        # Coverage probe — recognised by the COUNT(DISTINCT bib) shape used
        # in src/scix/citation_contexts_coverage.py.
        if "count(distinct" in sql_lower and "v_claim_edges" in sql_lower:
            seeds = list(params[0]) if params else []
            covered = sum(1 for s in seeds if s in self._coverage_bibcodes)
            self._last_rows = [(covered,)]
            return

        if "v_claim_edges" in sql_lower and "source_bibcode = %s" in sql_lower:
            source_bib = params[0]
            self._last_rows = list(self._hop_rows_by_source.get(source_bib, []))
            return

        if "v_claim_edges" in sql_lower and "target_bibcode = %s" in sql_lower:
            self._last_rows = list(self._forward_rows)
            return

        if "from papers" in sql_lower and "retraction" in sql_lower:
            wanted = list(params[0]) if params else []
            self._last_rows = [(b,) for b in wanted if b in self._retracted]
            return

        if "from papers p" in sql_lower:
            self._last_rows = list(self._candidate_rows)
            return

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


class _PoolCM:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    def __enter__(self) -> _FakeConnection:
        return self._conn

    def __exit__(self, *exc_info: Any) -> bool:
        return False


class _FakePool:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def connection(self) -> _PoolCM:
        return _PoolCM(_FakeConnection(self._cursor))


def _no_embed(_text: str) -> None:
    return None


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


# ---------------------------------------------------------------------------
# compute_coverage — the underlying helper
# ---------------------------------------------------------------------------


def test_compute_coverage_zero_when_no_seeds_in_v_claim_edges() -> None:
    cursor = _FakeCursor(coverage_bibcodes=set())
    out = compute_coverage(_FakeConnection(cursor), ["A", "B", "C"])
    assert out["covered_seeds"] == 0
    assert out["total_seeds"] == 3
    assert out["coverage_pct"] == 0.0
    assert COVERAGE_DOC_PATH in out["note"]


def test_compute_coverage_partial_when_some_seeds_in_v_claim_edges() -> None:
    cursor = _FakeCursor(coverage_bibcodes={"A", "C"})
    out = compute_coverage(_FakeConnection(cursor), ["A", "B", "C", "D"])
    assert out["covered_seeds"] == 2
    assert out["total_seeds"] == 4
    assert out["coverage_pct"] == pytest.approx(0.5)
    assert out["note"] == DEFAULT_COVERAGE_NOTE


def test_compute_coverage_handles_empty_seed_list() -> None:
    cursor = _FakeCursor(coverage_bibcodes=set())
    out = compute_coverage(_FakeConnection(cursor), [])
    assert out["covered_seeds"] == 0
    assert out["total_seeds"] == 0
    # Avoid div-by-zero — empty seeds means no coverage signal at all.
    assert out["coverage_pct"] == 0.0
    assert COVERAGE_DOC_PATH in out["note"]


def test_compute_coverage_dedupes_seed_bibcodes() -> None:
    cursor = _FakeCursor(coverage_bibcodes={"A"})
    # Caller passes duplicates; total_seeds reflects the unique count so
    # coverage_pct is well-defined.
    out = compute_coverage(_FakeConnection(cursor), ["A", "A", "B", "B"])
    assert out["total_seeds"] == 2
    assert out["covered_seeds"] == 1
    assert out["coverage_pct"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# claim_blame — coverage block on the response
# ---------------------------------------------------------------------------


def _coverage_keys() -> set[str]:
    return {"covered_seeds", "total_seeds", "coverage_pct", "note"}


def test_claim_blame_response_includes_coverage_block_when_uncovered() -> None:
    """When no seeds appear in citation_contexts, response still carries
    the coverage block — agents can distinguish 'no coverage' from
    'no events'."""
    cand = "2020NICHE.....1A"
    # No hop rows, no candidates in v_claim_edges, no forward rows.
    cursor = _FakeCursor(
        hop_rows_by_source={cand: []},
        coverage_bibcodes=set(),
    )
    pool = _FakePool(cursor)

    def seed(_conn: Any, _claim: str, _scope: ResearchScope, _limit: int):
        return [(cand, 2020, None)]

    out = claim_blame(
        "granular mechanics in microgravity",
        db_pool=pool,
        seed_candidates_fn=seed,
        embed_query_fn=_no_embed,
    )

    assert "coverage" in out
    assert _coverage_keys() <= set(out["coverage"].keys())
    assert out["coverage"]["covered_seeds"] == 0
    assert out["coverage"]["total_seeds"] == 1
    assert out["coverage"]["coverage_pct"] == 0.0
    assert COVERAGE_DOC_PATH in out["coverage"]["note"]


def test_claim_blame_response_includes_coverage_block_when_covered() -> None:
    """When seeds DO appear in v_claim_edges, covered_seeds reflects that."""
    cand_2010 = "2010ApJ...700..123A"
    cand_2023 = "2023ApJ...900..456B"
    earlier_target = "2005ApJ...600..001Z"

    hops = {
        cand_2010: [
            _hop_row(
                cand_2010,
                earlier_target,
                intent="result_comparison",
                source_year=2010,
                target_year=2005,
                char_offset=10,
            )
        ],
        cand_2023: [
            _hop_row(
                cand_2023,
                earlier_target,
                intent="background",
                source_year=2023,
                target_year=2005,
                char_offset=11,
            )
        ],
    }

    cursor = _FakeCursor(
        hop_rows_by_source=hops,
        # Both candidates appear in v_claim_edges (as source_bibcode);
        # the target also does (as target_bibcode). Coverage probe sees all
        # candidate seeds covered.
        coverage_bibcodes={cand_2010, cand_2023, earlier_target},
    )
    pool = _FakePool(cursor)

    def seed(_conn: Any, _claim: str, _scope: ResearchScope, _limit: int):
        return [(cand_2010, 2010, None), (cand_2023, 2023, None)]

    out = claim_blame(
        "test claim",
        db_pool=pool,
        seed_candidates_fn=seed,
        embed_query_fn=_no_embed,
    )

    assert "coverage" in out
    assert out["coverage"]["covered_seeds"] == 2
    assert out["coverage"]["total_seeds"] == 2
    assert out["coverage"]["coverage_pct"] == pytest.approx(1.0)


def test_claim_blame_coverage_block_partial() -> None:
    cand_covered = "2010ApJ...700..123A"
    cand_uncovered = "2020NICHE.....1B"
    target = "2005ApJ...600..001Z"

    hops = {
        cand_covered: [
            _hop_row(
                cand_covered,
                target,
                intent="result_comparison",
                source_year=2010,
                target_year=2005,
                char_offset=10,
            )
        ],
        cand_uncovered: [],
    }

    cursor = _FakeCursor(
        hop_rows_by_source=hops,
        coverage_bibcodes={cand_covered, target},  # cand_uncovered is absent
    )
    pool = _FakePool(cursor)

    def seed(_conn: Any, _claim: str, _scope: ResearchScope, _limit: int):
        return [(cand_covered, 2010, None), (cand_uncovered, 2020, None)]

    out = claim_blame(
        "test",
        db_pool=pool,
        seed_candidates_fn=seed,
        embed_query_fn=_no_embed,
    )

    assert out["coverage"]["covered_seeds"] == 1
    assert out["coverage"]["total_seeds"] == 2
    assert out["coverage"]["coverage_pct"] == pytest.approx(0.5)


def test_claim_blame_coverage_block_when_no_candidates_seeded() -> None:
    """Empty candidate set still produces a coverage block (with zeroes)."""
    cursor = _FakeCursor(coverage_bibcodes=set())
    pool = _FakePool(cursor)

    def seed(_conn: Any, _claim: str, _scope: ResearchScope, _limit: int):
        return []

    out = claim_blame(
        "totally niche query",
        db_pool=pool,
        seed_candidates_fn=seed,
        embed_query_fn=_no_embed,
    )
    assert "coverage" in out
    assert _coverage_keys() <= set(out["coverage"].keys())
    assert out["coverage"]["total_seeds"] == 0
    assert out["coverage"]["covered_seeds"] == 0
    assert out["coverage"]["coverage_pct"] == 0.0


# ---------------------------------------------------------------------------
# find_replications — coverage block on the response
# ---------------------------------------------------------------------------


def test_find_replications_response_includes_coverage_block_when_uncovered() -> None:
    """When the target is not in v_claim_edges at all, response includes
    coverage block with covered_seeds=0 (not just an empty citations list)."""
    cursor = _FakeCursor(forward_rows=[], coverage_bibcodes=set())
    pool = _FakePool(cursor)

    out = find_replications("2020UNCOVERED.....A", db_pool=pool)

    assert isinstance(out, dict)
    assert "citations" in out
    assert "coverage" in out
    assert _coverage_keys() <= set(out["coverage"].keys())
    assert out["coverage"]["total_seeds"] == 1
    assert out["coverage"]["covered_seeds"] == 0
    assert out["coverage"]["coverage_pct"] == 0.0
    assert COVERAGE_DOC_PATH in out["coverage"]["note"]
    assert out["citations"] == []


def test_find_replications_response_includes_coverage_block_when_covered() -> None:
    """When the target IS in v_claim_edges, coverage_pct == 1.0 even if
    it has zero forward citations matching the query."""
    cursor = _FakeCursor(
        forward_rows=[],
        coverage_bibcodes={"2010COVERED.....A"},
    )
    pool = _FakePool(cursor)

    out = find_replications("2010COVERED.....A", db_pool=pool)

    assert out["coverage"]["total_seeds"] == 1
    assert out["coverage"]["covered_seeds"] == 1
    assert out["coverage"]["coverage_pct"] == pytest.approx(1.0)


def test_find_replications_coverage_block_with_actual_citations() -> None:
    """Citations list populated AND coverage block present together."""
    rows = [
        ("2020CITER.....A", 2020, "result_comparison", "we confirm [REF]", None),
    ]
    cursor = _FakeCursor(
        forward_rows=rows,
        coverage_bibcodes={"2010ORIGIN.....A"},
    )
    pool = _FakePool(cursor)

    out = find_replications("2010ORIGIN.....A", db_pool=pool)

    assert len(out["citations"]) == 1
    assert out["coverage"]["covered_seeds"] == 1
    assert out["coverage"]["total_seeds"] == 1
    assert out["coverage"]["coverage_pct"] == pytest.approx(1.0)


def test_find_replications_coverage_note_references_doc_path() -> None:
    cursor = _FakeCursor(forward_rows=[], coverage_bibcodes=set())
    pool = _FakePool(cursor)
    out = find_replications("anything", db_pool=pool)
    assert COVERAGE_DOC_PATH in out["coverage"]["note"]
