"""Unit tests for ``scix.search.community_expand_search``.

Exercises the entity co-occurrence retrieval lane introduced by PRD
``docs/prd/prd_community_expand_search.md`` (bead xz4.1.40). Tests run
against a ``MagicMock`` connection — no database required.

The function issues three logical SQL stages, but the implementation may
combine them. Tests assert on observable contract rather than exact SQL
shape:

* Returned ``SearchResult`` papers, total, and metadata are correct for
  the fed-in mock rows.
* The SQL fragments that encode the structural invariants from the PRD
  (NULLS LAST tiebreak, NOT EXISTS seed-exclusion, HAVING min_cooccurrence)
  appear somewhere in the issued statements.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from scix.search import SearchFilters, SearchResult, community_expand_search

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cursor(side_effects: list[Any]) -> MagicMock:
    """Build a context-managed cursor whose execute/fetch* return ``side_effects`` in order.

    ``side_effects`` is a list of dicts, each describing what one cursor
    operation should return:

        {"fetchone": <value>}              # fetchone returns <value>
        {"fetchall": <iterable of dicts>}  # fetchall returns the list
    """
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    fetchone_returns: list[Any] = []
    fetchall_returns: list[Any] = []
    for effect in side_effects:
        if "fetchone" in effect:
            fetchone_returns.append(effect["fetchone"])
        elif "fetchall" in effect:
            fetchall_returns.append(effect["fetchall"])

    cursor.fetchone.side_effect = fetchone_returns or [None]
    cursor.fetchall.side_effect = fetchall_returns or [[]]
    return cursor


def _make_conn(cursors: list[MagicMock]) -> MagicMock:
    """Connection that hands out the supplied cursors in order on each .cursor() call."""
    conn = MagicMock()
    conn.cursor.side_effect = cursors
    return conn


def _captured_sql(cursor: MagicMock) -> str:
    """Concatenate every SQL string passed to cursor.execute, lower-cased.

    Tests assert on substrings of this so the implementation can split or
    inline stages without breaking the contract checks.
    """
    pieces: list[str] = []
    for call in cursor.execute.call_args_list:
        sql = call.args[0] if call.args else ""
        pieces.append(str(sql))
    return " ".join(pieces).lower()


def _captured_params(cursor: MagicMock) -> list[Any]:
    """Flatten every params list/tuple passed to cursor.execute, in order."""
    flat: list[Any] = []
    for call in cursor.execute.call_args_list:
        if len(call.args) >= 2:
            params = call.args[1]
            if isinstance(params, (list, tuple)):
                flat.extend(params)
            else:
                flat.append(params)
    return flat


# ---------------------------------------------------------------------------
# Fixtures: typical row shapes the implementation is allowed to consume
# ---------------------------------------------------------------------------

# Three neighbor entities, ranked by cooccur_count.
_NEIGHBOR_ROWS: list[dict[str, Any]] = [
    {"entity_id": 101, "canonical_name": "WFC3", "cooccur_count": 25},
    {"entity_id": 102, "canonical_name": "STIS", "cooccur_count": 18},
    {"entity_id": 103, "canonical_name": "ACS", "cooccur_count": 9},
]

# Three candidate papers, ranked by neighbor_coverage DESC, coverage_score DESC,
# pagerank DESC NULLS LAST.
_PAPER_ROWS: list[dict[str, Any]] = [
    {
        "bibcode": "2024ApJ...001A",
        "title": "WFC3 + STIS dual-instrument survey",
        "first_author": "Smith",
        "year": 2024,
        "citation_count": 42,
        "abstract": "WFC3 + STIS observations of HST targets.",
        "neighbor_coverage": 3,
        "coverage_score": 52,
        "best_cooccur": 25,
        "pagerank": 0.012,
    },
    {
        "bibcode": "2023ApJ...002B",
        "title": "STIS calibration update",
        "first_author": "Jones",
        "year": 2023,
        "citation_count": 30,
        "abstract": "STIS calibration revisions.",
        "neighbor_coverage": 2,
        "coverage_score": 27,
        "best_cooccur": 18,
        "pagerank": 0.008,
    },
    {
        "bibcode": "2022ApJ...003C",
        "title": "ACS deep field",
        "first_author": "Brown",
        "year": 2022,
        "citation_count": 12,
        "abstract": "ACS imaging program.",
        "neighbor_coverage": 1,
        "coverage_score": 9,
        "best_cooccur": 9,
        "pagerank": None,
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCommunityExpandSearch:
    """Six PRD §7 cases, plus signature/contract guards."""

    # 1. Happy path — neighbor ranking + paper return shape.
    def test_seed_resolution_pulls_neighbors(self) -> None:
        seed_count_cur = _make_cursor([{"fetchone": (5,)}])
        neighbors_cur = _make_cursor([{"fetchall": _NEIGHBOR_ROWS}])
        papers_cur = _make_cursor([{"fetchall": _PAPER_ROWS}])
        conn = _make_conn([seed_count_cur, neighbors_cur, papers_cur])

        result = community_expand_search(conn, seed_entity_id=999, top_k=20, min_cooccurrence=2)

        assert isinstance(result, SearchResult)
        assert result.total == len(_PAPER_ROWS)
        bibcodes = [p["bibcode"] for p in result.papers]
        assert bibcodes == [
            "2024ApJ...001A",
            "2023ApJ...002B",
            "2022ApJ...003C",
        ]

        # Metadata exposes seed + neighbor info per PRD §3.
        meta = result.metadata
        assert meta["seed_entity_id"] == 999
        assert meta["seed_paper_count"] == 5
        assert meta["neighbor_count"] == 3
        assert meta["truncated_seed_papers"] is False
        # Top-N neighbors echoed for observability (capped at 10 per PRD).
        echoed = meta.get("neighbors")
        assert isinstance(echoed, list)
        assert len(echoed) == 3
        assert echoed[0]["entity_id"] == 101
        assert echoed[0]["canonical_name"] == "WFC3"
        assert echoed[0]["cooccur_count"] == 25

        # No structured-error envelope on the happy path.
        assert "error_code" not in meta

        # Timing breakdown carries the per-stage ms keys called out in PRD §3.
        assert "cooccur_neighbors_ms" in result.timing_ms
        assert "cooccur_papers_ms" in result.timing_ms

    # 2. Seed-linked papers must be excluded from candidate output.
    def test_excludes_seed_linked_papers(self) -> None:
        seed_count_cur = _make_cursor([{"fetchone": (5,)}])
        neighbors_cur = _make_cursor([{"fetchall": _NEIGHBOR_ROWS}])
        papers_cur = _make_cursor([{"fetchall": _PAPER_ROWS}])
        conn = _make_conn([seed_count_cur, neighbors_cur, papers_cur])

        community_expand_search(conn, seed_entity_id=999)

        captured = _captured_sql(papers_cur)
        # The candidate-paper query must filter out papers tagged with the seed.
        assert "not exists" in captured
        # The seed entity_id must show up in the params list of the same query.
        assert 999 in _captured_params(papers_cur)

    # 3. Tiebreak rule: papers ranked by neighbor_coverage DESC,
    #    coverage_score DESC, pagerank DESC NULLS LAST. The tertiary
    #    pagerank tiebreak preserves "NULLS LAST" so missing pagerank
    #    rows always sort last on equal coverage.
    def test_pagerank_tiebreak(self) -> None:
        seed_count_cur = _make_cursor([{"fetchone": (5,)}])
        neighbors_cur = _make_cursor([{"fetchall": _NEIGHBOR_ROWS}])
        papers_cur = _make_cursor([{"fetchall": _PAPER_ROWS}])
        conn = _make_conn([seed_count_cur, neighbors_cur, papers_cur])

        community_expand_search(conn, seed_entity_id=999)

        captured = _captured_sql(papers_cur)
        # Coverage-first ranking with pagerank as the deepest tiebreak.
        assert "neighbor_coverage desc" in captured
        assert "pagerank" in captured and "desc nulls last" in captured

    # 4. Empty neighborhood → empty SearchResult, NOT a fallback to hybrid.
    def test_empty_neighborhood_returns_empty_result(self) -> None:
        seed_count_cur = _make_cursor([{"fetchone": (1,)}])
        neighbors_cur = _make_cursor([{"fetchall": []}])  # no co-occurrence
        # Third cursor must NOT be used: implementation should short-circuit.
        conn = MagicMock()
        conn.cursor.side_effect = [seed_count_cur, neighbors_cur]

        result = community_expand_search(conn, seed_entity_id=999)

        assert result.papers == []
        assert result.total == 0
        meta = result.metadata
        assert meta["seed_entity_id"] == 999
        assert meta["neighbor_count"] == 0
        # No structured-error code — empty neighborhood is a normal outcome.
        assert "error_code" not in meta
        # Implementation must not have asked for a third cursor (no Stage 3).
        assert conn.cursor.call_count == 2

    # 5. Super-hub seed → structured 'seed_too_broad' error envelope.
    def test_super_hub_seed_returns_structured_error(self) -> None:
        seed_count_cur = _make_cursor([{"fetchone": (60_000,)}])
        # Only one cursor should be opened — the count query — before the guard fires.
        conn = MagicMock()
        conn.cursor.side_effect = [seed_count_cur]

        result = community_expand_search(conn, seed_entity_id=999)

        assert result.papers == []
        assert result.total == 0
        meta = result.metadata
        assert meta.get("error_code") == "seed_too_broad"
        assert meta["seed_entity_id"] == 999
        assert meta["seed_paper_count"] == 60_000
        assert "hint" in meta
        # No further DB work after the guard fires.
        assert conn.cursor.call_count == 1

    # 6. min_cooccurrence threshold drops singleton neighbors.
    def test_min_cooccurrence_filters_singletons(self) -> None:
        seed_count_cur = _make_cursor([{"fetchone": (5,)}])
        neighbors_cur = _make_cursor([{"fetchall": _NEIGHBOR_ROWS}])
        papers_cur = _make_cursor([{"fetchall": _PAPER_ROWS}])
        conn = _make_conn([seed_count_cur, neighbors_cur, papers_cur])

        community_expand_search(conn, seed_entity_id=999, min_cooccurrence=3)

        neighbors_sql = _captured_sql(neighbors_cur)
        # The cooccurrence threshold must reach the SQL — test that:
        #   (a) the HAVING clause is present
        #   (b) the threshold value (3) is bound as a parameter to the same query
        assert "having" in neighbors_sql
        assert 3 in _captured_params(neighbors_cur)

    # ----- Additional contract guards -----

    def test_truncated_seed_papers_when_count_exceeds_cap(self) -> None:
        """seed_paper_count > seed_paper_cap must surface as truncated_seed_papers."""
        # 7,500 papers, cap 5,000 → not super-hub but truncated.
        seed_count_cur = _make_cursor([{"fetchone": (7_500,)}])
        neighbors_cur = _make_cursor([{"fetchall": _NEIGHBOR_ROWS}])
        papers_cur = _make_cursor([{"fetchall": _PAPER_ROWS}])
        conn = _make_conn([seed_count_cur, neighbors_cur, papers_cur])

        result = community_expand_search(conn, seed_entity_id=999, seed_paper_cap=5_000)

        assert result.metadata["truncated_seed_papers"] is True
        assert result.metadata["seed_paper_count"] == 7_500

    def test_filters_only_apply_to_output_papers(self) -> None:
        """SearchFilters must reach Stage 3 (papers query) only, not Stage 2 (neighbors)."""
        seed_count_cur = _make_cursor([{"fetchone": (5,)}])
        neighbors_cur = _make_cursor([{"fetchall": _NEIGHBOR_ROWS}])
        papers_cur = _make_cursor([{"fetchall": _PAPER_ROWS}])
        conn = _make_conn([seed_count_cur, neighbors_cur, papers_cur])

        filters = SearchFilters(year_min=2023, doctype="article")
        community_expand_search(conn, seed_entity_id=999, filters=filters)

        # year_min/doctype params must show up in the candidate-papers query
        # but NOT in the neighbors query.
        papers_params = _captured_params(papers_cur)
        neighbors_params = _captured_params(neighbors_cur)
        assert 2023 in papers_params
        assert "article" in papers_params
        assert 2023 not in neighbors_params
        assert "article" not in neighbors_params

    def test_signature_keyword_only_args(self) -> None:
        """Ensure the public signature matches the PRD."""
        import inspect

        sig = inspect.signature(community_expand_search)
        assert sig.parameters["seed_entity_id"].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        assert sig.parameters["top_k"].default == 20
        assert sig.parameters["min_cooccurrence"].default == 2
        assert sig.parameters["neighbor_limit"].default == 50
        assert sig.parameters["seed_paper_cap"].default == 5_000
        # PRD §10 follow-up — neighbor_entity_types is opt-in (default None).
        assert sig.parameters["neighbor_entity_types"].default is None

    def test_neighbor_entity_types_filter_reaches_stage2_sql(self) -> None:
        """When neighbor_entity_types is non-None, the entity_type filter
        must appear in the Stage-2 (neighbors) SQL — the opt-in lever for
        the R4 universal-concept-noise risk. The papers query must NOT
        carry the type filter (Stage-3 filtering is governed by SearchFilters).
        """
        seed_count_cur = _make_cursor([{"fetchone": (5,)}])
        neighbors_cur = _make_cursor([{"fetchall": _NEIGHBOR_ROWS}])
        papers_cur = _make_cursor([{"fetchall": _PAPER_ROWS}])
        conn = _make_conn([seed_count_cur, neighbors_cur, papers_cur])

        community_expand_search(
            conn,
            seed_entity_id=999,
            neighbor_entity_types=("instrument", "mission", "observatory"),
        )

        neighbors_sql = _captured_sql(neighbors_cur)
        assert "entity_type = any" in neighbors_sql
        # The list of allowed types must show up in the params for the same query.
        neighbors_params = _captured_params(neighbors_cur)
        assert any(
            isinstance(p, list) and set(p) == {"instrument", "mission", "observatory"}
            for p in neighbors_params
        )


@pytest.mark.parametrize("seed_count", [0, 1])
def test_zero_or_one_seed_paper_returns_empty(seed_count: int) -> None:
    """Edge case: a seed with no document_entities_canonical rows yields empty result without crash."""
    seed_count_cur = _make_cursor([{"fetchone": (seed_count,)}])
    neighbors_cur = _make_cursor([{"fetchall": []}])
    conn = MagicMock()
    conn.cursor.side_effect = [seed_count_cur, neighbors_cur]

    result = community_expand_search(conn, seed_entity_id=42)
    assert result.papers == []
    assert result.metadata["seed_entity_id"] == 42
    assert result.metadata["seed_paper_count"] == seed_count
