"""Tests for first-class working_set abstraction across MCP tools.

Bead scix_experiments-3uvn:
    - SessionState.add_bibcodes_to_working_set([...]) — bulk add with dedupe
      and FIFO cap at 200.
    - facet_counts / temporal_evolution / citation_traverse accept either an
      explicit ``bibcodes=[...]`` arg or fall through to focused papers in
      the session when omitted.
    - 3-turn agent flow (search -> expand -> characterize) where each turn
      implicitly consumes the previous turn's output.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from scix import mcp_server
from scix.session import SessionState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_session_state() -> Iterator[None]:
    """Clear the module-level session state between tests."""
    mcp_server._session_state.clear_working_set()
    mcp_server._session_state.clear_focused()
    yield
    mcp_server._session_state.clear_working_set()
    mcp_server._session_state.clear_focused()


def _make_conn(rows: list[tuple] | None = None) -> MagicMock:
    """Create a mock psycopg connection returning *rows* from fetchall."""
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchall.return_value = rows or []
    cur.fetchone.return_value = (1,)
    cur.__enter__ = lambda self: self
    cur.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cur
    return conn


# ---------------------------------------------------------------------------
# SessionState.add_bibcodes_to_working_set
# ---------------------------------------------------------------------------


class TestAddBibcodesToWorkingSet:
    def test_appends_new_bibcodes(self) -> None:
        state = SessionState()
        added = state.add_bibcodes_to_working_set(["A", "B", "C"], source_tool="search")
        assert added == 3
        ws = [e.bibcode for e in state.get_working_set()]
        assert ws == ["A", "B", "C"]

    def test_dedupes_within_call(self) -> None:
        state = SessionState()
        added = state.add_bibcodes_to_working_set(["A", "B", "A", "C", "B"], source_tool="search")
        # Dedupe keeps first occurrence of each bibcode (the second 'A' replaces
        # the first per existing add_to_working_set contract, but the count of
        # unique bibcodes added is 3).
        assert added == 3
        ws = [e.bibcode for e in state.get_working_set()]
        assert sorted(ws) == ["A", "B", "C"]

    def test_dedupes_against_existing(self) -> None:
        state = SessionState()
        state.add_to_working_set(bibcode="A", source_tool="prior")
        state.add_to_working_set(bibcode="B", source_tool="prior")
        added = state.add_bibcodes_to_working_set(["B", "C", "D"], source_tool="search")
        # B already present -> still merged but counted; C, D are new.
        ws = [e.bibcode for e in state.get_working_set()]
        assert sorted(ws) == ["A", "B", "C", "D"]
        # Returned count is the number of bibcodes processed in this call
        # that ended up in the working set (3 unique bibcodes processed).
        assert added == 3

    def test_caps_at_200(self) -> None:
        state = SessionState()
        # Add 250 bibcodes in one shot.
        bibcodes = [f"BIB{i:04d}" for i in range(250)]
        state.add_bibcodes_to_working_set(bibcodes, source_tool="search")
        ws = state.get_working_set()
        assert len(ws) == 200
        # FIFO cap: oldest 50 dropped, so first kept bibcode is BIB0050.
        ws_codes = [e.bibcode for e in ws]
        assert ws_codes[0] == "BIB0050"
        assert ws_codes[-1] == "BIB0249"

    def test_caps_with_existing_entries(self) -> None:
        state = SessionState()
        # Pre-fill with 150 entries via singular adds.
        for i in range(150):
            state.add_to_working_set(bibcode=f"OLD{i:04d}", source_tool="prior")
        # Bulk add 100 new entries — total would be 250, must cap at 200.
        state.add_bibcodes_to_working_set([f"NEW{i:04d}" for i in range(100)], source_tool="search")
        ws = state.get_working_set()
        assert len(ws) == 200
        ws_codes = [e.bibcode for e in ws]
        # 50 oldest OLD entries dropped; OLD0050..OLD0149 + NEW0000..NEW0099 remain.
        assert ws_codes[0] == "OLD0050"
        assert ws_codes[-1] == "NEW0099"

    def test_empty_list_is_noop(self) -> None:
        state = SessionState()
        added = state.add_bibcodes_to_working_set([], source_tool="search")
        assert added == 0
        assert state.get_working_set() == []

    def test_passes_source_tool_metadata(self) -> None:
        state = SessionState()
        state.add_bibcodes_to_working_set(
            ["A"], source_tool="search", source_context="query: galaxies"
        )
        entries = state.get_working_set()
        assert entries[0].source_tool == "search"
        assert entries[0].source_context == "query: galaxies"


# ---------------------------------------------------------------------------
# facet_counts working-set fall-through
# ---------------------------------------------------------------------------


class TestFacetCountsWorkingSet:
    @patch("scix.search.facet_counts")
    def test_explicit_bibcodes_scopes_query(self, mock_fc: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fc.return_value = SearchResult(
            papers=[], total=1, timing_ms={}, metadata={"facets": []}
        )
        conn = _make_conn()
        mcp_server._dispatch_tool(
            conn,
            "facet_counts",
            {"field": "year", "bibcodes": ["2024X", "2024Y"]},
        )
        # Verify the bibcodes kwarg was forwarded to search.facet_counts.
        kwargs = mock_fc.call_args.kwargs
        assert kwargs.get("bibcodes") == ["2024X", "2024Y"]

    @patch("scix.search.facet_counts")
    def test_falls_through_to_focused_papers(self, mock_fc: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fc.return_value = SearchResult(
            papers=[], total=1, timing_ms={}, metadata={"facets": []}
        )
        # Seed session via track_focused.
        mcp_server._session_state.track_focused("2024A")
        mcp_server._session_state.track_focused("2024B")
        conn = _make_conn()
        mcp_server._dispatch_tool(conn, "facet_counts", {"field": "doctype"})
        kwargs = mock_fc.call_args.kwargs
        assert kwargs.get("bibcodes") is not None
        assert sorted(kwargs["bibcodes"]) == ["2024A", "2024B"]

    @patch("scix.search.facet_counts")
    def test_no_bibcodes_no_session_returns_full_corpus(self, mock_fc: MagicMock) -> None:
        """Backward-compat: empty session and no bibcodes -> no scoping."""
        from scix.search import SearchResult

        mock_fc.return_value = SearchResult(
            papers=[], total=0, timing_ms={}, metadata={"facets": []}
        )
        conn = _make_conn()
        mcp_server._dispatch_tool(conn, "facet_counts", {"field": "doctype"})
        kwargs = mock_fc.call_args.kwargs
        # bibcodes should be None (or absent) — no scoping was applied.
        assert kwargs.get("bibcodes") is None

    @patch("scix.search.facet_counts")
    def test_explicit_bibcodes_overrides_session(self, mock_fc: MagicMock) -> None:
        from scix.search import SearchResult

        mock_fc.return_value = SearchResult(
            papers=[], total=1, timing_ms={}, metadata={"facets": []}
        )
        mcp_server._session_state.track_focused("SESSION_A")
        conn = _make_conn()
        mcp_server._dispatch_tool(
            conn,
            "facet_counts",
            {"field": "year", "bibcodes": ["EXPLICIT_X"]},
        )
        kwargs = mock_fc.call_args.kwargs
        assert kwargs.get("bibcodes") == ["EXPLICIT_X"]


# ---------------------------------------------------------------------------
# temporal_evolution working-set fall-through
# ---------------------------------------------------------------------------


class TestTemporalEvolutionWorkingSet:
    @patch("scix.search.temporal_evolution")
    def test_explicit_bibcodes_aggregates_citations(self, mock_te: MagicMock) -> None:
        from scix.search import SearchResult

        mock_te.return_value = SearchResult(
            papers=[],
            total=2,
            timing_ms={},
            metadata={"mode": "citations", "yearly_counts": []},
        )
        conn = _make_conn()
        mcp_server._dispatch_tool(
            conn,
            "temporal_evolution",
            {"bibcodes": ["2024A", "2024B"]},
        )
        kwargs = mock_te.call_args.kwargs
        assert kwargs.get("bibcodes") == ["2024A", "2024B"]

    @patch("scix.search.temporal_evolution")
    def test_falls_through_to_focused_papers(self, mock_te: MagicMock) -> None:
        from scix.search import SearchResult

        mock_te.return_value = SearchResult(
            papers=[],
            total=1,
            timing_ms={},
            metadata={"mode": "citations", "yearly_counts": []},
        )
        mcp_server._session_state.track_focused("FOC_A")
        mcp_server._session_state.track_focused("FOC_B")
        conn = _make_conn()
        mcp_server._dispatch_tool(conn, "temporal_evolution", {})
        kwargs = mock_te.call_args.kwargs
        assert kwargs.get("bibcodes") is not None
        assert sorted(kwargs["bibcodes"]) == ["FOC_A", "FOC_B"]

    @patch("scix.search.temporal_evolution")
    def test_legacy_bibcode_or_query_still_works(self, mock_te: MagicMock) -> None:
        """Backward-compat: when no bibcodes + no session, legacy path runs."""
        from scix.search import SearchResult

        mock_te.return_value = SearchResult(
            papers=[],
            total=1,
            timing_ms={},
            metadata={"mode": "citations", "yearly_counts": []},
        )
        conn = _make_conn()
        mcp_server._dispatch_tool(conn, "temporal_evolution", {"bibcode_or_query": "SOME_BIBCODE"})
        # Legacy path: bibcode_or_query is positional arg, bibcodes=None.
        args = mock_te.call_args.args
        kwargs = mock_te.call_args.kwargs
        # bibcode_or_query passed positionally after conn.
        assert "SOME_BIBCODE" in args
        assert kwargs.get("bibcodes") is None

    def test_missing_required_when_empty_session(self) -> None:
        """When working set empty and no bibcode_or_query/bibcodes given, error."""
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "temporal_evolution", {})
        result = json.loads(result_json)
        assert "error" in result

    @patch("scix.search.temporal_evolution")
    def test_caps_resolved_bibcodes_at_200(self, mock_te: MagicMock) -> None:
        """Cap the bibcode list passed to search.temporal_evolution at 200.

        Bead scix_experiments-k6w0: focused_papers caps at 500 (FIFO, from
        bead u0j1) and explicit ``args["bibcodes"]`` is unbounded, so the
        resolver can hand off up to 500+ bibcodes. The handler must apply
        the same 200-cap that ``find_gaps`` (mcp_server.py:4732) and
        ``synthesize_findings`` (``MAX_WORKING_SET_BIBCODES`` in
        synthesize.py) use, to keep the ANY(%s) array bounded consistently
        across tools.
        """
        from scix.search import SearchResult

        mock_te.return_value = SearchResult(
            papers=[],
            total=0,
            timing_ms={},
            metadata={"mode": "citations", "yearly_counts": []},
        )
        # Populate focused papers beyond the 200-cap.
        for i in range(250):
            mcp_server._session_state.track_focused(f"BIB{i:04d}")
        conn = _make_conn()
        mcp_server._dispatch_tool(conn, "temporal_evolution", {})
        kwargs = mock_te.call_args.kwargs
        assert kwargs.get("bibcodes") is not None
        assert len(kwargs["bibcodes"]) == 200

    @patch("scix.search.temporal_evolution")
    def test_caps_explicit_bibcodes_at_200(self, mock_te: MagicMock) -> None:
        """Same 200-cap applies when caller passes ``bibcodes=[...]`` explicitly."""
        from scix.search import SearchResult

        mock_te.return_value = SearchResult(
            papers=[],
            total=0,
            timing_ms={},
            metadata={"mode": "citations", "yearly_counts": []},
        )
        explicit = [f"EX{i:04d}" for i in range(300)]
        conn = _make_conn()
        mcp_server._dispatch_tool(conn, "temporal_evolution", {"bibcodes": explicit})
        kwargs = mock_te.call_args.kwargs
        assert kwargs.get("bibcodes") is not None
        assert len(kwargs["bibcodes"]) == 200
        # FIFO order preserved: first 200 of the explicit list.
        assert kwargs["bibcodes"] == explicit[:200]


# ---------------------------------------------------------------------------
# citation_traverse multi-bibcode mode
# ---------------------------------------------------------------------------


class TestCitationTraverseWorkingSet:
    @patch("scix.search.get_citations_batch")
    def test_explicit_bibcodes_returns_per_bibcode(self, mock_batch: MagicMock) -> None:
        mock_batch.return_value = {
            "A": [{"bibcode": "CITER_X"}],
            "B": [{"bibcode": "CITER_Y"}],
        }
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(
            conn,
            "citation_traverse",
            {"mode": "graph", "bibcodes": ["A", "B"]},
        )
        result = json.loads(result_json)
        assert "by_bibcode" in result
        assert "A" in result["by_bibcode"]
        assert "B" in result["by_bibcode"]
        # Batched: one query for the whole working set, not one per bibcode.
        assert mock_batch.call_count == 1

    @patch("scix.search.get_citations_batch")
    def test_falls_through_to_focused_papers(self, mock_batch: MagicMock) -> None:
        mock_batch.return_value = {"F1": [{"bibcode": "CITER"}]}
        mcp_server._session_state.track_focused("F1")
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(conn, "citation_traverse", {"mode": "graph"})
        result = json.loads(result_json)
        # When only one focused paper, falls back to single-bibcode response
        # OR a multi-paper by_bibcode result (implementation chooses).
        assert "by_bibcode" in result or "papers" in result

    @patch("scix.search.get_citations_batch")
    def test_large_working_set_is_one_query_not_n(self, mock_batch: MagicMock) -> None:
        """Regression (sd71): round-trips are bounded regardless of set size.

        The old per-bibcode loop fired one DB query per source paper (up to the
        FIFO cap of 200) and timed out on large working sets. The batched path
        must issue exactly one edge query for the whole set.
        """
        bibs = [f"BIB{i:04d}" for i in range(150)]
        mock_batch.return_value = {b: [{"bibcode": f"C{b}"}] for b in bibs}
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(
            conn, "citation_traverse", {"mode": "graph", "bibcodes": bibs}
        )
        result = json.loads(result_json)
        assert len(result["by_bibcode"]) == 150
        assert mock_batch.call_count == 1

    @patch("scix.search.get_references_batch")
    @patch("scix.search.get_citations_batch")
    def test_direction_both_returns_nested_shape(
        self, mock_fwd: MagicMock, mock_bwd: MagicMock
    ) -> None:
        """direction='both' preserves the per-direction nested payload shape."""
        mock_fwd.return_value = {"A": [{"bibcode": "CITER"}]}
        mock_bwd.return_value = {"A": [{"bibcode": "REF"}]}
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(
            conn,
            "citation_traverse",
            {"mode": "graph", "bibcodes": ["A"], "direction": "both"},
        )
        entry = json.loads(result_json)["by_bibcode"]["A"]
        assert entry["bibcode"] == "A"
        dirs = {d["direction"]: d["result"] for d in entry["directions"]}
        assert dirs["forward"]["papers"][0]["bibcode"] == "CITER"
        assert dirs["backward"]["papers"][0]["bibcode"] == "REF"
        assert mock_fwd.call_count == 1
        assert mock_bwd.call_count == 1

    def test_invalid_direction_returns_structured_error(self) -> None:
        """An invalid direction is rejected per bibcode without touching edges."""
        conn = _make_conn()
        result_json = mcp_server._dispatch_tool(
            conn,
            "citation_traverse",
            {"mode": "graph", "bibcodes": ["A"], "direction": "sideways"},
        )
        entry = json.loads(result_json)["by_bibcode"]["A"]
        assert entry["error_code"] == "invalid_direction"


class TestCitationEdgesBatch:
    """search-layer batched citation-edge query (sd71)."""

    def test_groups_rows_by_source_and_fills_missing(self) -> None:
        from scix import search

        rows = [
            {
                "bibcode": "CITER_X",
                "title": "t",
                "first_author": "a",
                "year": 2020,
                "citation_count": 5,
                "abstract": None,
                "src_bibcode": "A",
                "rn": 1,
            },
            {
                "bibcode": "CITER_Y",
                "title": "t",
                "first_author": "a",
                "year": 2021,
                "citation_count": 3,
                "abstract": None,
                "src_bibcode": "A",
                "rn": 2,
            },
        ]
        conn = _make_conn(rows)
        out = search.get_citations_batch(conn, ["A", "B"], limit=20)
        # 'A' gets both rows in rn order; 'B' (no edges) is present and empty.
        assert [p["bibcode"] for p in out["A"]] == ["CITER_X", "CITER_Y"]
        assert out["B"] == []

    def test_empty_bibcodes_skips_query(self) -> None:
        from scix import search

        conn = _make_conn()
        assert search.get_references_batch(conn, [], limit=20) == {}
        conn.cursor.assert_not_called()


# ---------------------------------------------------------------------------
# 3-turn agent flow: search -> expand -> characterize
# ---------------------------------------------------------------------------


class TestThreeTurnAgentFlow:
    @patch("scix.search.facet_counts")
    @patch("scix.search.get_paper")
    @patch("scix.search.lexical_search")
    def test_search_then_expand_then_characterize(
        self,
        mock_lex: MagicMock,
        mock_get_paper: MagicMock,
        mock_fc: MagicMock,
    ) -> None:
        """Turn 1: search seeds working set via auto-tracking.
        Turn 2: get_paper on top hit focuses it.
        Turn 3: facet_counts() with no args scopes to focused papers.
        """
        from scix.search import SearchResult

        # Turn 1: search returns 3 papers; auto_track adds them to seen.
        mock_lex.return_value = SearchResult(
            papers=[
                {"bibcode": "T1A"},
                {"bibcode": "T1B"},
                {"bibcode": "T1C"},
            ],
            total=3,
            timing_ms={},
        )
        conn = _make_conn()
        mcp_server._dispatch_tool(conn, "search", {"query": "q", "mode": "keyword"})

        # Turn 2: get_paper focuses two (and adds to working set).
        mock_get_paper.return_value = SearchResult(
            papers=[{"bibcode": "T1A"}], total=1, timing_ms={}
        )
        mcp_server._dispatch_tool(conn, "get_paper", {"bibcode": "T1A"})
        mcp_server._dispatch_tool(conn, "get_paper", {"bibcode": "T1B"})

        # Turn 3: facet_counts() with no bibcodes — should scope to focused.
        mock_fc.return_value = SearchResult(
            papers=[], total=1, timing_ms={}, metadata={"facets": []}
        )
        mcp_server._dispatch_tool(conn, "facet_counts", {"field": "doctype"})
        kwargs = mock_fc.call_args.kwargs
        assert kwargs.get("bibcodes") is not None
        assert sorted(kwargs["bibcodes"]) == ["T1A", "T1B"]
