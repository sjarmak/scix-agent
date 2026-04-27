"""Smoke tests for all 17 consolidated MCP tools.

These tests catch total breakage of any tool on every deploy. They:

1. Verify ``startup_self_test`` succeeds against a freshly-created server
   (17 tools, valid schemas) — 12 from the 2026-04-25 consolidation pass
   (search, concept_search, get_paper, read_paper, citation_traverse,
   citation_similarity, entity, entity_context, graph_context, find_gaps,
   temporal_evolution, facet_counts) + 2 PRD MH-4 tools (claim_blame,
   find_replications) + 1 section_retrieval tool from the
   section-embeddings-mcp-consolidation PRD + 2 paper_claims retrieval
   tools (read_paper_claims, find_claims) from the nanopub-claim-extraction
   PRD.
2. Call each of the 17 consolidated tools via ``_dispatch_tool`` with a
   minimal golden-path input and assert the returned JSON is a valid
   non-error response (no exception raised, no top-level ``error`` key).

The dispatch layer talks to ``scix.search`` functions directly — we mock
those at the module boundary so the smoke tests do not require database
state. Real integration tests live elsewhere; this suite is about
catching routing breakage and schema drift.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix.mcp_server import (
    EXPECTED_TOOLS,
    _dispatch_tool,
    _session_state,
    startup_self_test,
)
from scix.search import SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_result() -> SearchResult:
    """Empty but structurally-valid SearchResult for mocking."""
    return SearchResult(papers=[], total=0, timing_ms={"query_ms": 0.1})


def _assert_non_error(result_json: str, tool: str) -> dict[str, Any]:
    """Assert that the JSON response has no top-level 'error' key."""
    data = json.loads(result_json)
    assert isinstance(
        data, (dict, list)
    ), f"{tool}: response is not a dict or list, got {type(data).__name__}"
    if isinstance(data, dict):
        assert (
            "error" not in data
        ), f"{tool}: response contains top-level 'error': {data.get('error')}"
    return data if isinstance(data, dict) else {"items": data}


@pytest.fixture(autouse=True)
def _reset_session() -> Any:
    """Clear implicit session state between tests."""
    _session_state.clear_working_set()
    _session_state.clear_focused()
    yield
    _session_state.clear_working_set()
    _session_state.clear_focused()


@pytest.fixture
def mock_conn() -> MagicMock:
    """A MagicMock standing in for a psycopg connection."""
    conn = MagicMock()
    # Cursor context manager for tools that hit the DB directly (entity search).
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    cursor.execute.return_value = None
    conn.cursor.return_value = cursor
    return conn


# ---------------------------------------------------------------------------
# AC1-3: startup_self_test
# ---------------------------------------------------------------------------


class TestStartupSelfTest:
    """Validates the server's self-test catches missing/broken tools."""

    def test_expected_tools_has_20_entries(self) -> None:
        # 2026-04-25 consolidation: citation_graph + citation_chain merged
        # into citation_traverse (-1), find_similar_by_examples retired
        # (was opt-in, now hard-removed). Subsequent PRDs added
        # claim_blame, find_replications, section_retrieval, the two
        # paper_claims retrieval tools, and cited_by_intent. Bead cfh9
        # added synthesize_findings. Final = 20.
        assert len(EXPECTED_TOOLS) == 20
        assert len(set(EXPECTED_TOOLS)) == 20  # no duplicates

    def test_self_test_passes_on_fresh_server(self) -> None:
        """A freshly created server must pass the self-test.

        ``EXPECTED_TOOLS`` lists every registered tool, but
        ``list_tools()`` filters by ``_HIDDEN_TOOLS`` (default: tools whose
        backing data is not yet populated). The visible count is
        ``len(_expected_tool_set())`` — derive the assertion from it so
        future hide/unhide changes don't churn this test.
        """
        try:
            import mcp.types  # noqa: F401
        except ImportError:
            pytest.skip("mcp SDK not installed")

        from scix.mcp_server import _expected_tool_set

        expected_visible = _expected_tool_set()
        with patch("scix.mcp_server._init_model_impl"):
            status = startup_self_test()

        assert status["ok"] is True
        assert status["tool_count"] == len(expected_visible)
        assert status["errors"] == []
        assert sorted(expected_visible) == status["tool_names"]

    def test_self_test_raises_on_wrong_tool_count(self) -> None:
        """If list_tools() returns the wrong count, self-test raises."""
        try:
            from mcp.types import ListToolsRequest, ListToolsResult, Tool
        except ImportError:
            pytest.skip("mcp SDK not installed")

        # Build a fake server where the list_tools handler returns 16 tools
        # (one short of EXPECTED_TOOLS).
        fake_server = MagicMock()
        bad_tools = [
            Tool(
                name=f"tool_{i}",
                description="x",
                inputSchema={"type": "object", "properties": {}},
            )
            for i in range(16)
        ]

        async def bad_handler(_req: Any) -> Any:
            return ListToolsResult(tools=bad_tools)

        fake_server.request_handlers = {ListToolsRequest: bad_handler}

        with pytest.raises(RuntimeError, match="startup_self_test failed"):
            startup_self_test(server=fake_server)

    def test_self_test_raises_on_missing_expected_tool(self) -> None:
        """If an expected tool name is missing, self-test raises."""
        try:
            from mcp.types import ListToolsRequest, ListToolsResult, Tool
        except ImportError:
            pytest.skip("mcp SDK not installed")

        # 17 tools but one expected name replaced with a bogus one.
        swapped = list(EXPECTED_TOOLS)
        swapped[0] = "not_a_real_tool"
        bad_tools = [
            Tool(
                name=n,
                description="x",
                inputSchema={"type": "object", "properties": {}},
            )
            for n in swapped
        ]

        async def bad_handler(_req: Any) -> Any:
            return ListToolsResult(tools=bad_tools)

        fake_server = MagicMock()
        fake_server.request_handlers = {ListToolsRequest: bad_handler}

        with pytest.raises(RuntimeError, match="missing expected tools"):
            startup_self_test(server=fake_server)


# ---------------------------------------------------------------------------
# AC4-5: One smoke test per consolidated tool
# ---------------------------------------------------------------------------


class TestToolSmoke:
    """Golden-path smoke test for each of the 17 consolidated tools."""

    @patch("scix.mcp_server._log_query")
    @patch(
        "scix.mcp_server._hnsw_index_exists",
        return_value=True,
    )
    @patch(
        "scix.mcp_server.embed_batch",
        return_value=[[0.0] * 768],
    )
    @patch(
        "scix.mcp_server.load_model",
        return_value=(MagicMock(), MagicMock()),
    )
    @patch("scix.search.hybrid_search")
    def test_search(
        self,
        mock_hybrid: MagicMock,
        _mock_load: MagicMock,
        _mock_embed: MagicMock,
        _mock_guard: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_hybrid.return_value = _empty_result()
        out = _dispatch_tool(mock_conn, "search", {"query": "dark matter"})
        _assert_non_error(out, "search")

    @patch("scix.mcp_server._log_query")
    @patch(
        "scix.mcp_server._hnsw_index_exists",
        return_value=True,
    )
    @patch(
        "scix.mcp_server.embed_batch",
        return_value=[[0.0] * 768],
    )
    @patch(
        "scix.mcp_server.load_model",
        return_value=(MagicMock(), MagicMock()),
    )
    @patch("scix.search.hybrid_search")
    def test_search_with_entity_filters(
        self,
        mock_hybrid: MagicMock,
        _mock_load: MagicMock,
        _mock_embed: MagicMock,
        _mock_guard: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """xz4.1.27: MCP search with entity_types + entity_ids propagates to hybrid_search."""
        mock_hybrid.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "search",
            {
                "query": "papers about JWST instruments",
                "filters": {
                    "entity_types": ["instrument"],
                    "entity_ids": [27867],
                },
            },
        )
        _assert_non_error(out, "search")
        # Confirm the filter threaded through to hybrid_search.
        called_filters = mock_hybrid.call_args.kwargs["filters"]
        assert called_filters.entity_types == ("instrument",)
        assert called_filters.entity_ids == (27867,)

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.concept_search")
    def test_concept_search(
        self,
        mock_cs: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_cs.return_value = _empty_result()
        out = _dispatch_tool(mock_conn, "concept_search", {"query": "Galaxies"})
        _assert_non_error(out, "concept_search")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_paper")
    def test_get_paper(
        self,
        mock_gp: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_gp.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "get_paper",
            {"bibcode": "2024ApJ...962L..15J"},
        )
        _assert_non_error(out, "get_paper")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.read_paper_section")
    def test_read_paper(
        self,
        mock_rps: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_rps.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "read_paper",
            {"bibcode": "2024ApJ...962L..15J", "section": "introduction"},
        )
        _assert_non_error(out, "read_paper")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_citations")
    def test_citation_traverse(
        self,
        mock_cit: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """citation_traverse default mode='graph' routes to search.get_citations."""
        mock_cit.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_traverse",
            {
                "mode": "graph",
                "bibcode": "2024ApJ...962L..15J",
                "direction": "forward",
            },
        )
        _assert_non_error(out, "citation_traverse")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.citation_chain")
    def test_citation_traverse_chain_mode(
        self,
        mock_cc: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """citation_traverse mode='chain' routes to search.citation_chain."""
        mock_cc.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_traverse",
            {
                "mode": "chain",
                "source_bibcode": "2024ApJ...962L..15J",
                "target_bibcode": "2023ApJ...900L...1A",
                "max_depth": 3,
            },
        )
        _assert_non_error(out, "citation_traverse[chain]")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.co_citation_analysis")
    def test_citation_similarity(
        self,
        mock_cca: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_cca.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "citation_similarity",
            {"bibcode": "2024ApJ...962L..15J", "method": "co_citation"},
        )
        _assert_non_error(out, "citation_similarity")

    @patch("scix.mcp_server._log_query")
    def test_entity(
        self,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        # entity action=search hits conn.cursor() directly; mock_conn's
        # cursor already returns an empty fetchall, so we get an empty-but-
        # valid response.
        out = _dispatch_tool(
            mock_conn,
            "entity",
            {"action": "search", "entity_type": "methods", "query": "MCMC"},
        )
        _assert_non_error(out, "entity")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_entity_context")
    def test_entity_context(
        self,
        mock_gec: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_gec.return_value = _empty_result()
        out = _dispatch_tool(mock_conn, "entity_context", {"entity_id": 1})
        _assert_non_error(out, "entity_context")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.get_paper_metrics")
    def test_graph_context(
        self,
        mock_gpm: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_gpm.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "graph_context",
            {"bibcode": "2024ApJ...962L..15J"},
        )
        _assert_non_error(out, "graph_context")

    @patch("scix.mcp_server._log_query")
    def test_find_gaps(
        self,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        # find_gaps queries via conn.cursor(); mock_conn returns an empty
        # fetchall, yielding {"papers": [], "total": 0, ...}.
        out = _dispatch_tool(
            mock_conn,
            "find_gaps",
            {"resolution": "coarse", "limit": 5},
        )
        _assert_non_error(out, "find_gaps")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.temporal_evolution")
    def test_temporal_evolution(
        self,
        mock_te: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_te.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "temporal_evolution",
            {"bibcode_or_query": "dark matter"},
        )
        _assert_non_error(out, "temporal_evolution")

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.facet_counts")
    def test_facet_counts(
        self,
        mock_fc: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        mock_fc.return_value = _empty_result()
        out = _dispatch_tool(mock_conn, "facet_counts", {"field": "year"})
        _assert_non_error(out, "facet_counts")

    @patch("scix.mcp_server._log_query")
    def test_synthesize_findings(
        self,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """Bead cfh9: synthesize_findings dispatch returns the wire envelope.

        Empty working set + no DB rows yields a structurally-valid response
        with the assignment_coverage block and an empty sections list.
        """
        out = _dispatch_tool(
            mock_conn,
            "synthesize_findings",
            {"working_set_bibcodes": []},
        )
        data = _assert_non_error(out, "synthesize_findings")
        assert "sections" in data
        assert "unattributed_bibcodes" in data
        assert "assignment_coverage" in data
        # Cross-bead schema-collision guard: synthesize must NOT emit the
        # bare "coverage" key (reserved for claim_blame/find_replications).
        assert "coverage" not in data

    @patch("scix.mcp_server._log_query")
    def test_claim_blame(
        self,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """PRD MH-4: claim_blame dispatch returns a structured result."""
        # mock_conn's cursor.fetchall already returns [], so the seed query
        # comes back empty and the tool returns the empty-origin shape.
        out = _dispatch_tool(
            mock_conn,
            "claim_blame",
            {"claim_text": "local H0 measurement is 73 km/s/Mpc"},
        )
        data = _assert_non_error(out, "claim_blame")
        assert "origin" in data
        assert "lineage" in data
        assert "confidence" in data
        assert "retraction_warnings" in data

    @patch("scix.mcp_server._log_query")
    def test_find_replications(
        self,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """PRD MH-4: find_replications dispatch returns citations envelope."""
        out = _dispatch_tool(
            mock_conn,
            "find_replications",
            {"target_bibcode": "2011ApJ...730..119R"},
        )
        data = _assert_non_error(out, "find_replications")
        assert "citations" in data
        assert "total" in data

    @patch("scix.mcp_server._log_query")
    @patch(
        "scix.mcp_server._encode_section_query",
        return_value=[0.0] * 1024,
    )
    def test_section_retrieval(
        self,
        _mock_encode: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """PRD section-embeddings-mcp-consolidation: section_retrieval dispatch returns results envelope."""
        # mock_conn's cursor.fetchall returns []; both retrieval legs come
        # back empty and the tool returns {results: [], total: 0}.
        out = _dispatch_tool(
            mock_conn,
            "section_retrieval",
            {"query": "Hubble constant tension"},
        )
        data = _assert_non_error(out, "section_retrieval")
        assert "results" in data
        assert "total" in data
        assert data["total"] == 0

    @patch("scix.mcp_server._log_query")
    def test_read_paper_claims(
        self,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """PRD nanopub-claim-extraction: read_paper_claims dispatch returns claims envelope."""
        out = _dispatch_tool(
            mock_conn,
            "read_paper_claims",
            {"bibcode": "2024ApJ...962L..15J"},
        )
        data = _assert_non_error(out, "read_paper_claims")
        assert "claims" in data
        assert "total" in data
        assert data["total"] == 0

    @patch("scix.mcp_server._log_query")
    def test_find_claims(
        self,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """PRD nanopub-claim-extraction: find_claims dispatch returns claims envelope."""
        out = _dispatch_tool(
            mock_conn,
            "find_claims",
            {"query": "Hubble constant"},
        )
        data = _assert_non_error(out, "find_claims")
        assert "claims" in data
        assert "total" in data
        assert data["total"] == 0

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.facet_counts")
    def test_facet_counts_threads_entity_filters(
        self,
        mock_fc: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """xz4.1.27 HIGH follow-up: facet_counts must propagate entity filters, not drop them."""
        mock_fc.return_value = _empty_result()
        out = _dispatch_tool(
            mock_conn,
            "facet_counts",
            {
                "field": "year",
                "filters": {"entity_types": ["instrument"], "entity_ids": [27867]},
            },
        )
        _assert_non_error(out, "facet_counts")
        called_filters = mock_fc.call_args.kwargs["filters"]
        assert called_filters.entity_types == ("instrument",)
        assert called_filters.entity_ids == (27867,)

    @patch("scix.mcp_server._log_query")
    @patch(
        "scix.mcp_server._hnsw_index_exists",
        return_value=True,
    )
    @patch(
        "scix.mcp_server.embed_batch",
        return_value=[[0.0] * 768],
    )
    @patch(
        "scix.mcp_server.load_model",
        return_value=(MagicMock(), MagicMock()),
    )
    @patch("scix.search.hybrid_search")
    def test_search_returns_json_error_for_bad_entity_filter(
        self,
        mock_hybrid: MagicMock,
        _mock_load: MagicMock,
        _mock_embed: MagicMock,
        _mock_guard: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """xz4.1.27 HIGH follow-up: invalid filters must return JSON {"error": ...}, not raise."""
        out = _dispatch_tool(
            mock_conn,
            "search",
            {"query": "dark matter", "filters": {"entity_types": "instrument"}},
        )
        data = json.loads(out)
        assert "error" in data
        assert "entity_types" in data["error"]
        mock_hybrid.assert_not_called()

    @patch("scix.mcp_server._log_query")
    @patch("scix.search.facet_counts")
    def test_facet_counts_returns_json_error_for_bad_entity_filter(
        self,
        mock_fc: MagicMock,
        _mock_log: MagicMock,
        mock_conn: MagicMock,
    ) -> None:
        """xz4.1.27 HIGH follow-up: invalid filters on facet_counts also return JSON error."""
        out = _dispatch_tool(
            mock_conn,
            "facet_counts",
            {"field": "year", "filters": {"entity_ids": ["not-an-int"]}},
        )
        data = json.loads(out)
        assert "error" in data
        assert "entity_ids" in data["error"]
        mock_fc.assert_not_called()


# ---------------------------------------------------------------------------
# PRD chunk-embeddings-build: chunk_search MCP tool (Qdrant-gated)
# ---------------------------------------------------------------------------


def _list_registered_tool_names() -> list[str]:
    """Build a fresh server and return its currently advertised tool names.

    Mirrors what an MCP client sees on tools/list. Avoids importing
    ``startup_self_test`` to keep the gating tests independent from the
    self-test logic.
    """
    import asyncio

    from mcp.types import ListToolsRequest

    from scix.mcp_server import create_server

    with patch("scix.mcp_server._init_model_impl"):
        server = create_server(_run_self_test=False)
    handler = server.request_handlers[ListToolsRequest]
    result = asyncio.run(handler(ListToolsRequest(method="tools/list")))
    tools = result.root.tools if hasattr(result, "root") else result.tools
    return [t.name for t in tools]


def test_chunk_search_tool_listed_when_qdrant_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """chunk_search appears in list_tools when QDRANT_URL is set."""
    try:
        import mcp.types  # noqa: F401
    except ImportError:
        pytest.skip("mcp SDK not installed")

    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    # Force qdrant_tools.is_enabled() to return True even if the qdrant_client
    # package isn't installed in the test environment (it short-circuits on
    # QdrantClient is None otherwise).
    import scix.mcp_server as mcp_server_module

    if mcp_server_module._qdrant_tools is not None:
        monkeypatch.setattr(mcp_server_module._qdrant_tools, "is_enabled", lambda: True)
    else:
        monkeypatch.setattr(mcp_server_module, "_qdrant_enabled", lambda: True)

    names = _list_registered_tool_names()
    assert "chunk_search" in names


def test_chunk_search_tool_hidden_when_qdrant_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """chunk_search is NOT advertised when QDRANT_URL is unset."""
    try:
        import mcp.types  # noqa: F401
    except ImportError:
        pytest.skip("mcp SDK not installed")

    monkeypatch.delenv("QDRANT_URL", raising=False)
    # Belt-and-suspenders: force the gate to report disabled even if a real
    # qdrant_tools.is_enabled() implementation reads other env vars in future.
    import scix.mcp_server as mcp_server_module

    if mcp_server_module._qdrant_tools is not None:
        monkeypatch.setattr(mcp_server_module._qdrant_tools, "is_enabled", lambda: False)
    else:
        monkeypatch.setattr(mcp_server_module, "_qdrant_enabled", lambda: False)

    names = _list_registered_tool_names()
    assert "chunk_search" not in names


def test_chunk_search_dispatch_returns_qdrant_disabled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """call_tool('chunk_search', ...) returns {'error': 'qdrant_disabled'} when QDRANT_URL unset."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    import scix.mcp_server as mcp_server_module

    monkeypatch.setattr(mcp_server_module, "_qdrant_enabled", lambda: False)

    # call_tool acquires a pooled connection; stub _get_conn so we don't need
    # a live Postgres for the gate-only path.
    from contextlib import contextmanager

    fake_conn = MagicMock()

    @contextmanager
    def _fake_get_conn():
        yield fake_conn

    monkeypatch.setattr(mcp_server_module, "_get_conn", _fake_get_conn)
    monkeypatch.setattr(mcp_server_module, "_log_query", lambda *a, **k: None)
    monkeypatch.setattr(mcp_server_module, "_set_timeout", lambda *a, **k: None)

    out = mcp_server_module.call_tool("chunk_search", {"query": "mcmc"})
    data = json.loads(out)
    assert data.get("error") == "qdrant_disabled"


def test_chunk_search_dispatch_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """call_tool('chunk_search', ...) returns matches[] when Qdrant + INDUS are stubbed."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    import scix.mcp_server as mcp_server_module

    monkeypatch.setattr(mcp_server_module, "_qdrant_enabled", lambda: True)

    # Stub the INDUS embedder so we never touch the real model.
    monkeypatch.setattr("scix.embed.load_model", lambda *a, **k: (MagicMock(), MagicMock()))
    monkeypatch.setattr("scix.embed.embed_batch", lambda *a, **k: [[0.0] * 768])
    # Reset the cached embedder so the patched load_model is used.
    monkeypatch.setattr(mcp_server_module, "_indus_embedder", None)

    # Stub the Qdrant chunk-search + snippet helpers.
    from scix.qdrant_tools import ChunkHit

    sample_hit = ChunkHit(
        bibcode="2024ApJ...999X..01Z",
        chunk_id=42,
        section_idx=2,
        section_heading_norm="methods",
        section_heading="Methods",
        score=0.87,
        snippet=None,
        char_offset=120,
        n_tokens=180,
    )

    def _fake_chunk_search(vector, **kwargs):
        # Surface the parsed filters back so we can assert on them.
        _fake_chunk_search.last_call = {"vector": vector, **kwargs}
        return [sample_hit]

    def _fake_fetch_snippets(conn, hits, **kwargs):
        import dataclasses

        return [dataclasses.replace(h, snippet="abc methods MCMC ...") for h in hits]

    assert mcp_server_module._qdrant_tools is not None
    monkeypatch.setattr(
        mcp_server_module._qdrant_tools,
        "chunk_search_by_text",
        _fake_chunk_search,
    )
    monkeypatch.setattr(
        mcp_server_module._qdrant_tools,
        "fetch_chunk_snippets",
        _fake_fetch_snippets,
    )

    # Stub _get_conn so we don't need a live Postgres pool.
    from contextlib import contextmanager

    fake_conn = MagicMock()

    @contextmanager
    def _fake_get_conn():
        yield fake_conn

    monkeypatch.setattr(mcp_server_module, "_get_conn", _fake_get_conn)
    monkeypatch.setattr(mcp_server_module, "_log_query", lambda *a, **k: None)
    monkeypatch.setattr(mcp_server_module, "_set_timeout", lambda *a, **k: None)

    out = mcp_server_module.call_tool(
        "chunk_search",
        {
            "query": "mcmc cosmological parameters",
            "filters": {"section_heading": ["methods"]},
            "limit": 5,
        },
    )
    data = json.loads(out)
    assert "error" not in data, f"unexpected error: {data}"
    assert data["total"] == 1
    assert isinstance(data["matches"], list) and len(data["matches"]) == 1
    match = data["matches"][0]
    for key in ("bibcode", "chunk_id", "section_heading", "score", "snippet"):
        assert key in match, f"match missing key {key}: {match}"
    assert match["bibcode"] == "2024ApJ...999X..01Z"
    assert match["chunk_id"] == 42
    assert match["snippet"] == "abc methods MCMC ..."

    # filter_summary captures the applied filters + the limit.
    fs = data["filter_summary"]
    assert fs["limit"] == 5
    assert fs["section_heading"] == ["methods"]

    # And the parsed filter actually reached qdrant_tools.chunk_search_by_text.
    last = _fake_chunk_search.last_call
    assert last["section_heading_norm"] == ["methods"]
    assert last["limit"] == 5


# ---------------------------------------------------------------------------
# Meta: ensure every expected tool has a smoke test
# ---------------------------------------------------------------------------


def test_every_expected_tool_has_a_smoke_test() -> None:
    """Guard: if the expected tool list changes, this test reminds us."""
    smoke_test_methods = {
        name[len("test_") :] for name in dir(TestToolSmoke) if name.startswith("test_")
    }
    missing = set(EXPECTED_TOOLS) - smoke_test_methods
    assert not missing, f"Missing smoke tests for: {sorted(missing)}"
