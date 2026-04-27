"""Tests for the MCP `search` tool's disambiguator integration (work unit D1).

Covers the three dispatch branches wired in ``scix.mcp_server._handle_search``:

1. ``disambiguate=True`` + query contains an ambiguous mention
   → response shape is ``{"disambiguation": [...]}`` (no search run).
2. ``disambiguate=True`` + query has no ambiguous mentions
   → normal search response (``papers``/``total``/``timing_ms``).
3. ``disambiguate=False`` bypasses the check regardless of ambiguity
   → normal search response.

Plus a schema assertion that the tool's ``inputSchema`` advertises the new
``disambiguate`` boolean with default ``True``.

All tests mock ``disambiguate_query`` and the search implementation so they
run without a real DB — the behaviour under test is the dispatch/branching
logic inside ``_handle_search``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix.jit.disambiguator import EntityCandidate, MentionDisambiguation
from scix.mcp_server import _dispatch_tool
from scix.search import SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ambiguous_mentions() -> list[MentionDisambiguation]:
    """Return a MentionDisambiguation list with one ambiguous mention.

    Shape mirrors the "Hubble" case documented in the disambiguator's
    module docstring: one mission candidate, one person candidate, both
    above the default ``min_paper_count`` threshold.
    """
    return [
        MentionDisambiguation(
            mention="Hubble",
            ambiguous=True,
            default_type="mission",
            candidates=(
                EntityCandidate(
                    entity_id=42,
                    entity_type="mission",
                    display="Hubble Space Telescope",
                    score=1.0,
                    paper_count=45000,
                ),
                EntityCandidate(
                    entity_id=99,
                    entity_type="person",
                    display="Edwin Hubble",
                    score=0.003,
                    paper_count=120,
                ),
            ),
        ),
    ]


def _make_unambiguous_mentions() -> list[MentionDisambiguation]:
    """Return a MentionDisambiguation list with no ambiguous mentions.

    Represents the common case where mentions resolve cleanly to a single
    entity type (or no entity at all).
    """
    return [
        MentionDisambiguation(
            mention="JWST",
            ambiguous=False,
            default_type="mission",
            candidates=(
                EntityCandidate(
                    entity_id=7,
                    entity_type="mission",
                    display="James Webb Space Telescope",
                    score=1.0,
                    paper_count=8000,
                ),
            ),
        ),
    ]


def _stub_search_result() -> SearchResult:
    """Minimal SearchResult that mocks the normal search path."""
    return SearchResult(
        papers=[{"bibcode": "2024ApJ...001A", "title": "Stub paper"}],
        total=1,
        timing_ms={"query_ms": 0.1},
    )


# ---------------------------------------------------------------------------
# Dispatch-branch tests
# ---------------------------------------------------------------------------


@patch("scix.mcp_server.disambiguate_query")
def test_ambiguous_query_returns_disambiguation(mock_disambig: MagicMock) -> None:
    """AC2: disambiguate=True + ambiguous query → 'disambiguation' key, no search ran."""
    mock_disambig.return_value = _make_ambiguous_mentions()
    mock_conn = MagicMock()

    with patch("scix.search.hybrid_search") as mock_hybrid:
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            # bypass_unscoped_guard=True so the broad query reaches the
            # disambiguation step rather than tripping the unscoped guard
            # (bead scix_experiments-uerc).
            {
                "query": "Hubble observations of Bennu",
                "bypass_unscoped_guard": True,
            },
        )
        # Search implementation MUST NOT be invoked when disambiguation fires.
        mock_hybrid.assert_not_called()

    data = json.loads(result_json)

    assert "disambiguation" in data
    assert "papers" not in data
    assert "results" not in data
    assert "total" not in data  # SearchResult top-level fields should be absent

    disamb_list = data["disambiguation"]
    assert isinstance(disamb_list, list)
    assert len(disamb_list) == 1

    mention = disamb_list[0]
    assert mention["mention"] == "Hubble"
    assert mention["ambiguous"] is True
    assert mention["default_type"] == "mission"
    assert isinstance(mention["candidates"], list)
    assert len(mention["candidates"]) == 2

    # Candidate fields map 1:1 onto EntityCandidate dataclass fields.
    first_candidate = mention["candidates"][0]
    for key in ("entity_id", "entity_type", "display", "score", "paper_count"):
        assert key in first_candidate, f"missing candidate field: {key}"
    assert first_candidate["entity_type"] == "mission"
    assert first_candidate["display"] == "Hubble Space Telescope"

    mock_disambig.assert_called_once()


@patch("scix.mcp_server._hnsw_index_exists", return_value=False)
@patch("scix.mcp_server.disambiguate_query")
def test_unambiguous_query_returns_search(
    mock_disambig: MagicMock,
    _mock_hnsw: MagicMock,
) -> None:
    """AC3: disambiguate=True + no ambiguous mentions → normal search shape."""
    mock_disambig.return_value = _make_unambiguous_mentions()
    mock_conn = MagicMock()

    with patch("scix.search.hybrid_search", return_value=_stub_search_result()) as mock_hybrid:
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            # bypass_unscoped_guard=True for parity with the ambiguous-query
            # test; this case exercises the disambiguator-then-search path,
            # not the unscoped-broad-query guard.
            {
                "query": "JWST infrared spectroscopy",
                "bypass_unscoped_guard": True,
            },
        )
        mock_hybrid.assert_called_once()

    data = json.loads(result_json)

    assert "disambiguation" not in data
    # Current search shape: papers + total + timing_ms.
    assert "papers" in data
    assert "total" in data
    assert data["total"] == 1

    mock_disambig.assert_called_once()


@patch("scix.mcp_server._hnsw_index_exists", return_value=False)
@patch("scix.mcp_server.disambiguate_query")
def test_disambiguate_false_bypasses(
    mock_disambig: MagicMock,
    _mock_hnsw: MagicMock,
) -> None:
    """AC4: disambiguate=False → never invoke disambiguator, always run search."""
    # The disambiguator should not be invoked at all. Set a tripwire return
    # value that would fail the ambiguity assertion if the bypass were broken.
    mock_disambig.return_value = _make_ambiguous_mentions()
    mock_conn = MagicMock()

    with patch("scix.search.hybrid_search", return_value=_stub_search_result()) as mock_hybrid:
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            {
                "query": "Hubble observations of Bennu",
                "disambiguate": False,
                # bypass_unscoped_guard=True so the broad query reaches the
                # search path; we're testing the disambiguate=False bypass,
                # not the unscoped-broad-query guard.
                "bypass_unscoped_guard": True,
            },
        )
        mock_hybrid.assert_called_once()

    data = json.loads(result_json)

    assert "disambiguation" not in data
    assert "papers" in data
    assert "total" in data

    mock_disambig.assert_not_called()


@patch("scix.mcp_server._hnsw_index_exists", return_value=False)
@patch("scix.mcp_server.disambiguate_query")
def test_disambiguator_failure_falls_through_to_search(
    mock_disambig: MagicMock,
    _mock_hnsw: MagicMock,
) -> None:
    """Regression guard: a disambiguator exception must not break /search.

    If the entity tables are missing or the DB is unhappy, the search tool
    should degrade gracefully to the normal search path rather than
    surfacing an opaque error at the MCP boundary.
    """
    mock_disambig.side_effect = RuntimeError("simulated DB failure")
    mock_conn = MagicMock()

    with patch("scix.search.hybrid_search", return_value=_stub_search_result()) as mock_hybrid:
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            {"query": "Hubble observations"},
        )
        mock_hybrid.assert_called_once()

    data = json.loads(result_json)
    assert "disambiguation" not in data
    assert "papers" in data


# ---------------------------------------------------------------------------
# Schema assertion — must run without a DB.
# ---------------------------------------------------------------------------


def _get_search_tool_schema() -> dict[str, Any]:
    """Extract the ``search`` tool's ``inputSchema`` from the live MCP server."""
    import asyncio

    from mcp.types import ListToolsRequest

    from scix.mcp_server import create_server

    server = create_server(_run_self_test=False)
    handler = server.request_handlers[ListToolsRequest]
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(handler(ListToolsRequest(method="tools/list")))
    finally:
        loop.close()
    tools = result.root.tools if hasattr(result, "root") else result.tools
    search_tool = next(t for t in tools if t.name == "search")
    return search_tool.inputSchema  # type: ignore[no-any-return]


def test_schema_has_disambiguate_field() -> None:
    """AC1: the `search` tool's inputSchema advertises `disambiguate: bool = True`."""
    try:
        schema = _get_search_tool_schema()
    except (ImportError, AttributeError):
        pytest.skip("mcp SDK not installed or server API changed")

    props = schema["properties"]
    assert "disambiguate" in props, "search tool schema is missing 'disambiguate'"

    prop = props["disambiguate"]
    assert prop["type"] == "boolean"
    assert prop["default"] is True
    assert prop.get("description"), "disambiguate property must have a non-empty description"
