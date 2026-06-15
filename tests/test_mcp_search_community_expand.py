"""MCP integration tests for the ``search.community_expand`` lane.

Covers the wire format added by bead xz4.1.40:

* ``community_expand: bool`` argument is advertised in the search tool's
  ``inputSchema``.
* When ``community_expand=true``:
    - explicit single-id ``filters.entity_ids`` is honored as the seed;
    - free-text resolution to a single unambiguous entity is honored;
    - free-text with no resolvable entity returns a structured
      ``{"error_code": "community_expand_no_seed", ...}`` envelope;
    - free-text with multiple entity candidates returns the same
      envelope, plus a ``candidates`` list so the agent can pick one;
    - ``filters.entity_ids`` with >1 entries returns the same envelope;
    - super-hub guard fires from the underlying function and is lifted
      into a structured ``seed_too_broad`` envelope.

All tests stub the DB connection and ``community_expand_search``; no real
database or HNSW index is touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix import mcp_server
from scix.mcp_server import _dispatch_tool
from scix.search import SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _StubCandidate:
    """Mirror of :class:`scix.entity_resolver.ResolvedCandidate` for tests."""

    entity_id: int
    canonical_name: str
    entity_type: str = "instrument"
    source: str = "ads_facet"
    discipline: str = "astronomy"
    confidence: float = 0.9
    match_method: str = "exact"


def _happy_search_result() -> SearchResult:
    return SearchResult(
        papers=[
            {
                "bibcode": "2024ApJ...001A",
                "title": "WFC3 + STIS dual-instrument survey",
                "first_author": "Smith",
                "year": 2024,
                "citation_count": 42,
                "abstract_snippet": "WFC3 + STIS observations.",
                "cooccur_count": 25,
                "best_neighbor_id": 101,
            }
        ],
        total=1,
        timing_ms={"cooccur_neighbors_ms": 12.5, "cooccur_papers_ms": 18.0},
        metadata={
            "seed_entity_id": 999,
            "seed_paper_count": 5,
            "neighbor_count": 3,
            "truncated_seed_papers": False,
            "neighbors": [
                {"entity_id": 101, "canonical_name": "WFC3", "cooccur_count": 25},
            ],
        },
    )


def _seed_too_broad_result() -> SearchResult:
    return SearchResult(
        papers=[],
        total=0,
        timing_ms={"cooccur_neighbors_ms": 0.0, "cooccur_papers_ms": 0.0},
        metadata={
            "seed_entity_id": 999,
            "seed_paper_count": 60_000,
            "error_code": "seed_too_broad",
            "error": "seed entity has 60,000 linked papers — above the super-hub threshold of 50,000",
            "hint": "Narrow to a more specific entity.",
        },
    )


@pytest.fixture(autouse=True)
def _no_unscoped_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the unscoped-broad-query guard so community_expand path runs."""
    monkeypatch.setattr(mcp_server, "_is_unscoped_broad_query", lambda *a, **kw: False)


@pytest.fixture(autouse=True)
def _no_disambiguate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the in-query disambiguator — community_expand has its own seed flow."""
    monkeypatch.setattr(mcp_server, "_maybe_disambiguate", lambda *a, **kw: None)


@pytest.fixture
def conn() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# inputSchema advertises the new arg
# ---------------------------------------------------------------------------


def _get_search_tool_schema() -> dict[str, Any]:
    """Extract the live ``search`` tool's inputSchema from the MCP server.

    Mirrors the helper in tests/test_mcp_search.py so the schema is asserted
    against the real wire protocol the SDK exposes — not a hand-crafted dict.
    """
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


def test_search_tool_advertises_community_expand_arg() -> None:
    """The MCP search tool inputSchema must list community_expand: bool default false."""
    try:
        schema = _get_search_tool_schema()
    except (ImportError, AttributeError):
        pytest.skip("mcp SDK not installed or server API changed")

    props = schema["properties"]
    assert "community_expand" in props
    assert props["community_expand"]["type"] == "boolean"
    assert props["community_expand"].get("default") is False


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_community_expand_explicit_single_entity_id(conn: MagicMock) -> None:
    """filters.entity_ids=[999] with community_expand=true uses 999 as seed."""
    with patch.object(
        mcp_server.search,
        "community_expand_search",
        return_value=_happy_search_result(),
    ) as mock_fn:
        result_json = _dispatch_tool(
            conn,
            "search",
            {
                "query": "HST instrument suite",
                "community_expand": True,
                "filters": {"entity_ids": [999]},
            },
        )

    assert mock_fn.called
    kwargs = mock_fn.call_args.kwargs
    args = mock_fn.call_args.args
    # The seed entity_id must be passed positionally or via kwarg.
    assert 999 in args or kwargs.get("seed_entity_id") == 999

    payload = json.loads(result_json)
    assert payload["total"] == 1
    assert payload["papers"][0]["bibcode"] == "2024ApJ...001A"
    # Metadata must be propagated through.
    assert payload["metadata"]["seed_entity_id"] == 999
    assert payload["metadata"]["neighbor_count"] == 3


def test_community_expand_inferred_single_entity_match(conn: MagicMock) -> None:
    """Free-text query resolves to exactly one entity → run lane."""
    fake_resolver = MagicMock()
    fake_resolver.resolve.return_value = [_StubCandidate(entity_id=999, canonical_name="HST")]

    with (
        patch.object(mcp_server, "EntityResolver", return_value=fake_resolver),
        patch.object(
            mcp_server.search,
            "community_expand_search",
            return_value=_happy_search_result(),
        ) as mock_fn,
    ):
        result_json = _dispatch_tool(
            conn,
            "search",
            {"query": "HST", "community_expand": True},
        )

    assert mock_fn.called
    args = mock_fn.call_args.args
    kwargs = mock_fn.call_args.kwargs
    assert 999 in args or kwargs.get("seed_entity_id") == 999

    payload = json.loads(result_json)
    assert payload["total"] == 1


# ---------------------------------------------------------------------------
# Error envelopes
# ---------------------------------------------------------------------------


def test_no_seed_envelope_when_no_resolution(conn: MagicMock) -> None:
    """Free-text query with zero entity matches → community_expand_no_seed."""
    fake_resolver = MagicMock()
    fake_resolver.resolve.return_value = []

    with patch.object(mcp_server, "EntityResolver", return_value=fake_resolver):
        result_json = _dispatch_tool(
            conn,
            "search",
            {"query": "nonexistent entity term", "community_expand": True},
        )

    payload = json.loads(result_json)
    assert payload.get("error_code") == "community_expand_no_seed"
    assert "error" in payload
    assert "hint" in payload
    # Should not crash, and should not run search.
    assert "papers" not in payload or not payload.get("papers")


def test_no_seed_envelope_lists_candidates_when_ambiguous(conn: MagicMock) -> None:
    """Free-text query with multiple entity matches → no_seed envelope + candidates."""
    fake_resolver = MagicMock()
    fake_resolver.resolve.return_value = [
        _StubCandidate(entity_id=1, canonical_name="LIGO", entity_type="instrument"),
        _StubCandidate(entity_id=2, canonical_name="LIGO Voyager", entity_type="mission"),
    ]

    with patch.object(mcp_server, "EntityResolver", return_value=fake_resolver):
        result_json = _dispatch_tool(
            conn,
            "search",
            {"query": "LIGO", "community_expand": True},
        )

    payload = json.loads(result_json)
    assert payload.get("error_code") == "community_expand_no_seed"
    assert isinstance(payload.get("candidates"), list)
    assert len(payload["candidates"]) == 2
    ids = [c["entity_id"] for c in payload["candidates"]]
    assert ids == [1, 2]


def test_no_seed_envelope_when_explicit_entity_ids_is_multiple(conn: MagicMock) -> None:
    """filters.entity_ids with >1 entries is rejected — seed must be unique."""
    result_json = _dispatch_tool(
        conn,
        "search",
        {
            "query": "anything",
            "community_expand": True,
            "filters": {"entity_ids": [1, 2, 3]},
        },
    )
    payload = json.loads(result_json)
    assert payload.get("error_code") == "community_expand_no_seed"


def test_seed_too_broad_envelope_lifted_from_function(conn: MagicMock) -> None:
    """When community_expand_search returns metadata.error_code='seed_too_broad',
    the MCP layer must surface it as a top-level structured-error response."""
    with patch.object(
        mcp_server.search,
        "community_expand_search",
        return_value=_seed_too_broad_result(),
    ):
        result_json = _dispatch_tool(
            conn,
            "search",
            {
                "query": "Frequency",
                "community_expand": True,
                "filters": {"entity_ids": [999]},
            },
        )

    payload = json.loads(result_json)
    assert payload.get("error_code") == "seed_too_broad"
    assert payload["seed_paper_count"] == 60_000
    assert "hint" in payload


# ---------------------------------------------------------------------------
# Default-off
# ---------------------------------------------------------------------------


def test_community_expand_default_off_runs_normal_hybrid(conn: MagicMock) -> None:
    """Without community_expand=true, the lane is NOT invoked even if entity_ids is set."""
    fake_hybrid = SearchResult(papers=[{"bibcode": "TEST"}], total=1, timing_ms={"total_ms": 1.0})
    with (
        patch.object(mcp_server.search, "community_expand_search") as expand_fn,
        patch.object(mcp_server.search, "hybrid_search", return_value=fake_hybrid),
        patch.object(mcp_server, "_hnsw_index_exists", return_value=False),
    ):
        result_json = _dispatch_tool(
            conn,
            "search",
            {"query": "anything", "filters": {"entity_ids": [999]}},
        )

    assert not expand_fn.called
    payload = json.loads(result_json)
    assert payload["total"] == 1
