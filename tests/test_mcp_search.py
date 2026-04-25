"""Tests for the MCP `search` tool's cross-encoder rerank wiring (mcp-rerank-default).

Covers the env-var driven reranker selection plumbed through
``scix.mcp_server._handle_search``:

1. ``SCIX_RERANK_DEFAULT_MODEL`` defaults to ``'off'`` per the M1 negative
   result, so the default code path never constructs a ``CrossEncoderReranker``
   and ``hybrid_search`` is invoked with ``reranker=None``.
2. ``use_rerank=False`` skips the reranker even when a non-'off' model is
   configured.
3. With a non-'off' model AND ``use_rerank=True``, the reranker is invoked and
   the returned ordering differs from the un-reranked candidate set — while
   the candidate *set* (bibcodes) stays identical.
4. Reranker is only invoked when ``limit <= 20`` (PRD M3 latency envelope).
5. Unknown env values fall back to ``'off'`` with a warning (no construction).
6. The MCP `search` tool's inputSchema advertises ``use_rerank: bool = True``.

All tests mock the DB connection, the embedding pre-load, and the
``CrossEncoderReranker`` class so no model weights are downloaded.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix import mcp_server
from scix.mcp_server import _dispatch_tool, _reset_default_reranker_cache
from scix.search import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Deterministic candidate set used by the patched hybrid_search. Order here is
# the "un-reranked" ordering that the tests reason about.
_FIXTURE_CANDIDATES: list[dict[str, Any]] = [
    {"bibcode": f"2024ApJ...{i:03d}A", "title": f"Paper {i}", "rrf_score": 1.0 / (i + 1)}
    for i in range(10)
]


@pytest.fixture(autouse=True)
def _reset_reranker_singleton() -> None:
    """Drop the module-level reranker cache between tests so env changes apply."""
    _reset_default_reranker_cache()
    yield
    _reset_default_reranker_cache()


@pytest.fixture(autouse=True)
def _disable_disambiguator() -> None:
    """Patch the disambiguator so the search path never short-circuits."""
    with patch("scix.mcp_server.disambiguate_query", return_value=[]):
        yield


@pytest.fixture(autouse=True)
def _disable_hnsw() -> None:
    """Force the lexical-only path inside _handle_search — no embedding load."""
    with patch("scix.mcp_server._hnsw_index_exists", return_value=False):
        yield


def _make_hybrid_stub(
    candidates: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Return a MagicMock that mimics ``search.hybrid_search``.

    Applies the (optional) ``reranker`` kwarg to the candidate list before
    wrapping it in a SearchResult so the test can inspect ordering changes.
    """
    if candidates is None:
        candidates = list(_FIXTURE_CANDIDATES)

    def _impl(*_args: Any, **kwargs: Any) -> SearchResult:
        papers = list(candidates)
        reranker = kwargs.get("reranker")
        if reranker is not None:
            papers = reranker("query-text", papers)
        # Honour top_n so the limit > 20 test is realistic.
        top_n = kwargs.get("top_n")
        if top_n is not None:
            papers = papers[:top_n]
        return SearchResult(papers=papers, total=len(papers), timing_ms={"total_ms": 0.0})

    stub = MagicMock(side_effect=_impl)
    return stub


class _ReverseStubReranker:
    """Stand-in for ``CrossEncoderReranker`` that reverses paper order.

    The constructor records its model_name so tests can assert which alias was
    resolved. ``__call__`` reverses the candidate list, attaching a deterministic
    ``rerank_score`` so the new ordering is unambiguous.
    """

    constructed: list[str] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        type(self).constructed.append(model_name)

    def __call__(self, query: str, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {**p, "rerank_score": float(idx)}
            for idx, p in enumerate(reversed(papers))
        ]


@pytest.fixture
def _reset_stub_reranker_log() -> None:
    _ReverseStubReranker.constructed = []
    yield
    _ReverseStubReranker.constructed = []


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------


def test_default_env_off_passes_no_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3: default SCIX_RERANK_DEFAULT_MODEL='off' → reranker=None, no construction."""
    monkeypatch.delenv("SCIX_RERANK_DEFAULT_MODEL", raising=False)
    stub_hybrid = _make_hybrid_stub()
    mock_conn = MagicMock()

    with (
        patch("scix.mcp_server.search.hybrid_search", stub_hybrid),
        patch("scix.mcp_server.CrossEncoderReranker") as mock_cls,
    ):
        result_json = _dispatch_tool(
            mock_conn,
            "search",
            {"query": "exoplanet detection", "limit": 5},
        )

        # No CrossEncoderReranker construction in the default code path.
        mock_cls.assert_not_called()

    # hybrid_search received reranker=None.
    _, kwargs = stub_hybrid.call_args
    assert kwargs["reranker"] is None

    data = json.loads(result_json)
    assert data["total"] == 5
    # Order is the un-reranked candidate ordering.
    assert [p["bibcode"] for p in data["papers"]] == [
        c["bibcode"] for c in _FIXTURE_CANDIDATES[:5]
    ]


def test_use_rerank_false_skips_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC4(a): use_rerank=False bypasses the reranker even when a model is configured."""
    monkeypatch.setenv("SCIX_RERANK_DEFAULT_MODEL", "minilm")
    stub_hybrid = _make_hybrid_stub()
    mock_conn = MagicMock()

    with (
        patch("scix.mcp_server.search.hybrid_search", stub_hybrid),
        patch("scix.mcp_server.CrossEncoderReranker") as mock_cls,
    ):
        _dispatch_tool(
            mock_conn,
            "search",
            {"query": "exoplanet detection", "limit": 5, "use_rerank": False},
        )

        # use_rerank=False short-circuits before the factory is called, so no
        # CrossEncoderReranker is constructed.
        mock_cls.assert_not_called()

    _, kwargs = stub_hybrid.call_args
    assert kwargs["reranker"] is None


def test_use_rerank_true_with_stub_model_changes_order(
    monkeypatch: pytest.MonkeyPatch,
    _reset_stub_reranker_log: None,
) -> None:
    """AC4(b): set != 'off' AND use_rerank=True → ordering differs, candidate set identical."""
    monkeypatch.setenv("SCIX_RERANK_DEFAULT_MODEL", "minilm")

    # Run once with use_rerank=False to capture the un-reranked ordering.
    stub_hybrid_off = _make_hybrid_stub()
    mock_conn = MagicMock()
    with patch("scix.mcp_server.search.hybrid_search", stub_hybrid_off):
        off_json = _dispatch_tool(
            mock_conn,
            "search",
            {"query": "exoplanet detection", "limit": 10, "use_rerank": False},
        )
    off_bibs = [p["bibcode"] for p in json.loads(off_json)["papers"]]

    # Now run with use_rerank=True and a stub reranker that reverses order.
    stub_hybrid_on = _make_hybrid_stub()
    with (
        patch("scix.mcp_server.search.hybrid_search", stub_hybrid_on),
        patch("scix.mcp_server.CrossEncoderReranker", _ReverseStubReranker),
    ):
        on_json = _dispatch_tool(
            mock_conn,
            "search",
            {"query": "exoplanet detection", "limit": 10, "use_rerank": True},
        )
    on_bibs = [p["bibcode"] for p in json.loads(on_json)["papers"]]

    # Stub reranker was constructed exactly once with the resolved alias.
    assert _ReverseStubReranker.constructed == ["cross-encoder/ms-marco-MiniLM-L-12-v2"]

    # Candidate set is identical between the two paths.
    assert set(off_bibs) == set(on_bibs)
    # Ordering differs (the stub reverses).
    assert off_bibs != on_bibs
    assert on_bibs == list(reversed(off_bibs))


def test_top_k_above_20_skips_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC6: limit > 20 bypasses the reranker even when a model is configured."""
    monkeypatch.setenv("SCIX_RERANK_DEFAULT_MODEL", "minilm")
    # Provide 30 fixtures so limit=25 has room.
    candidates = [
        {"bibcode": f"2024ApJ...{i:03d}A", "title": f"Paper {i}"} for i in range(30)
    ]
    stub_hybrid = _make_hybrid_stub(candidates=candidates)
    mock_conn = MagicMock()

    with (
        patch("scix.mcp_server.search.hybrid_search", stub_hybrid),
        patch("scix.mcp_server.CrossEncoderReranker") as mock_cls,
    ):
        _dispatch_tool(
            mock_conn,
            "search",
            {"query": "exoplanet detection", "limit": 25, "use_rerank": True},
        )
        # Top-k guard fires before the factory is called.
        mock_cls.assert_not_called()

    _, kwargs = stub_hybrid.call_args
    assert kwargs["reranker"] is None


def test_unknown_env_value_falls_back_to_off(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown SCIX_RERANK_DEFAULT_MODEL value warns and falls back to 'off'."""
    monkeypatch.setenv("SCIX_RERANK_DEFAULT_MODEL", "totally-not-a-real-model")
    stub_hybrid = _make_hybrid_stub()
    mock_conn = MagicMock()

    with (
        caplog.at_level("WARNING", logger="scix.mcp_server"),
        patch("scix.mcp_server.search.hybrid_search", stub_hybrid),
        patch("scix.mcp_server.CrossEncoderReranker") as mock_cls,
    ):
        _dispatch_tool(
            mock_conn,
            "search",
            {"query": "x", "limit": 5, "use_rerank": True},
        )
        mock_cls.assert_not_called()

    _, kwargs = stub_hybrid.call_args
    assert kwargs["reranker"] is None
    assert any(
        "SCIX_RERANK_DEFAULT_MODEL" in record.message and "falling back to 'off'" in record.message
        for record in caplog.records
    )


def test_bge_large_alias_resolves(
    monkeypatch: pytest.MonkeyPatch, _reset_stub_reranker_log: None
) -> None:
    """The 'bge-large' env value resolves to the BAAI cross-encoder identifier."""
    monkeypatch.setenv("SCIX_RERANK_DEFAULT_MODEL", "bge-large")
    stub_hybrid = _make_hybrid_stub()
    mock_conn = MagicMock()

    with (
        patch("scix.mcp_server.search.hybrid_search", stub_hybrid),
        patch("scix.mcp_server.CrossEncoderReranker", _ReverseStubReranker),
    ):
        _dispatch_tool(
            mock_conn,
            "search",
            {"query": "x", "limit": 5, "use_rerank": True},
        )

    assert _ReverseStubReranker.constructed == ["BAAI/bge-reranker-large"]
    _, kwargs = stub_hybrid.call_args
    assert kwargs["reranker"] is not None


def test_singleton_reused_across_calls(
    monkeypatch: pytest.MonkeyPatch, _reset_stub_reranker_log: None
) -> None:
    """The reranker is constructed once per process, then reused."""
    monkeypatch.setenv("SCIX_RERANK_DEFAULT_MODEL", "minilm")
    stub_hybrid = _make_hybrid_stub()
    mock_conn = MagicMock()

    with (
        patch("scix.mcp_server.search.hybrid_search", stub_hybrid),
        patch("scix.mcp_server.CrossEncoderReranker", _ReverseStubReranker),
    ):
        for _ in range(3):
            _dispatch_tool(
                mock_conn,
                "search",
                {"query": "x", "limit": 5, "use_rerank": True},
            )

    # Constructed exactly once even though we made three search calls.
    assert _ReverseStubReranker.constructed == ["cross-encoder/ms-marco-MiniLM-L-12-v2"]


# ---------------------------------------------------------------------------
# Schema assertion (AC2: tool advertises use_rerank)
# ---------------------------------------------------------------------------


def _get_search_tool_schema() -> dict[str, Any]:
    """Extract the live ``search`` tool's inputSchema from the MCP server."""
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


def test_schema_has_use_rerank_field() -> None:
    """AC2: the search tool exposes use_rerank: bool = True in its inputSchema."""
    try:
        schema = _get_search_tool_schema()
    except (ImportError, AttributeError):
        pytest.skip("mcp SDK not installed or server API changed")

    props = schema["properties"]
    assert "use_rerank" in props, "search tool schema is missing 'use_rerank'"

    prop = props["use_rerank"]
    assert prop["type"] == "boolean"
    assert prop["default"] is True
    assert prop.get("description"), "use_rerank property must have a non-empty description"


# ---------------------------------------------------------------------------
# Direct unit test of the resolver (AC1: env mapping)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("off", None),
        ("OFF", None),
        ("minilm", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
        ("MiniLM", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
        ("bge-large", "BAAI/bge-reranker-large"),
        ("nonsense", None),
    ],
)
def test_resolve_default_reranker_model(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("SCIX_RERANK_DEFAULT_MODEL", raising=False)
    else:
        monkeypatch.setenv("SCIX_RERANK_DEFAULT_MODEL", value)
    assert mcp_server._resolve_default_reranker_model() == expected
