"""Unit tests for the ``section_retrieval`` MCP tool.

These tests must run on CPU-only CI: no GPU, no live DB, no real model.
They cover:

* RRF math correctness (incl. ordering, k constant, disjoint inputs).
* Snippet truncation at the 500-char cap.
* Filters apply (mocked DB; assert SQL params carry filter values).
* Tool registration via ``startup_self_test``.
* Returned object schema (exactly the required keys).
* No paid-API SDK imports anywhere along the new code path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix.mcp_server import (
    EXPECTED_TOOLS,
    _RRF_K_DEFAULT,
    _SNIPPET_MAX_CHARS,
    _dispatch_tool,
    _rrf_fuse,
    _section_filter_clauses,
    _truncate_snippet,
    startup_self_test,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestRrfFuse:
    """Reciprocal Rank Fusion math."""

    def test_rrf_default_k_is_60(self) -> None:
        """The PRD acceptance criterion fixes k=60."""
        assert _RRF_K_DEFAULT == 60

    def test_rrf_single_list_known_score(self) -> None:
        """A single ranked list scored alone matches 1/(k+rank) exactly."""
        ranked = ["a", "b", "c"]
        fused = _rrf_fuse([ranked], k_rrf=60)
        assert [k for (k, _s) in fused] == ["a", "b", "c"]
        assert fused[0][1] == pytest.approx(1.0 / (60 + 1))
        assert fused[1][1] == pytest.approx(1.0 / (60 + 2))
        assert fused[2][1] == pytest.approx(1.0 / (60 + 3))

    def test_rrf_two_lists_overlap(self) -> None:
        """Items that appear in both lists get the sum of their per-list scores."""
        list1 = ["a", "b", "c"]
        list2 = ["b", "a", "d"]
        fused = dict(_rrf_fuse([list1, list2], k_rrf=60))
        # a: 1/61 + 1/62
        # b: 1/62 + 1/61
        # c: 1/63
        # d: 1/63
        assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
        assert fused["c"] == pytest.approx(1 / 63)
        assert fused["d"] == pytest.approx(1 / 63)

    def test_rrf_orders_by_descending_score(self) -> None:
        """Higher-ranked combined items come first."""
        list1 = ["x", "y", "z"]
        list2 = ["z", "y", "x"]
        fused = _rrf_fuse([list1, list2], k_rrf=60)
        # x: 1/61 + 1/63; y: 1/62 + 1/62; z: 1/63 + 1/61.
        # x and z tie; y has 2/62 = 1/31 ~ 0.03226 vs x = 1/61+1/63
        # 1/61 + 1/63 = (63+61)/(61*63) = 124/3843 ≈ 0.03226.
        # All three tie within rounding; ordering must be stable by first-seen.
        keys = [k for (k, _s) in fused]
        assert set(keys) == {"x", "y", "z"}
        # Ensure the tie-break preserves first-seen order
        assert keys[0] == "x"  # x appeared first

    def test_rrf_disjoint_lists(self) -> None:
        """Items unique to one list still score 1/(k+rank) for that list."""
        fused = dict(_rrf_fuse([["a"], ["b"]], k_rrf=60))
        assert fused["a"] == pytest.approx(1 / 61)
        assert fused["b"] == pytest.approx(1 / 61)

    def test_rrf_empty_input(self) -> None:
        """Zero ranked lists, or all-empty lists, returns []."""
        assert _rrf_fuse([], k_rrf=60) == []
        assert _rrf_fuse([[], []], k_rrf=60) == []

    def test_rrf_rejects_non_positive_k(self) -> None:
        with pytest.raises(ValueError):
            _rrf_fuse([["a"]], k_rrf=0)
        with pytest.raises(ValueError):
            _rrf_fuse([["a"]], k_rrf=-1)


class TestTruncateSnippet:
    """Snippet truncation contract."""

    def test_default_cap_is_500_chars(self) -> None:
        assert _SNIPPET_MAX_CHARS == 500

    def test_under_limit_returns_unchanged(self) -> None:
        text = "x" * 250
        assert _truncate_snippet(text) == text
        assert len(_truncate_snippet(text)) == 250

    def test_at_limit_returns_unchanged(self) -> None:
        text = "y" * 500
        assert _truncate_snippet(text) == text

    def test_over_limit_truncates_to_500(self) -> None:
        text = "z" * 1000
        out = _truncate_snippet(text)
        assert len(out) <= _SNIPPET_MAX_CHARS
        assert len(out) == 500
        assert out == "z" * 500

    def test_far_over_limit_still_capped(self) -> None:
        text = "abc" * 100_000
        out = _truncate_snippet(text)
        assert len(out) == 500

    def test_none_returns_empty_string(self) -> None:
        assert _truncate_snippet(None) == ""

    def test_empty_returns_empty(self) -> None:
        assert _truncate_snippet("") == ""

    def test_custom_cap_respected(self) -> None:
        assert _truncate_snippet("hello world", max_chars=5) == "hello"


class TestSectionFilterClauses:
    """SQL fragment + param construction for filters."""

    def test_no_filters_emits_nothing(self) -> None:
        sql, params = _section_filter_clauses(None)
        assert sql == ""
        assert params == []
        sql, params = _section_filter_clauses({})
        assert sql == ""
        assert params == []

    def test_discipline_filter(self) -> None:
        sql, params = _section_filter_clauses({"discipline": "astrophysics"})
        assert "p.discipline = %s" in sql
        assert params == ["astrophysics"]

    def test_year_min_filter(self) -> None:
        sql, params = _section_filter_clauses({"year_min": 2020})
        assert "p.year >= %s" in sql
        assert params == [2020]

    def test_year_max_filter(self) -> None:
        sql, params = _section_filter_clauses({"year_max": 2024})
        assert "p.year <= %s" in sql
        assert params == [2024]

    def test_bibcode_prefix_filter(self) -> None:
        sql, params = _section_filter_clauses({"bibcode_prefix": "2024ApJ"})
        assert "p.bibcode LIKE %s" in sql
        assert params == ["2024ApJ%"]

    def test_all_filters_combined(self) -> None:
        sql, params = _section_filter_clauses(
            {
                "discipline": "astrophysics",
                "year_min": 2020,
                "year_max": 2024,
                "bibcode_prefix": "2024ApJ",
            }
        )
        # Order is fixed in the implementation: discipline, year_min, year_max, bibcode_prefix.
        assert params == ["astrophysics", 2020, 2024, "2024ApJ%"]
        assert "p.discipline = %s" in sql
        assert "p.year >= %s" in sql
        assert "p.year <= %s" in sql
        assert "p.bibcode LIKE %s" in sql

    def test_invalid_year_raises(self) -> None:
        with pytest.raises(ValueError):
            _section_filter_clauses({"year_min": "not-a-year"})


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Verify section_retrieval is wired into list_tools()."""

    def test_section_retrieval_in_expected_tools(self) -> None:
        assert "section_retrieval" in EXPECTED_TOOLS

    def test_section_retrieval_appears_in_list_tools(self) -> None:
        try:
            import mcp.types  # noqa: F401
        except ImportError:
            pytest.skip("mcp SDK not installed")

        with patch("scix.mcp_server._init_model_impl"):
            status = startup_self_test()

        assert status["ok"] is True
        assert "section_retrieval" in status["tool_names"]

    def test_section_retrieval_input_schema_shape(self) -> None:
        """Input schema must require ``query`` and accept ``k`` and ``filters``."""
        try:
            import asyncio

            from mcp.types import ListToolsRequest
        except ImportError:
            pytest.skip("mcp SDK not installed")

        with patch("scix.mcp_server._init_model_impl"):
            from scix.mcp_server import create_server

            server = create_server(_run_self_test=False)

        handler = server.request_handlers[ListToolsRequest]
        result = asyncio.run(handler(ListToolsRequest(method="tools/list")))
        tools = result.root.tools if hasattr(result, "root") else result.tools
        by_name = {t.name: t for t in tools}
        assert "section_retrieval" in by_name

        schema = by_name["section_retrieval"].inputSchema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "k" in schema["properties"]
        assert "filters" in schema["properties"]
        assert schema.get("required") == ["query"]
        # Filters object must accept the four documented keys
        filter_schema = schema["properties"]["filters"]
        assert filter_schema.get("type") == "object"
        for key in ("discipline", "year_min", "year_max", "bibcode_prefix"):
            assert key in filter_schema["properties"], f"missing filter key: {key}"


# ---------------------------------------------------------------------------
# End-to-end dispatch (mocked DB)
# ---------------------------------------------------------------------------


def _make_mock_cursor() -> MagicMock:
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.execute.return_value = None
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    return cursor


def _make_mock_conn(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


class _ScriptedCursor:
    """Cursor double that returns scripted fetchall results in order.

    Each call to ``execute`` advances a pointer; the next ``fetchall`` returns
    the next scripted batch. Records ``(sql, params)`` for each ``execute``.
    """

    def __init__(self, scripted: list[list[Any]]):
        self._scripted = list(scripted)
        self._pending: list[Any] | None = None
        self.calls: list[tuple[str, Any]] = []

    def __enter__(self) -> "_ScriptedCursor":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        self.calls.append((sql, params))
        # pop next scripted result for this execute. Non-SELECT execs just
        # have an empty list assigned; fetchall after that returns [].
        if self._scripted:
            self._pending = self._scripted.pop(0)
        else:
            self._pending = []

    def fetchall(self) -> list[Any]:
        if self._pending is None:
            return []
        out = self._pending
        self._pending = []
        return out

    def fetchone(self) -> Any:
        return None


class TestSectionRetrievalDispatch:
    """Black-box tests for ``_dispatch_tool(conn, 'section_retrieval', ...)``."""

    @patch("scix.mcp_server._log_query")
    @patch("scix.mcp_server._encode_section_query", return_value=[0.0] * 1024)
    def test_empty_results_envelope(
        self,
        _mock_encode: MagicMock,
        _mock_log: MagicMock,
    ) -> None:
        cursor = _make_mock_cursor()
        conn = _make_mock_conn(cursor)
        out = _dispatch_tool(
            conn,
            "section_retrieval",
            {"query": "Hubble tension", "k": 5},
        )
        data = json.loads(out)
        assert "error" not in data
        assert data["total"] == 0
        assert data["results"] == []

    @patch("scix.mcp_server._log_query")
    @patch("scix.mcp_server._encode_section_query", return_value=[0.0] * 1024)
    def test_filters_propagate_to_dense_and_bm25_sql(
        self,
        _mock_encode: MagicMock,
        _mock_log: MagicMock,
    ) -> None:
        cursor = _ScriptedCursor(
            [
                [],  # BEGIN
                [],  # SET hnsw.iterative_scan
                [],  # SET hnsw.ef_search
                [],  # dense SELECT
                [],  # COMMIT
                [],  # bm25 WITH ... SELECT
            ]
        )
        conn = _make_mock_conn(cursor)
        out = _dispatch_tool(
            conn,
            "section_retrieval",
            {
                "query": "JWST exoplanet atmospheres",
                "k": 3,
                "filters": {
                    "discipline": "astrophysics",
                    "year_min": 2022,
                    "year_max": 2024,
                    "bibcode_prefix": "2024",
                },
            },
        )
        data = json.loads(out)
        assert "error" not in data, data

        # Find the dense and bm25 SQL invocations and verify filter params
        # were threaded through.
        select_calls = [
            (sql, params)
            for (sql, params) in cursor.calls
            if isinstance(sql, str) and ("section_embeddings" in sql or "papers_fulltext" in sql)
        ]
        assert select_calls, "expected at least one SELECT against section tables"

        for sql, params in select_calls:
            # Filter clauses must appear in the SQL fragment.
            assert "p.discipline = %s" in sql
            assert "p.year >= %s" in sql
            assert "p.year <= %s" in sql
            assert "p.bibcode LIKE %s" in sql
            # Filter values must appear among the bound params.
            assert "astrophysics" in params
            assert 2022 in params
            assert 2024 in params
            assert "2024%" in params

    @patch("scix.mcp_server._log_query")
    @patch("scix.mcp_server._encode_section_query", return_value=[0.0] * 1024)
    def test_returned_object_schema(
        self,
        _mock_encode: MagicMock,
        _mock_log: MagicMock,
    ) -> None:
        # Script:
        #   BEGIN, SET, SET, dense SELECT (1 row), COMMIT,
        #   bm25 SELECT (1 row, same key as dense),
        #   payload SELECT (papers_fulltext.sections lookup),
        #   canonical_url SELECT (papers.identifier lookup).
        sections_blob = [
            {
                "heading": "Introduction",
                "text": "x" * 800,  # >500 chars; must be truncated
            },
        ]
        cursor = _ScriptedCursor(
            [
                [],  # BEGIN
                [],  # SET iterative_scan
                [],  # SET ef_search
                [("2024ApJ...001..1A", 0, 0.10)],  # dense
                [],  # COMMIT
                [("2024ApJ...001..1A", 0, 0.50)],  # bm25
                [("2024ApJ...001..1A", sections_blob)],  # payload
                [("2024ApJ...001..1A", ["2401.12345"])],  # canonical_url
            ]
        )
        conn = _make_mock_conn(cursor)

        out = _dispatch_tool(
            conn,
            "section_retrieval",
            {"query": "transit spectroscopy", "k": 5},
        )
        data = json.loads(out)
        assert "error" not in data, data
        assert data["total"] == 1
        item = data["results"][0]
        # Exact key set required by acceptance criterion 5.
        assert set(item.keys()) == {
            "bibcode",
            "section_heading",
            "snippet",
            "score",
            "canonical_url",
        }
        assert item["bibcode"] == "2024ApJ...001..1A"
        assert item["section_heading"] == "Introduction"
        assert isinstance(item["score"], float)
        assert item["score"] > 0.0
        assert len(item["snippet"]) <= _SNIPPET_MAX_CHARS
        assert len(item["snippet"]) == 500
        assert item["canonical_url"] == "https://arxiv.org/abs/2401.12345"

    @patch("scix.mcp_server._log_query")
    def test_invalid_query_returns_structured_error(
        self,
        _mock_log: MagicMock,
    ) -> None:
        cursor = _make_mock_cursor()
        conn = _make_mock_conn(cursor)
        out = _dispatch_tool(conn, "section_retrieval", {"query": ""})
        data = json.loads(out)
        assert "error" in data

    @patch("scix.mcp_server._log_query")
    @patch("scix.mcp_server._encode_section_query", return_value=[0.0] * 1024)
    def test_invalid_k_returns_structured_error(
        self,
        _mock_encode: MagicMock,
        _mock_log: MagicMock,
    ) -> None:
        cursor = _make_mock_cursor()
        conn = _make_mock_conn(cursor)
        out = _dispatch_tool(
            conn,
            "section_retrieval",
            {"query": "ok", "k": 0},
        )
        data = json.loads(out)
        assert "error" in data


# ---------------------------------------------------------------------------
# Import policy: no paid-API SDKs anywhere along the new code path
# ---------------------------------------------------------------------------


_FORBIDDEN_IMPORT_PATTERNS = (
    re.compile(r"^\s*import\s+openai\b", re.MULTILINE),
    re.compile(r"^\s*from\s+openai\b", re.MULTILINE),
    re.compile(r"^\s*import\s+cohere\b", re.MULTILINE),
    re.compile(r"^\s*from\s+cohere\b", re.MULTILINE),
    re.compile(r"^\s*import\s+anthropic\b", re.MULTILINE),
    re.compile(r"^\s*from\s+anthropic\b", re.MULTILINE),
    re.compile(r"^\s*import\s+voyageai\b", re.MULTILINE),
    re.compile(r"^\s*from\s+voyageai\b", re.MULTILINE),
)


def test_mcp_server_does_not_import_paid_sdks() -> None:
    """No paid-API SDK imports may appear in the MCP server module."""
    src = Path(__file__).resolve().parents[1] / "src" / "scix" / "mcp_server.py"
    text = src.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_IMPORT_PATTERNS:
        match = pattern.search(text)
        assert match is None, (
            f"forbidden import pattern {pattern.pattern!r} found in mcp_server.py "
            f"at line {text[:match.start()].count(chr(10)) + 1 if match else None}"
        )


def test_section_pipeline_does_not_import_paid_sdks() -> None:
    """Reused encoder loader must remain SDK-free."""
    src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scix"
        / "embeddings"
        / "section_pipeline.py"
    )
    text = src.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_IMPORT_PATTERNS:
        assert pattern.search(text) is None, (
            f"forbidden import pattern {pattern.pattern!r} found in section_pipeline.py"
        )
