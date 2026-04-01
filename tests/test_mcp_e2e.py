"""End-to-end tests for the MCP server using the mcp SDK.

Tests the full MCP server lifecycle: create_server -> list_tools -> call_tool.
Requires:
  - mcp SDK installed (pip install mcp)
  - Running scix database with data

Skips gracefully if either dependency is unavailable.
"""

from __future__ import annotations

import json
import os

import psycopg
import pytest

# Skip entire module if mcp SDK is not installed
mcp = pytest.importorskip("mcp", reason="mcp SDK not installed")

from mcp.types import TextContent  # noqa: E402

DSN = os.environ.get("SCIX_DSN", "dbname=scix")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_available():
    """Check database availability once for the module."""
    try:
        c = psycopg.connect(DSN)
        c.close()
        return True
    except psycopg.OperationalError:
        pytest.skip("scix database not available")


@pytest.fixture(scope="module")
def server(db_available):
    """Create the MCP server once for the module."""
    from scix.mcp_server import create_server

    return create_server()


@pytest.fixture(scope="module")
def list_tools_handler(server):
    """Extract the list_tools handler from the server."""
    # The server registers handlers via decorators; access internal registry
    return server.list_tools_handlers[0] if server.list_tools_handlers else None


@pytest.fixture(scope="module")
def call_tool_handler(server):
    """Extract the call_tool handler from the server."""
    return server.call_tool_handlers[0] if server.call_tool_handlers else None


def _has_papers_direct() -> bool:
    """Check if papers exist (direct connection, not via fixture)."""
    try:
        c = psycopg.connect(DSN)
        with c.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM papers LIMIT 1)")
            result = cur.fetchone()[0]
        c.close()
        return result
    except Exception:
        return False


def _get_any_bibcode_direct() -> str | None:
    """Get a bibcode directly from the database."""
    try:
        c = psycopg.connect(DSN)
        with c.cursor() as cur:
            cur.execute("SELECT bibcode FROM papers LIMIT 1")
            row = cur.fetchone()
        c.close()
        return row[0] if row else None
    except Exception:
        return None


def _get_any_author_direct() -> str | None:
    """Get an author name directly from the database."""
    try:
        c = psycopg.connect(DSN)
        with c.cursor() as cur:
            cur.execute("SELECT first_author FROM papers WHERE first_author IS NOT NULL LIMIT 1")
            row = cur.fetchone()
        c.close()
        return row[0] if row else None
    except Exception:
        return None


def _parse_tool_result(result: list[TextContent]) -> dict:
    """Parse the TextContent list returned by call_tool into a dict."""
    assert len(result) >= 1
    assert result[0].type == "text"
    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestListTools:
    @pytest.mark.asyncio
    async def test_lists_all_eight_tools(self, list_tools_handler) -> None:
        if list_tools_handler is None:
            pytest.skip("No list_tools handler registered")
        tools = await list_tools_handler()
        tool_names = {t.name for t in tools}
        expected = {
            "semantic_search",
            "keyword_search",
            "get_paper",
            "get_citations",
            "get_references",
            "get_author_papers",
            "facet_counts",
            "co_citation_analysis",
            "bibliographic_coupling",
            "citation_chain",
            "temporal_evolution",
            "get_paper_metrics",
            "explore_community",
            "concept_search",
            "health_check",
        }
        assert expected == tool_names

    @pytest.mark.asyncio
    async def test_tools_have_input_schemas(self, list_tools_handler) -> None:
        if list_tools_handler is None:
            pytest.skip("No list_tools handler registered")
        tools = await list_tools_handler()
        for tool in tools:
            assert tool.inputSchema is not None
            assert "type" in tool.inputSchema


@pytest.mark.integration
class TestCallToolHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        result = await call_tool_handler("health_check", {})
        data = _parse_tool_result(result)
        assert data["db"] == "ok"
        assert "model_cached" in data
        assert "pool" in data


@pytest.mark.integration
class TestCallToolKeywordSearch:
    @pytest.mark.asyncio
    async def test_keyword_search(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        if not _has_papers_direct():
            pytest.skip("No papers in database")
        result = await call_tool_handler("keyword_search", {"terms": "galaxy", "limit": 3})
        data = _parse_tool_result(result)
        assert "papers" in data
        assert "total" in data
        assert "timing_ms" in data

    @pytest.mark.asyncio
    async def test_keyword_search_with_filters(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        if not _has_papers_direct():
            pytest.skip("No papers in database")
        result = await call_tool_handler(
            "keyword_search",
            {"terms": "star", "filters": {"year_min": 2023}, "limit": 3},
        )
        data = _parse_tool_result(result)
        assert "papers" in data


@pytest.mark.integration
class TestCallToolGetPaper:
    @pytest.mark.asyncio
    async def test_existing_paper(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        bibcode = _get_any_bibcode_direct()
        if not bibcode:
            pytest.skip("No papers in database")
        result = await call_tool_handler("get_paper", {"bibcode": bibcode})
        data = _parse_tool_result(result)
        assert data["total"] == 1
        assert data["papers"][0]["bibcode"] == bibcode

    @pytest.mark.asyncio
    async def test_nonexistent_paper(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        result = await call_tool_handler("get_paper", {"bibcode": "NONEXISTENT_XYZ_999"})
        data = _parse_tool_result(result)
        assert data["total"] == 0


@pytest.mark.integration
class TestCallToolGetCitations:
    @pytest.mark.asyncio
    async def test_get_citations(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        bibcode = _get_any_bibcode_direct()
        if not bibcode:
            pytest.skip("No papers in database")
        result = await call_tool_handler("get_citations", {"bibcode": bibcode, "limit": 5})
        data = _parse_tool_result(result)
        assert "papers" in data
        assert "timing_ms" in data


@pytest.mark.integration
class TestCallToolGetReferences:
    @pytest.mark.asyncio
    async def test_get_references(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        bibcode = _get_any_bibcode_direct()
        if not bibcode:
            pytest.skip("No papers in database")
        result = await call_tool_handler("get_references", {"bibcode": bibcode, "limit": 5})
        data = _parse_tool_result(result)
        assert "papers" in data
        assert "timing_ms" in data


@pytest.mark.integration
class TestCallToolGetAuthorPapers:
    @pytest.mark.asyncio
    async def test_get_author_papers(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        author = _get_any_author_direct()
        if not author:
            pytest.skip("No authors in database")
        result = await call_tool_handler("get_author_papers", {"author_name": author})
        data = _parse_tool_result(result)
        assert "papers" in data
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_get_author_papers_with_year_range(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        author = _get_any_author_direct()
        if not author:
            pytest.skip("No authors in database")
        result = await call_tool_handler(
            "get_author_papers",
            {"author_name": author, "year_min": 2020, "year_max": 2025},
        )
        data = _parse_tool_result(result)
        assert "papers" in data


@pytest.mark.integration
class TestCallToolFacetCounts:
    @pytest.mark.asyncio
    async def test_year_facets(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        if not _has_papers_direct():
            pytest.skip("No papers in database")
        result = await call_tool_handler("facet_counts", {"field": "year", "limit": 10})
        data = _parse_tool_result(result)
        assert "metadata" in data
        assert data["metadata"]["facet_field"] == "year"

    @pytest.mark.asyncio
    async def test_doctype_facets(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        if not _has_papers_direct():
            pytest.skip("No papers in database")
        result = await call_tool_handler("facet_counts", {"field": "doctype"})
        data = _parse_tool_result(result)
        assert "metadata" in data
        assert data["metadata"]["facet_field"] == "doctype"


@pytest.mark.integration
class TestCallToolSemanticSearch:
    @pytest.mark.asyncio
    async def test_semantic_search_error_or_result(self, call_tool_handler) -> None:
        """semantic_search returns papers if model available, error dict otherwise."""
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        result = await call_tool_handler("semantic_search", {"query": "dark energy", "limit": 3})
        data = _parse_tool_result(result)
        assert "papers" in data or "error" in data


@pytest.mark.integration
class TestCallToolUnknown:
    @pytest.mark.asyncio
    async def test_unknown_tool(self, call_tool_handler) -> None:
        if call_tool_handler is None:
            pytest.skip("No call_tool handler registered")
        result = await call_tool_handler("nonexistent_tool_xyz", {})
        data = _parse_tool_result(result)
        assert "error" in data
        assert "Unknown tool" in data["error"]
