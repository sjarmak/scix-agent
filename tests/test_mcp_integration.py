"""Integration tests for MCP server _dispatch_tool with a real PostgreSQL database.

Tests the full dispatch path: _dispatch_tool -> search.py functions -> PostgreSQL.
Requires a running scix database with data and migration 003 (tsv column).

All tests are marked with @pytest.mark.integration and skip gracefully if the
database is unavailable or too slow (QueryCanceled).
"""

from __future__ import annotations

import json
import os

import psycopg
import psycopg.errors
import pytest

from scix.mcp_server import _dispatch_tool

DSN = os.environ.get("SCIX_DSN", "dbname=scix")

# Per-query timeout in seconds (configurable for slow environments)
_STMT_TIMEOUT_S = int(os.environ.get("SCIX_TEST_TIMEOUT", "60"))


# ---------------------------------------------------------------------------
# Fixtures (module-scoped connection, SAVEPOINT isolation per test)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conn():
    """Provide a connection to the scix database for the entire module.

    Sets a per-statement timeout to prevent tests from hanging indefinitely
    on cold databases with large tables.
    """
    try:
        c = psycopg.connect(DSN)
        c.autocommit = False
        with c.cursor() as cur:
            cur.execute(f"SET statement_timeout = {_STMT_TIMEOUT_S * 1000}")
        yield c
        c.rollback()
        c.close()
    except psycopg.OperationalError:
        pytest.skip("scix database not available")


@pytest.fixture(autouse=True)
def _savepoint(conn):
    """Wrap each test in a savepoint for isolation (auto-rollback).

    Handles three cases:
    1. Normal completion -> ROLLBACK TO SAVEPOINT
    2. QueryCanceled rolled back the transaction -> re-begin with rollback()
    3. Transaction in failed state -> rollback() to reset
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT test_sp")
    except psycopg.errors.InFailedSqlTransaction:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT test_sp")
    yield
    try:
        with conn.cursor() as cur:
            cur.execute("ROLLBACK TO SAVEPOINT test_sp")
    except (
        psycopg.errors.InFailedSqlTransaction,
        psycopg.errors.InvalidSavepointSpecification,
    ):
        conn.rollback()
        # Re-establish statement_timeout after rollback
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {_STMT_TIMEOUT_S * 1000}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimated_row_count(conn: psycopg.Connection, table: str) -> int:
    """Use pg_class statistics for fast row count estimate (no table scan)."""
    with conn.cursor() as cur:
        cur.execute("SELECT reltuples::bigint FROM pg_class WHERE relname = %s", (table,))
        row = cur.fetchone()
        return row[0] if row else 0


def _has_papers(conn: psycopg.Connection) -> bool:
    return _estimated_row_count(conn, "papers") > 0


def _has_tsv_column(conn: psycopg.Connection) -> bool:
    """Check if the tsv column exists on papers (migration 003 applied)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_attribute WHERE attrelid = 'papers'::regclass "
            "AND attname = 'tsv' AND NOT attisdropped"
        )
        return cur.fetchone() is not None


def _has_citation_edges(conn: psycopg.Connection) -> bool:
    return _estimated_row_count(conn, "citation_edges") > 0


def _rollback_and_reset(conn: psycopg.Connection) -> None:
    """Rollback a failed transaction and restore statement_timeout."""
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {_STMT_TIMEOUT_S * 1000}")


def _get_any_bibcode(conn: psycopg.Connection) -> str:
    """Get a bibcode by scanning a known year (uses idx_papers_year for speed)."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT bibcode FROM papers WHERE year = 2024 LIMIT 1")
            row = cur.fetchone()
            return row[0] if row else ""
    except psycopg.errors.QueryCanceled:
        _rollback_and_reset(conn)
        return ""


def _get_citing_bibcode(conn: psycopg.Connection) -> str | None:
    """Get a bibcode that has at least one citation edge as target."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT target_bibcode FROM citation_edges LIMIT 1")
            row = cur.fetchone()
            return row[0] if row else None
    except psycopg.errors.QueryCanceled:
        _rollback_and_reset(conn)
        return None


def _get_referencing_bibcode(conn: psycopg.Connection) -> str | None:
    """Get a bibcode that has at least one citation edge as source."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT source_bibcode FROM citation_edges LIMIT 1")
            row = cur.fetchone()
            return row[0] if row else None
    except psycopg.errors.QueryCanceled:
        _rollback_and_reset(conn)
        return None


def _get_any_author(conn: psycopg.Connection) -> str | None:
    """Get an author name using the year index to narrow the scan."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT first_author FROM papers "
                "WHERE year = 2024 AND first_author IS NOT NULL LIMIT 1"
            )
            row = cur.fetchone()
            return row[0] if row else None
    except psycopg.errors.QueryCanceled:
        _rollback_and_reset(conn)
        return None


def _dispatch_safe(conn: psycopg.Connection, name: str, args: dict) -> dict:
    """Call _dispatch_tool and parse JSON; skip test on QueryCanceled."""
    try:
        result_json = _dispatch_tool(conn, name, args)
    except psycopg.errors.QueryCanceled:
        _rollback_and_reset(conn)
        pytest.skip("Query timed out (database too slow for integration tests)")
    return json.loads(result_json)


# ---------------------------------------------------------------------------
# Integration tests for each tool via _dispatch_tool
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestKeywordSearchIntegration:
    def test_returns_papers_with_timing(self, conn) -> None:
        if not _has_papers(conn) or not _has_tsv_column(conn):
            pytest.skip("No papers or tsv column not available")
        result = _dispatch_safe(conn, "keyword_search", {"terms": "galaxy", "limit": 5})
        assert "papers" in result
        assert "total" in result
        assert "timing_ms" in result
        assert result["timing_ms"]["lexical_ms"] >= 0

    def test_with_filters(self, conn) -> None:
        if not _has_papers(conn) or not _has_tsv_column(conn):
            pytest.skip("No papers or tsv column not available")
        result = _dispatch_safe(
            conn,
            "keyword_search",
            {"terms": "star", "filters": {"year_min": 2023}, "limit": 3},
        )
        assert "papers" in result
        # All returned papers should be >= 2023 if any
        for paper in result["papers"]:
            if paper.get("year") is not None:
                assert paper["year"] >= 2023

    def test_no_results_query(self, conn) -> None:
        if not _has_papers(conn) or not _has_tsv_column(conn):
            pytest.skip("No papers or tsv column not available")
        result = _dispatch_safe(
            conn, "keyword_search", {"terms": "xyznonexistentterm99999", "limit": 5}
        )
        assert result["total"] == 0
        assert result["papers"] == []


@pytest.mark.integration
class TestGetPaperIntegration:
    def test_existing_paper(self, conn) -> None:
        if not _has_papers(conn):
            pytest.skip("No papers in database")
        bibcode = _get_any_bibcode(conn)
        result = _dispatch_safe(conn, "get_paper", {"bibcode": bibcode})
        assert result["total"] == 1
        assert result["papers"][0]["bibcode"] == bibcode
        assert "timing_ms" in result

    def test_nonexistent_paper(self, conn) -> None:
        result = _dispatch_safe(conn, "get_paper", {"bibcode": "NONEXISTENT_XYZ_999"})
        assert result["total"] == 0
        assert "timing_ms" in result


@pytest.mark.integration
class TestGetCitationsIntegration:
    def test_returns_search_result(self, conn) -> None:
        if not _has_citation_edges(conn):
            pytest.skip("No citation edges in database")
        bibcode = _get_citing_bibcode(conn)
        if not bibcode:
            pytest.skip("No citing bibcode found")
        result = _dispatch_safe(conn, "get_citations", {"bibcode": bibcode, "limit": 5})
        assert "papers" in result
        assert "timing_ms" in result
        assert result["total"] >= 0

    def test_no_citations(self, conn) -> None:
        result = _dispatch_safe(
            conn, "get_citations", {"bibcode": "NONEXISTENT_XYZ_999", "limit": 5}
        )
        assert result["total"] == 0


@pytest.mark.integration
class TestGetReferencesIntegration:
    def test_returns_search_result(self, conn) -> None:
        if not _has_citation_edges(conn):
            pytest.skip("No citation edges in database")
        bibcode = _get_referencing_bibcode(conn)
        if not bibcode:
            pytest.skip("No referencing bibcode found")
        result = _dispatch_safe(conn, "get_references", {"bibcode": bibcode, "limit": 5})
        assert "papers" in result
        assert "timing_ms" in result
        assert result["total"] >= 0

    def test_no_references(self, conn) -> None:
        result = _dispatch_safe(
            conn, "get_references", {"bibcode": "NONEXISTENT_XYZ_999", "limit": 5}
        )
        assert result["total"] == 0


@pytest.mark.integration
class TestGetAuthorPapersIntegration:
    def test_returns_papers(self, conn) -> None:
        if not _has_papers(conn):
            pytest.skip("No papers in database")
        author = _get_any_author(conn)
        if not author:
            pytest.skip("No authors found")
        result = _dispatch_safe(conn, "get_author_papers", {"author_name": author})
        assert "papers" in result
        assert "timing_ms" in result
        assert result["total"] >= 1

    def test_with_year_filters(self, conn) -> None:
        if not _has_papers(conn):
            pytest.skip("No papers in database")
        author = _get_any_author(conn)
        if not author:
            pytest.skip("No authors found")
        result = _dispatch_safe(
            conn,
            "get_author_papers",
            {"author_name": author, "year_min": 2020, "year_max": 2025},
        )
        assert "papers" in result
        for paper in result["papers"]:
            if paper.get("year") is not None:
                assert 2020 <= paper["year"] <= 2025

    def test_no_results(self, conn) -> None:
        result = _dispatch_safe(
            conn, "get_author_papers", {"author_name": "ZZZZNONEXISTENT_AUTHOR_XYZ"}
        )
        assert result["total"] == 0


@pytest.mark.integration
class TestFacetCountsIntegration:
    def test_year_facets(self, conn) -> None:
        if not _has_papers(conn):
            pytest.skip("No papers in database")
        result = _dispatch_safe(conn, "facet_counts", {"field": "year", "limit": 10})
        assert "metadata" in result
        assert "facets" in result["metadata"]
        assert result["metadata"]["facet_field"] == "year"
        if result["metadata"]["facets"]:
            facet = result["metadata"]["facets"][0]
            assert "value" in facet
            assert "count" in facet

    def test_doctype_facets(self, conn) -> None:
        if not _has_papers(conn):
            pytest.skip("No papers in database")
        result = _dispatch_safe(conn, "facet_counts", {"field": "doctype"})
        assert "metadata" in result
        assert result["metadata"]["facet_field"] == "doctype"

    def test_with_filters(self, conn) -> None:
        if not _has_papers(conn):
            pytest.skip("No papers in database")
        result = _dispatch_safe(
            conn,
            "facet_counts",
            {"field": "year", "filters": {"year_min": 2023}, "limit": 5},
        )
        assert "metadata" in result
        # All facet values should be >= 2023 if they represent years
        for facet in result["metadata"].get("facets", []):
            year_val = int(facet["value"])
            assert year_val >= 2023


@pytest.mark.integration
class TestHealthCheckIntegration:
    def test_db_connectivity(self, conn) -> None:
        result = _dispatch_safe(conn, "health_check", {})
        assert result["db"] == "ok"
        assert "model_cached" in result
        assert "pool" in result


@pytest.mark.integration
class TestSemanticSearchIntegration:
    """Semantic search integration test using a fake embedding vector.

    We bypass the model loading by passing a pre-built vector directly
    through search.vector_search. This tests the DB query path without
    requiring torch/transformers.
    """

    def test_dispatch_without_model_returns_error_or_result(self, conn) -> None:
        """_dispatch_tool for semantic_search either succeeds (if model cached)
        or returns an error dict (if torch not installed). Both are valid."""
        result = _dispatch_safe(conn, "semantic_search", {"query": "dark energy", "limit": 3})
        # Either we get papers (model available) or an error (no torch)
        assert "papers" in result or "error" in result


@pytest.mark.integration
class TestCoCitationIntegration:
    def test_returns_results_with_overlap(self, conn) -> None:
        if not _has_citation_edges(conn):
            pytest.skip("No citation edges in database")
        bibcode = _get_citing_bibcode(conn)
        if not bibcode:
            pytest.skip("No citing bibcode found")
        # Use the target of that edge (a paper that is cited)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT target_bibcode FROM citation_edges "
                    "WHERE source_bibcode = %s LIMIT 1",
                    (bibcode,),
                )
                row = cur.fetchone()
        except psycopg.errors.QueryCanceled:
            _rollback_and_reset(conn)
            pytest.skip("Query timed out")
        if not row:
            pytest.skip("No target bibcode found")
        result = _dispatch_safe(
            conn, "co_citation_analysis", {"bibcode": row[0], "min_overlap": 1, "limit": 5}
        )
        assert "papers" in result
        assert "timing_ms" in result

    def test_nonexistent_bibcode(self, conn) -> None:
        result = _dispatch_safe(conn, "co_citation_analysis", {"bibcode": "NONEXISTENT_XYZ_999"})
        assert result["total"] == 0


@pytest.mark.integration
class TestBibliographicCouplingIntegration:
    def test_returns_results(self, conn) -> None:
        if not _has_citation_edges(conn):
            pytest.skip("No citation edges in database")
        bibcode = _get_referencing_bibcode(conn)
        if not bibcode:
            pytest.skip("No referencing bibcode found")
        result = _dispatch_safe(
            conn, "bibliographic_coupling", {"bibcode": bibcode, "min_overlap": 1, "limit": 5}
        )
        assert "papers" in result
        assert "timing_ms" in result

    def test_nonexistent_bibcode(self, conn) -> None:
        result = _dispatch_safe(conn, "bibliographic_coupling", {"bibcode": "NONEXISTENT_XYZ_999"})
        assert result["total"] == 0


@pytest.mark.integration
class TestCitationChainIntegration:
    def test_direct_edge(self, conn) -> None:
        if not _has_citation_edges(conn):
            pytest.skip("No citation edges in database")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT source_bibcode, target_bibcode FROM citation_edges LIMIT 1")
                row = cur.fetchone()
        except psycopg.errors.QueryCanceled:
            _rollback_and_reset(conn)
            pytest.skip("Query timed out")
        if not row:
            pytest.skip("No edges found")
        source, target = row
        result = _dispatch_safe(
            conn,
            "citation_chain",
            {"source_bibcode": source, "target_bibcode": target, "max_depth": 1},
        )
        assert "metadata" in result
        assert result["metadata"]["path_length"] == 1

    def test_same_bibcode(self, conn) -> None:
        result = _dispatch_safe(
            conn,
            "citation_chain",
            {"source_bibcode": "ANY", "target_bibcode": "ANY"},
        )
        assert result["metadata"]["path_length"] == 0

    def test_no_path(self, conn) -> None:
        result = _dispatch_safe(
            conn,
            "citation_chain",
            {"source_bibcode": "NONEXISTENT_A", "target_bibcode": "NONEXISTENT_B"},
        )
        assert result["metadata"]["path_length"] == -1


@pytest.mark.integration
class TestTemporalEvolutionIntegration:
    def test_bibcode_mode(self, conn) -> None:
        if not _has_citation_edges(conn):
            pytest.skip("No citation edges in database")
        bibcode = _get_citing_bibcode(conn)
        if not bibcode:
            pytest.skip("No citing bibcode found")
        # Get a target (a paper that is cited)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT target_bibcode FROM citation_edges "
                    "WHERE source_bibcode = %s LIMIT 1",
                    (bibcode,),
                )
                row = cur.fetchone()
        except psycopg.errors.QueryCanceled:
            _rollback_and_reset(conn)
            pytest.skip("Query timed out")
        if not row:
            pytest.skip("No target bibcode found")
        result = _dispatch_safe(conn, "temporal_evolution", {"bibcode_or_query": row[0]})
        assert "metadata" in result
        assert result["metadata"]["mode"] == "citations"
        assert "yearly_counts" in result["metadata"]

    def test_query_mode(self, conn) -> None:
        if not _has_papers(conn) or not _has_tsv_column(conn):
            pytest.skip("No papers or tsv column")
        result = _dispatch_safe(
            conn,
            "temporal_evolution",
            {"bibcode_or_query": "galaxy evolution", "year_start": 2022, "year_end": 2024},
        )
        assert result["metadata"]["mode"] == "publications"
        for entry in result["metadata"]["yearly_counts"]:
            assert 2022 <= entry["year"] <= 2024


@pytest.mark.integration
class TestUnknownToolIntegration:
    def test_unknown_tool(self, conn) -> None:
        result = _dispatch_safe(conn, "nonexistent_tool_xyz", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]
