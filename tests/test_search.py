"""Unit and integration tests for search module.

Unit tests cover query construction, filter generation, RRF fusion, and timing.
Integration tests (marked with @pytest.mark.integration) require a running scix database.
"""

from __future__ import annotations

import psycopg
import pytest
from helpers import (
    DSN,
    get_cited_bibcode,
    get_citing_bibcode,
    has_citation_edges,
    has_papers,
    has_tsv_column,
)

from scix.search import (
    SearchFilters,
    SearchResult,
    _elapsed_ms,
    bibliographic_coupling,
    citation_chain,
    co_citation_analysis,
    facet_counts,
    get_paper,
    lexical_search,
    rrf_fuse,
    temporal_evolution,
)

# ---------------------------------------------------------------------------
# Unit tests (no database required)
# ---------------------------------------------------------------------------


class TestSearchFilters:
    def test_empty_filters(self) -> None:
        f = SearchFilters()
        clause, params = f.to_where_clause()
        assert clause == ""
        assert params == []

    def test_year_range(self) -> None:
        f = SearchFilters(year_min=2022, year_max=2024)
        clause, params = f.to_where_clause()
        assert "year >= %s" in clause
        assert "year <= %s" in clause
        assert params == [2022, 2024]

    def test_year_min_only(self) -> None:
        f = SearchFilters(year_min=2023)
        clause, params = f.to_where_clause()
        assert "year >= %s" in clause
        assert "year <=" not in clause
        assert params == [2023]

    def test_arxiv_class_filter(self) -> None:
        f = SearchFilters(arxiv_class="astro-ph.SR")
        clause, params = f.to_where_clause()
        assert "arxiv_class @> ARRAY[%s]" in clause
        assert params == ["astro-ph.SR"]

    def test_doctype_filter(self) -> None:
        f = SearchFilters(doctype="article")
        clause, params = f.to_where_clause()
        assert "doctype = %s" in clause
        assert params == ["article"]

    def test_first_author_filter(self) -> None:
        f = SearchFilters(first_author="Einstein")
        clause, params = f.to_where_clause()
        assert "first_author ILIKE %s" in clause
        assert params == ["%Einstein%"]

    def test_combined_filters(self) -> None:
        f = SearchFilters(year_min=2020, year_max=2025, doctype="article", arxiv_class="gr-qc")
        clause, params = f.to_where_clause()
        assert clause.startswith(" AND ")
        # Should have 4 conditions
        assert clause.count("AND") == 4  # leading AND + 3 joins
        assert len(params) == 4

    def test_custom_table_alias(self) -> None:
        f = SearchFilters(year_min=2023)
        clause, _ = f.to_where_clause(table_alias="papers")
        assert "papers.year >= %s" in clause

    def test_frozen_dataclass(self) -> None:
        f = SearchFilters(year_min=2023)
        with pytest.raises(AttributeError):
            f.year_min = 2024  # type: ignore[misc]


class TestSearchResult:
    def test_construction(self) -> None:
        r = SearchResult(
            papers=[{"bibcode": "test"}],
            total=1,
            timing_ms={"query_ms": 5.0},
        )
        assert r.total == 1
        assert r.timing_ms["query_ms"] == 5.0
        assert r.metadata == {}

    def test_with_metadata(self) -> None:
        r = SearchResult(
            papers=[],
            total=0,
            timing_ms={"query_ms": 1.0},
            metadata={"facet_field": "year"},
        )
        assert r.metadata["facet_field"] == "year"

    def test_frozen(self) -> None:
        r = SearchResult(papers=[], total=0, timing_ms={})
        with pytest.raises(AttributeError):
            r.total = 5  # type: ignore[misc]

    def test_timing_ms_always_present(self) -> None:
        """Every SearchResult must have timing_ms dict."""
        r = SearchResult(papers=[], total=0, timing_ms={"query_ms": 0.0})
        assert isinstance(r.timing_ms, dict)
        assert len(r.timing_ms) > 0


class TestElapsedMs:
    def test_returns_positive_float(self) -> None:
        import time

        t0 = time.perf_counter()
        time.sleep(0.01)
        ms = _elapsed_ms(t0)
        assert ms > 0
        assert isinstance(ms, float)


class TestRRFFuse:
    def test_single_list(self) -> None:
        papers = [
            {"bibcode": "A", "title": "Paper A"},
            {"bibcode": "B", "title": "Paper B"},
        ]
        result = rrf_fuse([papers], top_n=10)
        assert len(result) == 2
        assert result[0]["bibcode"] == "A"
        assert "rrf_score" in result[0]

    def test_two_lists_boost_overlap(self) -> None:
        list1 = [{"bibcode": "A"}, {"bibcode": "B"}, {"bibcode": "C"}]
        list2 = [{"bibcode": "B"}, {"bibcode": "D"}, {"bibcode": "A"}]
        result = rrf_fuse([list1, list2], k=60, top_n=10)
        bibcodes = [r["bibcode"] for r in result]
        assert "A" in bibcodes
        assert "B" in bibcodes
        assert "C" in bibcodes
        assert "D" in bibcodes

    def test_overlap_paper_ranks_higher(self) -> None:
        """Paper in both lists should score higher than paper in only one."""
        list1 = [{"bibcode": "A"}, {"bibcode": "B"}]
        list2 = [{"bibcode": "B"}, {"bibcode": "C"}]
        result = rrf_fuse([list1, list2], k=60, top_n=10)
        # B appears in both lists at rank 2 and 1 -> highest RRF score
        scores = {r["bibcode"]: r["rrf_score"] for r in result}
        assert scores["B"] > scores["C"]

    def test_top_n_limits_results(self) -> None:
        papers = [{"bibcode": f"P{i}"} for i in range(10)]
        result = rrf_fuse([papers], top_n=3)
        assert len(result) == 3

    def test_empty_lists(self) -> None:
        result = rrf_fuse([[]], top_n=10)
        assert result == []

    def test_rrf_scores_are_deterministic(self) -> None:
        papers = [{"bibcode": "A"}, {"bibcode": "B"}]
        r1 = rrf_fuse([papers], k=60, top_n=10)
        r2 = rrf_fuse([papers], k=60, top_n=10)
        assert r1[0]["rrf_score"] == r2[0]["rrf_score"]
        assert r1[1]["rrf_score"] == r2[1]["rrf_score"]

    def test_k_parameter_affects_scores(self) -> None:
        papers = [{"bibcode": "A"}]
        r_small_k = rrf_fuse([papers], k=1, top_n=10)
        r_large_k = rrf_fuse([papers], k=1000, top_n=10)
        # Smaller k gives higher RRF scores (1/(k+rank))
        assert r_small_k[0]["rrf_score"] > r_large_k[0]["rrf_score"]


class TestFacetFieldValidation:
    def test_invalid_field_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported facet field"):
            facet_counts(None, "invalid_field")  # type: ignore[arg-type]

    def test_sql_injection_blocked(self) -> None:
        with pytest.raises(ValueError, match="Unsupported facet field"):
            facet_counts(None, "year; DROP TABLE papers")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration tests (require running scix database with data)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conn():
    """Provide a connection to the scix database."""
    try:
        c = psycopg.connect(DSN)
        c.autocommit = False
        yield c
        c.rollback()
        c.close()
    except psycopg.OperationalError:
        pytest.skip("scix database not available")


@pytest.fixture(autouse=True)
def _savepoint(conn):
    """Wrap each test in a savepoint for isolation."""
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT test_sp")
    yield
    with conn.cursor() as cur:
        cur.execute("ROLLBACK TO SAVEPOINT test_sp")


@pytest.mark.integration
class TestLexicalSearchIntegration:
    def test_returns_search_result_with_timing(self, conn) -> None:
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column not available")
        # Use 'english' config as fallback if scix_english not available
        result = lexical_search(conn, "copper", ts_config="english")
        assert isinstance(result, SearchResult)
        assert "lexical_ms" in result.timing_ms
        assert result.timing_ms["lexical_ms"] >= 0

    def test_returns_stubs_with_scores(self, conn) -> None:
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column not available")
        result = lexical_search(conn, "copper", ts_config="english")
        if result.total > 0:
            paper = result.papers[0]
            assert "bibcode" in paper
            assert "score" in paper


@pytest.mark.integration
class TestGetPaperIntegration:
    def test_existing_paper(self, conn) -> None:
        if not has_papers(conn):
            pytest.skip("No papers in database")
        # Get any bibcode
        with conn.cursor() as cur:
            cur.execute("SELECT bibcode FROM papers LIMIT 1")
            bibcode = cur.fetchone()[0]
        result = get_paper(conn, bibcode)
        assert isinstance(result, SearchResult)
        assert result.total == 1
        assert "query_ms" in result.timing_ms

    def test_nonexistent_paper(self, conn) -> None:
        result = get_paper(conn, "NONEXISTENT_BIBCODE_XYZ")
        assert isinstance(result, SearchResult)
        assert result.total == 0
        assert "query_ms" in result.timing_ms


@pytest.mark.integration
class TestFacetCountsIntegration:
    def test_year_facets(self, conn) -> None:
        if not has_papers(conn):
            pytest.skip("No papers in database")
        result = facet_counts(conn, "year")
        assert isinstance(result, SearchResult)
        assert "query_ms" in result.timing_ms
        assert "facets" in result.metadata
        if result.metadata["facets"]:
            facet = result.metadata["facets"][0]
            assert "value" in facet
            assert "count" in facet

    def test_doctype_facets(self, conn) -> None:
        if not has_papers(conn):
            pytest.skip("No papers in database")
        result = facet_counts(conn, "doctype")
        assert isinstance(result, SearchResult)
        assert result.timing_ms["query_ms"] >= 0


# ---------------------------------------------------------------------------
# Integration tests for graph analysis functions
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCoCitationAnalysisIntegration:
    def test_returns_search_result_with_timing(self, conn) -> None:
        if not has_citation_edges(conn):
            pytest.skip("No citation edges in database")
        bibcode = get_cited_bibcode(conn)
        if not bibcode:
            pytest.skip("No cited bibcode found")
        result = co_citation_analysis(conn, bibcode, min_overlap=1, limit=5)
        assert isinstance(result, SearchResult)
        assert "query_ms" in result.timing_ms
        assert result.timing_ms["query_ms"] >= 0

    def test_overlap_count_in_results(self, conn) -> None:
        if not has_citation_edges(conn):
            pytest.skip("No citation edges in database")
        bibcode = get_cited_bibcode(conn)
        if not bibcode:
            pytest.skip("No cited bibcode found")
        result = co_citation_analysis(conn, bibcode, min_overlap=1, limit=5)
        for paper in result.papers:
            assert "overlap_count" in paper
            assert paper["overlap_count"] >= 1

    def test_nonexistent_bibcode(self, conn) -> None:
        result = co_citation_analysis(conn, "NONEXISTENT_XYZ_999")
        assert result.total == 0
        assert result.papers == []


@pytest.mark.integration
class TestBibliographicCouplingIntegration:
    def test_returns_search_result_with_timing(self, conn) -> None:
        if not has_citation_edges(conn):
            pytest.skip("No citation edges in database")
        bibcode = get_citing_bibcode(conn)
        if not bibcode:
            pytest.skip("No citing bibcode found")
        result = bibliographic_coupling(conn, bibcode, min_overlap=1, limit=5)
        assert isinstance(result, SearchResult)
        assert "query_ms" in result.timing_ms

    def test_shared_refs_in_results(self, conn) -> None:
        if not has_citation_edges(conn):
            pytest.skip("No citation edges in database")
        bibcode = get_citing_bibcode(conn)
        if not bibcode:
            pytest.skip("No citing bibcode found")
        result = bibliographic_coupling(conn, bibcode, min_overlap=1, limit=5)
        for paper in result.papers:
            assert "shared_refs" in paper
            assert paper["shared_refs"] >= 1

    def test_nonexistent_bibcode(self, conn) -> None:
        result = bibliographic_coupling(conn, "NONEXISTENT_XYZ_999")
        assert result.total == 0


@pytest.mark.integration
class TestCitationChainIntegration:
    def test_same_bibcode_returns_zero_length(self, conn) -> None:
        result = citation_chain(conn, "ANY_BIBCODE", "ANY_BIBCODE")
        assert result.metadata["path_length"] == 0
        assert result.metadata["path_bibcodes"] == ["ANY_BIBCODE"]

    def test_no_path_returns_negative_one(self, conn) -> None:
        result = citation_chain(conn, "NONEXISTENT_A", "NONEXISTENT_B", max_depth=2)
        assert result.metadata["path_length"] == -1
        assert result.metadata["path_bibcodes"] == []
        assert result.total == 0

    def test_direct_citation_path(self, conn) -> None:
        if not has_citation_edges(conn):
            pytest.skip("No citation edges in database")
        # Get a known direct edge
        with conn.cursor() as cur:
            cur.execute("SELECT source_bibcode, target_bibcode FROM citation_edges LIMIT 1")
            row = cur.fetchone()
        if not row:
            pytest.skip("No edges found")
        source, target = row
        result = citation_chain(conn, source, target, max_depth=1)
        assert result.metadata["path_length"] == 1
        assert result.metadata["path_bibcodes"][0] == source
        assert result.metadata["path_bibcodes"][-1] == target

    def test_timing_metadata(self, conn) -> None:
        result = citation_chain(conn, "A", "B", max_depth=1)
        assert "query_ms" in result.timing_ms
        assert result.timing_ms["query_ms"] >= 0

    def test_max_depth_zero_raises(self, conn) -> None:
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            citation_chain(conn, "A", "B", max_depth=0)


@pytest.mark.integration
class TestTemporalEvolutionIntegration:
    def test_bibcode_mode(self, conn) -> None:
        if not has_papers(conn) or not has_citation_edges(conn):
            pytest.skip("No papers or citation edges")
        bibcode = get_cited_bibcode(conn)
        if not bibcode:
            pytest.skip("No cited bibcode found")
        result = temporal_evolution(conn, bibcode)
        assert isinstance(result, SearchResult)
        assert result.metadata["mode"] == "citations"
        assert "yearly_counts" in result.metadata
        assert "query_ms" in result.timing_ms

    def test_query_mode(self, conn) -> None:
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column")
        result = temporal_evolution(conn, "galaxy formation")
        assert result.metadata["mode"] == "publications"
        assert "yearly_counts" in result.metadata

    def test_year_range_filter(self, conn) -> None:
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column")
        result = temporal_evolution(conn, "galaxy", year_start=2023, year_end=2024)
        for entry in result.metadata["yearly_counts"]:
            assert 2023 <= entry["year"] <= 2024

    def test_bibcode_found_metadata(self, conn) -> None:
        if not has_papers(conn) or not has_citation_edges(conn):
            pytest.skip("No papers or citation edges")
        bibcode = get_cited_bibcode(conn)
        if not bibcode:
            pytest.skip("No cited bibcode found")
        result = temporal_evolution(conn, bibcode)
        assert result.metadata["bibcode_found"] is True

    def test_nonexistent_bibcode_falls_to_query(self, conn) -> None:
        result = temporal_evolution(conn, "NONEXISTENT_XYZ_999")
        assert result.metadata["mode"] == "publications"
        assert result.metadata["bibcode_found"] is False
