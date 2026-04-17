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
    SELECTIVITY_THRESHOLD,
    SearchFilters,
    SearchResult,
    _elapsed_ms,
    _estimate_filter_selectivity,
    _filter_first_vector_search,
    _model_has_embeddings,
    bibliographic_coupling,
    citation_chain,
    co_citation_analysis,
    facet_counts,
    get_paper,
    hybrid_search,
    lexical_search,
    rrf_fuse,
    temporal_evolution,
    vector_search,
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


class TestHybridSearchDefaultModel:
    """Verify that hybrid_search and vector_search default to indus."""

    def test_hybrid_search_default_model(self) -> None:
        import inspect

        sig = inspect.signature(hybrid_search)
        assert sig.parameters["model_name"].default == "indus"

    def test_vector_search_default_model(self) -> None:
        import inspect

        sig = inspect.signature(vector_search)
        assert sig.parameters["model_name"].default == "indus"


class TestCardinalityRouting:
    """Verify filter-first fallback triggers for selective filters."""

    def test_selectivity_threshold_is_one_percent(self) -> None:
        assert SELECTIVITY_THRESHOLD == 0.01

    def test_estimate_no_filters_returns_one(self) -> None:
        """Empty filters should return selectivity 1.0 without hitting DB."""
        result = _estimate_filter_selectivity(None, SearchFilters())  # type: ignore[arg-type]
        assert result == 1.0

    def test_cardinality_routing_uses_filter_first(self) -> None:
        """When selectivity < threshold, hybrid_search should use filter-first CTE."""
        from unittest.mock import MagicMock, patch

        mock_conn = MagicMock()

        fake_embedding = [0.1] * 768
        selective_filters = SearchFilters(year_min=2026, year_max=2026, doctype="article")

        # Make _estimate_filter_selectivity return very low selectivity
        # Make _model_has_embeddings return False (skip OpenAI)
        # Make _filter_first_vector_search return a result
        # Make lexical_search return a result
        fake_search_result = SearchResult(
            papers=[{"bibcode": "TEST"}],
            total=1,
            timing_ms={"vector_ms": 1.0},
            metadata={"filter_first": True},
        )
        fake_lex_result = SearchResult(
            papers=[],
            total=0,
            timing_ms={"lexical_ms": 1.0},
        )

        with (
            patch("scix.search._estimate_filter_selectivity", return_value=0.001) as mock_est,
            patch(
                "scix.search._filter_first_vector_search",
                return_value=fake_search_result,
            ) as mock_ff,
            patch("scix.search.lexical_search", return_value=fake_lex_result),
            patch("scix.search._model_has_embeddings", return_value=False),
        ):
            result = hybrid_search(
                mock_conn,
                "test query",
                query_embedding=fake_embedding,
                filters=selective_filters,
            )

            # filter-first should have been called
            mock_est.assert_called_once_with(mock_conn, selective_filters)
            mock_ff.assert_called_once()

            # Result should contain the fused papers
            assert isinstance(result, SearchResult)
            assert "vector_ms" in result.timing_ms

    def test_cardinality_routing_uses_hnsw_for_broad_filters(self) -> None:
        """When selectivity >= threshold, hybrid_search should use normal vector_search."""
        from unittest.mock import MagicMock, patch

        mock_conn = MagicMock()

        fake_embedding = [0.1] * 768
        broad_filters = SearchFilters(year_min=2020)

        fake_vec_result = SearchResult(
            papers=[{"bibcode": "TEST"}],
            total=1,
            timing_ms={"vector_ms": 1.0},
            metadata={"iterative_scan": True},
        )
        fake_lex_result = SearchResult(
            papers=[],
            total=0,
            timing_ms={"lexical_ms": 1.0},
        )

        with (
            patch("scix.search._estimate_filter_selectivity", return_value=0.5),
            patch("scix.search.vector_search", return_value=fake_vec_result) as mock_vs,
            patch("scix.search.lexical_search", return_value=fake_lex_result),
            patch("scix.search._model_has_embeddings", return_value=False),
        ):
            result = hybrid_search(
                mock_conn,
                "test query",
                query_embedding=fake_embedding,
                filters=broad_filters,
            )

            # Normal vector_search should have been called
            mock_vs.assert_called_once()
            assert isinstance(result, SearchResult)


class TestOpenAISignalSkipped:
    """Verify OpenAI embedding signal is skipped when model has 0 rows."""

    def test_openai_skipped_when_no_rows(self) -> None:
        from unittest.mock import MagicMock, patch

        mock_conn = MagicMock()

        fake_lex_result = SearchResult(
            papers=[{"bibcode": "A"}],
            total=1,
            timing_ms={"lexical_ms": 1.0},
        )

        with (
            patch("scix.search.lexical_search", return_value=fake_lex_result),
            patch("scix.search._model_has_embeddings", return_value=False) as mock_check,
            patch("scix.search.vector_search") as mock_vs,
        ):
            result = hybrid_search(
                mock_conn,
                "test query",
                openai_embedding=[0.1] * 3072,
            )

            # _model_has_embeddings should be called for text-embedding-3-large
            mock_check.assert_called_once_with(mock_conn, "text-embedding-3-large")

            # vector_search should NOT be called (no primary embedding, no openai)
            mock_vs.assert_not_called()

            # OpenAI timing should be 0
            assert result.timing_ms["openai_vector_ms"] == 0.0


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
def _savepoint(request, conn):
    """Wrap integration tests in a savepoint for isolation. No-op for unit tests."""
    if "integration" not in {m.name for m in request.node.iter_markers()}:
        yield
        return
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

    def test_query_mode_returns_buckets_with_anchors(self, conn) -> None:
        """Query-mode must return per-year buckets with anchor papers — the core fix for o30."""
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column")
        result = temporal_evolution(conn, "galaxy formation", year_start=2022, year_end=2024)
        assert result.metadata["mode"] == "publications"
        buckets = result.metadata.get("buckets", [])
        if not buckets:
            pytest.skip("No matching publications for query in year range")
        assert "buckets" in result.metadata
        assert isinstance(buckets, list)

        for bucket in buckets:
            assert set(bucket.keys()) >= {"year", "count", "anchors", "communities"}
            assert isinstance(bucket["year"], int)
            assert bucket["count"] >= 1
            assert isinstance(bucket["anchors"], list)
            assert isinstance(bucket["communities"], list)

            for anchor in bucket["anchors"]:
                assert "bibcode" in anchor
                assert "title" in anchor
                assert "year" in anchor
            assert len(bucket["anchors"]) <= 5

    def test_query_mode_anchors_ordered_by_pagerank(self, conn) -> None:
        """Anchors within a bucket must be ranked by pagerank DESC NULLS LAST."""
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column")
        result = temporal_evolution(conn, "black hole", year_start=2023, year_end=2024)
        for bucket in result.metadata.get("buckets", []):
            anchors = bucket["anchors"]
            if len(anchors) < 2:
                continue
            ranks = [a.get("pagerank") if a.get("pagerank") is not None else -1.0 for a in anchors]
            assert ranks == sorted(
                ranks, reverse=True
            ), f"anchors for year {bucket['year']} not ordered by pagerank: {ranks}"

    def test_query_mode_buckets_respect_year_range(self, conn) -> None:
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column")
        result = temporal_evolution(conn, "galaxy", year_start=2023, year_end=2024)
        for bucket in result.metadata.get("buckets", []):
            assert 2023 <= bucket["year"] <= 2024

    def test_bibcode_mode_no_buckets(self, conn) -> None:
        """Bibcode mode is unchanged — still returns yearly_counts only, no buckets."""
        if not has_papers(conn) or not has_citation_edges(conn):
            pytest.skip("No papers or citation edges")
        bibcode = get_cited_bibcode(conn)
        if not bibcode:
            pytest.skip("No cited bibcode found")
        result = temporal_evolution(conn, bibcode)
        assert result.metadata["mode"] == "citations"
        assert "yearly_counts" in result.metadata
        assert "buckets" not in result.metadata

    def test_query_mode_anchors_do_not_exceed_bucket_count(self, conn) -> None:
        """len(anchors) must never exceed the bucket's total count."""
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column")
        result = temporal_evolution(conn, "galaxy formation", year_start=2022, year_end=2024)
        for bucket in result.metadata.get("buckets", []):
            assert len(bucket["anchors"]) <= bucket["count"], (
                f"year {bucket['year']}: {len(bucket['anchors'])} anchors "
                f"but only {bucket['count']} matching papers"
            )

    def test_query_mode_communities_have_labels(self, conn) -> None:
        """Communities in buckets must always carry a non-null label (None is filtered out)."""
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column")
        result = temporal_evolution(conn, "galaxy formation", year_start=2022, year_end=2024)
        for bucket in result.metadata.get("buckets", []):
            for community in bucket["communities"]:
                assert community["label"] is not None
                assert community["anchor_count"] >= 1
