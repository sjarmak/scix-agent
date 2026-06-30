"""Unit and integration tests for search module.

Unit tests cover query construction, filter generation, RRF fusion, and timing.
Integration tests (marked with @pytest.mark.integration) require a running scix database.
"""

from __future__ import annotations

import logging

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
    _LEXICAL_POOL_DEFAULT,
    _LEXICAL_RANK_FLAG_DEFAULT,
    _TS_CONFIG_WHITELIST,
    SELECTIVITY_THRESHOLD,
    SearchFilters,
    SearchResult,
    _elapsed_ms,
    _estimate_filter_selectivity,
    _filter_first_vector_search,
    _resolve_lexical_pool,
    _resolve_lexical_rank_flag,
    _score_sections_ts_rank,
    bibliographic_coupling,
    citation_chain,
    co_citation_analysis,
    facet_counts,
    get_paper,
    hybrid_search,
    lexical_search,
    lit_review,
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


class TestEntityFilters:
    """xz4.1.27: entity_types / entity_ids filters on SearchFilters."""

    def test_defaults_are_none(self) -> None:
        f = SearchFilters()
        assert f.entity_types is None
        assert f.entity_ids is None

    def test_entity_types_accepts_list(self) -> None:
        f = SearchFilters(entity_types=["instrument", "mission"])
        # Normalized to tuple for frozen immutability.
        assert f.entity_types == ("instrument", "mission")

    def test_entity_ids_accepts_list(self) -> None:
        f = SearchFilters(entity_ids=[1, 2, 3])
        assert f.entity_ids == (1, 2, 3)

    def test_empty_list_treated_as_none(self) -> None:
        """Empty lists should NOT filter out everything — treat as no-filter."""
        f = SearchFilters(entity_types=[], entity_ids=[])
        assert f.entity_types is None
        assert f.entity_ids is None
        clause, params = f.to_entity_filter_clause("p")
        assert clause == ""
        assert params == []

    def test_no_entity_filter_clause_when_none(self) -> None:
        f = SearchFilters()
        clause, params = f.to_entity_filter_clause("p")
        assert clause == ""
        assert params == []

    def test_entity_ids_only_clause(self) -> None:
        f = SearchFilters(entity_ids=[42, 99])
        clause, params = f.to_entity_filter_clause("p")
        assert clause.startswith(" AND EXISTS")
        assert "document_entities_canonical" in clause
        assert "dec.bibcode = p.bibcode" in clause
        assert "dec.entity_id = ANY(%s)" in clause
        assert params == [[42, 99]]

    def test_entity_types_only_clause(self) -> None:
        f = SearchFilters(entity_types=["instrument"])
        clause, params = f.to_entity_filter_clause("p")
        assert "EXISTS" in clause
        assert "entities" in clause
        assert "e.entity_type = ANY(%s)" in clause
        assert params == [["instrument"]]

    def test_combined_entity_filters(self) -> None:
        f = SearchFilters(entity_types=["instrument"], entity_ids=[42])
        clause, params = f.to_entity_filter_clause("p")
        assert clause.startswith(" AND EXISTS")
        # Both constraints must appear.
        assert "entity_id = ANY(%s)" in clause
        assert "entity_type = ANY(%s)" in clause
        # Order: entity_ids then entity_types.
        assert params == [[42], ["instrument"]]

    def test_custom_table_alias_in_entity_clause(self) -> None:
        f = SearchFilters(entity_ids=[1])
        clause, _ = f.to_entity_filter_clause("papers")
        assert "dec.bibcode = papers.bibcode" in clause

    def test_entity_types_validates_str(self) -> None:
        with pytest.raises(TypeError, match="entity_types"):
            SearchFilters(entity_types=[1, 2])  # type: ignore[list-item]

    def test_entity_ids_validates_int(self) -> None:
        with pytest.raises(TypeError, match="entity_ids"):
            SearchFilters(entity_ids=["a", "b"])  # type: ignore[list-item]

    def test_frozen_entity_filters(self) -> None:
        f = SearchFilters(entity_ids=[1])
        with pytest.raises(AttributeError):
            f.entity_ids = (2,)  # type: ignore[misc]

    def test_to_where_clause_unaffected(self) -> None:
        """Entity filters must NOT leak into to_where_clause — they live on a separate method."""
        f = SearchFilters(year_min=2020, entity_types=["instrument"], entity_ids=[42])
        clause, params = f.to_where_clause("p")
        # Entity filters belong to to_entity_filter_clause, not to_where_clause.
        assert "entity_id" not in clause
        assert "entity_type" not in clause
        assert params == [2020]


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

    def test_entity_filters_reach_sql(self) -> None:
        """xz4.1.27 HIGH follow-up: facet_counts must thread entity filters into SQL,
        not silently drop them."""
        from unittest.mock import MagicMock

        from scix.search import SearchFilters

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = []
        conn.cursor.return_value = cursor

        filters = SearchFilters(entity_types=("instrument",), entity_ids=(42,))
        facet_counts(conn, "year", filters=filters)

        assert cursor.execute.called
        sql, params = cursor.execute.call_args.args
        assert (
            "document_entities_canonical" in sql
        ), "facet_counts dropped the entity filter — the EXISTS clause is missing"
        assert [42] in params
        assert ["instrument"] in params

    def test_entity_filters_reach_sql_for_array_fields(self) -> None:
        """Same guarantee for array-unnest facet fields (arxiv_class, etc.)."""
        from unittest.mock import MagicMock

        from scix.search import SearchFilters

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = []
        conn.cursor.return_value = cursor

        filters = SearchFilters(entity_ids=(42,))
        facet_counts(conn, "arxiv_class", filters=filters)

        assert cursor.execute.called
        sql, params = cursor.execute.call_args.args
        assert "document_entities_canonical" in sql
        assert [42] in params


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

    def test_cardinality_routing_uses_filter_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When selectivity < threshold, hybrid_search should use filter-first CTE."""
        from unittest.mock import MagicMock, patch

        # Filter-first is the pg lane; it only routes when the Qdrant gate is off.
        monkeypatch.delenv("QDRANT_URL", raising=False)
        mock_conn = MagicMock()

        fake_embedding = [0.1] * 768
        selective_filters = SearchFilters(year_min=2026, year_max=2026, doctype="article")

        # Make _estimate_filter_selectivity return very low selectivity
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

    def test_cardinality_routing_uses_hnsw_for_broad_filters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When selectivity >= threshold, hybrid_search should use normal vector_search."""
        from unittest.mock import MagicMock, patch

        monkeypatch.delenv("QDRANT_URL", raising=False)
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


class TestQdrantFilteredRouting:
    """Filtered dense lanes must not touch the dropped paper_embeddings table
    when the Qdrant gate is active (bead miw9, jg4a FAIL 1 follow-up)."""

    def test_filter_first_delegates_to_qdrant_when_gated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under QDRANT_URL, _filter_first_vector_search must route to the
        Qdrant lane and issue no SQL (paper_embeddings was dropped, ADR-013)."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        mock_conn = MagicMock()
        fake_result = SearchResult(papers=[], total=0, timing_ms={"vector_ms": 1.0})

        filters = SearchFilters(year_min=2026)
        with patch("scix.search._vector_search_qdrant", return_value=fake_result) as mock_qdrant:
            result = _filter_first_vector_search(
                mock_conn,
                [0.1] * 768,
                model_name="indus",
                filters=filters,
                limit=5,
            )

        mock_qdrant.assert_called_once()
        # model_name/filters/limit must survive the delegation handoff.
        kwargs = mock_qdrant.call_args.kwargs
        assert kwargs["model_name"] == "indus"
        assert kwargs["filters"] is filters
        assert kwargs["limit"] == 5
        assert result is fake_result
        mock_conn.cursor.assert_not_called()

    def test_filter_first_uses_pg_when_not_gated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without QDRANT_URL the legacy pg path is preserved (rollback lane)."""
        from unittest.mock import MagicMock

        monkeypatch.delenv("QDRANT_URL", raising=False)
        conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = []
        conn.cursor.return_value = cursor

        _filter_first_vector_search(
            conn,
            [0.1] * 768,
            model_name="indus",
            filters=SearchFilters(year_min=2026),
            limit=5,
        )

        sql, _params = cursor.execute.call_args.args
        assert "paper_embeddings" in sql

    def test_hybrid_search_skips_selectivity_probe_when_gated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under QDRANT_URL, hybrid_search must not run the selectivity probe
        or the filter-first CTE — filtered dense goes through vector_search,
        whose Qdrant lane applies filters as a PG post-filter."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        mock_conn = MagicMock()
        selective_filters = SearchFilters(year_min=2026, year_max=2026, doctype="article")
        fake_vec_result = SearchResult(
            papers=[{"bibcode": "TEST"}],
            total=1,
            timing_ms={"vector_ms": 1.0},
            metadata={"backend": "qdrant_dense"},
        )
        fake_lex_result = SearchResult(papers=[], total=0, timing_ms={"lexical_ms": 1.0})
        fake_body_result = SearchResult(papers=[], total=0, timing_ms={"body_lexical_ms": 1.0})

        with (
            patch("scix.search._estimate_filter_selectivity") as mock_est,
            patch("scix.search._filter_first_vector_search") as mock_ff,
            patch("scix.search.vector_search", return_value=fake_vec_result) as mock_vs,
            patch("scix.search.lexical_search", return_value=fake_lex_result),
            patch("scix.search.lexical_search_body", return_value=fake_body_result),
        ):
            result = hybrid_search(
                mock_conn,
                "test query",
                query_embedding=[0.1] * 768,
                filters=selective_filters,
            )

        mock_est.assert_not_called()
        mock_ff.assert_not_called()
        mock_vs.assert_called_once()
        assert mock_vs.call_args.kwargs["filters"] is selective_filters
        assert isinstance(result, SearchResult)


class TestHybridSearchEntityFilterWiring:
    """xz4.1.27: Entity filter params must propagate through hybrid_search subcalls."""

    def test_entity_filter_threads_through_lexical(self) -> None:
        from unittest.mock import MagicMock, patch

        mock_conn = MagicMock()
        entity_filter = SearchFilters(entity_types=["instrument"], entity_ids=[42])

        fake_lex = SearchResult(papers=[], total=0, timing_ms={"lexical_ms": 1.0})
        fake_body = SearchResult(papers=[], total=0, timing_ms={"body_lexical_ms": 1.0})

        with (
            patch("scix.search.lexical_search", return_value=fake_lex) as mock_lex,
            patch("scix.search.lexical_search_body", return_value=fake_body),
        ):
            hybrid_search(mock_conn, "jwst", filters=entity_filter)
            assert mock_lex.call_count == 1
            # SearchFilters with entity fields must be passed through.
            called_filters = mock_lex.call_args.kwargs.get("filters")
            assert called_filters is not None
            assert called_filters.entity_types == ("instrument",)
            assert called_filters.entity_ids == (42,)

    def test_entity_filter_threads_through_vector(self) -> None:
        from unittest.mock import MagicMock, patch

        mock_conn = MagicMock()
        entity_filter = SearchFilters(entity_ids=[42])
        fake_embedding = [0.1] * 768

        fake_lex = SearchResult(papers=[], total=0, timing_ms={"lexical_ms": 1.0})
        fake_body = SearchResult(papers=[], total=0, timing_ms={"body_lexical_ms": 1.0})
        fake_vec = SearchResult(
            papers=[{"bibcode": "X"}],
            total=1,
            timing_ms={"vector_ms": 1.0},
        )

        with (
            patch("scix.search.lexical_search", return_value=fake_lex),
            patch("scix.search.lexical_search_body", return_value=fake_body),
            patch("scix.search._estimate_filter_selectivity", return_value=0.5),
            patch("scix.search.vector_search", return_value=fake_vec) as mock_vs,
        ):
            hybrid_search(
                mock_conn,
                "q",
                query_embedding=fake_embedding,
                filters=entity_filter,
            )
            mock_vs.assert_called_once()
            assert mock_vs.call_args.kwargs["filters"].entity_ids == (42,)

    def test_default_behavior_unchanged_without_entity_filters(self) -> None:
        """Backward compat: searches with no entity filters behave exactly as before."""
        from unittest.mock import MagicMock, patch

        mock_conn = MagicMock()
        plain_filter = SearchFilters(year_min=2020)

        fake_lex = SearchResult(papers=[], total=0, timing_ms={"lexical_ms": 1.0})
        fake_body = SearchResult(papers=[], total=0, timing_ms={"body_lexical_ms": 1.0})

        with (
            patch("scix.search.lexical_search", return_value=fake_lex) as mock_lex,
            patch("scix.search.lexical_search_body", return_value=fake_body),
        ):
            hybrid_search(mock_conn, "x", filters=plain_filter)
            called = mock_lex.call_args.kwargs["filters"]
            assert called.entity_types is None
            assert called.entity_ids is None


class TestHybridSearchLaneErrorHandling:
    """uq28: hybrid_search must split a recoverable QueryCanceled (drop the
    lane, surface the drop in metadata) from a tx-abort / other psycopg.Error
    (propagate — never silently return a degraded RRF result)."""

    @staticmethod
    def _fake_lex() -> SearchResult:
        return SearchResult(
            papers=[{"bibcode": "2024ApJ...1A"}], total=1, timing_ms={"lexical_ms": 1.0}
        )

    def test_body_query_canceled_drops_lane_and_records_signal(self) -> None:
        from unittest.mock import MagicMock, patch

        conn = MagicMock()

        def _timeout(*args: object, **kwargs: object) -> object:
            raise psycopg.errors.QueryCanceled("statement timeout")

        with (
            patch("scix.search.lexical_search", return_value=self._fake_lex()),
            patch("scix.search.lexical_search_body", side_effect=_timeout),
        ):
            result = hybrid_search(conn, "dark matter")

        # The body lane is dropped, but the search still returns the main lane
        # and the drop is surfaced so the caller knows the RRF is degraded.
        assert result.total == 1
        assert result.metadata.get("dropped_lanes") == ["body_bm25"]

    def test_body_tx_abort_propagates(self) -> None:
        from unittest.mock import MagicMock, patch

        conn = MagicMock()

        def _abort(*args: object, **kwargs: object) -> object:
            # Class-25 tx-abort: the outer connection is poisoned. Swallowing
            # this would make every later lane fail silently.
            raise psycopg.errors.InFailedSqlTransaction("tx aborted")

        with (
            patch("scix.search.lexical_search", return_value=self._fake_lex()),
            patch("scix.search.lexical_search_body", side_effect=_abort),
            pytest.raises(psycopg.errors.InFailedSqlTransaction),
        ):
            hybrid_search(conn, "dark matter")

    def test_clean_run_has_no_dropped_lanes_key(self) -> None:
        from unittest.mock import MagicMock, patch

        conn = MagicMock()
        fake_body = SearchResult(papers=[], total=0, timing_ms={"body_lexical_ms": 1.0})

        with (
            patch("scix.search.lexical_search", return_value=self._fake_lex()),
            patch("scix.search.lexical_search_body", return_value=fake_body),
        ):
            result = hybrid_search(conn, "dark matter")

        assert "dropped_lanes" not in result.metadata


class TestLitReviewCitationExpansionErrorHandling:
    """otsm (DEEP_AUDIT A1): the citation-expansion loop in lit_review must wrap
    each get_references/get_citations call in a savepoint and catch only a
    recoverable QueryCanceled (statement_timeout) — dropping just that lane.
    Any other psycopg.Error (e.g. a class-25 tx-abort) must propagate rather
    than be swallowed by a bare ``except Exception``, which would let a poisoned
    connection silently degrade every later query in the same call."""

    @staticmethod
    def _seed_result() -> SearchResult:
        # Two seeds so the expansion loop iterates; bibcodes drive the working set.
        return SearchResult(
            papers=[{"bibcode": "2024ApJ...1A"}, {"bibcode": "2024ApJ...2B"}],
            total=2,
            timing_ms={"query_ms": 1.0},
        )

    @staticmethod
    def _empty_conn() -> object:
        """A MagicMock connection whose savepoints don't suppress exceptions and
        whose Step 4-6 cursors return empty result sets so lit_review can finish."""
        from unittest.mock import MagicMock

        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = []
        cur.fetchone.return_value = (0,)
        conn.cursor.return_value = cur
        # conn.transaction() is a MagicMock context manager; its default __exit__
        # returns False, so a QueryCanceled raised inside it propagates to our
        # except clause rather than being silently suppressed.
        return conn

    def test_query_canceled_lane_is_dropped_and_review_completes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from unittest.mock import MagicMock, patch

        conn = self._empty_conn()

        def _timeout(*args: object, **kwargs: object) -> object:
            raise psycopg.errors.QueryCanceled("statement timeout")

        # __name__ mirrors the real function attribute the warning log reads.
        refs = MagicMock(side_effect=_timeout, __name__="get_references")
        expansion = SearchResult(
            papers=[{"bibcode": "2020ApJ...9X", "year": 2020}], total=1, timing_ms={}
        )
        cites = MagicMock(return_value=expansion, __name__="get_citations")

        with (
            patch("scix.search.hybrid_search", return_value=self._seed_result()),
            patch("scix.search.get_references", refs),
            patch("scix.search.get_citations", cites),
            caplog.at_level(logging.WARNING, logger="scix.search"),
        ):
            result = lit_review(conn, "dark matter", expansion_seeds=1, expand_per_seed=5)

        # The references lane timed out and was dropped, but lit_review still
        # returns: seeds intact, the surviving citations lane folded in, and the
        # drop logged at WARNING.
        ws = result.metadata["working_set_bibcodes"]
        assert "2024ApJ...1A" in ws
        assert "2020ApJ...9X" in ws  # from the citations lane that survived
        assert "timed out" in caplog.text

    def test_tx_abort_in_expansion_propagates(self) -> None:
        from unittest.mock import MagicMock, patch

        conn = self._empty_conn()

        def _abort(*args: object, **kwargs: object) -> object:
            # Class-25 tx-abort: the outer connection is poisoned. A bare
            # ``except Exception`` would swallow this and let every later query
            # in the call fail silently.
            raise psycopg.errors.InFailedSqlTransaction("tx aborted")

        refs = MagicMock(side_effect=_abort, __name__="get_references")

        with (
            patch("scix.search.hybrid_search", return_value=self._seed_result()),
            patch("scix.search.get_references", refs),
            pytest.raises(psycopg.errors.InFailedSqlTransaction),
        ):
            lit_review(conn, "dark matter", expansion_seeds=1, expand_per_seed=5)


class TestScoreSectionsTsRank:
    """uq28: the section scorer degrades to the Python proxy only on a real
    OperationalError (logged at WARNING, visible at prod INFO), and lets other
    DB errors propagate so genuine bugs are not masked."""

    @staticmethod
    def _conn_with_execute_error(exc: Exception) -> object:
        from unittest.mock import MagicMock

        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.execute.side_effect = exc
        conn.cursor.return_value = cur
        return conn

    def test_operational_error_falls_back_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        conn = self._conn_with_execute_error(psycopg.OperationalError("timeout"))
        with caplog.at_level(logging.WARNING, logger="scix.search"):
            scores = _score_sections_ts_rank(conn, ["dark matter halos"], "dark matter")
        assert len(scores) == 1
        assert all(isinstance(s, float) for s in scores)
        assert "falling back to Python proxy" in caplog.text

    def test_query_canceled_subclass_also_falls_back(self) -> None:
        # QueryCanceled is an OperationalError subclass, so a per-statement
        # timeout in the section scorer still degrades rather than propagates.
        conn = self._conn_with_execute_error(psycopg.errors.QueryCanceled("timeout"))
        scores = _score_sections_ts_rank(conn, ["text one", "text two"], "q")
        assert len(scores) == 2

    def test_programming_error_propagates(self) -> None:
        conn = self._conn_with_execute_error(psycopg.ProgrammingError("bad sql"))
        with pytest.raises(psycopg.ProgrammingError):
            _score_sections_ts_rank(conn, ["text"], "q")

    def test_empty_sections_short_circuit(self) -> None:
        from unittest.mock import MagicMock

        assert _score_sections_ts_rank(MagicMock(), [], "q") == []


class TestLexicalSearchCandidatePool:
    """Unit tests for the lexical_search candidate-pool cap (bead 3t37)."""

    def test_default_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("SCIX_LEXICAL_POOL", raising=False)
        assert _resolve_lexical_pool() == _LEXICAL_POOL_DEFAULT

    def test_explicit_integer(self, monkeypatch) -> None:
        monkeypatch.setenv("SCIX_LEXICAL_POOL", "250")
        assert _resolve_lexical_pool() == 250

    def test_whitespace_tolerated(self, monkeypatch) -> None:
        monkeypatch.setenv("SCIX_LEXICAL_POOL", "  750  ")
        assert _resolve_lexical_pool() == 750

    @pytest.mark.parametrize("token", ["inf", "INF", "all", "None"])
    def test_unbounded_tokens_return_none(self, monkeypatch, token) -> None:
        monkeypatch.setenv("SCIX_LEXICAL_POOL", token)
        assert _resolve_lexical_pool() is None

    def test_non_integer_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("SCIX_LEXICAL_POOL", "lots")
        assert _resolve_lexical_pool() == _LEXICAL_POOL_DEFAULT

    @pytest.mark.parametrize("token", ["0", "-1"])
    def test_non_positive_falls_back_to_default(self, monkeypatch, token) -> None:
        monkeypatch.setenv("SCIX_LEXICAL_POOL", token)
        assert _resolve_lexical_pool() == _LEXICAL_POOL_DEFAULT

    def test_ts_config_whitelist_contents(self) -> None:
        assert "scix_english" in _TS_CONFIG_WHITELIST
        assert "english" in _TS_CONFIG_WHITELIST

    def test_ts_config_injection_rejected(self) -> None:
        # The whitelist guard runs before any DB access, so conn is never
        # touched — passing None proves the ValueError fires first.
        with pytest.raises(ValueError, match="ts_config must be one of"):
            lexical_search(None, "galaxy", ts_config="english'); DROP TABLE papers--")

    def test_ts_config_unknown_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="ts_config must be one of"):
            lexical_search(None, "galaxy", ts_config="simple")


class TestLexicalRankFlag:
    """Unit tests for the SCIX_LEXICAL_RANK_FLAG env hook (bead q9k5)."""

    def test_default_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("SCIX_LEXICAL_RANK_FLAG", raising=False)
        assert _resolve_lexical_rank_flag() == _LEXICAL_RANK_FLAG_DEFAULT

    @pytest.mark.parametrize("token,expected", [("0", 0), ("32", 32), ("33", 33), ("48", 48)])
    def test_valid_flags(self, monkeypatch, token, expected) -> None:
        monkeypatch.setenv("SCIX_LEXICAL_RANK_FLAG", token)
        assert _resolve_lexical_rank_flag() == expected

    def test_whitespace_tolerated(self, monkeypatch) -> None:
        monkeypatch.setenv("SCIX_LEXICAL_RANK_FLAG", "  48  ")
        assert _resolve_lexical_rank_flag() == 48

    def test_non_integer_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("SCIX_LEXICAL_RANK_FLAG", "dense")
        assert _resolve_lexical_rank_flag() == _LEXICAL_RANK_FLAG_DEFAULT

    @pytest.mark.parametrize("token", ["-1", "64", "100"])
    def test_bits_outside_mask_fall_back_to_default(self, monkeypatch, token) -> None:
        # 64 sets a bit above the defined 0..63 normalization mask; -1 is negative.
        monkeypatch.setenv("SCIX_LEXICAL_RANK_FLAG", token)
        assert _resolve_lexical_rank_flag() == _LEXICAL_RANK_FLAG_DEFAULT


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

    def test_pool_cap_bounds_result_count(self, conn, monkeypatch) -> None:
        """A tiny SCIX_LEXICAL_POOL caps the ranked candidate set (bead 3t37)."""
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column not available")
        monkeypatch.setenv("SCIX_LEXICAL_POOL", "2")
        result = lexical_search(conn, "galaxy", limit=20, ts_config="english")
        # cand CTE caps the match set at 2 rows before ts_rank_cd runs.
        assert result.total <= 2

    def test_common_term_completes_under_default_pool(self, conn) -> None:
        """The regression: a common single-token term must not time out.

        Without the candidate-pool cap, 'galaxy' forces ts_rank_cd over the
        full match set and exceeds the 30s statement timeout. With the default
        pool it returns promptly.
        """
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column not available")
        result = lexical_search(conn, "galaxy", ts_config="english")
        assert isinstance(result, SearchResult)
        assert result.timing_ms["lexical_ms"] >= 0

    def test_unbounded_pool_still_returns_results(self, conn, monkeypatch) -> None:
        """SCIX_LEXICAL_POOL=INF disables the cap (LIMIT NULL) for eval use."""
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column not available")
        monkeypatch.setenv("SCIX_LEXICAL_POOL", "INF")
        result = lexical_search(conn, "copper", limit=5, ts_config="english")
        assert isinstance(result, SearchResult)
        assert result.total <= 5


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


# ---------------------------------------------------------------------------
# Integration tests for entity-aware filters (xz4.1.27)
# Read-only — safe on production DSN.
# ---------------------------------------------------------------------------


def _has_entity_data(conn: psycopg.Connection) -> bool:
    """True when the entity graph has any linked documents to filter on."""
    with conn.cursor() as cur:
        cur.execute("SELECT EXISTS(" "SELECT 1 FROM document_entities_canonical LIMIT 1" ")")
        return cur.fetchone()[0]


def _pick_entity(conn: psycopg.Connection, entity_type: str) -> tuple[int, str] | None:
    """Return (entity_id, canonical_name) for an entity of the given type that has links."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id, e.canonical_name
            FROM entities e
            WHERE e.entity_type = %s
              AND EXISTS (
                  SELECT 1 FROM document_entities_canonical dec
                  WHERE dec.entity_id = e.id LIMIT 1
              )
            LIMIT 1
            """,
            (entity_type,),
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row else None


@pytest.mark.integration
class TestEntityFilterIntegration:
    """End-to-end entity filter behaviour against real indexes."""

    def test_entity_type_filter_returns_only_linked_papers(self, conn) -> None:
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column")
        if not _has_entity_data(conn):
            pytest.skip("No document_entities_canonical rows")

        entity_filter = SearchFilters(entity_types=["instrument"])
        result = lexical_search(
            conn, "telescope", filters=entity_filter, limit=10, ts_config="english"
        )
        assert isinstance(result, SearchResult)
        # All returned bibcodes must be linked to at least one instrument entity.
        if result.total == 0:
            pytest.skip("No matching papers for 'telescope' + instrument filter")
        with conn.cursor() as cur:
            for paper in result.papers:
                cur.execute(
                    """
                    SELECT EXISTS(
                        SELECT 1
                        FROM document_entities_canonical dec
                        JOIN entities e ON dec.entity_id = e.id
                        WHERE dec.bibcode = %s
                          AND e.entity_type = 'instrument'
                    )
                    """,
                    (paper["bibcode"],),
                )
                linked = cur.fetchone()[0]
                assert linked, (
                    f"bibcode {paper['bibcode']} returned by entity_type filter "
                    "but has no instrument link in document_entities_canonical"
                )

    def test_entity_id_filter_returns_only_that_entity(self, conn) -> None:
        if not has_papers(conn):
            pytest.skip("No papers")
        picked = _pick_entity(conn, "instrument")
        if picked is None:
            pytest.skip("No instrument entity with links")
        eid, _ = picked

        entity_filter = SearchFilters(entity_ids=[eid])
        result = lexical_search(
            conn, "observation", filters=entity_filter, limit=10, ts_config="english"
        )
        if result.total == 0:
            pytest.skip("No matching papers")
        with conn.cursor() as cur:
            for paper in result.papers:
                cur.execute(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM document_entities_canonical
                        WHERE bibcode = %s AND entity_id = %s
                    )
                    """,
                    (paper["bibcode"], eid),
                )
                assert cur.fetchone()[
                    0
                ], f"bibcode {paper['bibcode']} returned but not linked to entity_id={eid}"

    def test_combined_entity_filters(self, conn) -> None:
        """entity_types + entity_ids AND together."""
        picked = _pick_entity(conn, "instrument")
        if picked is None:
            pytest.skip("No instrument entity")
        eid, _ = picked

        # Non-matching combination: an 'instrument' entity_id filtered by
        # entity_type='mission' should return nothing.
        mismatch_filter = SearchFilters(entity_ids=[eid], entity_types=["mission"])
        result = lexical_search(
            conn, "observation", filters=mismatch_filter, limit=10, ts_config="english"
        )
        assert result.total == 0, (
            "Combining entity_id of type 'instrument' with entity_types=['mission'] "
            "must return zero rows."
        )

    def test_default_behavior_unchanged(self, conn) -> None:
        """Searches without entity filters behave identically to current production."""
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column")
        baseline = lexical_search(
            conn, "galaxy", filters=SearchFilters(), limit=5, ts_config="english"
        )
        no_filters = lexical_search(conn, "galaxy", limit=5, ts_config="english")
        assert [p["bibcode"] for p in baseline.papers] == [p["bibcode"] for p in no_filters.papers]

    def test_empty_list_behaves_like_none(self, conn) -> None:
        """filter with empty lists must NOT cull all results — treated as no-op."""
        if not has_papers(conn) or not has_tsv_column(conn):
            pytest.skip("No papers or tsv column")
        baseline = lexical_search(
            conn, "star", filters=SearchFilters(), limit=5, ts_config="english"
        )
        empty_entity = lexical_search(
            conn,
            "star",
            filters=SearchFilters(entity_types=[], entity_ids=[]),
            limit=5,
            ts_config="english",
        )
        assert [p["bibcode"] for p in baseline.papers] == [
            p["bibcode"] for p in empty_entity.papers
        ]


# ---------------------------------------------------------------------------
# MCP tool schema exposes entity_types / entity_ids (xz4.1.27)
# ---------------------------------------------------------------------------


class TestMCPSearchEntityFilterSchema:
    """MCP `search` tool must expose entity_types and entity_ids via filters schema."""

    def test_filters_schema_includes_entity_fields(self) -> None:
        from scix.mcp_server import _FILTERS_SCHEMA

        props = _FILTERS_SCHEMA["properties"]
        assert "entity_types" in props
        assert "entity_ids" in props
        # entity_types: array of strings
        assert props["entity_types"]["type"] == "array"
        assert props["entity_types"]["items"]["type"] == "string"
        # entity_ids: array of integers
        assert props["entity_ids"]["type"] == "array"
        assert props["entity_ids"]["items"]["type"] == "integer"

    def test_parse_filters_threads_entity_fields(self) -> None:
        from scix.mcp_server import _parse_filters

        parsed = _parse_filters(
            {
                "year_min": 2020,
                "entity_types": ["instrument"],
                "entity_ids": [27867],
            }
        )
        assert parsed.entity_types == ("instrument",)
        assert parsed.entity_ids == (27867,)
        assert parsed.year_min == 2020

    def test_parse_filters_missing_entity_keys(self) -> None:
        from scix.mcp_server import _parse_filters

        parsed = _parse_filters({"year_min": 2020})
        assert parsed.entity_types is None
        assert parsed.entity_ids is None

    def test_parse_filters_caps_oversized_entity_lists(self) -> None:
        """MCP boundary caps list size to prevent abuse."""
        from scix.mcp_server import MAX_ENTITY_FILTER_ITEMS, _parse_filters

        oversized_types = [f"type_{i}" for i in range(MAX_ENTITY_FILTER_ITEMS + 50)]
        oversized_ids = list(range(MAX_ENTITY_FILTER_ITEMS + 50))
        with pytest.raises(ValueError, match="entity_types"):
            _parse_filters({"entity_types": oversized_types})
        with pytest.raises(ValueError, match="entity_ids"):
            _parse_filters({"entity_ids": oversized_ids})

    def test_parse_filters_rejects_non_list_entity_types(self) -> None:
        from scix.mcp_server import _parse_filters

        with pytest.raises(ValueError, match="entity_types"):
            _parse_filters({"entity_types": "instrument"})

    def test_parse_filters_rejects_non_integer_entity_ids(self) -> None:
        from scix.mcp_server import _parse_filters

        with pytest.raises(ValueError, match="entity_ids"):
            _parse_filters({"entity_ids": ["abc"]})


class TestVectorSearchQdrantLane:
    """Internals of _vector_search_qdrant (bead 5jtf): over-fetch policy,
    PG post-filter, exact-mode env control, and score/order mapping."""

    @staticmethod
    def _fake_client(bibcodes_scores: list[tuple[str, float]]):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        points = [SimpleNamespace(payload={"bibcode": b}, score=s) for b, s in bibcodes_scores]
        client = MagicMock()
        client.query_points.return_value = SimpleNamespace(points=points)
        return client

    @staticmethod
    def _fake_conn(rows: list[dict]):
        from unittest.mock import MagicMock

        cursor = MagicMock()
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = rows
        conn = MagicMock()
        conn.cursor.return_value = cursor
        return conn, cursor

    @staticmethod
    def _row(bibcode: str) -> dict:
        return {
            "bibcode": bibcode,
            "title": f"Title {bibcode}",
            "first_author": "Author",
            "year": 2020,
            "citation_count": 1,
            "abstract": None,
        }

    def test_scores_and_order_follow_qdrant_ranking(self) -> None:
        from unittest.mock import patch

        from scix.search import _vector_search_qdrant

        client = self._fake_client([("B1", 0.9), ("B2", 0.8), ("B3", 0.7)])
        # PG returns the join rows in arbitrary order; ranking must come
        # from Qdrant, not from the SQL result order.
        conn, cursor = self._fake_conn([self._row("B3"), self._row("B1"), self._row("B2")])

        with patch("scix.search._get_qdrant_dense_client", return_value=client):
            result = _vector_search_qdrant(
                conn, [0.1] * 768, model_name="indus", filters=None, limit=3, ef_search=100
            )

        assert [p["bibcode"] for p in result.papers] == ["B1", "B2", "B3"]
        assert [p["score"] for p in result.papers] == [0.9, 0.8, 0.7]
        assert result.metadata["backend"] == "qdrant_dense"
        # Unfiltered queries must not over-fetch.
        assert result.metadata["fetch_n"] == 3
        assert client.query_points.call_args.kwargs["limit"] == 3
        # The metadata join must not touch the dropped paper_embeddings table.
        sql, _params = cursor.execute.call_args.args
        assert "paper_embeddings" not in sql
        assert "ANY" in sql

    def test_filters_trigger_capped_overfetch(self) -> None:
        from unittest.mock import patch

        from scix.search import _vector_search_qdrant

        client = self._fake_client([])
        conn, _cursor = self._fake_conn([])
        filters = SearchFilters(year_min=2020)

        with patch("scix.search._get_qdrant_dense_client", return_value=client):
            result = _vector_search_qdrant(
                conn, [0.1] * 768, model_name="indus", filters=filters, limit=20, ef_search=100
            )
            assert result.metadata["fetch_n"] == 200  # 10x limit

            result = _vector_search_qdrant(
                conn, [0.1] * 768, model_name="indus", filters=filters, limit=80, ef_search=100
            )
            assert result.metadata["fetch_n"] == 500  # capped, not 800

    def test_pg_post_filter_drops_rows_and_truncates_to_limit(self) -> None:
        from unittest.mock import patch

        from scix.search import _vector_search_qdrant

        client = self._fake_client(
            [("B1", 0.9), ("B2", 0.8), ("B3", 0.7), ("B4", 0.6), ("B5", 0.5)]
        )
        # B2 filtered out by SQL (or absent from papers): survivors are
        # B1, B3, B4, B5 — truncated to limit=2 in Qdrant rank order.
        conn, _cursor = self._fake_conn(
            [self._row("B1"), self._row("B3"), self._row("B4"), self._row("B5")]
        )

        with patch("scix.search._get_qdrant_dense_client", return_value=client):
            result = _vector_search_qdrant(
                conn,
                [0.1] * 768,
                model_name="indus",
                filters=SearchFilters(year_min=2020),
                limit=2,
                ef_search=100,
            )

        assert [p["bibcode"] for p in result.papers] == ["B1", "B3"]
        assert result.total == 2

    def test_default_search_params_use_hnsw_ef_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import patch

        from scix.search import _vector_search_qdrant

        monkeypatch.delenv("SCIX_QDRANT_EXACT", raising=False)
        client = self._fake_client([])
        conn, _cursor = self._fake_conn([])

        with patch("scix.search._get_qdrant_dense_client", return_value=client):
            _vector_search_qdrant(
                conn, [0.1] * 768, model_name="indus", filters=None, limit=50, ef_search=10
            )

        params = client.query_points.call_args.kwargs["search_params"]
        assert params.exact is not True
        # ef must never drop below the requested limit.
        assert params.hnsw_ef == 50

    def test_exact_env_forces_exact_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import patch

        from scix.search import _vector_search_qdrant

        monkeypatch.setenv("SCIX_QDRANT_EXACT", "1")
        client = self._fake_client([])
        conn, _cursor = self._fake_conn([])

        with patch("scix.search._get_qdrant_dense_client", return_value=client):
            _vector_search_qdrant(
                conn, [0.1] * 768, model_name="indus", filters=None, limit=5, ef_search=100
            )

        params = client.query_points.call_args.kwargs["search_params"]
        assert params.exact is True

    def test_points_missing_bibcode_payload_are_skipped_not_crashed(self) -> None:
        """A Qdrant point whose payload lacks 'bibcode' (transitional
        payload-backfill window, bead 8m0a) must be skipped + logged, not
        raise KeyError that crashes the whole search request (bead lq32)."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from scix.search import _vector_search_qdrant

        # Mix of well-formed points and degenerate ones: a None payload, an
        # empty-dict payload, and a payload with unrelated keys only.
        points = [
            SimpleNamespace(payload={"bibcode": "B1"}, score=0.9),
            SimpleNamespace(payload=None, score=0.85),
            SimpleNamespace(payload={}, score=0.8),
            SimpleNamespace(payload={"other": "x"}, score=0.75),
            SimpleNamespace(payload={"bibcode": "B2"}, score=0.7),
        ]
        client = MagicMock()
        client.query_points.return_value = SimpleNamespace(points=points)
        conn, _cursor = self._fake_conn([self._row("B1"), self._row("B2")])

        with patch("scix.search._get_qdrant_dense_client", return_value=client):
            result = _vector_search_qdrant(
                conn, [0.1] * 768, model_name="indus", filters=None, limit=10, ef_search=100
            )

        # Only the two bibcode-bearing points survive, in Qdrant rank order.
        assert [p["bibcode"] for p in result.papers] == ["B1", "B2"]
        assert [p["score"] for p in result.papers] == [0.9, 0.7]
