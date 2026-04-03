"""Tests for dual-model (SPECTER2 + OpenAI) hybrid search with RRF fusion.

Covers: dual-model fusion, single-model fallback, circuit breaker behaviour.
All tests are unit tests using mocked database connections and vector_search.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix.search import SearchResult, hybrid_search, rrf_fuse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_papers(prefix: str, n: int = 3) -> list[dict[str, Any]]:
    """Create a list of fake paper dicts with unique bibcodes."""
    return [
        {
            "bibcode": f"{prefix}_{i}",
            "title": f"Paper {prefix} {i}",
            "first_author": "Author",
            "year": 2024,
            "citation_count": i * 10,
            "score": 1.0 / (i + 1),
        }
        for i in range(n)
    ]


def _make_search_result(
    prefix: str,
    n: int = 3,
    timing_key: str = "vector_ms",
) -> SearchResult:
    papers = _make_papers(prefix, n)
    return SearchResult(
        papers=papers,
        total=len(papers),
        timing_ms={timing_key: 5.0},
    )


# ---------------------------------------------------------------------------
# Test: dual-model fusion (both SPECTER2 and OpenAI embeddings provided)
# ---------------------------------------------------------------------------


class TestDualModelFusion:
    """When both query_embedding and openai_embedding are supplied,
    hybrid_search should call vector_search twice and fuse three lists."""

    @patch("scix.search.vector_search")
    @patch("scix.search.lexical_search")
    def test_dual_model_produces_three_lists(
        self,
        mock_lexical: MagicMock,
        mock_vector: MagicMock,
    ) -> None:
        mock_lexical.return_value = _make_search_result("lex", timing_key="lexical_ms")
        mock_vector.side_effect = [
            _make_search_result("specter"),
            _make_search_result("openai"),
        ]
        conn = MagicMock()

        result = hybrid_search(
            conn,
            "dark energy",
            query_embedding=[0.1] * 768,
            openai_embedding=[0.2] * 3072,
        )

        # vector_search called twice: once for specter2, once for openai
        assert mock_vector.call_count == 2
        # Check model_name args
        specter_call = mock_vector.call_args_list[0]
        openai_call = mock_vector.call_args_list[1]
        assert (
            specter_call.kwargs.get("model_name", specter_call[1].get("model_name", "specter2"))
            == "specter2"
        )
        assert openai_call.kwargs["model_name"] == "text-embedding-3-large"

        assert result.total > 0
        assert "openai_vector_ms" in result.timing_ms
        assert result.timing_ms["openai_vector_ms"] > 0

    @patch("scix.search.vector_search")
    @patch("scix.search.lexical_search")
    def test_dual_model_fused_results_include_all_sources(
        self,
        mock_lexical: MagicMock,
        mock_vector: MagicMock,
    ) -> None:
        mock_lexical.return_value = _make_search_result("lex", n=2, timing_key="lexical_ms")
        mock_vector.side_effect = [
            _make_search_result("specter", n=2),
            _make_search_result("openai", n=2),
        ]
        conn = MagicMock()

        result = hybrid_search(
            conn,
            "dark energy",
            query_embedding=[0.1] * 768,
            openai_embedding=[0.2] * 3072,
            top_n=50,
        )

        bibcodes = {p["bibcode"] for p in result.papers}
        # All 6 unique papers should appear in fused results
        assert len(bibcodes) == 6


# ---------------------------------------------------------------------------
# Test: single-model fallback (only SPECTER2, no OpenAI)
# ---------------------------------------------------------------------------


class TestSingleModelFallback:
    """When openai_embedding is None, only SPECTER2 + lexical should be used."""

    @patch("scix.search.vector_search")
    @patch("scix.search.lexical_search")
    def test_no_openai_embedding_single_vector_call(
        self,
        mock_lexical: MagicMock,
        mock_vector: MagicMock,
    ) -> None:
        mock_lexical.return_value = _make_search_result("lex", timing_key="lexical_ms")
        mock_vector.return_value = _make_search_result("specter")
        conn = MagicMock()

        result = hybrid_search(
            conn,
            "dark energy",
            query_embedding=[0.1] * 768,
        )

        assert mock_vector.call_count == 1
        assert result.timing_ms["openai_vector_ms"] == 0.0

    @patch("scix.search.lexical_search")
    def test_no_embeddings_lexical_only(
        self,
        mock_lexical: MagicMock,
    ) -> None:
        mock_lexical.return_value = _make_search_result("lex", timing_key="lexical_ms")
        conn = MagicMock()

        result = hybrid_search(conn, "dark energy")

        assert result.total > 0
        assert result.timing_ms["vector_ms"] == 0.0
        assert result.timing_ms["openai_vector_ms"] == 0.0


# ---------------------------------------------------------------------------
# Test: circuit breaker (OpenAI vector search raises exception)
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """If OpenAI vector search fails, hybrid_search should not crash
    and should fall back to SPECTER2+lexical only."""

    @patch("scix.search.vector_search")
    @patch("scix.search.lexical_search")
    def test_openai_failure_falls_back_gracefully(
        self,
        mock_lexical: MagicMock,
        mock_vector: MagicMock,
    ) -> None:
        mock_lexical.return_value = _make_search_result("lex", timing_key="lexical_ms")

        # First call (SPECTER2) succeeds, second call (OpenAI) raises
        mock_vector.side_effect = [
            _make_search_result("specter"),
            RuntimeError("OpenAI embedding column missing"),
        ]
        conn = MagicMock()

        result = hybrid_search(
            conn,
            "dark energy",
            query_embedding=[0.1] * 768,
            openai_embedding=[0.2] * 3072,
        )

        # Should not crash; results should come from specter + lexical
        assert result.total > 0
        assert result.timing_ms["openai_vector_ms"] == 0.0
        assert result.timing_ms["vector_ms"] > 0

    @patch("scix.search.vector_search")
    @patch("scix.search.lexical_search")
    def test_openai_failure_logs_warning(
        self,
        mock_lexical: MagicMock,
        mock_vector: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_lexical.return_value = _make_search_result("lex", timing_key="lexical_ms")
        mock_vector.side_effect = [
            _make_search_result("specter"),
            Exception("DB connection lost"),
        ]
        conn = MagicMock()

        import logging

        with caplog.at_level(logging.WARNING, logger="scix.search"):
            hybrid_search(
                conn,
                "dark energy",
                query_embedding=[0.1] * 768,
                openai_embedding=[0.2] * 3072,
            )

        assert any("OpenAI vector search failed" in msg for msg in caplog.messages)
