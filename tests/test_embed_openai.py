"""Tests for OpenAI embedding functions in scix.embed."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scix.embed import embed_openai, embed_query_openai

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding_response(vectors: list[list[float]]) -> SimpleNamespace:
    """Build a fake OpenAI embeddings.create() response."""
    data = [SimpleNamespace(embedding=vec, index=i) for i, vec in enumerate(vectors)]
    return SimpleNamespace(
        data=data,
        model="text-embedding-3-large",
        usage=SimpleNamespace(prompt_tokens=10, total_tokens=10),
    )


def _fake_vector(dim: int = 1024, value: float = 0.1) -> list[float]:
    return [value] * dim


# ---------------------------------------------------------------------------
# Tests: embed_openai
# ---------------------------------------------------------------------------


class TestEmbedOpenai:
    """Tests for embed_openai()."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("scix.embed.OpenAI", create=True)
    def test_single_text(self, mock_openai_cls: MagicMock) -> None:
        """embed_openai returns correct embeddings for a single text."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        expected_vec = _fake_vector(1024)
        mock_client.embeddings.create.return_value = _make_embedding_response([expected_vec])

        with patch("scix.embed._get_openai_client", return_value=mock_client):
            result = embed_openai(["hello world"])

        assert len(result) == 1
        assert result[0] == expected_vec
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large",
            input=["hello world"],
            dimensions=1024,
        )

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_batch_texts(self) -> None:
        """embed_openai handles multiple texts."""
        vecs = [_fake_vector(1024, v) for v in (0.1, 0.2, 0.3)]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embedding_response(vecs)

        with patch("scix.embed._get_openai_client", return_value=mock_client):
            result = embed_openai(["a", "b", "c"])

        assert len(result) == 3
        assert result[1] == vecs[1]

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_empty_list(self) -> None:
        """embed_openai returns empty list for empty input without API call."""
        result = embed_openai([])
        assert result == []

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_dimension_validation_failure(self) -> None:
        """embed_openai raises ValueError when returned dimensions mismatch."""
        wrong_vec = _fake_vector(512)  # Wrong dimension
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embedding_response([wrong_vec])

        with patch("scix.embed._get_openai_client", return_value=mock_client):
            with pytest.raises(ValueError, match="512 dimensions, expected 1024"):
                embed_openai(["test"])

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_custom_dimensions(self) -> None:
        """embed_openai respects custom dimensions parameter."""
        vec = _fake_vector(256)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embedding_response([vec])

        with patch("scix.embed._get_openai_client", return_value=mock_client):
            result = embed_openai(["test"], dimensions=256)

        assert len(result[0]) == 256
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large",
            input=["test"],
            dimensions=256,
        )


# ---------------------------------------------------------------------------
# Tests: missing API key
# ---------------------------------------------------------------------------


class TestMissingApiKey:
    """Tests for OPENAI_API_KEY validation."""

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self) -> None:
        """embed_openai raises ValueError when OPENAI_API_KEY is not set."""
        # Ensure the env var is truly absent
        import os

        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            embed_openai(["test"])

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_query_raises(self) -> None:
        """embed_query_openai raises ValueError when OPENAI_API_KEY is not set."""
        import os

        os.environ.pop("OPENAI_API_KEY", None)
        embed_query_openai.cache_clear()
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            embed_query_openai("test")


# ---------------------------------------------------------------------------
# Tests: embed_query_openai (caching)
# ---------------------------------------------------------------------------


class TestEmbedQueryOpenai:
    """Tests for embed_query_openai() including LRU cache behavior."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_returns_list_of_floats(self) -> None:
        """embed_query_openai returns a list of floats."""
        vec = _fake_vector(1024)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embedding_response([vec])

        embed_query_openai.cache_clear()
        with patch("scix.embed._get_openai_client", return_value=mock_client):
            result = embed_query_openai("hello")

        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(v, float) for v in result)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_cache_hit(self) -> None:
        """embed_query_openai caches results — second call does not invoke API."""
        vec = _fake_vector(1024)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embedding_response([vec])

        embed_query_openai.cache_clear()
        with patch("scix.embed._get_openai_client", return_value=mock_client):
            result1 = embed_query_openai("cache test query")
            result2 = embed_query_openai("cache test query")

        assert result1 == result2
        # API should only be called once due to caching
        assert mock_client.embeddings.create.call_count == 1

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_cache_miss_different_text(self) -> None:
        """embed_query_openai calls API for different inputs."""
        vec = _fake_vector(1024)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embedding_response([vec])

        embed_query_openai.cache_clear()
        with patch("scix.embed._get_openai_client", return_value=mock_client):
            embed_query_openai("query alpha")
            embed_query_openai("query beta")

        assert mock_client.embeddings.create.call_count == 2
