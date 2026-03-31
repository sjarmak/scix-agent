"""Unit tests for the embedding pipeline (input preparation, no GPU required)."""

from __future__ import annotations

import pytest

from scix.embed import _vec_to_pgvector, prepare_input


class TestPrepareInput:
    def test_title_and_abstract(self) -> None:
        result = prepare_input("2024ApJ...001A", "Solar Flares", "We study solar flares.")
        assert result is not None
        assert result.bibcode == "2024ApJ...001A"
        assert result.text == "Solar Flares [SEP] We study solar flares."
        assert result.input_type == "title_abstract"
        assert len(result.source_hash) == 64  # SHA-256

    def test_title_only(self) -> None:
        result = prepare_input("2024ApJ...002B", "Gravitational Waves", None)
        assert result is not None
        assert result.text == "Gravitational Waves"
        assert result.input_type == "title_only"

    def test_empty_abstract_treated_as_title_only(self) -> None:
        result = prepare_input("2024ApJ...003C", "Dark Matter", "")
        assert result is not None
        # Empty string is falsy, so treated as title_only
        assert result.input_type == "title_only"
        assert result.text == "Dark Matter"

    def test_no_title_returns_none(self) -> None:
        assert prepare_input("2024ApJ...004D", None, "Some abstract") is None
        assert prepare_input("2024ApJ...005E", "", "Some abstract") is None

    def test_source_hash_deterministic(self) -> None:
        r1 = prepare_input("bib1", "Title", "Abstract")
        r2 = prepare_input("bib2", "Title", "Abstract")
        assert r1 is not None and r2 is not None
        # Same text -> same hash, even with different bibcodes
        assert r1.source_hash == r2.source_hash

    def test_source_hash_differs_for_different_text(self) -> None:
        r1 = prepare_input("bib1", "Title A", "Abstract")
        r2 = prepare_input("bib1", "Title B", "Abstract")
        assert r1 is not None and r2 is not None
        assert r1.source_hash != r2.source_hash

    def test_frozen_dataclass(self) -> None:
        result = prepare_input("bib1", "Title", "Abstract")
        assert result is not None
        with pytest.raises(AttributeError):
            result.text = "Modified"  # type: ignore[misc]


class TestVecToPgvector:
    def test_simple_vector(self) -> None:
        vec = [1.0, 2.5, -0.3]
        result = _vec_to_pgvector(vec)
        assert result == "[1.0,2.5,-0.3]"

    def test_empty_vector(self) -> None:
        result = _vec_to_pgvector([])
        assert result == "[]"

    def test_single_element(self) -> None:
        result = _vec_to_pgvector([0.5])
        assert result == "[0.5]"


class TestLoadModelCache:
    """Tests for load_model caching behavior."""

    @pytest.fixture(autouse=True)
    def _patch_ml_imports(self):
        """Patch torch/transformers imports and clear model cache around each test."""
        from unittest.mock import MagicMock, patch

        from scix.embed import _model_cache

        _model_cache.clear()

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        self.mock_auto_model = MagicMock()
        self.mock_auto_model.from_pretrained.return_value = mock_model
        self.mock_auto_tokenizer = MagicMock()
        self.mock_auto_tokenizer.from_pretrained.return_value = MagicMock()
        self.mock_torch = MagicMock()
        self.mock_torch.cuda.is_available.return_value = False

        fake_transformers = type(
            "mod",
            (),
            {"AutoModel": self.mock_auto_model, "AutoTokenizer": self.mock_auto_tokenizer},
        )()

        with patch.dict(
            "sys.modules",
            {"transformers": fake_transformers, "torch": self.mock_torch},
        ):
            yield

        _model_cache.clear()

    def test_cache_returns_same_objects(self) -> None:
        """Second call with same args should return cached (model, tokenizer)."""
        from scix.embed import load_model

        result1 = load_model("specter2", device="cpu")
        result2 = load_model("specter2", device="cpu")

        assert result1 is result2
        assert self.mock_auto_model.from_pretrained.call_count == 1

    def test_auto_resolves_to_cpu_shares_cache(self) -> None:
        """auto→cpu on non-GPU machine should share cache with explicit cpu."""
        from scix.embed import _model_cache, load_model

        load_model("specter2", device="cpu")
        load_model("specter2", device="auto")

        # auto resolves to cpu (mock has cuda.is_available=False),
        # so both calls share the same cache entry
        assert len(_model_cache) == 1
        assert self.mock_auto_model.from_pretrained.call_count == 1

    def test_clear_model_cache(self) -> None:
        """clear_model_cache should empty the cache."""
        from scix.embed import _model_cache, clear_model_cache, load_model

        load_model("specter2", device="cpu")
        assert len(_model_cache) == 1

        clear_model_cache()
        assert len(_model_cache) == 0

    def test_unknown_model_raises(self) -> None:
        """Unknown model name should raise ValueError, not touch cache."""
        from scix.embed import _model_cache, load_model

        with pytest.raises(ValueError, match="Unknown model"):
            load_model("nonexistent_model", device="cpu")

        assert len(_model_cache) == 0
