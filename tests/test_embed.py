"""Unit tests for the embedding pipeline (input preparation, no GPU required)."""

from __future__ import annotations

import pytest

from scix.embed import EmbeddingInput, _vec_to_pgvector, prepare_input


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
