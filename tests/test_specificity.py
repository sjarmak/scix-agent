"""Tests for information-theoretic entity specificity scoring."""

from __future__ import annotations

import json
import math

import pytest

from scix.specificity import (
    DEFAULT_THRESHOLD,
    ScoredEntity,
    classify_entity,
    score_entities,
    specificity_score,
)

# ---------------------------------------------------------------------------
# specificity_score
# ---------------------------------------------------------------------------


class TestSpecificityScore:
    """Unit tests for the specificity_score function."""

    def test_basic_computation(self) -> None:
        score = specificity_score("test", df=100, N=1000)
        assert score == pytest.approx(math.log(1000 / 100))

    def test_high_frequency_low_score(self) -> None:
        """An entity in 50% of papers gets log(2) ~ 0.693."""
        score = specificity_score("water", df=50000, N=100000)
        assert score == pytest.approx(math.log(2))
        assert score < DEFAULT_THRESHOLD

    def test_low_frequency_high_score(self) -> None:
        """An entity in 0.5% of papers gets log(200) ~ 5.3."""
        score = specificity_score("James Webb Space Telescope", df=500, N=100000)
        assert score == pytest.approx(math.log(200))
        assert score > DEFAULT_THRESHOLD

    def test_df_equals_N(self) -> None:
        """Entity appearing in all papers gets score 0."""
        score = specificity_score("data", df=100000, N=100000)
        assert score == pytest.approx(0.0)

    def test_df_one(self) -> None:
        """Entity appearing in 1 paper gets log(N)."""
        score = specificity_score("unique entity", df=1, N=100000)
        assert score == pytest.approx(math.log(100000))

    def test_invalid_df_zero(self) -> None:
        with pytest.raises(ValueError, match="df must be positive"):
            specificity_score("bad", df=0, N=100)

    def test_invalid_df_negative(self) -> None:
        with pytest.raises(ValueError, match="df must be positive"):
            specificity_score("bad", df=-1, N=100)

    def test_invalid_N_zero(self) -> None:
        with pytest.raises(ValueError, match="N must be positive"):
            specificity_score("bad", df=1, N=0)

    def test_invalid_df_exceeds_N(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed N"):
            specificity_score("bad", df=200, N=100)


# ---------------------------------------------------------------------------
# classify_entity
# ---------------------------------------------------------------------------


class TestClassifyEntity:
    """Unit tests for the classify_entity function."""

    def test_high_score_keeps(self) -> None:
        assert classify_entity(5.0) == "keep"

    def test_low_score_filters(self) -> None:
        assert classify_entity(1.0) == "filter"

    def test_exact_threshold_keeps(self) -> None:
        assert classify_entity(DEFAULT_THRESHOLD) == "keep"

    def test_just_below_threshold_filters(self) -> None:
        assert classify_entity(DEFAULT_THRESHOLD - 0.001) == "filter"

    def test_custom_threshold(self) -> None:
        assert classify_entity(3.0, threshold=4.0) == "filter"
        assert classify_entity(5.0, threshold=4.0) == "keep"


# ---------------------------------------------------------------------------
# ScoredEntity
# ---------------------------------------------------------------------------


class TestScoredEntity:
    """Tests for the ScoredEntity frozen dataclass."""

    def test_creation(self) -> None:
        se = ScoredEntity(entity="test", score=3.5, classification="keep")
        assert se.entity == "test"
        assert se.score == 3.5
        assert se.classification == "keep"

    def test_frozen(self) -> None:
        se = ScoredEntity(entity="test", score=3.5, classification="keep")
        with pytest.raises(AttributeError):
            se.score = 1.0  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ScoredEntity(entity="x", score=1.0, classification="filter")
        b = ScoredEntity(entity="x", score=1.0, classification="filter")
        assert a == b


# ---------------------------------------------------------------------------
# score_entities
# ---------------------------------------------------------------------------


class TestScoreEntities:
    """Tests for the score_entities batch scoring function."""

    def test_returns_list_of_scored_entities(self) -> None:
        freqs = [("water", 50000), ("JWST", 500)]
        result = score_entities(freqs, N=100000)
        assert all(isinstance(s, ScoredEntity) for s in result)

    def test_sorted_by_score_descending(self) -> None:
        freqs = [("common", 90000), ("rare", 10), ("medium", 5000)]
        result = score_entities(freqs, N=100000)
        scores = [s.score for s in result]
        assert scores == sorted(scores, reverse=True)

    def test_classification_matches_threshold(self) -> None:
        freqs = [("water", 50000), ("JWST", 500)]
        result = score_entities(freqs, N=100000)
        by_name = {s.entity: s for s in result}
        # water: df=50000/N=100000 -> log(2) ~ 0.693 < 2.3 -> filter
        assert by_name["water"].classification == "filter"
        # JWST normalizes to "james webb space telescope"
        assert by_name["james webb space telescope"].classification == "keep"

    def test_normalization_merges_duplicates(self) -> None:
        """HST and hst should merge into one normalized entity."""
        freqs = [("HST", 100), ("hst", 50)]
        result = score_entities(freqs, N=10000)
        entities = [s.entity for s in result]
        assert "hubble space telescope" in entities
        assert len(entities) == 1
        # Merged df = 150
        se = result[0]
        assert se.score == pytest.approx(math.log(10000 / 150))

    def test_normalization_disabled(self) -> None:
        freqs = [("HST", 100), ("hst", 50)]
        result = score_entities(freqs, N=10000, normalize=False)
        entities = [s.entity for s in result]
        assert "HST" in entities
        assert "hst" in entities
        assert len(entities) == 2

    def test_custom_threshold(self) -> None:
        freqs = [("entity_a", 1000)]
        # log(10000/1000) = log(10) ~ 2.3
        result_default = score_entities(freqs, N=10000, normalize=False)
        assert result_default[0].classification == "keep"  # score >= threshold

        result_strict = score_entities(freqs, N=10000, threshold=5.0, normalize=False)
        assert result_strict[0].classification == "filter"

    def test_empty_input(self) -> None:
        result = score_entities([], N=100000)
        assert result == []

    def test_zero_df_skipped(self) -> None:
        freqs = [("zero_entity", 0)]
        result = score_entities(freqs, N=100000, normalize=False)
        assert result == []


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """Tests derived directly from the acceptance criteria."""

    def test_water_low_specificity(self) -> None:
        """specificity_score('water', df=50000, N=100000) returns low score below threshold."""
        score = specificity_score("water", df=50000, N=100000)
        assert score < DEFAULT_THRESHOLD

    def test_jwst_high_specificity(self) -> None:
        """specificity_score('James Webb Space Telescope', df=500, N=100000) returns high score."""
        score = specificity_score("James Webb Space Telescope", df=500, N=100000)
        assert score > DEFAULT_THRESHOLD

    def test_score_entities_returns_scored_entity_with_classification(self) -> None:
        """score_entities() returns ScoredEntity(entity, score, classification)."""
        freqs = [("water", 50000), ("JWST", 500)]
        result = score_entities(freqs, N=100000)
        for se in result:
            assert hasattr(se, "entity")
            assert hasattr(se, "score")
            assert hasattr(se, "classification")
            assert se.classification in ("keep", "filter")

    def test_output_json_serializable(self) -> None:
        """Each ScoredEntity can be serialized to JSON with entity/score/classification keys."""
        from dataclasses import asdict

        freqs = [("water", 50000), ("JWST", 500)]
        result = score_entities(freqs, N=100000)
        for se in result:
            d = asdict(se)
            assert "entity" in d
            assert "score" in d
            assert "classification" in d
            # Round-trip through JSON
            serialized = json.dumps(d)
            parsed = json.loads(serialized)
            assert parsed["entity"] == se.entity

    def test_default_threshold_is_log10(self) -> None:
        assert DEFAULT_THRESHOLD == pytest.approx(math.log(10))
