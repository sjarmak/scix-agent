"""Tests for the dbl.3 NER quality profile module."""

from __future__ import annotations

import pytest

from scix.extract.ner_quality_profile import (
    CLASSIFIER_FILTERED_PRECISION,
    LEXICAL_PRECISION_DEFAULT,
    UNFILTERED_PRECISION,
    _classify_era,
    precision_band,
    precision_estimate,
    quality_summary,
)


class TestEraClassification:
    def test_modern_year(self) -> None:
        assert _classify_era(2024) == "modern"
        assert _classify_era(2010) == "modern"

    def test_pre_1990_year(self) -> None:
        assert _classify_era(1985) == "pre_1990"
        assert _classify_era(1900) == "pre_1990"
        assert _classify_era(2009) == "pre_1990"  # 2000s also lumped in

    def test_none_falls_to_pre(self) -> None:
        assert _classify_era(None) == "pre_1990"


class TestPrecisionEstimate:
    def test_lexical_source_uses_default(self) -> None:
        # Any non-gliner source returns the lexical default regardless
        # of agreement / year.
        assert precision_estimate("software", "canonical_exact") == LEXICAL_PRECISION_DEFAULT
        assert precision_estimate("dataset", "keyword_exact_lower") == LEXICAL_PRECISION_DEFAULT

    def test_gliner_unfiltered_modern(self) -> None:
        # Method on modern papers should be ~0.87 unfiltered.
        p = precision_estimate("method", "gliner", agreement=None, year=2023)
        assert p == pytest.approx(UNFILTERED_PRECISION[("method", "modern")])

    def test_gliner_unfiltered_pre(self) -> None:
        p = precision_estimate("dataset", "gliner", agreement=None, year=1980)
        assert p == pytest.approx(UNFILTERED_PRECISION[("dataset", "pre_1990")])

    def test_gliner_filtered_modern(self) -> None:
        p = precision_estimate("instrument", "gliner", agreement=True, year=2024)
        assert p == pytest.approx(CLASSIFIER_FILTERED_PRECISION[("instrument", "modern")])
        assert p == 1.0  # 4/4 in the modern eval

    def test_disagreement_drops_to_floor(self) -> None:
        # Classifier disagreed → low confidence.
        p = precision_estimate("method", "gliner", agreement=False, year=2024)
        assert p <= 0.3

    def test_unknown_type_falls_back(self) -> None:
        # An entity_type we never measured should get a conservative
        # default, not crash.
        p = precision_estimate("unmeasured_type", "gliner", agreement=None, year=2024)
        assert 0.0 <= p <= 1.0

    def test_modern_clearly_better_than_pre_for_software(self) -> None:
        # Sanity: modern software (61%) > pre-1990 software (22%).
        modern = precision_estimate("software", "gliner", agreement=None, year=2024)
        pre = precision_estimate("software", "gliner", agreement=None, year=1985)
        assert modern > pre


class TestPrecisionBand:
    def test_high_band(self) -> None:
        assert precision_band(0.95) == "high"
        assert precision_band(0.85) == "high"

    def test_medium_band(self) -> None:
        assert precision_band(0.75) == "medium"

    def test_low_band(self) -> None:
        assert precision_band(0.55) == "low"

    def test_noisy_band(self) -> None:
        assert precision_band(0.20) == "noisy"
        assert precision_band(0.45) == "noisy"


class TestQualitySummary:
    def test_returns_serializable(self) -> None:
        import json

        s = quality_summary()
        # Round-trips cleanly through JSON (no surprise types).
        json.dumps(s)
        assert "unfiltered" in s
        assert "classifier_filtered" in s
        assert "lexical_default" in s
        assert "notes" in s

    def test_all_known_types_in_summary(self) -> None:
        summary = quality_summary()
        unfilt_keys = set(summary["unfiltered"].keys())
        # Should have at least one bucket for every type we tested.
        for et in (
            "method",
            "chemical",
            "location",
            "organism",
            "instrument",
            "mission",
            "gene",
            "software",
            "dataset",
        ):
            matching = [k for k in unfilt_keys if k.startswith(f"{et}/")]
            assert matching, f"no precision data for type {et}"
