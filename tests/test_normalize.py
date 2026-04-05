"""Tests for the deterministic entity normalization pipeline."""

from __future__ import annotations

import pytest

from scix.normalize import (
    ALIAS_MAP,
    NormalizationResult,
    normalize_batch,
    normalize_entity,
)

# ---------------------------------------------------------------------------
# Stage 1: Unicode NFKC + lowercase + strip
# ---------------------------------------------------------------------------


class TestStage1Unicode:
    """Stage 1: Unicode NFKC normalization, lowercasing, and stripping."""

    def test_lowercase(self) -> None:
        assert normalize_entity("Density Functional Theory") == "density functional theory"

    def test_strip_whitespace(self) -> None:
        assert normalize_entity("  galaxy clusters  ") == "galaxy clusters"

    def test_unicode_nfkc(self) -> None:
        # NFKC decomposes ligatures and compatibility characters
        # \ufb01 = fi ligature -> "fi"
        assert normalize_entity("\ufb01tting") == "fitting"

    def test_fullwidth_characters(self) -> None:
        # Fullwidth latin letters (e.g. \uff21 = fullwidth 'A')
        assert normalize_entity("\uff21\uff22\uff23") == "abc"

    def test_empty_string(self) -> None:
        assert normalize_entity("") == ""

    def test_only_whitespace(self) -> None:
        assert normalize_entity("   \t\n  ") == ""


# ---------------------------------------------------------------------------
# Stage 2: Punctuation normalization
# ---------------------------------------------------------------------------


class TestStage2Punctuation:
    """Stage 2: Hyphens to spaces, remove possessives."""

    def test_hyphen_to_space(self) -> None:
        assert normalize_entity("density-functional theory") == "density functional theory"

    def test_en_dash_to_space(self) -> None:
        # \u2013 = en dash
        assert normalize_entity("mass\u2013luminosity") == "mass luminosity"

    def test_em_dash_to_space(self) -> None:
        # \u2014 = em dash
        assert normalize_entity("star\u2014forming") == "star forming"

    def test_possessive_removal(self) -> None:
        assert normalize_entity("Hubble's Law") == "hubble law"

    def test_case_equivalence_with_hyphens(self) -> None:
        assert normalize_entity("density-functional theory") == normalize_entity(
            "Density Functional Theory"
        )


# ---------------------------------------------------------------------------
# Stage 3: Alias resolution
# ---------------------------------------------------------------------------


class TestStage3Alias:
    """Stage 3: Alias resolution via curated dictionary."""

    def test_mcmc_alias(self) -> None:
        assert normalize_entity("  MCMC ") == "markov chain monte carlo"

    def test_hst_alias(self) -> None:
        assert normalize_entity("HST") == "hubble space telescope"

    def test_jwst_alias(self) -> None:
        assert normalize_entity("JWST") == "james webb space telescope"

    def test_sdss_alias(self) -> None:
        assert normalize_entity("SDSS") == "sloan digital sky survey"

    def test_2mass_alias(self) -> None:
        assert normalize_entity("2MASS") == "two micron all sky survey"

    def test_wise_alias(self) -> None:
        assert normalize_entity("WISE") == "wide field infrared survey explorer"

    def test_xmm_newton_alias(self) -> None:
        # XMM-Newton: hyphen becomes space, then alias matches "xmm newton"
        assert normalize_entity("XMM-Newton") == "x ray multi mirror mission"

    def test_pca_alias(self) -> None:
        assert normalize_entity("PCA") == "principal component analysis"

    def test_cmb_alias(self) -> None:
        assert normalize_entity("CMB") == "cosmic microwave background"

    def test_no_alias_passthrough(self) -> None:
        # Strings not in ALIAS_MAP pass through unchanged (just normalized)
        assert normalize_entity("stellar mass") == "stellar mass"

    def test_alias_map_has_minimum_entries(self) -> None:
        assert len(ALIAS_MAP) >= 30


# ---------------------------------------------------------------------------
# Stage 4: Whitespace collapse
# ---------------------------------------------------------------------------


class TestStage4Whitespace:
    """Stage 4: Collapse whitespace runs to single space."""

    def test_multiple_spaces(self) -> None:
        assert normalize_entity("star   formation   rate") == "star formation rate"

    def test_tabs_and_newlines(self) -> None:
        assert normalize_entity("dark\t\tenergy\nsurvey") == "dark energy survey"

    def test_mixed_whitespace(self) -> None:
        assert (
            normalize_entity("  cosmic \t microwave \n background  ")
            == "cosmic microwave background"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """All transformations must be deterministic."""

    @pytest.mark.parametrize(
        "entity",
        [
            "Density-Functional Theory",
            "  MCMC ",
            "HST",
            "XMM-Newton",
            "stellar mass function",
            "",
            "2MASS",
        ],
    )
    def test_same_input_same_output(self, entity: str) -> None:
        first = normalize_entity(entity)
        second = normalize_entity(entity)
        assert first == second


# ---------------------------------------------------------------------------
# Acceptance criteria: exact matches
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """Tests taken directly from the acceptance criteria."""

    def test_dft_case_hyphen_equivalence(self) -> None:
        assert normalize_entity("density-functional theory") == normalize_entity(
            "Density Functional Theory"
        )

    def test_mcmc_whitespace_alias(self) -> None:
        assert normalize_entity("  MCMC ") == "markov chain monte carlo"

    def test_hst_alias(self) -> None:
        assert normalize_entity("HST") == "hubble space telescope"


# ---------------------------------------------------------------------------
# normalize_batch
# ---------------------------------------------------------------------------


class TestNormalizeBatch:
    """Tests for normalize_batch() including the denormalization map."""

    def test_returns_normalization_result(self) -> None:
        result = normalize_batch(["HST", "JWST"])
        assert isinstance(result, NormalizationResult)

    def test_canonical_forms_length(self) -> None:
        entities = ["HST", "Hubble Space Telescope", "hst"]
        result = normalize_batch(entities)
        assert len(result.canonical_forms) == len(entities)

    def test_canonical_forms_order(self) -> None:
        entities = ["MCMC", "PCA", "stellar mass"]
        result = normalize_batch(entities)
        assert result.canonical_forms == (
            "markov chain monte carlo",
            "principal component analysis",
            "stellar mass",
        )

    def test_original_map_contains_all_originals(self) -> None:
        entities = ["HST", "hst", "Hubble Space Telescope"]
        result = normalize_batch(entities)
        canonical = "hubble space telescope"
        assert canonical in result.original_map
        assert result.original_map[canonical] == frozenset(entities)

    def test_original_map_is_frozen(self) -> None:
        result = normalize_batch(["HST"])
        # MappingProxyType is not directly mutable
        with pytest.raises(TypeError):
            result.original_map["new_key"] = frozenset(["x"])  # type: ignore[index]

    def test_batch_deduplication_reduces_unique_count(self) -> None:
        """A batch of 100 entities with known duplicates reduces unique count >30%."""
        # Build a list of 100 entities with heavy duplication via case/hyphen/alias variants
        base_pairs: list[tuple[str, ...]] = [
            ("HST", "hst", "Hubble Space Telescope", "hubble space telescope"),
            ("JWST", "jwst", "James Webb Space Telescope"),
            ("MCMC", "mcmc", "Markov Chain Monte Carlo", "markov chain monte carlo"),
            ("SDSS", "sdss", "Sloan Digital Sky Survey"),
            ("PCA", "pca", "Principal Component Analysis"),
            ("CMB", "cmb", "Cosmic Microwave Background"),
            ("ALMA", "alma", "Atacama Large Millimeter Array"),
            ("WISE", "wise", "Wide-field Infrared Survey Explorer"),
            ("2MASS", "2mass", "Two Micron All Sky Survey"),
            ("TESS", "tess", "Transiting Exoplanet Survey Satellite"),
            ("DES", "des", "Dark Energy Survey"),
            ("XMM-Newton", "xmm newton", "XMM", "xmm"),
            ("density-functional theory", "Density Functional Theory", "DFT"),
            ("star formation rate", "Star Formation Rate", "SFR", "sfr"),
            ("active galactic nucleus", "Active Galactic Nucleus", "AGN", "agn"),
        ]
        entities: list[str] = []
        for group in base_pairs:
            entities.extend(group)
        # Pad to reach 100
        while len(entities) < 100:
            entities.append("filler entity")

        assert len(entities) >= 100

        result = normalize_batch(entities[:100])
        unique_inputs = len(set(entities[:100]))
        unique_outputs = len(set(result.canonical_forms))

        reduction = 1.0 - (unique_outputs / unique_inputs)
        assert reduction > 0.30, (
            f"Expected >30% reduction in unique count, got {reduction:.1%} "
            f"({unique_inputs} -> {unique_outputs})"
        )

    def test_empty_batch(self) -> None:
        result = normalize_batch([])
        assert result.canonical_forms == ()
        assert dict(result.original_map) == {}
