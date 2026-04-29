"""Unit tests for ``scix.extract.surface_normalize.normalize_surface``.

The normalizer is the pre-canonicalization step that strips markup
artifacts and Unicode variants. Tests cover each of the four stages
(tag strip, entity decode, NFKC, punctuation fold) plus the integration
path through ``canonicalize``.
"""

from __future__ import annotations

import pytest

from scix.extract.ner_pass import canonicalize
from scix.extract.surface_normalize import normalize_surface


class TestTagStripping:
    def test_strips_sub(self) -> None:
        assert normalize_surface("co<sub>2</sub>") == "co2"

    def test_strips_sup(self) -> None:
        assert normalize_surface("co<sup>2</sup>") == "co2"

    def test_strips_uppercase_sub(self) -> None:
        assert normalize_surface("CO<SUB>2</SUB>") == "CO2"

    def test_strips_nested_supersub(self) -> None:
        assert (
            normalize_surface("11-hydroxy-<sup>9</sup>-tetrahydrocannabinol")
            == "11-hydroxy-9-tetrahydrocannabinol"
        )

    def test_strips_italic_and_bold(self) -> None:
        assert normalize_surface("<i>E. coli</i>") == "E. coli"
        assert normalize_surface("<b>PyTorch</b>") == "PyTorch"

    def test_strips_tag_with_attributes(self) -> None:
        assert normalize_surface('co<sub class="x">2</sub>') == "co2"

    def test_preserves_non_html_angle_brackets(self) -> None:
        # We only strip a fixed list of inline-formatting tags. Angle
        # brackets that are not those tags must pass through.
        assert normalize_surface("a<b") == "a<b"
        assert normalize_surface("3<5") == "3<5"

    def test_does_not_strip_unknown_tag(self) -> None:
        # Outside the formatting whitelist, leave it alone.
        assert normalize_surface("<chemical>NaCl</chemical>") == "<chemical>NaCl</chemical>"


class TestHtmlEntityDecode:
    def test_named_amp(self) -> None:
        assert normalize_surface("ar&amp;d") == "ar&d"

    def test_named_lt_gt(self) -> None:
        assert normalize_surface("a&lt;b&gt;c") == "a<b>c"

    def test_named_nbsp_becomes_space(self) -> None:
        assert normalize_surface("Hubble&nbsp;Space") == "Hubble Space"

    def test_named_dash_entities(self) -> None:
        assert normalize_surface("a&ndash;b") == "a-b"
        assert normalize_surface("a&mdash;b") == "a-b"

    def test_numeric_decimal(self) -> None:
        assert normalize_surface("Greek &#946; particle") == "Greek β particle"

    def test_numeric_hex(self) -> None:
        assert normalize_surface("Greek &#x3B2; particle") == "Greek β particle"

    def test_unknown_named_entity_passes_through(self) -> None:
        # We only decode a known list to avoid mistakes; other named
        # entities are left untouched so the bug surface stays small.
        assert normalize_surface("rare &foobar; entity") == "rare &foobar; entity"


class TestNfkcNormalization:
    def test_subscript_digit_to_ascii(self) -> None:
        # NFKC pulls fullwidth/subscript digits down to ASCII.
        assert normalize_surface("C₆H₁₂O₆") == "C6H12O6"

    def test_superscript_digit_to_ascii(self) -> None:
        assert normalize_surface("x² + y²") == "x2 + y2"

    def test_fullwidth_letters_to_ascii(self) -> None:
        # Fullwidth A-Z block (Ａ..Ｚ) collapses to ASCII.
        assert normalize_surface("ＡＢＣ") == "ABC"

    def test_ligatures_decomposed(self) -> None:
        assert normalize_surface("ﬁsh") == "fish"  # fi ligature


class TestPunctuationFold:
    def test_en_dash_to_hyphen(self) -> None:
        assert normalize_surface("iron–59") == "iron-59"

    def test_em_dash_to_hyphen(self) -> None:
        assert normalize_surface("soil—root") == "soil-root"

    def test_minus_sign_to_hyphen(self) -> None:
        assert normalize_surface("co−2") == "co-2"

    def test_nb_hyphen_to_hyphen(self) -> None:
        assert normalize_surface("n‑butylenes") == "n-butylenes"

    def test_curly_quotes_to_straight(self) -> None:
        assert normalize_surface("‘a’") == "'a'"
        assert normalize_surface("“a”") == '"a"'

    def test_nbsp_to_space(self) -> None:
        assert normalize_surface("a b") == "a b"

    def test_thin_space_to_space(self) -> None:
        assert normalize_surface("a b") == "a b"

    def test_zero_width_chars_dropped(self) -> None:
        assert normalize_surface("a​b") == "ab"
        assert normalize_surface("a‌b") == "ab"
        assert normalize_surface("a﻿") == "a"

    def test_soft_hyphen_dropped(self) -> None:
        # Soft hyphen is a layout-only artifact; don't keep it in the
        # canonical key.
        assert normalize_surface("co­2") == "co2"


class TestPreservesUsefulText:
    def test_does_not_lowercase(self) -> None:
        # Lowercasing is downstream in canonicalize(); normalize_surface
        # leaves case alone.
        assert normalize_surface("CRISPR-Cas9") == "CRISPR-Cas9"

    def test_does_not_collapse_whitespace_runs(self) -> None:
        # Whitespace collapsing is downstream in canonicalize().
        assert normalize_surface("Hubble  Space") == "Hubble  Space"

    def test_does_not_strip_outer_whitespace(self) -> None:
        assert normalize_surface("  Hubble  ") == "  Hubble  "

    def test_preserves_meaningful_punctuation(self) -> None:
        assert normalize_surface("p53") == "p53"
        assert normalize_surface("CRISPR-Cas9") == "CRISPR-Cas9"
        assert normalize_surface("NF-kB") == "NF-kB"


class TestCanonicalizeIntegration:
    """The corpus-observed merges driven by surface_normalize."""

    def test_co_subscript_collapses_with_co2(self) -> None:
        # The flagship case from bead scix_experiments-06xc:
        # 'co<sub>2</sub>' must collapse to the same canonical as 'co2'.
        assert canonicalize("co<sub>2</sub>") == canonicalize("co2") == "co2"

    def test_iron_endash_collapses_with_iron_dash(self) -> None:
        # Unicode dash variants of the same chemical isotope label.
        assert canonicalize("iron–59") == canonicalize("iron-59") == "iron-59"

    def test_html_entity_collapses(self) -> None:
        assert canonicalize("ar&amp;d") == canonicalize("ar&d") == "ar&d"

    def test_subscript_glyph_collapses(self) -> None:
        # Unicode subscript digits and ASCII forms must agree.
        assert canonicalize("CO₂") == canonicalize("CO2") == "co2"

    def test_nbsp_collapses_with_space(self) -> None:
        assert (
            canonicalize("hubble space telescope")
            == canonicalize("Hubble Space Telescope")
            == "hubble space telescope"
        )

    def test_zwsp_collapses(self) -> None:
        assert canonicalize("py​torch") == canonicalize("pytorch") == "pytorch"

    @pytest.mark.parametrize(
        "noisy,expected",
        [
            ("co<sub>2</sub>", "co2"),
            ("CO<SUB>2</SUB>", "co2"),
            ("co₂", "co2"),
            ("iron–59", "iron-59"),
            ("Hubble Space", "hubble space"),
            ("PyTorch (Paszke et al., 2019)", "pytorch"),
        ],
    )
    def test_corpus_examples(self, noisy: str, expected: str) -> None:
        assert canonicalize(noisy) == expected
