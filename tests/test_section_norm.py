"""Tests for ``scix.extract.chunk_pass.section_norm``.

Covers all nine acceptance criteria from the section-norm work unit:

1. Module exists and is importable.
2. ``normalize_heading`` exposes the documented canonical vocabulary.
3. ``HEADING_MAP`` is a ``dict[str, str]`` with pre-normalized keys.
4. Mapping covers the spec'd synonym list (one explicit case per name).
5. Case-insensitive lookup.
6. Whitespace-tolerant lookup (collapse runs, strip ends).
7. Unknown headings collapse to ``'other'``.
8. ``None`` / empty / whitespace-only collapse to ``'unknown'``.
9. Pure-function: same input -> same output, no module mutation.
"""

from __future__ import annotations

import pytest

from scix.extract.chunk_pass import section_norm
from scix.extract.chunk_pass.section_norm import (
    CANONICAL_HEADINGS,
    HEADING_MAP,
    OTHER,
    UNKNOWN,
    normalize_heading,
)


# ---------------------------------------------------------------------------
# Criterion 1: module exists & is importable
# ---------------------------------------------------------------------------


def test_module_importable() -> None:
    """The module imports cleanly with no I/O side effects."""
    assert section_norm is not None
    assert hasattr(section_norm, "normalize_heading")
    assert hasattr(section_norm, "HEADING_MAP")


# ---------------------------------------------------------------------------
# Criterion 2: canonical vocabulary is exactly the documented set
# ---------------------------------------------------------------------------


EXPECTED_CANONICAL = {
    "abstract",
    "introduction",
    "background",
    "related_work",
    "methods",
    "experiments",
    "data",
    "results",
    "discussion",
    "conclusion",
    "acknowledgments",
    "references",
    "appendix",
    "other",
}


def test_canonical_vocabulary_matches_spec() -> None:
    assert set(CANONICAL_HEADINGS) == EXPECTED_CANONICAL


def test_normalize_returns_only_canonical_or_unknown() -> None:
    """Every output of ``normalize_heading`` is in canonical set or 'unknown'."""
    samples = [
        "Methods",
        "introduction",
        "Bogus heading not in any synonym list",
        "",
        None,
        "Discussion",
    ]
    for raw in samples:
        out = normalize_heading(raw)
        assert out in CANONICAL_HEADINGS or out == UNKNOWN, (
            f"normalize_heading({raw!r}) returned non-canonical {out!r}"
        )


# ---------------------------------------------------------------------------
# Criterion 3: HEADING_MAP shape
# ---------------------------------------------------------------------------


def test_heading_map_is_dict_str_str() -> None:
    assert isinstance(HEADING_MAP, dict)
    assert HEADING_MAP, "HEADING_MAP should not be empty"
    for k, v in HEADING_MAP.items():
        assert isinstance(k, str), f"HEADING_MAP key {k!r} not a str"
        assert isinstance(v, str), f"HEADING_MAP value {v!r} not a str"


def test_heading_map_keys_are_pre_normalized() -> None:
    """Keys must be lowercase + whitespace-collapsed.

    Otherwise lookups in :func:`normalize_heading` would silently miss on
    obvious variants of the keys themselves.
    """
    for key in HEADING_MAP:
        assert key == key.lower(), f"HEADING_MAP key {key!r} is not lowercase"
        assert key == " ".join(key.split()), (
            f"HEADING_MAP key {key!r} has non-canonical whitespace"
        )


def test_heading_map_values_are_canonical() -> None:
    """Every value in HEADING_MAP must be a canonical label."""
    for key, value in HEADING_MAP.items():
        assert value in CANONICAL_HEADINGS, (
            f"HEADING_MAP[{key!r}] = {value!r} is not a canonical label"
        )


# ---------------------------------------------------------------------------
# Criterion 4: explicit synonym coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        # methods
        ("method", "methods"),
        ("methodology", "methods"),
        ("materials and methods", "methods"),
        ("experimental procedures", "methods"),
        # results
        ("result", "results"),
        ("findings", "results"),
        # discussion
        ("discussions", "discussion"),
        # conclusion
        ("conclusions", "conclusion"),
        ("concluding remarks", "conclusion"),
        ("summary", "conclusion"),
        # introduction
        ("intro", "introduction"),
        # data
        ("datasets", "data"),
        ("observations", "data"),
        ("sample", "data"),
    ],
)
def test_named_synonyms_map_to_canonical(raw: str, expected: str) -> None:
    assert normalize_heading(raw) == expected


# ---------------------------------------------------------------------------
# Criterion 5: case-insensitivity
# ---------------------------------------------------------------------------


def test_case_insensitive_methods() -> None:
    assert normalize_heading("METHODS") == "methods"
    assert normalize_heading("methods") == "methods"
    assert normalize_heading("Methods") == "methods"
    assert normalize_heading("METHODS") == normalize_heading("methods") == "methods"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("INTRODUCTION", "introduction"),
        ("Introduction", "introduction"),
        ("RESULTS", "results"),
        ("Results", "results"),
        ("DISCUSSION", "discussion"),
        ("References", "references"),
        ("APPENDIX", "appendix"),
    ],
)
def test_case_insensitive_canonicals(raw: str, expected: str) -> None:
    assert normalize_heading(raw) == expected


# ---------------------------------------------------------------------------
# Criterion 6: whitespace-tolerance
# ---------------------------------------------------------------------------


def test_whitespace_tolerant_materials_and_methods() -> None:
    assert normalize_heading("  Materials  and  Methods  ") == "methods"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  methods", "methods"),
        ("methods  ", "methods"),
        ("   methods   ", "methods"),
        ("Materials\tand\tMethods", "methods"),  # tabs
        ("Materials\nand\nMethods", "methods"),  # newlines
        ("Related   Work", "related_work"),
        ("  related work  ", "related_work"),
    ],
)
def test_whitespace_tolerant_variants(raw: str, expected: str) -> None:
    assert normalize_heading(raw) == expected


# ---------------------------------------------------------------------------
# Criterion 7: unknown -> 'other'
# ---------------------------------------------------------------------------


def test_unknown_heading_returns_other() -> None:
    assert normalize_heading("Lorem Ipsum Section") == "other"


@pytest.mark.parametrize(
    "raw",
    [
        "Lorem Ipsum Section",
        "Quux",
        "12345",
        "Section That Definitely Does Not Exist",
        "the future of fish",
    ],
)
def test_unknown_headings_collapse_to_other(raw: str) -> None:
    assert normalize_heading(raw) == OTHER == "other"


# ---------------------------------------------------------------------------
# Criterion 8: None / empty / whitespace -> 'unknown'
# ---------------------------------------------------------------------------


def test_none_returns_unknown() -> None:
    assert normalize_heading(None) == "unknown"


def test_empty_string_returns_unknown() -> None:
    assert normalize_heading("") == "unknown"


@pytest.mark.parametrize("raw", ["", " ", "   ", "\t", "\n", " \t\n "])
def test_whitespace_only_returns_unknown(raw: str) -> None:
    assert normalize_heading(raw) == UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# Criterion 9: purity / determinism
# ---------------------------------------------------------------------------


def test_normalize_is_deterministic() -> None:
    """Repeated calls return the same output."""
    for raw in ["Methods", "Lorem", None, "  results  ", ""]:
        first = normalize_heading(raw)
        for _ in range(5):
            assert normalize_heading(raw) == first


def test_normalize_does_not_mutate_heading_map() -> None:
    """Calling ``normalize_heading`` must not alter the lookup table."""
    snapshot = dict(HEADING_MAP)
    for raw in [
        "Methods",
        "intro",
        "Brand New Heading Never Seen Before",
        None,
        "",
        "  Materials and Methods  ",
    ]:
        normalize_heading(raw)
    assert HEADING_MAP == snapshot


# ---------------------------------------------------------------------------
# Sanity: the module's own constants are wired up correctly
# ---------------------------------------------------------------------------


def test_unknown_constant_value() -> None:
    assert UNKNOWN == "unknown"


def test_other_constant_value() -> None:
    assert OTHER == "other"
