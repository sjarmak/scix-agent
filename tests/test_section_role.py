"""Tests for section_role — rule-based section header role classifier.

Covers:
* unit tests on canonical section names from :mod:`scix.section_parser`
* numbering / Roman / "Section N:" prefix stripping
* compound headers ("Results and Discussion", "Materials and Methods")
* fixture-based accuracy >= 0.80 over 50 hand-crafted headers
* integration test: ``read_paper_section(role='method')`` selects the
  methods-classified section (mocked DB)
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scix import search
from scix.section_role import (
    ROLE_BACKGROUND,
    ROLE_CONCLUSION,
    ROLE_METHOD,
    ROLE_OTHER,
    ROLE_RESULT,
    ROLES,
    classify_section_role,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "section_roles_50.jsonl"


# ---------------------------------------------------------------------------
# Unit tests on canonical section names
# ---------------------------------------------------------------------------


class TestCanonicalNames:
    """The canonical names emitted by section_parser should classify cleanly."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("introduction", ROLE_BACKGROUND),
            ("methods", ROLE_METHOD),
            ("observations", ROLE_METHOD),
            ("data", ROLE_METHOD),
            ("results", ROLE_RESULT),
            ("discussion", ROLE_CONCLUSION),
            ("conclusions", ROLE_CONCLUSION),
            ("summary", ROLE_CONCLUSION),
            ("abstract", ROLE_OTHER),
            ("acknowledgments", ROLE_OTHER),
            ("references", ROLE_OTHER),
            ("full", ROLE_OTHER),
        ],
    )
    def test_canonical_mapping(self, name: str, expected: str) -> None:
        assert classify_section_role(name) == expected


class TestNumberingStripping:
    """Leading numbering of various flavours should be tolerated."""

    @pytest.mark.parametrize(
        "header,expected",
        [
            ("1. Introduction", ROLE_BACKGROUND),
            ("2.1 Data Reduction", ROLE_METHOD),
            ("3.4.5. Methods", ROLE_METHOD),
            ("III. Results and Discussion", ROLE_CONCLUSION),
            ("IV. Discussion", ROLE_CONCLUSION),
            ("Section 4: Methodology", ROLE_METHOD),
            ("Section 1: Introduction", ROLE_BACKGROUND),
            ("Chapter 2: Observations", ROLE_METHOD),
        ],
    )
    def test_strips_numbering(self, header: str, expected: str) -> None:
        assert classify_section_role(header) == expected


class TestCompoundHeaders:
    """Compound headers should resolve via priority order."""

    def test_results_and_discussion_is_conclusion(self) -> None:
        # Discussion is the synthesis signal; "Results and Discussion" is
        # conventionally the combined synthesis section.
        assert classify_section_role("Results and Discussion") == ROLE_CONCLUSION

    def test_materials_and_methods_is_method(self) -> None:
        assert classify_section_role("Materials and Methods") == ROLE_METHOD

    def test_summary_and_conclusions_is_conclusion(self) -> None:
        assert classify_section_role("Summary and Conclusions") == ROLE_CONCLUSION


class TestEdgeCases:
    def test_empty_string(self) -> None:
        assert classify_section_role("") == ROLE_OTHER

    def test_only_numbering(self) -> None:
        assert classify_section_role("2.1") == ROLE_OTHER

    def test_unknown_header(self) -> None:
        assert classify_section_role("Foobar Quux") == ROLE_OTHER

    def test_non_string_input(self) -> None:
        # Defensive: classifier should not crash on None.
        assert classify_section_role(None) == ROLE_OTHER  # type: ignore[arg-type]

    def test_role_set_is_complete(self) -> None:
        assert ROLES == {
            ROLE_BACKGROUND,
            ROLE_METHOD,
            ROLE_RESULT,
            ROLE_CONCLUSION,
            ROLE_OTHER,
        }


# ---------------------------------------------------------------------------
# Fixture accuracy gate
# ---------------------------------------------------------------------------


def _load_fixture() -> list[dict[str, str]]:
    """Read the 50-pair fixture from disk."""
    with FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_fixture_has_exactly_50_pairs() -> None:
    pairs = _load_fixture()
    assert len(pairs) == 50, f"Expected 50 fixture pairs, got {len(pairs)}"


def test_fixture_all_roles_valid() -> None:
    pairs = _load_fixture()
    for p in pairs:
        assert p["role"] in ROLES, f"Invalid role in fixture: {p}"


def test_accuracy_on_fixture() -> None:
    """Classifier must achieve >= 0.80 accuracy on the 50-pair fixture."""
    pairs = _load_fixture()
    correct = 0
    misses: list[tuple[str, str, str]] = []
    for p in pairs:
        predicted = classify_section_role(p["header"])
        if predicted == p["role"]:
            correct += 1
        else:
            misses.append((p["header"], p["role"], predicted))
    accuracy = correct / len(pairs)
    assert accuracy >= 0.80, (
        f"Accuracy {accuracy:.2f} < 0.80 — misses ({len(misses)}): {misses}"
    )


# ---------------------------------------------------------------------------
# Signature contract: read_paper_section must accept role=...
# ---------------------------------------------------------------------------


def test_read_paper_section_accepts_role_param() -> None:
    sig = inspect.signature(search.read_paper_section)
    assert "role" in sig.parameters, (
        "read_paper_section must accept a 'role' keyword argument"
    )
    param = sig.parameters["role"]
    assert param.default is None, "role must default to None to preserve behavior"


# ---------------------------------------------------------------------------
# Integration test: role='method' selects a methods-classified section
# ---------------------------------------------------------------------------


class _MockCursor:
    """Minimal context-manager cursor that returns a fixed row."""

    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row
        self._executed: list[tuple[str, tuple]] = []

    def __enter__(self) -> "_MockCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: tuple = ()) -> None:
        self._executed.append((sql, params))

    def fetchone(self) -> dict[str, Any] | None:
        return self._row


def _make_mock_conn(body: str, *, abstract: str = "", title: str = "Test Paper") -> MagicMock:
    """Build a mock psycopg connection that returns a single paper row."""
    conn = MagicMock()
    row = {"body": body, "abstract": abstract, "title": title}

    # First cursor call returns the paper row; subsequent calls (e.g. the
    # ADR-006 provenance check) hit papers_fulltext, which we want to
    # behave as "no row" — return None. We achieve this by handing out
    # different cursors per invocation.
    cursors = iter([_MockCursor(row), _MockCursor(None)])

    def cursor_factory(*args: Any, **kwargs: Any) -> _MockCursor:
        try:
            return next(cursors)
        except StopIteration:
            return _MockCursor(None)

    conn.cursor.side_effect = cursor_factory
    return conn


def test_read_paper_section_role_method_selects_methods_section() -> None:
    body = (
        "1. Introduction\n"
        "We study star formation in molecular clouds.\n"
        "\n"
        "2. Methods\n"
        "We used the VLT to observe 42 protostellar cores.\n"
        "\n"
        "3. Results\n"
        "We detected outflows in 38 of the targets.\n"
        "\n"
        "4. Conclusions\n"
        "Outflows are ubiquitous in young protostars.\n"
    )
    conn = _make_mock_conn(body)

    result = search.read_paper_section(conn, "2024TEST..1.....X", role="method")

    assert result.total == 1, f"Expected one paper, got metadata={result.metadata}"
    paper = result.papers[0]
    assert paper["section_name"] == "methods"
    assert classify_section_role(paper["section_name"]) == ROLE_METHOD
    assert "VLT" in paper["section_text"]


def test_read_paper_section_role_none_preserves_default_behavior() -> None:
    """When role=None, behavior matches section='full' default."""
    body = "1. Introduction\nFoo\n\n2. Methods\nBar\n"
    conn = _make_mock_conn(body)
    result = search.read_paper_section(conn, "2024TEST..1.....X")

    assert result.total == 1
    paper = result.papers[0]
    assert paper["section_name"] == "full"


def test_read_paper_section_role_no_match_returns_error() -> None:
    """Requesting a role with no matching parsed section returns error metadata."""
    body = "Abstract\nShort abstract only.\n"  # Will parse as 'full' (no headers)
    conn = _make_mock_conn(body)

    result = search.read_paper_section(conn, "2024TEST..1.....X", role="method")

    # 'full' classifies as 'other', so role='method' has no match.
    assert result.total == 0
    assert "error" in result.metadata
    assert "method" in result.metadata["error"]
