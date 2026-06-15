"""Tests for ``scix.oa_gate`` — the OA/preprint predicate.

Mirrors the SQL function ``papers_is_oa_or_preprint(p papers)`` defined in
``migrations/068_papers_is_oa_or_preprint.sql``. Lets the body-AI pipeline
test suite assert predicate semantics without a live DB.

The migration file's predicate text is also pinned here — if either side
drifts, both branches must be updated together.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scix.oa_gate import (
    SQL_FUNCTION_NAME,
    is_oa_or_preprint,
)

_MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "migrations" / "068_papers_is_oa_or_preprint.sql"
)


# ---------------------------------------------------------------------------
# Predicate semantics — Python mirror of the SQL function
# ---------------------------------------------------------------------------


class TestIsOaOrPreprintPredicate:
    def test_property_contains_openaccess_returns_true(self) -> None:
        assert is_oa_or_preprint(property_=["OPENACCESS"], arxiv_class=None) is True

    def test_property_contains_openaccess_among_others(self) -> None:
        assert (
            is_oa_or_preprint(
                property_=["EPRINT_OPENACCESS", "OPENACCESS", "TOC"],
                arxiv_class=None,
            )
            is True
        )

    def test_property_without_openaccess_falls_through(self) -> None:
        # Has OA-related substrings but not the literal token.
        assert (
            is_oa_or_preprint(
                property_=["EPRINT_OPENACCESS", "PUB_OPENACCESS"],
                arxiv_class=None,
            )
            is False
        )

    def test_non_empty_arxiv_class_returns_true(self) -> None:
        assert is_oa_or_preprint(property_=None, arxiv_class=["astro-ph"]) is True

    def test_property_and_arxiv_class_both_qualify(self) -> None:
        assert (
            is_oa_or_preprint(
                property_=["OPENACCESS"],
                arxiv_class=["astro-ph", "cs.AI"],
            )
            is True
        )

    def test_property_qualifies_with_empty_arxiv_class(self) -> None:
        assert is_oa_or_preprint(property_=["OPENACCESS"], arxiv_class=[]) is True

    def test_arxiv_class_qualifies_with_no_oa_property(self) -> None:
        assert is_oa_or_preprint(property_=["TOC"], arxiv_class=["astro-ph"]) is True

    def test_both_none_returns_false(self) -> None:
        assert is_oa_or_preprint(property_=None, arxiv_class=None) is False

    def test_both_empty_arrays_returns_false(self) -> None:
        # Mirrors PG: array_length({}, 1) is NULL — wrapped in COALESCE so the
        # SQL result is FALSE; the Python mirror returns the same.
        assert is_oa_or_preprint(property_=[], arxiv_class=[]) is False

    def test_property_empty_arxiv_none_returns_false(self) -> None:
        assert is_oa_or_preprint(property_=[], arxiv_class=None) is False

    def test_property_none_arxiv_empty_returns_false(self) -> None:
        assert is_oa_or_preprint(property_=None, arxiv_class=[]) is False


# ---------------------------------------------------------------------------
# Migration file pin — the SQL function name + predicate must stay aligned
# with what scripts call and what oa_gate.py mirrors.
# ---------------------------------------------------------------------------


class TestMigrationFileContract:
    def test_migration_file_exists(self) -> None:
        assert _MIGRATION_PATH.exists(), (
            f"Expected migration at {_MIGRATION_PATH}; if it moved, update "
            "tests/test_oa_gate.py to point at the new path."
        )

    def test_migration_defines_expected_function_name(self) -> None:
        sql = _MIGRATION_PATH.read_text()
        assert SQL_FUNCTION_NAME == "papers_is_oa_or_preprint"
        assert re.search(
            rf"\bFUNCTION\s+{re.escape(SQL_FUNCTION_NAME)}\s*\(",
            sql,
            re.IGNORECASE,
        ), "migration must define the canonical SQL function"

    def test_migration_predicate_includes_both_branches(self) -> None:
        # The predicate must reference both `property` and `arxiv_class`
        # (the two branches of the OA/preprint test). If either is missing,
        # the gate is broken.
        sql = _MIGRATION_PATH.read_text()
        assert "OPENACCESS" in sql, "migration must check for OPENACCESS property"
        assert "property" in sql, "migration must reference the property column"
        assert "arxiv_class" in sql, "migration must reference the arxiv_class column"
        assert "array_length" in sql, "migration must use array_length for the arxiv_class branch"
        assert "COALESCE" in sql, (
            "migration must wrap branches in COALESCE so the result is " "BOOLEAN (not tri-valued)"
        )

    def test_migration_creates_partial_index_on_body_not_null(self) -> None:
        sql = _MIGRATION_PATH.read_text()
        assert "CONCURRENTLY" in sql, (
            "expression index must be CONCURRENTLY (32M-row papers table; "
            "regular CREATE INDEX would block writes)"
        )
        assert "idx_papers_is_oa" in sql
        # Partial index keeps it small — bead spec ~150 MB.
        assert re.search(r"WHERE\s+body\s+IS\s+NOT\s+NULL", sql, re.IGNORECASE)

    def test_migration_has_no_top_level_transaction(self) -> None:
        # CREATE INDEX CONCURRENTLY cannot run inside a transaction — same
        # convention as migration 054. Migration runner must execute this
        # file with autocommit.
        sql = _MIGRATION_PATH.read_text()
        # Allow BEGIN/COMMIT inside DO blocks if any future revision needs
        # them, but the file should not start the script with BEGIN.
        body = "\n".join(
            line for line in sql.splitlines() if not line.lstrip().startswith("--") and line.strip()
        )
        assert not re.match(r"\s*BEGIN\s*;", body, re.IGNORECASE), (
            "migration must not wrap the whole file in a transaction "
            "(CREATE INDEX CONCURRENTLY would fail)"
        )


# ---------------------------------------------------------------------------
# Sanity: the Python helper must reject nonsense input shapes
# ---------------------------------------------------------------------------


class TestIsOaOrPreprintRejectsNonList:
    def test_property_must_be_list_or_none(self) -> None:
        with pytest.raises(TypeError):
            is_oa_or_preprint(property_="OPENACCESS", arxiv_class=None)  # type: ignore[arg-type]

    def test_arxiv_class_must_be_list_or_none(self) -> None:
        with pytest.raises(TypeError):
            is_oa_or_preprint(property_=None, arxiv_class="astro-ph")  # type: ignore[arg-type]
