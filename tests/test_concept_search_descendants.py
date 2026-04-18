"""Integration tests for UAT-hierarchy-aware concept_search descendant expansion.

Covers:
    - Recursive CTE terminates at depth CONCEPT_DESCENDANT_MAX_DEPTH (6) on a
      deep fixture hierarchy (root + 7 descendants).
    - ``include_descendants=True`` returns a strict superset of
      ``include_descendants=False`` on a corpus with descendants.
    - A leaf concept (no descendants) returns the same result set either way.

All writes go to ``SCIX_TEST_DSN``; tests skip if it is unset or points at the
production database.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from helpers import get_test_dsn

from scix.search import CONCEPT_DESCENDANT_MAX_DEPTH, concept_search

# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

# Depth-0 .. depth-7 concept chain (8 levels total -> 7 descendants).
# CONCEPT_DESCENDANT_MAX_DEPTH bounds expansion to depth 6, so a depth-7
# descendant must NOT appear in the superset results.
_DEPTH_TOTAL = 7  # levels beyond root; produces root + 7 descendants
assert _DEPTH_TOTAL > CONCEPT_DESCENDANT_MAX_DEPTH, (
    "fixture must extend past the depth cap to exercise truncation"
)

_CONCEPT_PREFIX = "http://example.test/uat-depth-fixture/"
_BIBCODE_PREFIX = "9999TEST.conf..00"  # yields 9999TEST.conf..000 .. 007
_LEAF_CONCEPT = "http://example.test/uat-depth-fixture/leaf"
_LEAF_BIBCODE = "9999TEST.leaf..000"


def _concept_id(depth: int) -> str:
    return f"{_CONCEPT_PREFIX}L{depth}"


def _bibcode(depth: int) -> str:
    # 3-digit zero-padded suffix, so each depth gets a unique bibcode.
    return f"{_BIBCODE_PREFIX}{depth:01d}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_dsn() -> str:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set or points at production DB")
    return dsn


@pytest.fixture
def conn(test_dsn: str) -> Iterator[psycopg.Connection]:
    c = psycopg.connect(test_dsn)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def hierarchy(conn: psycopg.Connection) -> Iterator[None]:
    """Build a depth-7 concept chain + a disconnected leaf concept.

    Layout::

        root=L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> L7
        (each Ln tagged on paper 9999TEST.conf..00n)

        leaf (no descendants, tagged on 9999TEST.leaf..000)

    All rows are inserted transactionally; teardown rolls the transaction back
    so the test DB is left clean even if assertions fail.
    """
    with conn.cursor() as cur:
        # Clean slate for any prior aborted runs. Use delete-by-prefix so we
        # don't touch unrelated fixture data that might live in scix_test.
        cur.execute(
            "DELETE FROM paper_uat_mappings WHERE bibcode LIKE %s",
            (f"{_BIBCODE_PREFIX}%",),
        )
        cur.execute(
            "DELETE FROM paper_uat_mappings WHERE bibcode = %s",
            (_LEAF_BIBCODE,),
        )
        cur.execute(
            "DELETE FROM uat_relationships WHERE parent_id LIKE %s OR child_id LIKE %s",
            (f"{_CONCEPT_PREFIX}%", f"{_CONCEPT_PREFIX}%"),
        )
        cur.execute(
            "DELETE FROM uat_concepts WHERE concept_id LIKE %s",
            (f"{_CONCEPT_PREFIX}%",),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode LIKE %s OR bibcode = %s",
            (f"{_BIBCODE_PREFIX}%", _LEAF_BIBCODE),
        )
        conn.commit()

        # Insert concepts L0..L7
        for depth in range(_DEPTH_TOTAL + 1):
            cur.execute(
                """
                INSERT INTO uat_concepts (concept_id, preferred_label, level)
                VALUES (%s, %s, %s)
                """,
                (_concept_id(depth), f"fixture-L{depth}", depth),
            )
        # Insert leaf concept
        cur.execute(
            """
            INSERT INTO uat_concepts (concept_id, preferred_label, level)
            VALUES (%s, %s, %s)
            """,
            (_LEAF_CONCEPT, "fixture-leaf", 0),
        )

        # Insert parent->child relationships along the chain
        for depth in range(_DEPTH_TOTAL):
            cur.execute(
                "INSERT INTO uat_relationships (parent_id, child_id) VALUES (%s, %s)",
                (_concept_id(depth), _concept_id(depth + 1)),
            )

        # Insert one paper per depth and tag it with that concept
        for depth in range(_DEPTH_TOTAL + 1):
            cur.execute(
                """
                INSERT INTO papers (bibcode, title, first_author, year, citation_count)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (_bibcode(depth), f"Fixture paper L{depth}", "Fixture, A.", 2099, depth),
            )
            cur.execute(
                """
                INSERT INTO paper_uat_mappings (bibcode, concept_id, match_type)
                VALUES (%s, %s, 'exact')
                """,
                (_bibcode(depth), _concept_id(depth)),
            )

        # Insert leaf paper + mapping
        cur.execute(
            """
            INSERT INTO papers (bibcode, title, first_author, year, citation_count)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (_LEAF_BIBCODE, "Fixture leaf paper", "Fixture, L.", 2099, 42),
        )
        cur.execute(
            """
            INSERT INTO paper_uat_mappings (bibcode, concept_id, match_type)
            VALUES (%s, %s, 'exact')
            """,
            (_LEAF_BIBCODE, _LEAF_CONCEPT),
        )
        conn.commit()

    try:
        yield
    finally:
        # Teardown: explicit deletes (reverse FK order) to leave the DB clean.
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM paper_uat_mappings WHERE bibcode LIKE %s OR bibcode = %s",
                (f"{_BIBCODE_PREFIX}%", _LEAF_BIBCODE),
            )
            cur.execute(
                "DELETE FROM uat_relationships WHERE parent_id LIKE %s OR child_id LIKE %s",
                (f"{_CONCEPT_PREFIX}%", f"{_CONCEPT_PREFIX}%"),
            )
            cur.execute(
                "DELETE FROM uat_concepts WHERE concept_id LIKE %s OR concept_id = %s",
                (f"{_CONCEPT_PREFIX}%", _LEAF_CONCEPT),
            )
            cur.execute(
                "DELETE FROM papers WHERE bibcode LIKE %s OR bibcode = %s",
                (f"{_BIBCODE_PREFIX}%", _LEAF_BIBCODE),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _bibcodes(result) -> set[str]:
    return {p["bibcode"] for p in result.papers}


def test_recursive_cte_terminates_at_depth_cap(
    conn: psycopg.Connection, hierarchy: None
) -> None:
    """include_descendants=True from the root must include papers at depths
    1..CONCEPT_DESCENDANT_MAX_DEPTH plus the root (depth 0), but NOT the
    deeper L(cap+1) paper — proving the recursion stopped at the cap.
    """
    result = concept_search(
        conn,
        query=_concept_id(0),
        include_descendants=True,
        limit=100,
    )
    found = _bibcodes(result)

    # Depths 0..cap must be present.
    for depth in range(CONCEPT_DESCENDANT_MAX_DEPTH + 1):
        assert _bibcode(depth) in found, (
            f"expected depth-{depth} paper in expansion (cap={CONCEPT_DESCENDANT_MAX_DEPTH})"
        )

    # Anything past the cap must be excluded.
    for depth in range(CONCEPT_DESCENDANT_MAX_DEPTH + 1, _DEPTH_TOTAL + 1):
        assert _bibcode(depth) not in found, (
            f"depth-{depth} paper leaked past cap={CONCEPT_DESCENDANT_MAX_DEPTH}"
        )


def test_descendants_true_is_strict_superset_of_false(
    conn: psycopg.Connection, hierarchy: None
) -> None:
    """For a concept with descendants, expansion must be a strict superset."""
    expanded = _bibcodes(
        concept_search(conn, query=_concept_id(0), include_descendants=True, limit=100)
    )
    exact_only = _bibcodes(
        concept_search(conn, query=_concept_id(0), include_descendants=False, limit=100)
    )

    # Exact-only returns just the root-tagged paper.
    assert exact_only == {_bibcode(0)}
    # Expanded must include that, plus at least one descendant -> strict superset.
    assert exact_only < expanded, "descendant expansion is not a strict superset"


def test_leaf_concept_same_either_way(conn: psycopg.Connection, hierarchy: None) -> None:
    """A leaf concept has no descendants — both paths must return the same set."""
    with_desc = _bibcodes(
        concept_search(conn, query=_LEAF_CONCEPT, include_descendants=True, limit=100)
    )
    without_desc = _bibcodes(
        concept_search(conn, query=_LEAF_CONCEPT, include_descendants=False, limit=100)
    )
    assert with_desc == without_desc == {_LEAF_BIBCODE}
