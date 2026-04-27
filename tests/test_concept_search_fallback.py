"""Integration tests for concept_search free-text fallback (scix_experiments-2ixv).

Surfaces 2026-04-27: ``concept_search('granular mechanics in microgravity')``
returns 0 papers because the natural-language phrase doesn't map to any
controlled-vocabulary concept. The tool name implies broader concept retrieval,
so an agent reaches for it and hits an empty response.

Fix: when the vocabulary router yields zero hits, fall through to
``hybrid_search(query)`` and return those papers tagged with
``metadata.fallback='hybrid_search'`` so the agent can see what happened.

Coverage:

* Free-text query that resolves to no vocabulary concept yields >0 papers via
  the hybrid_search fallback (the bead's acceptance criterion #1).
* The vocabulary path still wins when a concept resolves (back-compat).
* Fallback can be disabled via ``fallback=False`` for callers that want
  legacy strict behavior.
* Garbage/unresolvable query still returns an empty list (not an error)
  when ``fallback=False``.

Writes only to ``SCIX_TEST_DSN``; skips when it is unset or points at
production.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from helpers import get_test_dsn

from scix.search import concept_search

# A query the bead specifically calls out — natural-language, multi-word, and
# does not appear as a preferred/alt label in any seeded controlled vocabulary
# in the test database. Used to exercise the fallback branch.
_BEAD_FREETEXT_QUERY = "granular mechanics in microgravity"

# Synthetic seed bibcodes for the fallback corpus. Lexical tokens overlap
# with the bead query so plainto_tsquery picks them up. Distinct from the
# router-test bibcodes used in test_concept_search_router.py.
_FALLBACK_BIBCODES = (
    "9999FALL.test..001",
    "9999FALL.test..002",
    "9999FALL.test..003",
)

_FALLBACK_VOCAB_PREFIX = "DBL7-FALLBACK-TEST-"
_FALLBACK_UAT_PREFIX = "http://example.test/uat-fallback-fixture/"
_FALLBACK_UAT_BIBCODE = "9999FALL.test..uat"


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
def freetext_corpus(conn: psycopg.Connection) -> Iterator[None]:
    """Seed papers whose tsvector matches the bead's free-text query.

    The seeded titles/abstracts contain lexical tokens overlapping the
    canonical bead query so PostgreSQL ``plainto_tsquery`` picks them up
    via the production tsvector path. We deliberately do NOT seed any
    matching vocabulary concept — the whole point of the fallback test
    is that the vocabulary lookup returns 0 hits.
    """
    rows = (
        (
            _FALLBACK_BIBCODES[0],
            "Granular mechanics in microgravity environments",
            "Smith, A.",
            2024,
            42,
            "We study granular materials behavior in microgravity, focusing on "
            "mechanics of grain flow under reduced gravity.",
        ),
        (
            _FALLBACK_BIBCODES[1],
            "Microgravity granular flow experiments on the ISS",
            "Jones, B.",
            2023,
            17,
            "Experiments on the International Space Station investigated "
            "granular mechanics in microgravity, complementing terrestrial work.",
        ),
        (
            _FALLBACK_BIBCODES[2],
            "Mechanics of granular media: parabolic flight observations",
            "Kim, C.",
            2025,
            8,
            "Parabolic flight microgravity tests of granular mechanics under "
            "reduced effective gravity provide insight into bulk grain behavior.",
        ),
    )
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM papers WHERE bibcode = ANY(%s)",
            (list(_FALLBACK_BIBCODES),),
        )
        for bib, title, author, year, cites, abstract in rows:
            cur.execute(
                """
                INSERT INTO papers (
                    bibcode, title, first_author, year, citation_count, abstract
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (bib, title, author, year, cites, abstract),
            )
        conn.commit()
    try:
        yield
    finally:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM papers WHERE bibcode = ANY(%s)",
                (list(_FALLBACK_BIBCODES),),
            )
        conn.commit()


@pytest.fixture
def vocab_hit_seed(conn: psycopg.Connection) -> Iterator[None]:
    """Seed a UAT concept + paper so the vocab path resolves and fallback skips.

    Used to verify the fallback is NOT triggered when vocabulary lookup
    succeeds — preserving back-compat with the existing router tests.
    """
    label = "fallback-test-galaxy-cluster"
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM paper_uat_mappings WHERE bibcode = %s",
            (_FALLBACK_UAT_BIBCODE,),
        )
        cur.execute(
            "DELETE FROM uat_concepts WHERE concept_id LIKE %s",
            (f"{_FALLBACK_UAT_PREFIX}%",),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode = %s",
            (_FALLBACK_UAT_BIBCODE,),
        )
        cur.execute(
            """
            INSERT INTO uat_concepts (concept_id, preferred_label, level)
            VALUES (%s, %s, %s)
            """,
            (f"{_FALLBACK_UAT_PREFIX}cluster", label, 0),
        )
        cur.execute(
            """
            INSERT INTO papers (bibcode, title, first_author, year, citation_count)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (_FALLBACK_UAT_BIBCODE, "UAT fallback fixture", "Fix, U.", 2099, 3),
        )
        cur.execute(
            """
            INSERT INTO paper_uat_mappings (bibcode, concept_id, match_type)
            VALUES (%s, %s, 'exact')
            """,
            (_FALLBACK_UAT_BIBCODE, f"{_FALLBACK_UAT_PREFIX}cluster"),
        )
        conn.commit()
    try:
        yield label
    finally:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM paper_uat_mappings WHERE bibcode = %s",
                (_FALLBACK_UAT_BIBCODE,),
            )
            cur.execute(
                "DELETE FROM uat_concepts WHERE concept_id LIKE %s",
                (f"{_FALLBACK_UAT_PREFIX}%",),
            )
            cur.execute(
                "DELETE FROM papers WHERE bibcode = %s",
                (_FALLBACK_UAT_BIBCODE,),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_freetext_query_falls_back_to_lexical(
    conn: psycopg.Connection, freetext_corpus: None
) -> None:
    """Bead acceptance #1: the canonical free-text query yields >0 papers
    via the lexical fallback when no vocabulary concept resolves."""
    result = concept_search(conn, _BEAD_FREETEXT_QUERY, limit=20)

    # Vocabulary lookup must have returned no hits — this is the bug shape.
    assert result.metadata["concept_found"] is False
    assert result.metadata["concepts"] == []

    # But the fallback must have run and returned papers.
    assert result.metadata["fallback"] == "lexical_search"
    assert result.total > 0
    assert len(result.papers) > 0

    # The seeded fixtures should appear in the fallback's lexical results.
    returned = {p["bibcode"] for p in result.papers}
    assert returned & set(_FALLBACK_BIBCODES), (
        f"Expected at least one seeded bibcode in fallback results; got {returned}"
    )


def test_vocab_hit_does_not_trigger_fallback(
    conn: psycopg.Connection, vocab_hit_seed: str
) -> None:
    """When the vocabulary path resolves a concept, the fallback must not
    run and the result preserves legacy semantics."""
    label = vocab_hit_seed
    result = concept_search(conn, label, limit=10)

    assert result.metadata["concept_found"] is True
    assert result.metadata["concept_vocabulary"] == "uat"
    # Fallback flag is absent (or explicitly None) when vocab path wins.
    assert result.metadata.get("fallback") is None
    # UAT-driven papers were returned (back-compat).
    assert _FALLBACK_UAT_BIBCODE in {p["bibcode"] for p in result.papers}


def test_fallback_disabled_returns_empty(
    conn: psycopg.Connection, freetext_corpus: None
) -> None:
    """``fallback=False`` preserves legacy behavior: 0 vocab hits → empty."""
    result = concept_search(conn, _BEAD_FREETEXT_QUERY, limit=20, fallback=False)

    assert result.metadata["concept_found"] is False
    assert result.metadata["concepts"] == []
    assert result.metadata.get("fallback") is None
    assert result.papers == []
    assert result.total == 0


def test_unresolvable_garbage_query_returns_empty(conn: psycopg.Connection) -> None:
    """A nonsense query that matches neither vocabulary nor lexical corpus
    returns an empty list (not an error). The fallback runs but legitimately
    finds nothing."""
    result = concept_search(
        conn,
        "qzzz-mxxx-no-such-token-yyyy-9876543210-aaaa-bbbb",
        limit=5,
    )
    assert result.metadata["concept_found"] is False
    assert result.metadata["concepts"] == []
    # Fallback ran but found nothing — papers list is empty, not an error.
    assert result.papers == []
    assert result.total == 0
    # Fallback metadata still present so callers can see we tried.
    assert result.metadata.get("fallback") == "lexical_search"


def test_empty_query_returns_empty_no_fallback(conn: psycopg.Connection) -> None:
    """Empty query short-circuits — no vocabulary lookup, no fallback."""
    result = concept_search(conn, "", limit=5)
    assert result.papers == []
    assert result.total == 0
    assert result.metadata["concept_found"] is False
    # We must not run a lexical fallback on the empty string.
    assert result.metadata.get("fallback") is None


def test_vocabulary_restricted_still_falls_back(
    conn: psycopg.Connection, freetext_corpus: None
) -> None:
    """Even with vocabulary restricted, a 0-hit query falls back to lexical."""
    result = concept_search(
        conn,
        _BEAD_FREETEXT_QUERY,
        vocabulary=["mesh"],  # biomedical vocab, will not match the query
        limit=20,
    )
    assert result.metadata["concept_found"] is False
    assert result.metadata["fallback"] == "lexical_search"
    assert result.total > 0
