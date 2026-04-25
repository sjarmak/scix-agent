"""Integration tests for the multi-vocabulary concept_search router (dbl.7).

Covers:
    - vocabulary parameter normalization (None / str / list / unknown)
    - Cross-vocab dispatch: a query that hits ACM CCS + OpenAlex returns hits
      tagged with both vocabularies in metadata.concepts.
    - Vocabulary restriction: vocabulary=['acm_ccs'] suppresses other-vocab hits.
    - URI lookup against external_uri.
    - Alternate-label hit ranks below exact preferred-label hit.
    - Backwards compatibility: a UAT preferred-label match still returns
      ``papers`` with descendant expansion.

Writes only to ``SCIX_TEST_DSN``; skips when it is unset or points at
production.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from helpers import get_test_dsn

from scix.search import (
    CONCEPT_VOCABULARIES,
    _normalize_vocabulary_arg,
    concept_search,
)

# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

_VOCAB_FIXTURE_PREFIX = "dbl7_test_"  # vocabulary names start with this prefix
_CONCEPT_PREFIX = "DBL7-TEST-"  # concept_id prefix; deterministic cleanup
_UAT_FIXTURE_PREFIX = "http://example.test/uat-router-fixture/"
_UAT_BIBCODE = "9999RTRT.test..001"


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


def _seed_real_vocab_concept(
    cur: psycopg.Cursor,
    vocabulary: str,
    concept_id: str,
    preferred_label: str,
    alternate_labels: list[str] | None = None,
    external_uri: str | None = None,
) -> None:
    """Insert one concept under an EXISTING (real) vocabulary code.

    The router only allows queries against vocabularies in
    :data:`CONCEPT_VOCABULARIES`, so we cannot use a synthetic vocab name.
    Instead we seed concepts under the real vocab code with a deterministic
    concept_id prefix so teardown can DELETE without touching real data.
    """
    cur.execute(
        """
        INSERT INTO concepts (
            vocabulary, concept_id, preferred_label,
            alternate_labels, external_uri
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (vocabulary, concept_id) DO UPDATE
            SET preferred_label = EXCLUDED.preferred_label,
                alternate_labels = EXCLUDED.alternate_labels,
                external_uri = EXCLUDED.external_uri
        """,
        (
            vocabulary,
            concept_id,
            preferred_label,
            alternate_labels or [],
            external_uri,
        ),
    )


@pytest.fixture
def cross_vocab_seed(conn: psycopg.Connection) -> Iterator[None]:
    """Seed test concepts in real vocabularies, scoped by concept_id prefix.

    All concept_ids start with ``DBL7-TEST-`` so teardown can delete them
    without touching real ingested vocabulary data. Vocabulary rows are
    upserted with placeholder license/source metadata so the test DB does
    not need the dbl.1 ingest to have run; the vocabulary rows are left in
    place at teardown (no cascade) since they're harmless metadata.
    """
    required = ("acm_ccs", "openalex", "msc")
    with conn.cursor() as cur:
        for v in required:
            cur.execute(
                """
                INSERT INTO vocabularies (vocabulary, name, license, source_url)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (vocabulary) DO NOTHING
                """,
                (v, f"test-{v}", "test-license", "https://example.test/"),
            )
        conn.commit()

    with conn.cursor() as cur:
        # Clean any prior leftovers first.
        cur.execute("DELETE FROM concepts WHERE concept_id LIKE %s", (f"{_CONCEPT_PREFIX}%",))
        # ACM CCS: exact preferred label hit.
        _seed_real_vocab_concept(
            cur,
            vocabulary="acm_ccs",
            concept_id=f"{_CONCEPT_PREFIX}ACM-1",
            preferred_label="Transformer attention",
            alternate_labels=["self-attention"],
            external_uri="https://example.test/acm/transformer-attention",
        )
        # OpenAlex: same query as alt-label only -> lower score than ACM exact.
        _seed_real_vocab_concept(
            cur,
            vocabulary="openalex",
            concept_id=f"{_CONCEPT_PREFIX}OA-1",
            preferred_label="Neural sequence models",
            alternate_labels=["transformer attention", "BERT"],
        )
        # MSC: unrelated; should not hit on 'transformer attention'.
        _seed_real_vocab_concept(
            cur,
            vocabulary="msc",
            concept_id=f"{_CONCEPT_PREFIX}MSC-1",
            preferred_label="Galois theory",
        )
        conn.commit()

    try:
        yield
    finally:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM concepts WHERE concept_id LIKE %s",
                (f"{_CONCEPT_PREFIX}%",),
            )
        conn.commit()


@pytest.fixture
def uat_seed(conn: psycopg.Connection) -> Iterator[None]:
    """Seed one UAT concept + paper for the legacy backward-compat path."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM paper_uat_mappings WHERE bibcode = %s",
            (_UAT_BIBCODE,),
        )
        cur.execute(
            "DELETE FROM uat_concepts WHERE concept_id LIKE %s",
            (f"{_UAT_FIXTURE_PREFIX}%",),
        )
        cur.execute(
            "DELETE FROM papers WHERE bibcode = %s",
            (_UAT_BIBCODE,),
        )
        cur.execute(
            """
            INSERT INTO uat_concepts (concept_id, preferred_label, level)
            VALUES (%s, %s, %s)
            """,
            (f"{_UAT_FIXTURE_PREFIX}router-galaxies", "router-test-galaxies", 0),
        )
        cur.execute(
            """
            INSERT INTO papers (bibcode, title, first_author, year, citation_count)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (_UAT_BIBCODE, "UAT router fixture", "Fixture, R.", 2099, 7),
        )
        cur.execute(
            """
            INSERT INTO paper_uat_mappings (bibcode, concept_id, match_type)
            VALUES (%s, %s, 'exact')
            """,
            (_UAT_BIBCODE, f"{_UAT_FIXTURE_PREFIX}router-galaxies"),
        )
        conn.commit()

    try:
        yield
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM paper_uat_mappings WHERE bibcode = %s", (_UAT_BIBCODE,))
            cur.execute(
                "DELETE FROM uat_concepts WHERE concept_id LIKE %s",
                (f"{_UAT_FIXTURE_PREFIX}%",),
            )
            cur.execute("DELETE FROM papers WHERE bibcode = %s", (_UAT_BIBCODE,))
        conn.commit()


# ---------------------------------------------------------------------------
# Pure unit tests
# ---------------------------------------------------------------------------


class TestNormalizeVocabularyArg:
    def test_none_returns_all(self) -> None:
        assert _normalize_vocabulary_arg(None) == CONCEPT_VOCABULARIES

    def test_empty_list_returns_all(self) -> None:
        assert _normalize_vocabulary_arg([]) == CONCEPT_VOCABULARIES

    def test_single_string(self) -> None:
        assert _normalize_vocabulary_arg("uat") == ("uat",)

    def test_list_preserves_order(self) -> None:
        assert _normalize_vocabulary_arg(["openalex", "uat"]) == ("openalex", "uat")

    def test_dedupes(self) -> None:
        assert _normalize_vocabulary_arg(["uat", "uat", "openalex"]) == (
            "uat",
            "openalex",
        )

    def test_unknown_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown vocabulary"):
            _normalize_vocabulary_arg("not_a_real_vocab")


# ---------------------------------------------------------------------------
# Integration tests: cross-vocab routing
# ---------------------------------------------------------------------------


def test_cross_vocab_returns_tagged_hits(conn: psycopg.Connection, cross_vocab_seed: None) -> None:
    """A query hitting two vocabularies returns both, tagged + ranked."""
    result = concept_search(conn, "transformer attention", limit=20)

    # Restrict to seeded fixtures only — real ingested data may also match.
    fixture_hits = [
        h for h in result.metadata["concepts"] if h["concept_id"].startswith(_CONCEPT_PREFIX)
    ]
    vocabs_seen = {h["vocabulary"] for h in fixture_hits}
    assert "acm_ccs" in vocabs_seen, "ACM CCS preferred-label hit missing"
    assert "openalex" in vocabs_seen, "OpenAlex alt-label hit missing"
    assert "msc" not in vocabs_seen, "MSC must not match unrelated query"

    # ACM exact-label hit (1.0) must rank above OpenAlex alt-label hit (0.7).
    acm = next(h for h in fixture_hits if h["vocabulary"] == "acm_ccs")
    oa = next(h for h in fixture_hits if h["vocabulary"] == "openalex")
    assert acm["score"] > oa["score"]
    assert acm["score"] == 1.0
    assert oa["score"] == 0.7


def test_vocabulary_filter_restricts_search(
    conn: psycopg.Connection, cross_vocab_seed: None
) -> None:
    """vocabulary=['acm_ccs'] must exclude OpenAlex hits."""
    result = concept_search(conn, "transformer attention", vocabulary=["acm_ccs"], limit=20)
    fixture_hits = [
        h for h in result.metadata["concepts"] if h["concept_id"].startswith(_CONCEPT_PREFIX)
    ]
    assert {h["vocabulary"] for h in fixture_hits} == {"acm_ccs"}
    assert result.metadata["vocabularies_searched"] == ["acm_ccs"]


def test_uri_lookup_via_external_uri(conn: psycopg.Connection, cross_vocab_seed: None) -> None:
    """Querying by external_uri must resolve the concept."""
    result = concept_search(
        conn,
        "https://example.test/acm/transformer-attention",
        vocabulary=["acm_ccs"],
        limit=5,
    )
    assert result.metadata["concept_found"]
    assert result.metadata["concept_id"] == f"{_CONCEPT_PREFIX}ACM-1"
    assert result.metadata["concept_vocabulary"] == "acm_ccs"


def test_alt_label_only_match(conn: psycopg.Connection, cross_vocab_seed: None) -> None:
    """A query that hits only an alternate label still returns the concept."""
    result = concept_search(conn, "BERT", vocabulary=["openalex"], limit=5)
    fixture_hits = [
        h for h in result.metadata["concepts"] if h["concept_id"].startswith(_CONCEPT_PREFIX)
    ]
    assert any(h["concept_id"] == f"{_CONCEPT_PREFIX}OA-1" for h in fixture_hits)


def test_no_hits_returns_empty_with_metadata(conn: psycopg.Connection) -> None:
    """Unknown query returns empty result with concept_found=False."""
    result = concept_search(
        conn,
        "this-string-cannot-possibly-match-anything-zzz-9876",
        limit=5,
    )
    assert result.papers == []
    assert result.metadata["concept_found"] is False
    assert result.metadata["concepts"] == []
    assert "vocabularies_searched" in result.metadata


# ---------------------------------------------------------------------------
# Integration tests: UAT backwards compatibility
# ---------------------------------------------------------------------------


def test_uat_match_returns_papers(conn: psycopg.Connection, uat_seed: None) -> None:
    """A UAT-only label hit must still return ``papers`` (legacy behavior)."""
    result = concept_search(conn, "router-test-galaxies", limit=10)
    assert result.metadata["concept_found"]
    assert result.metadata["concept_vocabulary"] == "uat"
    assert {p["bibcode"] for p in result.papers} == {_UAT_BIBCODE}
    # Legacy metadata keys preserved.
    assert "concept_id" in result.metadata
    assert "concept_label" in result.metadata
    assert result.metadata["include_descendants"] is True


def test_uat_wins_score_tie_against_other_vocab(conn: psycopg.Connection, uat_seed: None) -> None:
    """When UAT and another vocab tie at 1.0, UAT must win and return papers.

    Preserves backward compatibility: legacy callers querying labels that
    exist verbatim in both UAT and a newer vocabulary (e.g. "Galaxies"
    -> UAT + PhySH) must still get UAT-driven paper retrieval since UAT
    is the only vocab with paper mappings today.
    """
    with conn.cursor() as cur:
        # Seed a PhySH concept with the same preferred label as the UAT one,
        # so both should hit at score=1.0 on the same query.
        cur.execute("""
            INSERT INTO vocabularies (vocabulary, name, license, source_url)
            VALUES ('physh', 'test-physh', 'test-license', 'https://example.test/')
            ON CONFLICT (vocabulary) DO NOTHING
            """)
        cur.execute(
            """
            INSERT INTO concepts (vocabulary, concept_id, preferred_label)
            VALUES ('physh', %s, 'router-test-galaxies')
            ON CONFLICT (vocabulary, concept_id) DO UPDATE
                SET preferred_label = EXCLUDED.preferred_label
            """,
            (f"{_CONCEPT_PREFIX}PHYSH-tie",),
        )
        conn.commit()
    try:
        result = concept_search(conn, "router-test-galaxies", limit=10)
        assert result.metadata["concept_vocabulary"] == "uat"
        assert {p["bibcode"] for p in result.papers} == {_UAT_BIBCODE}
        # Both vocabs should appear in the candidates list.
        vocabs_in_hits = {h["vocabulary"] for h in result.metadata["concepts"]}
        assert {"uat", "physh"}.issubset(vocabs_in_hits)
    finally:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM concepts WHERE concept_id = %s",
                (f"{_CONCEPT_PREFIX}PHYSH-tie",),
            )
        conn.commit()


def test_uat_vocabulary_restriction(conn: psycopg.Connection, uat_seed: None) -> None:
    """vocabulary='uat' restricts search to UAT only."""
    result = concept_search(conn, "router-test-galaxies", vocabulary="uat", limit=10)
    assert result.metadata["concept_found"]
    assert result.metadata["vocabularies_searched"] == ["uat"]
    assert all(h["vocabulary"] == "uat" for h in result.metadata["concepts"])


def test_non_uat_vocab_returns_empty_papers(
    conn: psycopg.Connection, cross_vocab_seed: None
) -> None:
    """Best hit in a non-UAT vocab returns empty papers (no mappings yet)."""
    result = concept_search(conn, "transformer attention", vocabulary=["acm_ccs"], limit=5)
    assert result.metadata["concept_found"]
    assert result.metadata["concept_vocabulary"] == "acm_ccs"
    assert result.papers == []
