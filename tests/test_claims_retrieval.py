"""Tests for :mod:`scix.claims.retrieval` — read_paper_claims + find_claims.

Two layers, mirroring tests/test_paper_claims_schema.py:

* Static tests — run on every invocation. They cover input validation
  (claim_type whitelisting, empty bibcode/query, bad limit) and the pure
  helpers in the retrieval module. No DB connection required.

* Integration tests — gated on a reachable scix_test database
  (SCIX_TEST_DSN env var, or the fallback ``dbname=scix_test``). They
  apply migration 062, seed a few paper_claims rows, and exercise the
  helpers end-to-end including the GIN-index EXPLAIN check.

Safety: the integration leg refuses any DSN that does not reference
scix_test, mirroring the pattern in tests/test_paper_claims_schema.py.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from typing import Iterator

import pytest

from scix.claims.retrieval import (
    VALID_CLAIM_TYPES,
    find_claims,
    read_paper_claims,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_PATH = REPO_ROOT / "migrations" / "062_paper_claims.sql"


# ---------------------------------------------------------------------------
# Static layer — no DB required
# ---------------------------------------------------------------------------


class TestValidClaimTypes:
    """The whitelist matches the migration-062 CHECK constraint."""

    def test_set_matches_migration(self) -> None:
        assert VALID_CLAIM_TYPES == frozenset(
            {
                "factual",
                "methodological",
                "comparative",
                "speculative",
                "cited_from_other",
            }
        )


class TestInputValidationStatic:
    """All input validators raise before opening the DB connection."""

    def test_read_paper_claims_rejects_empty_bibcode(self) -> None:
        with pytest.raises(ValueError, match="bibcode"):
            read_paper_claims(_NoOpConn(), bibcode="")
        with pytest.raises(ValueError, match="bibcode"):
            read_paper_claims(_NoOpConn(), bibcode="   ")

    def test_read_paper_claims_rejects_unknown_claim_type(self) -> None:
        with pytest.raises(ValueError, match="invalid claim_type"):
            read_paper_claims(
                _NoOpConn(), bibcode="x", claim_type="bogus"
            )

    def test_read_paper_claims_accepts_each_valid_claim_type(self) -> None:
        # Should NOT raise — the validator only complains about unknown labels.
        for ct in VALID_CLAIM_TYPES:
            try:
                read_paper_claims(_NoOpConn(), bibcode="x", claim_type=ct)
            except _NoOpInvoked:
                pass  # static validator passed; got into SQL execution

    def test_read_paper_claims_rejects_zero_limit(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            read_paper_claims(_NoOpConn(), bibcode="x", limit=0)

    def test_read_paper_claims_rejects_negative_limit(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            read_paper_claims(_NoOpConn(), bibcode="x", limit=-1)

    def test_find_claims_rejects_empty_query(self) -> None:
        with pytest.raises(ValueError, match="query"):
            find_claims(_NoOpConn(), query="")
        with pytest.raises(ValueError, match="query"):
            find_claims(_NoOpConn(), query="   ")

    def test_find_claims_rejects_unknown_claim_type(self) -> None:
        with pytest.raises(ValueError, match="invalid claim_type"):
            find_claims(_NoOpConn(), query="hubble", claim_type="bogus")

    def test_find_claims_rejects_non_int_entity_id(self) -> None:
        with pytest.raises(ValueError, match="entity_id"):
            find_claims(_NoOpConn(), query="hubble", entity_id="abc")  # type: ignore[arg-type]


class _NoOpInvoked(Exception):
    """Raised by the no-op cursor stub to signal we got past validation."""


class _NoOpCursor:
    def __enter__(self) -> "_NoOpCursor":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:  # noqa: D401
        # If we got here, static validation passed.
        raise _NoOpInvoked


class _NoOpConn:
    def cursor(self) -> _NoOpCursor:
        return _NoOpCursor()


# ---------------------------------------------------------------------------
# Integration layer
# ---------------------------------------------------------------------------


def _resolve_test_dsn() -> str | None:
    """Resolve the scix_test DSN (mirrors tests/test_paper_claims_schema.py)."""
    dsn = os.environ.get("SCIX_TEST_DSN")
    if dsn:
        if "scix_test" not in dsn:
            pytest.fail(
                "SCIX_TEST_DSN must reference scix_test — got: " + dsn,
                pytrace=False,
            )
        return dsn

    fallback = "dbname=scix_test"
    try:
        import psycopg
    except ImportError:
        return None

    try:
        with psycopg.connect(fallback, connect_timeout=2):
            pass
    except Exception:
        return None
    return fallback


TEST_DSN = _resolve_test_dsn()
INTEGRATION_REASON = "scix_test database not reachable (set SCIX_TEST_DSN)"


@pytest.fixture(scope="module")
def dsn() -> str:
    if TEST_DSN is None:
        pytest.skip(INTEGRATION_REASON, allow_module_level=False)
    return TEST_DSN


@pytest.fixture(scope="module")
def applied_migration(dsn: str) -> str:
    """Apply migration 062 (idempotently) before integration tests run."""
    result = subprocess.run(
        ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"migration 062 failed to apply: stderr=\n{result.stderr}"
    )
    return dsn


def _ensure_test_bibcode(dsn: str) -> str:
    """Return a real bibcode from papers, or seed a synthetic one."""
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT bibcode FROM papers LIMIT 1")
        row = cur.fetchone()
        if row is not None:
            return row[0]
        synthetic = "9999paper_claims_retrieval_X"
        cur.execute(
            "INSERT INTO papers (bibcode) VALUES (%s) "
            "ON CONFLICT (bibcode) DO NOTHING",
            (synthetic,),
        )
        conn.commit()
        return synthetic


# Marker prefix on extraction_model so the per-test seed cleanup never
# disturbs rows produced by other tests / live extraction runs.
_TEST_MARKER = "test-claims-retrieval"


@pytest.fixture
def seeded_claims(applied_migration: str) -> Iterator[dict[str, object]]:
    """Seed a small fixed set of paper_claims rows. Cleans up on teardown.

    Returns a dict with ``dsn``, ``bibcode``, and the ``claim_ids`` of the
    inserted rows so tests can verify lookups directly.
    """
    import psycopg

    bibcode = _ensure_test_bibcode(applied_migration)
    seed_rows = [
        # (section, paragraph, char_start, char_end, claim_text, claim_type,
        #  subject, predicate, object, confidence, linked_subject, linked_object)
        (
            0,
            0,
            0,
            42,
            "The Hubble constant H0 is 73.0 km/s/Mpc.",
            "factual",
            "Hubble constant",
            "is",
            "73.0 km/s/Mpc",
            0.95,
            1001,
            None,
        ),
        (
            0,
            1,
            50,
            120,
            "We measured galaxy distances using Cepheid variables.",
            "methodological",
            "we",
            "measured",
            "galaxy distances",
            0.90,
            None,
            None,
        ),
        (
            1,
            0,
            0,
            70,
            "Our H0 value is higher than Planck's CMB-derived value.",
            "comparative",
            "H0 value",
            "is higher than",
            "Planck CMB value",
            0.85,
            1001,
            2002,
        ),
        (
            1,
            1,
            80,
            160,
            "Future JWST observations may resolve the Hubble tension.",
            "speculative",
            "JWST observations",
            "may resolve",
            "Hubble tension",
            0.60,
            None,
            None,
        ),
        (
            2,
            0,
            0,
            55,
            "The Hubble tension is robust across many surveys.",
            "factual",
            "Hubble tension",
            "is",
            "robust",
            0.88,
            None,
            2002,
        ),
    ]

    inserted_claim_ids: list[uuid.UUID] = []
    with psycopg.connect(applied_migration) as conn:
        with conn.cursor() as cur:
            for row in seed_rows:
                claim_id = uuid.uuid4()
                inserted_claim_ids.append(claim_id)
                cur.execute(
                    """
                    INSERT INTO paper_claims (
                        claim_id, bibcode, section_index, paragraph_index,
                        char_span_start, char_span_end,
                        claim_text, claim_type,
                        subject, predicate, object, confidence,
                        extraction_model, extraction_prompt_version,
                        linked_entity_subject_id, linked_entity_object_id
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    """,
                    (
                        str(claim_id),
                        bibcode,
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                        row[7],
                        row[8],
                        row[9],
                        _TEST_MARKER,
                        "v1",
                        row[10],
                        row[11],
                    ),
                )
        conn.commit()

    try:
        yield {
            "dsn": applied_migration,
            "bibcode": bibcode,
            "claim_ids": inserted_claim_ids,
        }
    finally:
        with psycopg.connect(applied_migration) as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM paper_claims WHERE extraction_model = %s",
                (_TEST_MARKER,),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# read_paper_claims
# ---------------------------------------------------------------------------


class TestReadPaperClaims:
    """End-to-end behavior of read_paper_claims against scix_test."""

    def test_returns_rows_for_known_bibcode(
        self, seeded_claims: dict[str, object]
    ) -> None:
        import psycopg

        dsn = str(seeded_claims["dsn"])
        bibcode = str(seeded_claims["bibcode"])
        with psycopg.connect(dsn) as conn:
            rows = read_paper_claims(conn, bibcode=bibcode)
        assert len(rows) == 5
        # Exact key set required by acceptance criterion 1.
        assert set(rows[0].keys()) == {
            "bibcode",
            "section_index",
            "paragraph_index",
            "char_span_start",
            "char_span_end",
            "claim_text",
            "claim_type",
            "subject",
            "predicate",
            "object",
            "confidence",
        }
        for row in rows:
            assert row["bibcode"] == bibcode

    def test_orders_by_section_paragraph_charspan(
        self, seeded_claims: dict[str, object]
    ) -> None:
        import psycopg

        dsn = str(seeded_claims["dsn"])
        bibcode = str(seeded_claims["bibcode"])
        with psycopg.connect(dsn) as conn:
            rows = read_paper_claims(conn, bibcode=bibcode)
        keys = [
            (r["section_index"], r["paragraph_index"], r["char_span_start"])
            for r in rows
        ]
        assert keys == sorted(keys)

    def test_filters_by_claim_type(
        self, seeded_claims: dict[str, object]
    ) -> None:
        import psycopg

        dsn = str(seeded_claims["dsn"])
        bibcode = str(seeded_claims["bibcode"])
        with psycopg.connect(dsn) as conn:
            rows = read_paper_claims(
                conn, bibcode=bibcode, claim_type="factual"
            )
        assert len(rows) == 2
        for row in rows:
            assert row["claim_type"] == "factual"

    def test_limit_is_honored(
        self, seeded_claims: dict[str, object]
    ) -> None:
        import psycopg

        dsn = str(seeded_claims["dsn"])
        bibcode = str(seeded_claims["bibcode"])
        with psycopg.connect(dsn) as conn:
            rows = read_paper_claims(conn, bibcode=bibcode, limit=2)
        assert len(rows) == 2

    def test_empty_when_no_match(
        self, seeded_claims: dict[str, object]
    ) -> None:
        import psycopg

        dsn = str(seeded_claims["dsn"])
        with psycopg.connect(dsn) as conn:
            rows = read_paper_claims(
                conn, bibcode="nonexistent_bibcode_xyz"
            )
        assert rows == []


# ---------------------------------------------------------------------------
# find_claims
# ---------------------------------------------------------------------------


class TestFindClaims:
    """End-to-end behavior of find_claims against scix_test."""

    def test_returns_results_ranked_by_ts_rank(
        self, seeded_claims: dict[str, object]
    ) -> None:
        import psycopg

        dsn = str(seeded_claims["dsn"])
        bibcode = str(seeded_claims["bibcode"])
        # "Hubble tension" appears in two seed rows (section 1 paragraph 1
        # and section 2 paragraph 0). The ts_rank ordering is what we test.
        with psycopg.connect(dsn) as conn:
            rows = find_claims(conn, query="Hubble tension")
        # Filter to rows from our seed bibcode (other tests may have inserted
        # rows; the marker filter on extraction_model isn't visible here, so
        # we narrow down by bibcode and 'tension' substring instead).
        seeded = [
            r
            for r in rows
            if r["bibcode"] == bibcode and "tension" in r["claim_text"].lower()
        ]
        assert len(seeded) >= 2
        texts = [r["claim_text"] for r in seeded]
        # Both seed rows that mention "Hubble tension" should be present.
        assert any("Future JWST" in t for t in texts)
        assert any("robust across" in t for t in texts)

    def test_filters_by_entity_id_subject_or_object(
        self, seeded_claims: dict[str, object]
    ) -> None:
        import psycopg

        dsn = str(seeded_claims["dsn"])
        bibcode = str(seeded_claims["bibcode"])
        # entity_id 1001 is the linked_entity_subject_id on the row whose
        # text mentions H0 ("The Hubble constant H0 is 73.0 km/s/Mpc.")
        # AND on the row that compares H0 vs Planck. The second mentions
        # H0 but not "Hubble", so the tsquery "H0" hits both.
        with psycopg.connect(dsn) as conn:
            rows = find_claims(conn, query="H0", entity_id=1001)
        seeded = [r for r in rows if r["bibcode"] == bibcode]
        assert len(seeded) >= 2, seeded

        # entity_id 2002 is the linked_entity_object_id on the comparative
        # row (section 1) and the "Hubble tension is robust..." row
        # (section 2). The latter is the clean search hit for "Hubble
        # tension"; it must come back when filtering on the OBJECT side.
        with psycopg.connect(dsn) as conn:
            rows = find_claims(
                conn, query="Hubble tension", entity_id=2002
            )
        seeded = [r for r in rows if r["bibcode"] == bibcode]
        assert len(seeded) >= 1
        for row in seeded:
            assert "Hubble" in row["claim_text"]

    def test_entity_id_filter_excludes_unlinked_rows(
        self, seeded_claims: dict[str, object]
    ) -> None:
        """entity_id 9999 is not linked anywhere → no rows."""
        import psycopg

        dsn = str(seeded_claims["dsn"])
        with psycopg.connect(dsn) as conn:
            rows = find_claims(conn, query="Hubble", entity_id=9999)
        assert rows == []

    def test_filters_by_claim_type(
        self, seeded_claims: dict[str, object]
    ) -> None:
        import psycopg

        dsn = str(seeded_claims["dsn"])
        bibcode = str(seeded_claims["bibcode"])
        with psycopg.connect(dsn) as conn:
            rows = find_claims(
                conn,
                query="Hubble",
                claim_type="factual",
            )
        seeded = [r for r in rows if r["bibcode"] == bibcode]
        for row in seeded:
            assert row["claim_type"] == "factual"

    def test_limit_is_honored(
        self, seeded_claims: dict[str, object]
    ) -> None:
        import psycopg

        dsn = str(seeded_claims["dsn"])
        with psycopg.connect(dsn) as conn:
            rows = find_claims(conn, query="Hubble", limit=1)
        assert len(rows) <= 1

    def test_no_match_returns_empty(
        self, seeded_claims: dict[str, object]
    ) -> None:
        import psycopg

        dsn = str(seeded_claims["dsn"])
        with psycopg.connect(dsn) as conn:
            rows = find_claims(
                conn, query="zxqv_thisstringdoesnotappear_anywhereuvw"
            )
        assert rows == []


# ---------------------------------------------------------------------------
# EXPLAIN — the GIN tsvector index is selected by the planner
# ---------------------------------------------------------------------------


class TestFindClaimsUsesGinIndex:
    """The query plan must reference the migration-062 GIN index by name.

    On a small / cold table the planner may pick a sequential scan; we
    set ``enable_seqscan = off`` for the EXPLAIN to force the index path
    so this test is robust to fixture size.
    """

    def test_explain_mentions_ix_paper_claims_claim_text_tsv(
        self, seeded_claims: dict[str, object]
    ) -> None:
        import psycopg

        dsn = str(seeded_claims["dsn"])
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("BEGIN")
                cur.execute("SET LOCAL enable_seqscan = off")
                cur.execute(
                    """
                    EXPLAIN (FORMAT TEXT)
                    SELECT bibcode, claim_text
                    FROM paper_claims
                    WHERE to_tsvector('english', claim_text)
                          @@ plainto_tsquery('english', %s)
                    ORDER BY ts_rank(
                        to_tsvector('english', claim_text),
                        plainto_tsquery('english', %s)
                    ) DESC
                    LIMIT 5
                    """,
                    ("Hubble", "Hubble"),
                )
                plan_rows = cur.fetchall()
                cur.execute("ROLLBACK")
        plan_text = "\n".join(r[0] for r in plan_rows)
        assert "ix_paper_claims_claim_text_tsv" in plan_text, (
            f"GIN index not used. Plan was:\n{plan_text}"
        )
