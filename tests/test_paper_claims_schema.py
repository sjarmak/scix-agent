"""Tests for migration 062_paper_claims.sql — paper_claims schema.

Two layers:

1. Static SQL parse — the migration file exists, is wrapped in BEGIN/COMMIT,
   uses IF NOT EXISTS everywhere, declares the documented columns and the
   CHECK constraint, and ends with the LOGGED-table guard. Always runs.

2. Integration — apply the migration to scix_test, then assert via
   information_schema / pg_indexes that the schema is what we documented;
   verify the CHECK constraint fires for a bogus claim_type; round-trip a
   valid INSERT; and re-apply the migration to confirm idempotency.

The integration leg uses the SCIX_TEST_DSN env var with a fallback to
"dbname=scix_test" (matching the project convention used by the other
migration tests in this repo). It is skipped if no scix_test DB is
reachable, with the static-parse layer providing minimal coverage.

Safety: the integration leg refuses any DSN that doesn't reference
scix_test, mirroring tests/test_halfvec_migration.py.
"""

from __future__ import annotations

import os
import re
import subprocess
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_PATH = REPO_ROOT / "migrations" / "062_paper_claims.sql"


# ---------------------------------------------------------------------------
# Static layer — always runs, no DB required
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sql_content() -> str:
    """Read the migration file once for the whole module."""
    return MIGRATION_PATH.read_text()


class TestMigrationFileStatic:
    """Static checks against the SQL text. No DB connection needed."""

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.is_file(), f"{MIGRATION_PATH} must exist"

    def test_wrapped_in_transaction(self, sql_content: str) -> None:
        assert "BEGIN;" in sql_content
        assert "COMMIT;" in sql_content
        # COMMIT must come after BEGIN
        assert sql_content.rindex("COMMIT;") > sql_content.index("BEGIN;")

    def test_creates_table_if_not_exists(self, sql_content: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS paper_claims" in sql_content

    def test_all_documented_columns_present(self, sql_content: str) -> None:
        """Each documented column appears in the CREATE TABLE block."""
        # Just check the column name + type token shows up in the create-table
        # body. We assert types via information_schema in the integration leg.
        for col_token in [
            "claim_id",
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
            "extraction_model",
            "extraction_prompt_version",
            "extracted_at",
            "linked_entity_subject_id",
            "linked_entity_object_id",
        ]:
            assert col_token in sql_content, f"missing column: {col_token}"

    def test_claim_type_check_constraint(self, sql_content: str) -> None:
        """CHECK constraint enumerates the 5 documented claim types."""
        for label in (
            "factual",
            "methodological",
            "comparative",
            "speculative",
            "cited_from_other",
        ):
            assert f"'{label}'" in sql_content, f"claim_type missing: {label}"
        assert "CHECK" in sql_content
        assert "claim_type" in sql_content

    def test_bibcode_fk(self, sql_content: str) -> None:
        assert "REFERENCES papers(bibcode)" in sql_content

    def test_uuid_default(self, sql_content: str) -> None:
        assert "gen_random_uuid()" in sql_content

    def test_extracted_at_default_now(self, sql_content: str) -> None:
        assert re.search(r"extracted_at\s+timestamptz\s+NOT\s+NULL\s+DEFAULT\s+now\(\)", sql_content)

    def test_indexes_use_if_not_exists_and_naming_convention(self, sql_content: str) -> None:
        """All 5 indexes use IF NOT EXISTS and the ix_paper_claims_* prefix."""
        for idx in (
            "ix_paper_claims_bibcode_section",
            "ix_paper_claims_linked_entity_subject_id",
            "ix_paper_claims_linked_entity_object_id",
            "ix_paper_claims_claim_type",
            "ix_paper_claims_claim_text_tsv",
        ):
            assert f"CREATE INDEX IF NOT EXISTS {idx}" in sql_content, idx

    def test_gin_index_on_tsvector(self, sql_content: str) -> None:
        assert "USING GIN" in sql_content
        assert "to_tsvector('english', claim_text)" in sql_content

    def test_logged_guard_block_present(self, sql_content: str) -> None:
        """Mirrors migration 041's relpersistence='p' assertion."""
        assert "DO $$" in sql_content
        assert "relpersistence" in sql_content
        assert "RAISE EXCEPTION" in sql_content
        assert "must be LOGGED" in sql_content


# ---------------------------------------------------------------------------
# Integration layer — applies migration to scix_test
# ---------------------------------------------------------------------------


def _resolve_test_dsn() -> str | None:
    """Resolve the scix_test DSN.

    Order:
    1. SCIX_TEST_DSN env var.
    2. Fallback "dbname=scix_test" if reachable (mirrors the convention
       used elsewhere in tests/).
    Returns None if no scix_test DB is reachable.
    """
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
        f"migration 062 failed to apply: stderr=\n{result.stderr}\nstdout=\n{result.stdout}"
    )
    return dsn


# Documented (column_name, expected_data_type, is_nullable) per docs/schema/paper_claims.md.
# data_type values are the strings information_schema.columns returns.
EXPECTED_COLUMNS: list[tuple[str, str, str]] = [
    ("claim_id", "uuid", "NO"),
    ("bibcode", "text", "NO"),
    ("section_index", "integer", "NO"),
    ("paragraph_index", "integer", "NO"),
    ("char_span_start", "integer", "NO"),
    ("char_span_end", "integer", "NO"),
    ("claim_text", "text", "NO"),
    ("claim_type", "text", "NO"),
    ("subject", "text", "YES"),
    ("predicate", "text", "YES"),
    ("object", "text", "YES"),
    ("confidence", "real", "YES"),
    ("extraction_model", "text", "NO"),
    ("extraction_prompt_version", "text", "NO"),
    ("extracted_at", "timestamp with time zone", "NO"),
    ("linked_entity_subject_id", "bigint", "YES"),
    ("linked_entity_object_id", "bigint", "YES"),
]

EXPECTED_INDEXES: list[str] = [
    "ix_paper_claims_bibcode_section",
    "ix_paper_claims_linked_entity_subject_id",
    "ix_paper_claims_linked_entity_object_id",
    "ix_paper_claims_claim_type",
    "ix_paper_claims_claim_text_tsv",
]


class TestPaperClaimsColumnsExist:
    """All documented columns present with the documented types/nullability."""

    def test_columns_match_docs(self, applied_migration: str) -> None:
        import psycopg

        with psycopg.connect(applied_migration) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'paper_claims'
                ORDER BY ordinal_position
                """
            )
            rows = cur.fetchall()

        actual = {(name, dtype, nullable) for (name, dtype, nullable) in rows}
        for expected in EXPECTED_COLUMNS:
            assert expected in actual, (
                f"missing or mismatched column: expected={expected}; "
                f"actual_columns={[r for r in rows if r[0] == expected[0]]}"
            )


class TestPaperClaimsIndexesExist:
    """All 5 documented indexes are present in pg_indexes."""

    def test_indexes_match_docs(self, applied_migration: str) -> None:
        import psycopg

        with psycopg.connect(applied_migration) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'paper_claims'"
            )
            actual = {row[0] for row in cur.fetchall()}

        for idx in EXPECTED_INDEXES:
            assert idx in actual, f"missing index: {idx}; got={sorted(actual)}"

    def test_gin_tsvector_index_definition(self, applied_migration: str) -> None:
        import psycopg

        with psycopg.connect(applied_migration) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT indexdef FROM pg_indexes "
                "WHERE indexname = 'ix_paper_claims_claim_text_tsv'"
            )
            row = cur.fetchone()
            assert row is not None
            assert "gin" in row[0].lower()
            assert "to_tsvector" in row[0].lower()


class TestPaperClaimsCheckConstraint:
    """claim_type CHECK constraint blocks unknown labels."""

    def test_bogus_claim_type_raises_check_violation(self, applied_migration: str) -> None:
        import psycopg
        from psycopg import errors

        bibcode = _ensure_test_bibcode(applied_migration)

        with psycopg.connect(applied_migration) as conn, conn.cursor() as cur:
            with pytest.raises(errors.CheckViolation):
                cur.execute(
                    """
                    INSERT INTO paper_claims (
                        bibcode, section_index, paragraph_index,
                        char_span_start, char_span_end,
                        claim_text, claim_type,
                        extraction_model, extraction_prompt_version
                    ) VALUES (
                        %s, 0, 0, 0, 10,
                        'bogus claim', 'bogus',
                        'test-model', 'v0'
                    )
                    """,
                    (bibcode,),
                )
            # don't commit; raise leaves us in aborted txn — rollback on context exit


class TestPaperClaimsRoundTrip:
    """A valid INSERT round-trips; values come back as inserted."""

    def test_insert_and_select_back(self, applied_migration: str) -> None:
        import psycopg

        bibcode = _ensure_test_bibcode(applied_migration)
        claim_id = uuid.uuid4()

        with psycopg.connect(applied_migration) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO paper_claims (
                        claim_id, bibcode, section_index, paragraph_index,
                        char_span_start, char_span_end,
                        claim_text, claim_type,
                        subject, predicate, object,
                        confidence,
                        extraction_model, extraction_prompt_version
                    ) VALUES (
                        %s, %s, 1, 2, 100, 250,
                        'Galaxies are gravitationally bound.', 'factual',
                        'galaxies', 'are', 'gravitationally bound',
                        0.93,
                        'claude-opus-4-7', 'v1'
                    )
                    """,
                    (str(claim_id), bibcode),
                )
            conn.commit()

            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT bibcode, section_index, paragraph_index,
                               char_span_start, char_span_end,
                               claim_text, claim_type,
                               subject, predicate, object,
                               confidence,
                               extraction_model, extraction_prompt_version,
                               linked_entity_subject_id, linked_entity_object_id
                        FROM paper_claims
                        WHERE claim_id = %s
                        """,
                        (str(claim_id),),
                    )
                    row = cur.fetchone()

                assert row is not None
                assert row[0] == bibcode
                assert row[1] == 1
                assert row[2] == 2
                assert row[3] == 100
                assert row[4] == 250
                assert row[5] == "Galaxies are gravitationally bound."
                assert row[6] == "factual"
                assert row[7] == "galaxies"
                assert row[8] == "are"
                assert row[9] == "gravitationally bound"
                assert row[10] == pytest.approx(0.93, rel=1e-4)
                assert row[11] == "claude-opus-4-7"
                assert row[12] == "v1"
                assert row[13] is None
                assert row[14] is None
            finally:
                # Cleanup the row we inserted so reruns stay clean.
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM paper_claims WHERE claim_id = %s",
                        (str(claim_id),),
                    )
                conn.commit()


class TestPaperClaimsIdempotency:
    """Re-applying the migration succeeds (CREATE ... IF NOT EXISTS everywhere)."""

    def test_second_apply_succeeds(self, applied_migration: str) -> None:
        # applied_migration already ran the migration once; run it again here.
        result = subprocess.run(
            ["psql", applied_migration, "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_PATH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"second migration apply failed: stderr=\n{result.stderr}\nstdout=\n{result.stdout}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_test_bibcode(dsn: str) -> str:
    """Return a real bibcode from papers, or seed a synthetic one for the test.

    paper_claims.bibcode is FK to papers(bibcode); inserts need a real row.
    Prefer an existing paper to avoid mutating scix_test for a schema test.
    """
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT bibcode FROM papers LIMIT 1")
        row = cur.fetchone()
        if row is not None:
            return row[0]

        # Empty papers table — seed a synthetic bibcode.
        synthetic = "9999paper_claims_test_A"
        cur.execute(
            "INSERT INTO papers (bibcode) VALUES (%s) ON CONFLICT (bibcode) DO NOTHING",
            (synthetic,),
        )
        conn.commit()
        return synthetic
