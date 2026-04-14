"""Schema tests for migrations 038 (papers_external_ids) and 039 (papers_ads_body).

Verifies:
- Both tables exist and are LOGGED (never UNLOGGED per feedback_unlogged_tables).
- Primary keys and foreign keys to papers(bibcode) are in place.
- Expected indexes exist (btree for external_ids, GIN for papers_ads_body.tsv).
- papers_ads_body.tsv is a generated stored tsvector column.
- papers_external_ids.updated_at trigger updates the column on UPDATE.

These tests require SCIX_TEST_DSN to be set to a non-production database.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import is_production_dsn  # noqa: E402

TEST_DSN = os.environ.get("SCIX_TEST_DSN")

pytestmark = pytest.mark.skipif(
    TEST_DSN is None or (TEST_DSN is not None and is_production_dsn(TEST_DSN)),
    reason="Destructive schema tests require non-production SCIX_TEST_DSN.",
)


@pytest.fixture(scope="module")
def conn() -> Iterator[psycopg.Connection]:
    """Autocommit connection; tests assume migrations 038+039 are already applied
    to SCIX_TEST_DSN via scripts/setup_db.sh.
    """
    assert TEST_DSN is not None
    if is_production_dsn(TEST_DSN):
        pytest.skip("Refuses to run schema tests against production DSN.")
    c = psycopg.connect(TEST_DSN)
    c.autocommit = True
    try:
        yield c
    finally:
        c.close()


# ---------------------------------------------------------------------------
# LOGGED-ness: critical check. UNLOGGED tables are truncated on crash recovery
# and have destroyed 32M SPECTER2 embeddings once already.
# See feedback_unlogged_tables memory and migration 023.
# ---------------------------------------------------------------------------


class TestLoggedness:
    def test_papers_external_ids_is_logged(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT relpersistence FROM pg_class WHERE relname = 'papers_external_ids'")
            row = cur.fetchone()
            assert row is not None, "papers_external_ids table missing"
            assert row[0] == "p", (
                f"papers_external_ids must be LOGGED (relpersistence='p'), "
                f"got relpersistence='{row[0]}'. UNLOGGED tables truncate on crash."
            )

    def test_papers_ads_body_is_logged(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT relpersistence FROM pg_class WHERE relname = 'papers_ads_body'")
            row = cur.fetchone()
            assert row is not None, "papers_ads_body table missing"
            assert row[0] == "p", (
                f"papers_ads_body must be LOGGED (relpersistence='p'), "
                f"got relpersistence='{row[0]}'. UNLOGGED tables truncate on crash."
            )


# ---------------------------------------------------------------------------
# papers_external_ids structure
# ---------------------------------------------------------------------------


class TestPapersExternalIdsSchema:
    def test_primary_key_is_bibcode(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT a.attname
                FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
                WHERE t.relname = 'papers_external_ids' AND c.contype = 'p'
                """)
            cols = [r[0] for r in cur.fetchall()]
            assert cols == ["bibcode"]

    def test_foreign_key_to_papers(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT confrelid::regclass::text
                FROM pg_constraint
                WHERE conrelid = 'papers_external_ids'::regclass AND contype = 'f'
                """)
            refs = [r[0] for r in cur.fetchall()]
            assert "papers" in refs, "Expected FK from papers_external_ids to papers"

    def test_required_columns_present(self, conn: psycopg.Connection) -> None:
        expected = {
            "bibcode": "text",
            "doi": "text",
            "arxiv_id": "text",
            "openalex_id": "text",
            "s2_corpus_id": "bigint",
            "s2_paper_id": "text",
            "pmcid": "text",
            "pmid": "bigint",
            "has_ads_body": "boolean",
            "has_arxiv_source": "boolean",
            "has_ar5iv_html": "boolean",
            "has_s2orc_body": "boolean",
            "openalex_has_pdf_url": "boolean",
            "updated_at": "timestamp with time zone",
        }
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'papers_external_ids'
                """)
            actual = dict(cur.fetchall())
        for col, dtype in expected.items():
            assert col in actual, f"Missing column {col}"
            assert actual[col] == dtype, f"Column {col}: expected {dtype}, got {actual[col]}"

    def test_has_flags_default_false(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, column_default, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'papers_external_ids'
                  AND (column_name LIKE 'has\\_%' ESCAPE '\\'
                       OR column_name = 'openalex_has_pdf_url')
                """)
            rows = cur.fetchall()
        assert rows, "Expected has_* / openalex_has_pdf_url columns on papers_external_ids"
        for name, default, nullable in rows:
            assert nullable == "NO", f"{name} should be NOT NULL"
            assert (
                default is not None and "false" in default.lower()
            ), f"{name} should default to false, got {default}"

    def test_btree_indexes_on_external_ids(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT indexdef
                FROM pg_indexes
                WHERE tablename = 'papers_external_ids'
                """)
            defs = [r[0] for r in cur.fetchall()]
        indexed_cols = {"doi", "arxiv_id", "openalex_id", "s2_corpus_id"}
        for col in indexed_cols:
            assert any(f"({col})" in d for d in defs), f"Missing index on {col}: {defs}"

    def test_updated_at_advances_on_update(self, conn: psycopg.Connection) -> None:
        """updated_at must advance when a row is UPDATEd (trigger, not just DEFAULT)."""
        with conn.cursor() as cur:
            cur.execute("SELECT bibcode FROM papers LIMIT 1")
            row = cur.fetchone()
            if row is None:
                # Insert a minimal papers row for this test so FK is satisfied.
                cur.execute(
                    "INSERT INTO papers (bibcode) VALUES (%s) ON CONFLICT DO NOTHING",
                    ("2099TEST.upd.at",),
                )
                bibcode = "2099TEST.upd.at"
            else:
                bibcode = row[0]

            cur.execute(
                """
                INSERT INTO papers_external_ids (bibcode, doi) VALUES (%s, %s)
                ON CONFLICT (bibcode) DO UPDATE SET doi = EXCLUDED.doi
                RETURNING updated_at
                """,
                (bibcode, "10.test/mig38.initial"),
            )
            ts_before = cur.fetchone()[0]

            cur.execute("SELECT pg_sleep(0.01)")  # ensure a tick passes
            cur.execute(
                "UPDATE papers_external_ids SET doi = %s WHERE bibcode = %s RETURNING updated_at",
                ("10.test/mig38.updated", bibcode),
            )
            ts_after = cur.fetchone()[0]
            # Cleanup
            cur.execute("DELETE FROM papers_external_ids WHERE bibcode = %s", (bibcode,))
            if bibcode == "2099TEST.upd.at":
                cur.execute("DELETE FROM papers WHERE bibcode = %s", (bibcode,))

        assert ts_after > ts_before, (
            f"papers_external_ids.updated_at must advance on UPDATE: "
            f"before={ts_before}, after={ts_after}"
        )


# ---------------------------------------------------------------------------
# papers_ads_body structure
# ---------------------------------------------------------------------------


class TestPapersAdsBodySchema:
    def test_primary_key_is_bibcode(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT a.attname
                FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
                WHERE t.relname = 'papers_ads_body' AND c.contype = 'p'
                """)
            cols = [r[0] for r in cur.fetchall()]
            assert cols == ["bibcode"]

    def test_foreign_key_to_papers(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT confrelid::regclass::text
                FROM pg_constraint
                WHERE conrelid = 'papers_ads_body'::regclass AND contype = 'f'
                """)
            refs = [r[0] for r in cur.fetchall()]
            assert "papers" in refs, "Expected FK from papers_ads_body to papers"

    def test_required_columns(self, conn: psycopg.Connection) -> None:
        expected_cols = {"bibcode", "body_text", "body_length", "harvested_at", "tsv"}
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'papers_ads_body'
                """)
            actual = {r[0] for r in cur.fetchall()}
        missing = expected_cols - actual
        assert not missing, f"Missing columns on papers_ads_body: {missing}"

    def test_body_text_not_null(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT is_nullable
                FROM information_schema.columns
                WHERE table_name = 'papers_ads_body' AND column_name = 'body_text'
                """)
            assert cur.fetchone()[0] == "NO"

    def test_tsv_is_generated_stored(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT is_generated, generation_expression
                FROM information_schema.columns
                WHERE table_name = 'papers_ads_body' AND column_name = 'tsv'
                """)
            row = cur.fetchone()
            assert row is not None, "tsv column missing"
            assert row[0] == "ALWAYS", f"tsv should be GENERATED ALWAYS, got {row[0]}"
            assert (
                row[1] is not None and "to_tsvector" in row[1]
            ), f"tsv generation expression should call to_tsvector, got: {row[1]}"

    def test_gin_index_on_tsv(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT indexdef
                FROM pg_indexes
                WHERE tablename = 'papers_ads_body'
                """)
            defs = [r[0] for r in cur.fetchall()]
        assert any(
            "gin" in d.lower() and "tsv" in d for d in defs
        ), f"Expected a GIN index on papers_ads_body.tsv. Found: {defs}"
