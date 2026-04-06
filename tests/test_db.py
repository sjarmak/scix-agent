"""Integration tests for DB helpers: IndexManager and IngestLog."""

from __future__ import annotations

import os

import psycopg
import pytest

from scix.db import IndexManager, IngestLog, get_connection

TEST_DSN = os.environ.get("SCIX_TEST_DSN")
DSN = TEST_DSN or os.environ.get("SCIX_DSN", "dbname=scix")

_PRODUCTION_DB_NAMES = {"scix"}


def _is_production_dsn(dsn: str) -> bool:
    """Return True if DSN appears to point at a production database."""
    for token in dsn.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key.strip() == "dbname" and value.strip() in _PRODUCTION_DB_NAMES:
                return True
    return False


# SAFETY: tests that drop indexes on the live papers table must never run
# against production. A killed/interrupted drop_and_recreate roundtrip would
# leave the production papers table without indexes.
_skip_destructive = pytest.mark.skipif(
    TEST_DSN is None or _is_production_dsn(DSN),
    reason=(
        "Destructive index tests require SCIX_TEST_DSN pointing to a non-production "
        "database (refuses dbname=scix). Skipped to protect production indexes."
    ),
)


@pytest.fixture()
def conn():
    """Per-test connection. IndexManager and IngestLog commit internally,
    so we can't use savepoints. Instead, clean up explicitly."""
    with psycopg.connect(DSN) as c:
        c.autocommit = False
        yield c


@pytest.mark.integration
class TestGetConnection:
    def test_connects_successfully(self) -> None:
        c = get_connection(DSN)
        try:
            with c.cursor() as cur:
                cur.execute("SELECT 1")
                assert cur.fetchone()[0] == 1
        finally:
            c.close()


@pytest.mark.integration
class TestIndexManager:
    def test_get_non_pk_indexes(self, conn) -> None:
        mgr = IndexManager(conn, "papers")
        indexes = mgr.get_non_pk_indexes()
        names = {idx.name for idx in indexes}
        assert "idx_papers_year" in names
        assert "idx_papers_authors" in names
        assert "papers_pkey" not in names

    def test_each_index_has_create_statement(self, conn) -> None:
        mgr = IndexManager(conn, "papers")
        indexes = mgr.get_non_pk_indexes()
        for idx in indexes:
            assert idx.definition.upper().startswith("CREATE INDEX")
            assert idx.table == "papers"

    @_skip_destructive
    def test_drop_and_recreate_roundtrip(self, conn) -> None:
        mgr = IndexManager(conn, "papers")
        original = mgr.get_non_pk_indexes()
        assert len(original) > 0

        dropped = mgr.drop_indexes()
        assert len(dropped) == len(original)

        remaining = mgr.get_non_pk_indexes()
        assert len(remaining) == 0

        # Always recreate, even on failure
        try:
            mgr.recreate_indexes(dropped)
            restored = mgr.get_non_pk_indexes()
            assert {idx.name for idx in restored} == {idx.name for idx in original}
        except Exception:
            mgr.recreate_indexes(dropped)
            raise


@pytest.mark.integration
class TestIngestLog:
    @pytest.fixture(autouse=True)
    def _cleanup(self, conn):
        yield
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ingest_log WHERE filename LIKE 'test_%%'")
        conn.commit()

    def test_is_complete_false_for_unknown(self, conn) -> None:
        log = IngestLog(conn)
        assert log.is_complete("test_nonexistent.jsonl") is False

    def test_start_and_check_in_progress(self, conn) -> None:
        log = IngestLog(conn)
        log.start("test_file.jsonl")
        assert log.is_complete("test_file.jsonl") is False
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM ingest_log WHERE filename = %s", ("test_file.jsonl",))
            assert cur.fetchone()[0] == "in_progress"

    def test_update_counts(self, conn) -> None:
        log = IngestLog(conn)
        log.start("test_file.jsonl")
        log.update_counts("test_file.jsonl", records=5000, errors=3, edges=15000)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT records_loaded, errors_skipped, edges_loaded FROM ingest_log WHERE filename = %s",
                ("test_file.jsonl",),
            )
            assert cur.fetchone() == (5000, 3, 15000)

    def test_finish_marks_complete(self, conn) -> None:
        log = IngestLog(conn)
        log.start("test_file.jsonl")
        log.finish("test_file.jsonl")
        assert log.is_complete("test_file.jsonl") is True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT finished_at FROM ingest_log WHERE filename = %s",
                ("test_file.jsonl",),
            )
            assert cur.fetchone()[0] is not None

    def test_mark_failed(self, conn) -> None:
        log = IngestLog(conn)
        log.start("test_file.jsonl")
        log.mark_failed("test_file.jsonl")
        assert log.is_complete("test_file.jsonl") is False
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM ingest_log WHERE filename = %s", ("test_file.jsonl",))
            assert cur.fetchone()[0] == "failed"

    def test_start_resets_on_re_ingest(self, conn) -> None:
        log = IngestLog(conn)
        log.start("test_file.jsonl")
        log.update_counts("test_file.jsonl", records=100, errors=1, edges=200)
        log.mark_failed("test_file.jsonl")
        # Re-start should reset
        log.start("test_file.jsonl")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT records_loaded, errors_skipped, edges_loaded, status, finished_at FROM ingest_log WHERE filename = %s",
                ("test_file.jsonl",),
            )
            row = cur.fetchone()
            assert row[0] == 0
            assert row[1] == 0
            assert row[2] == 0
            assert row[3] == "in_progress"
            assert row[4] is None
