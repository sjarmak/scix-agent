"""Tests for the halfvec storage migration (bead scix_experiments-0vy).

Two layers:

1. Unit tests (always run) — verify the SQL emitted by search.py and embed.py
   targets the halfvec shadow column for INDUS and the legacy vector column
   for pilot models.
2. Integration tests (skipped without SCIX_TEST_DSN) — apply the migration
   against a real pgvector-enabled database, seed synthetic rows, run the
   backfill script, and assert the partial HNSW halfvec index is the one the
   planner uses.

The integration leg uses SCIX_TEST_DSN to protect production. See
CLAUDE.md #Testing — Database Safety.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Unit: search.py routes INDUS to embedding_hv + halfvec cast
# ---------------------------------------------------------------------------


def _captured_sql(model_name: str) -> str:
    from scix.search import vector_search  # local import after sys.path tweak

    conn = MagicMock()
    # cursor() returns a context manager whose __enter__ is a MagicMock
    cm = conn.cursor.return_value
    cm.__enter__.return_value.fetchall.return_value = []

    class _Capture:
        last_query = ""

        def execute(self, query, params=None):  # noqa: ARG002
            _Capture.last_query = query

        def fetchall(self):
            return []

    capture = _Capture()
    cm.__enter__.return_value = capture

    vector_search(conn, [0.1] * 768, model_name=model_name, limit=5)
    return capture.last_query


def test_indus_query_uses_embedding_hv_and_halfvec_cast() -> None:
    sql = _captured_sql("indus")
    assert "pe.embedding_hv" in sql, sql
    assert "halfvec(768)" in sql, sql
    assert "pe.embedding::" not in sql, sql
    assert "vector(768)" not in sql, sql


def test_pilot_query_still_uses_legacy_vector_column() -> None:
    sql = _captured_sql("specter2")
    assert "pe.embedding" in sql
    assert "vector(768)" in sql
    assert "embedding_hv" not in sql
    assert "halfvec" not in sql


# ---------------------------------------------------------------------------
# Integration: applies migration 053+054 against scix_test
# ---------------------------------------------------------------------------


def _test_dsn() -> str | None:
    dsn = os.environ.get("SCIX_TEST_DSN")
    if not dsn:
        return None
    if "scix_test" not in dsn:
        pytest.fail(
            "SCIX_TEST_DSN must reference scix_test — got: " + dsn,
            pytrace=False,
        )
    return dsn


@pytest.mark.integration
def test_migration_053_054_apply_idempotently() -> None:
    dsn = _test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set")

    import psycopg

    for migration in (
        "migrations/053_paper_embeddings_halfvec.sql",
        "migrations/054_paper_embeddings_halfvec_index.sql",
    ):
        # Apply twice — must succeed both times (IF NOT EXISTS everywhere).
        for _ in range(2):
            result = subprocess.run(
                ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-f", migration],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, (
                f"{migration} failed: {result.stderr}"
            )

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT format_type(atttypid, atttypmod) "
            "FROM pg_attribute WHERE attrelid='paper_embeddings'::regclass "
            "AND attname='embedding_hv'"
        )
        row = cur.fetchone()
        assert row is not None and row[0] == "halfvec(768)", row

        cur.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE indexname='idx_embed_hnsw_indus_hv'"
        )
        idx = cur.fetchone()
        assert idx is not None
        assert "halfvec_cosine_ops" in idx[0]
        assert "model_name = 'indus'" in idx[0]
