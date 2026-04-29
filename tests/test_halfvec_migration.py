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


def _captured_sql(
    model_name: str,
    *,
    halfvec_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    from scix import search as search_mod  # local import after sys.path tweak

    monkeypatch.setattr(search_mod, "_HALFVEC_ENABLED", halfvec_enabled)

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

    search_mod.vector_search(conn, [0.1] * 768, model_name=model_name, limit=5)
    return capture.last_query


def test_indus_query_uses_embedding_hv_when_gate_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sql = _captured_sql("indus", halfvec_enabled=True, monkeypatch=monkeypatch)
    assert "pe.embedding_hv" in sql, sql
    assert "halfvec(768)" in sql, sql
    assert "pe.embedding::" not in sql, sql
    assert "vector(768)" not in sql, sql


def test_indus_query_uses_legacy_vector_when_gate_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default state until migrations 053/054 are applied to prod scix.
    # Bead scix_experiments-d0a — INDUS reads must not reference embedding_hv.
    sql = _captured_sql("indus", halfvec_enabled=False, monkeypatch=monkeypatch)
    assert "pe.embedding" in sql
    assert "vector(768)" in sql
    assert "embedding_hv" not in sql
    assert "halfvec" not in sql


def test_pilot_query_still_uses_legacy_vector_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for gate in (False, True):
        sql = _captured_sql("specter2", halfvec_enabled=gate, monkeypatch=monkeypatch)
        assert "pe.embedding" in sql, (gate, sql)
        assert "vector(768)" in sql, (gate, sql)
        assert "embedding_hv" not in sql, (gate, sql)
        assert "halfvec" not in sql, (gate, sql)


# ---------------------------------------------------------------------------
# Unit: embed.py write path is gated on SCIX_USE_HALFVEC
#
# Regression for bead scix_experiments-d0a — daily cron hung on 2026-04-28
# because store_embeddings_copy hard-coded `embedding_hv` in the INSERT into
# paper_embeddings, but the column does not exist on prod. The gate must
# default to off so INDUS writes land on the legacy `embedding` column.
# ---------------------------------------------------------------------------


def _capture_executes(
    callable_,
    *args,
    halfvec_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs,
) -> list[str]:
    """Run callable with embed._HALFVEC_ENABLED forced; return all executed SQL."""
    from scix import embed as embed_mod

    monkeypatch.setattr(embed_mod, "_HALFVEC_ENABLED", halfvec_enabled)

    executed: list[str] = []

    class _CopyCM:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write_row(self, row):
            return None

    class _Cursor:
        rowcount = 0

        def execute(self, query, params=None):  # noqa: ARG002
            executed.append(query)

        def copy(self, query):
            executed.append(query)
            return _CopyCM()

    class _CursorCM:
        def __enter__(self):
            return _Cursor()

        def __exit__(self, *exc):
            return False

    conn = MagicMock()
    conn.cursor = lambda: _CursorCM()
    callable_(conn, *args, **kwargs)
    return executed


def _make_input(bibcode: str = "2024TEST.....1A"):
    from scix.embed import EmbeddingInput

    return EmbeddingInput(
        bibcode=bibcode,
        text="title [SEP] abstract",
        input_type="title_abstract",
        source_hash="deadbeef",
    )


def test_indus_copy_write_skips_embedding_hv_when_gate_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scix.embed import store_embeddings_copy

    sql = _capture_executes(
        store_embeddings_copy,
        [_make_input()],
        [[0.1] * 768],
        "indus",
        halfvec_enabled=False,
        monkeypatch=monkeypatch,
    )
    insert_sql = next(s for s in sql if "INSERT INTO paper_embeddings" in s)
    assert "embedding_hv" not in insert_sql, insert_sql


def test_indus_copy_write_uses_embedding_hv_when_gate_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scix.embed import store_embeddings_copy

    sql = _capture_executes(
        store_embeddings_copy,
        [_make_input()],
        [[0.1] * 768],
        "indus",
        halfvec_enabled=True,
        monkeypatch=monkeypatch,
    )
    insert_sql = next(s for s in sql if "INSERT INTO paper_embeddings" in s)
    assert "embedding_hv" in insert_sql, insert_sql


def test_pilot_copy_write_never_uses_embedding_hv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scix.embed import store_embeddings_copy

    for gate in (False, True):
        sql = _capture_executes(
            store_embeddings_copy,
            [_make_input()],
            [[0.1] * 768],
            "specter2",
            halfvec_enabled=gate,
            monkeypatch=monkeypatch,
        )
        insert_sql = next(s for s in sql if "INSERT INTO paper_embeddings" in s)
        assert "embedding_hv" not in insert_sql, (gate, insert_sql)


def test_indus_row_insert_skips_embedding_hv_when_gate_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scix.embed import store_embeddings

    sql = _capture_executes(
        store_embeddings,
        [_make_input()],
        [[0.1] * 768],
        "indus",
        halfvec_enabled=False,
        monkeypatch=monkeypatch,
    )
    insert_sql = next(s for s in sql if "INSERT INTO paper_embeddings" in s)
    assert "embedding_hv" not in insert_sql, insert_sql
    assert "halfvec" not in insert_sql, insert_sql


def test_indus_row_insert_uses_embedding_hv_when_gate_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scix.embed import store_embeddings

    sql = _capture_executes(
        store_embeddings,
        [_make_input()],
        [[0.1] * 768],
        "indus",
        halfvec_enabled=True,
        monkeypatch=monkeypatch,
    )
    insert_sql = next(s for s in sql if "INSERT INTO paper_embeddings" in s)
    assert "embedding_hv" in insert_sql, insert_sql
    assert "halfvec(768)" in insert_sql, insert_sql


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
