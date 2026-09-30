from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import throwaway_db

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


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
    """Replay 001 + 053 + 054 into a throwaway database.

    These migrations extend ``paper_embeddings``, which ADR-015 dropped from
    production; migration 074 records the retirement. The files are still part
    of the history and must stay replayable, so this asserts against a database
    built from the chain rather than against the shared scix_test — where the
    table's presence depended on whichever module replayed 001 first.
    """
    if _test_dsn() is None:
        pytest.skip("SCIX_TEST_DSN not set")

    import psycopg

    migrations = [
        "001_initial_schema.sql",
        "053_paper_embeddings_halfvec.sql",
        "054_paper_embeddings_halfvec_index.sql",
    ]
    with throwaway_db(migrations, REPO_ROOT) as dsn:
        # Re-apply 053/054 — must succeed a second time (IF NOT EXISTS everywhere).
        for migration in migrations[1:]:
            result = subprocess.run(
                ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-f", f"migrations/{migration}"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, f"{migration} failed: {result.stderr}"

        _assert_halfvec_shape(psycopg, dsn)


def _assert_halfvec_shape(psycopg, dsn: str) -> None:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT format_type(atttypid, atttypmod) "
            "FROM pg_attribute WHERE attrelid='paper_embeddings'::regclass "
            "AND attname='embedding_hv'"
        )
        row = cur.fetchone()
        assert row is not None and row[0] == "halfvec(768)", row

        cur.execute("SELECT indexdef FROM pg_indexes " "WHERE indexname='idx_embed_hnsw_indus_hv'")
        idx = cur.fetchone()
        assert idx is not None
        assert "halfvec_cosine_ops" in idx[0]
        assert "model_name = 'indus'" in idx[0]
