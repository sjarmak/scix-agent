"""Tests for M8 fusion MV — migration 033 + src/scix/fusion_mv.py."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import psycopg
import pytest

from scix import fusion_mv
from tests.helpers import get_test_dsn

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = REPO_ROOT / "migrations" / "033_fusion_mv.sql"

# Initial (uncalibrated) weights — must match migration 033's tier_weight().
SEED_TIER_WEIGHTS = {
    1: 0.98,
    2: 0.85,
    3: 0.92,
    4: 0.50,
    5: 0.88,
}
DEFAULT_WEIGHT = 0.50


# ---------------------------------------------------------------------------
# Static migration-file checks (no DB required)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def migration_sql() -> str:
    return MIGRATION_PATH.read_text(encoding="utf-8")


class TestMigrationFileText:
    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.is_file()

    def test_wrapped_in_transaction(self, migration_sql: str) -> None:
        assert "BEGIN;" in migration_sql
        assert "COMMIT;" in migration_sql

    def test_tier_weight_function_declared_immutable(self, migration_sql: str) -> None:
        assert "CREATE OR REPLACE FUNCTION tier_weight" in migration_sql
        assert "IMMUTABLE" in migration_sql
        assert "LEAKPROOF" in migration_sql
        assert "PARALLEL SAFE" in migration_sql

    def test_tier_weight_seed_values(self, migration_sql: str) -> None:
        for weight in ("0.98", "0.85", "0.92", "0.50", "0.88"):
            assert weight in migration_sql, f"missing seed weight {weight}"

    def test_calibration_log_table(self, migration_sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS tier_weight_calibration_log" in migration_sql
        assert "placeholder_2026-04-12" in migration_sql

    def test_fusion_state_table(self, migration_sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS fusion_mv_state" in migration_sql
        assert "CHECK (id = 1)" in migration_sql
        assert "last_refresh_at" in migration_sql

    def test_materialized_view_created(self, migration_sql: str) -> None:
        assert "CREATE MATERIALIZED VIEW document_entities_canonical" in migration_sql
        # Noisy-OR formula components
        assert "1 - exp(" in migration_sql
        assert "sum(" in migration_sql
        assert "ln(" in migration_sql
        assert "LEAST(" in migration_sql  # clamp

    def test_required_indexes(self, migration_sql: str) -> None:
        assert "CREATE UNIQUE INDEX" in migration_sql
        assert "(bibcode, entity_id)" in migration_sql
        assert "(entity_id, fused_confidence DESC)" in migration_sql
        # bibcode-only index
        assert "idx_dec_bibcode\n    ON document_entities_canonical (bibcode)" in migration_sql


# ---------------------------------------------------------------------------
# DB-backed integration tests (require SCIX_TEST_DSN)
# ---------------------------------------------------------------------------

requires_test_dsn = pytest.mark.skipif(
    get_test_dsn() is None,
    reason="SCIX_TEST_DSN not set (or points at production) — skipping integration tests",
)


def _apply_migration(conn: psycopg.Connection) -> None:
    """Apply migration 033 (idempotent)."""
    sql_text = MIGRATION_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql_text)
    conn.commit()


def _reset_entity_rows(conn: psycopg.Connection) -> None:
    """Drop any test data in document_entities / entities left from a prior run."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM document_entities WHERE bibcode LIKE 'TEST.u08%'")
        cur.execute("DELETE FROM entities WHERE canonical_name LIKE 'u08-test-%'")
    conn.commit()


def _insert_entity(conn: psycopg.Connection, name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, source)
            VALUES (%s, 'mission', 'u08-test')
            RETURNING id
            """,
            (name,),
        )
        row = cur.fetchone()
    conn.commit()
    assert row is not None
    return int(row[0])


def _closed_form(pairs: list[tuple[float, int]]) -> float:
    """Reference noisy-OR computation in pure Python."""
    s = 0.0
    for confidence, tier in pairs:
        weight = SEED_TIER_WEIGHTS.get(tier, DEFAULT_WEIGHT)
        clamped = min(0.9999, max(0.0, confidence * weight))
        s += math.log(1.0 - clamped)
    return 1.0 - math.exp(s)


@pytest.fixture(scope="module")
def db_conn() -> psycopg.Connection:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set")
    conn = psycopg.connect(dsn)
    conn.autocommit = False
    _apply_migration(conn)
    _reset_entity_rows(conn)
    yield conn
    _reset_entity_rows(conn)
    conn.close()


@requires_test_dsn
class TestTierWeightFunction:
    def test_values_match_seed_table(self, db_conn: psycopg.Connection) -> None:
        with db_conn.cursor() as cur:
            for tier, expected in SEED_TIER_WEIGHTS.items():
                cur.execute("SELECT tier_weight(%s::SMALLINT)", (tier,))
                got = float(cur.fetchone()[0])
                assert got == pytest.approx(expected, abs=1e-12)

    def test_unknown_tier_returns_default(self, db_conn: psycopg.Connection) -> None:
        with db_conn.cursor() as cur:
            cur.execute("SELECT tier_weight(0::SMALLINT)")
            assert float(cur.fetchone()[0]) == pytest.approx(DEFAULT_WEIGHT)
            cur.execute("SELECT tier_weight(99::SMALLINT)")
            assert float(cur.fetchone()[0]) == pytest.approx(DEFAULT_WEIGHT)

    def test_declared_immutable_and_parallel_safe(self, db_conn: psycopg.Connection) -> None:
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT provolatile, proparallel, proleakproof
                  FROM pg_proc
                 WHERE proname = 'tier_weight'
                """)
            row = cur.fetchone()
        assert row is not None
        provolatile, proparallel, proleakproof = row
        assert provolatile == "i", f"expected IMMUTABLE (i), got {provolatile!r}"
        assert proparallel == "s", f"expected PARALLEL SAFE (s), got {proparallel!r}"
        assert proleakproof is True


@requires_test_dsn
class TestCalibrationLog:
    def test_initial_row_present(self, db_conn: psycopg.Connection) -> None:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT version, weights FROM tier_weight_calibration_log "
                "WHERE version = 'placeholder_2026-04-12'"
            )
            row = cur.fetchone()
        assert row is not None
        version, weights = row
        assert version == "placeholder_2026-04-12"
        assert weights["1"] == 0.98
        assert weights["5"] == 0.88


@requires_test_dsn
class TestMaterializedView:
    def test_unique_index_on_bibcode_entity(self, db_conn: psycopg.Connection) -> None:
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT indexname, indexdef
                  FROM pg_indexes
                 WHERE tablename = 'document_entities_canonical'
                """)
            rows = cur.fetchall()
        names = [r[0] for r in rows]
        assert "idx_dec_bibcode_entity" in names
        assert "idx_dec_entity_fused" in names
        assert "idx_dec_bibcode" in names
        # Confirm uniqueness
        unique_def = next(defn for name, defn in rows if name == "idx_dec_bibcode_entity")
        assert "UNIQUE" in unique_def.upper()

    def test_refresh_concurrently_succeeds(self, db_conn: psycopg.Connection) -> None:
        # REFRESH CONCURRENTLY requires autocommit.
        dsn = get_test_dsn()
        assert dsn is not None
        with psycopg.connect(dsn) as c:
            c.autocommit = True
            with c.cursor() as cur:
                cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY document_entities_canonical")

    def test_fused_matches_closed_form_five_tiers(self, db_conn: psycopg.Connection) -> None:
        _reset_entity_rows(db_conn)
        entity_id = _insert_entity(db_conn, "u08-test-five-tiers")
        bibcode = "TEST.u08.five"
        # Five tiers, five link_types so the PK (bibcode, entity_id, link_type, tier)
        # is unique for each row.
        seed = [
            ("keyword_exact", 1, 0.95),
            ("alias_ctx", 2, 0.80),
            ("legacy_kw", 3, 0.60),
            ("llm_adj", 4, 0.70),
            ("jit", 5, 0.85),
        ]
        with db_conn.cursor() as cur:
            for link_type, tier, conf in seed:
                cur.execute(
                    """
                    INSERT INTO document_entities
                        (bibcode, entity_id, link_type, confidence, tier, tier_version)
                    VALUES (%s, %s, %s, %s, %s, 1)
                    """,
                    (bibcode, entity_id, link_type, conf, tier),
                )
        db_conn.commit()

        # Refresh via the helper (non-concurrent first run is fine because
        # mark_dirty -> refresh_if_due(0) takes CONCURRENTLY path).
        fusion_mv.mark_dirty(db_conn)
        refreshed = fusion_mv.refresh_if_due(db_conn, min_interval_seconds=0)
        assert refreshed is True

        # Read back via raw psql (not through ORM) — the lint forbids the
        # SELECT literal so we annotate it.
        dsn = get_test_dsn()
        assert dsn is not None
        with psycopg.connect(dsn) as c:
            with c.cursor() as cur:
                cur.execute(
                    "SELECT fused_confidence, link_count "  # noqa: resolver-lint
                    "FROM document_entities_canonical "
                    "WHERE bibcode = %s AND entity_id = %s",
                    (bibcode, entity_id),
                )
                row = cur.fetchone()
        assert row is not None, "MV row missing after refresh"
        got, link_count = row
        expected = _closed_form([(conf, tier) for _, tier, conf in seed])
        assert float(got) == pytest.approx(expected, abs=1e-9)
        assert link_count == 5


@requires_test_dsn
class TestRefreshHelper:
    def test_mark_dirty_sets_flag(self, db_conn: psycopg.Connection) -> None:
        with db_conn.cursor() as cur:
            cur.execute("UPDATE fusion_mv_state SET dirty = false WHERE id = 1")
        db_conn.commit()

        fusion_mv.mark_dirty(db_conn)

        with db_conn.cursor() as cur:
            cur.execute("SELECT dirty FROM fusion_mv_state WHERE id = 1")
            (dirty,) = cur.fetchone()
        assert dirty is True

    def test_rate_limiter_blocks_second_call(self, db_conn: psycopg.Connection) -> None:
        fusion_mv.mark_dirty(db_conn)
        # First call with a 0-second window runs the refresh.
        assert fusion_mv.refresh_if_due(db_conn, min_interval_seconds=0) is True
        # Second call with a 1-hour window is rate-limited (dirty is now false
        # and last_refresh_at is recent, so both guards apply).
        assert fusion_mv.refresh_if_due(db_conn, min_interval_seconds=3600) is False

    def test_not_dirty_is_noop(self, db_conn: psycopg.Connection) -> None:
        # Ensure state row exists and flip dirty off.
        fusion_mv.refresh_if_due(db_conn, min_interval_seconds=0)
        with db_conn.cursor() as cur:
            cur.execute("UPDATE fusion_mv_state SET dirty = false WHERE id = 1")
        db_conn.commit()
        assert fusion_mv.refresh_if_due(db_conn, min_interval_seconds=0) is False


@requires_test_dsn
class TestLatency:
    def test_entity_topk_under_100ms(self, db_conn: psycopg.Connection) -> None:
        _reset_entity_rows(db_conn)
        entity_id = _insert_entity(db_conn, "u08-test-latency")

        rows = []
        for i in range(100):
            bibcode = f"TEST.u08.lat.{i:04d}"
            tier = (i % 5) + 1  # tiers 1..5
            link_type = f"lt{i}"  # unique per row so the PK is unique
            confidence = 0.50 + (i % 50) / 200.0  # 0.50 .. 0.745
            rows.append((bibcode, entity_id, link_type, confidence, tier))

        with db_conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO document_entities
                    (bibcode, entity_id, link_type, confidence, tier, tier_version)
                VALUES (%s, %s, %s, %s, %s, 1)
                """,
                rows,
            )
        db_conn.commit()

        fusion_mv.mark_dirty(db_conn)
        assert fusion_mv.refresh_if_due(db_conn, min_interval_seconds=0) is True

        dsn = get_test_dsn()
        assert dsn is not None
        with psycopg.connect(dsn) as c:
            with c.cursor() as cur:
                # Warm the planner / cache.
                cur.execute(
                    "SELECT bibcode, fused_confidence "  # noqa: resolver-lint
                    "FROM document_entities_canonical "
                    "WHERE entity_id = %s "
                    "ORDER BY fused_confidence DESC LIMIT 20",
                    (entity_id,),
                )
                cur.fetchall()
                start = time.perf_counter()
                cur.execute(
                    "SELECT bibcode, fused_confidence "  # noqa: resolver-lint
                    "FROM document_entities_canonical "
                    "WHERE entity_id = %s "
                    "ORDER BY fused_confidence DESC LIMIT 20",
                    (entity_id,),
                )
                hits = cur.fetchall()
                elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert len(hits) == 20
        assert elapsed_ms < 100.0, f"query took {elapsed_ms:.2f}ms (>=100ms)"
