"""Integration tests for scix.harvest_promotion.promote_harvest.

These tests require SCIX_TEST_DSN pointing at a non-production database
(canonical: ``dbname=scix_test``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import pytest

# Ensure the local `src/` layout is importable without a full install.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from helpers import is_production_dsn  # noqa: E402

from scix.harvest_promotion import (  # noqa: E402
    PER_SOURCE_FLOORS,
    PromotionResult,
    promote_harvest,
)

TEST_DSN = os.environ.get("SCIX_TEST_DSN")

pytestmark = pytest.mark.skipif(
    TEST_DSN is None or (TEST_DSN is not None and is_production_dsn(TEST_DSN)),
    reason=("promote_harvest tests require SCIX_TEST_DSN pointing at a non-production DB"),
)


TEST_SOURCE_CLEAN = "PROMOTE_TEST_CLEAN"
TEST_SOURCE_SHRINK = "PROMOTE_TEST_SHRINK"
TEST_SOURCE_ORPHAN = "PROMOTE_TEST_ORPHAN"
TEST_BIBCODE_PREFIX = "PRTEST.."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    assert TEST_DSN is not None
    c = psycopg.connect(TEST_DSN)
    c.autocommit = False
    try:
        yield c
    finally:
        c.close()


def _cleanup(conn: psycopg.Connection) -> None:
    """Remove any rows from prior tests. Idempotent."""
    with conn.cursor() as cur:
        sources = (TEST_SOURCE_CLEAN, TEST_SOURCE_SHRINK, TEST_SOURCE_ORPHAN)
        cur.execute(
            "DELETE FROM document_entities WHERE bibcode LIKE %s",
            (TEST_BIBCODE_PREFIX + "%",),
        )
        cur.execute("DELETE FROM papers WHERE bibcode LIKE %s", (TEST_BIBCODE_PREFIX + "%",))
        cur.execute(
            "DELETE FROM entity_identifiers WHERE entity_id IN "
            "(SELECT id FROM entities WHERE source = ANY(%s))",
            (list(sources),),
        )
        cur.execute(
            "DELETE FROM entity_aliases WHERE entity_id IN "
            "(SELECT id FROM entities WHERE source = ANY(%s))",
            (list(sources),),
        )
        cur.execute("DELETE FROM entities WHERE source = ANY(%s)", (list(sources),))
        cur.execute("DELETE FROM entities_staging WHERE source = ANY(%s)", (list(sources),))
        cur.execute(
            "DELETE FROM entity_aliases_staging WHERE source = ANY(%s)",
            (list(sources),),
        )
        cur.execute(
            "DELETE FROM entity_identifiers_staging WHERE source = ANY(%s)",
            (list(sources),),
        )
        cur.execute("DELETE FROM harvest_runs WHERE source = ANY(%s)", (list(sources),))
    conn.commit()


@pytest.fixture(autouse=True)
def _clean_before_and_after(conn):
    _cleanup(conn)
    yield
    _cleanup(conn)


def _create_run(conn: psycopg.Connection, source: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO harvest_runs (source, status) VALUES (%s, 'running') RETURNING id",
            (source,),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return int(run_id)


def _insert_staging_entities(
    conn: psycopg.Connection,
    run_id: int,
    source: str,
    n: int,
    name_prefix: str = "name",
) -> None:
    with conn.cursor() as cur:
        for i in range(n):
            cur.execute(
                """
                INSERT INTO entities_staging (
                    staging_run_id, canonical_name, entity_type, source, properties
                )
                VALUES (%s, %s, 'concept', %s, '{}'::jsonb)
                """,
                (run_id, f"{name_prefix}-{i}", source),
            )
    conn.commit()


def _insert_prod_entities(
    conn: psycopg.Connection,
    source: str,
    n: int,
    name_prefix: str = "prod",
) -> list[int]:
    ids: list[int] = []
    with conn.cursor() as cur:
        for i in range(n):
            cur.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, source, properties)
                VALUES (%s, 'concept', %s, '{}'::jsonb)
                RETURNING id
                """,
                (f"{name_prefix}-{i}", source),
            )
            ids.append(int(cur.fetchone()[0]))
    conn.commit()
    return ids


def _seed_papers(conn: psycopg.Connection, bibcodes: list[str]) -> None:
    """Insert minimal paper rows so document_entities FKs are satisfied."""
    with conn.cursor() as cur:
        for bc in bibcodes:
            cur.execute(
                "INSERT INTO papers (bibcode) VALUES (%s) ON CONFLICT DO NOTHING",
                (bc,),
            )
    conn.commit()


def _attach_doc_entities(conn: psycopg.Connection, entity_id: int, n: int) -> None:
    bibcodes = [f"{TEST_BIBCODE_PREFIX}{entity_id:04d}{i:06d}" for i in range(n)]
    _seed_papers(conn, bibcodes)
    with conn.cursor() as cur:
        for bc in bibcodes:
            cur.execute(
                """
                INSERT INTO document_entities (bibcode, entity_id, link_type, tier)
                VALUES (%s, %s, 'mention', 0)
                ON CONFLICT DO NOTHING
                """,
                (bc, entity_id),
            )
    conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPromoteHarvestClean:
    def test_clean_run_is_promoted_atomically(self, conn):
        run_id = _create_run(conn, TEST_SOURCE_CLEAN)
        _insert_staging_entities(conn, run_id, TEST_SOURCE_CLEAN, 10, name_prefix="clean")

        # No production entities for this source — zero shrinkage.
        result = promote_harvest(
            run_id,
            dsn=TEST_DSN,
            # Override floors so our synthetic test-source isn't blocked.
            floors={TEST_SOURCE_CLEAN: 0},
        )
        assert isinstance(result, PromotionResult)
        assert result.accepted, f"expected accepted, got reason={result.reason} diff={result.diff}"
        assert result.reason is None

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entities WHERE source = %s", (TEST_SOURCE_CLEAN,))
            prod_count = cur.fetchone()[0]
            cur.execute("SELECT status FROM harvest_runs WHERE id = %s", (run_id,))
            status = cur.fetchone()[0]
        assert prod_count == 10
        assert status == "promoted"


class TestPromoteHarvestShrinkage:
    def test_canonical_shrinkage_rejects_and_preserves_staging(self, conn):
        # Seed production with 100 entities, staging with 50 -> 50% shrinkage.
        _insert_prod_entities(conn, TEST_SOURCE_SHRINK, 100, name_prefix="p")
        run_id = _create_run(conn, TEST_SOURCE_SHRINK)
        _insert_staging_entities(conn, run_id, TEST_SOURCE_SHRINK, 50, name_prefix="s")

        result = promote_harvest(
            run_id,
            dsn=TEST_DSN,
            floors={TEST_SOURCE_SHRINK: 0},
        )
        assert not result.accepted
        assert result.reason == "canonical_shrinkage"
        assert "canonical_shrinkage" in (result.reason or "")
        assert result.diff.get("canonical_shrinkage") is not None
        assert float(result.diff["canonical_shrinkage"]) > 0.02

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM entities_staging WHERE staging_run_id = %s",
                (run_id,),
            )
            assert cur.fetchone()[0] == 50, "staging rows must be preserved on reject"
            cur.execute("SELECT status FROM harvest_runs WHERE id = %s", (run_id,))
            assert cur.fetchone()[0] == "rejected_by_diff"


class TestPromoteHarvestOrphan:
    def test_orphan_prevention_rejects(self, conn):
        # Seed production with one heavily-referenced entity.
        prod_ids = _insert_prod_entities(conn, TEST_SOURCE_ORPHAN, 1, name_prefix="heavy")
        heavy_id = prod_ids[0]

        # Attach 1500 document_entities rows to put it over the 1000 threshold.
        _attach_doc_entities(conn, heavy_id, 1500)

        # Staging run for the same source that does NOT include the heavy entity.
        # Use a larger staging set so shrinkage is not the reason.
        run_id = _create_run(conn, TEST_SOURCE_ORPHAN)
        _insert_staging_entities(conn, run_id, TEST_SOURCE_ORPHAN, 5, name_prefix="other")

        result = promote_harvest(
            run_id,
            dsn=TEST_DSN,
            floors={TEST_SOURCE_ORPHAN: 0},
            # Allow huge shrinkage so the orphan check is what fires.
            canonical_shrinkage_max=1.0,
            alias_shrinkage_max=1.0,
        )
        assert not result.accepted
        assert result.reason == "orphan_violation"
        violations = result.diff.get("orphan_violations") or []
        assert len(violations) >= 1
        assert any(v.get("canonical_name") == "heavy-0" for v in violations)

        with conn.cursor() as cur:
            cur.execute("SELECT status FROM harvest_runs WHERE id = %s", (run_id,))
            assert cur.fetchone()[0] == "rejected_by_diff"


class TestPerSourceFloors:
    def test_known_broken_source_has_floor_zero(self):
        assert PER_SOURCE_FLOORS["SsODNet"] == 0
        assert PER_SOURCE_FLOORS["CMR"] == 0
        assert PER_SOURCE_FLOORS["SBDB"] == 0

    def test_per_source_floors_match_prd(self):
        assert PER_SOURCE_FLOORS["VizieR"] == 55000
        assert PER_SOURCE_FLOORS["GCMD"] == 9000
        assert PER_SOURCE_FLOORS["PwC"] == 7500
        assert PER_SOURCE_FLOORS["ASCL"] == 3500
        assert PER_SOURCE_FLOORS["PhySH"] == 3500
        assert PER_SOURCE_FLOORS["AAS"] == 600
        assert PER_SOURCE_FLOORS["SPASE"] == 200
