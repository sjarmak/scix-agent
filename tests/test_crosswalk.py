"""Tests for SPDF-SPASE crosswalk."""

from __future__ import annotations

import os

import psycopg
import pytest

from helpers import DSN, is_production_dsn

from scix.crosswalk import (
    CrosswalkEntry,
    lookup_by_spase_id,
    lookup_by_spdf_id,
    upsert_crosswalk,
)

TEST_DSN = os.environ.get("SCIX_TEST_DSN")

_skip_destructive = pytest.mark.skipif(
    TEST_DSN is None,
    reason="Crosswalk tests require SCIX_TEST_DSN (writes to crosswalk table).",
)


# ---------------------------------------------------------------------------
# Unit tests (no database)
# ---------------------------------------------------------------------------


class TestCrosswalkEntry:
    def test_frozen(self) -> None:
        entry = CrosswalkEntry(id=1, spdf_id="AC_H2_MFI", spase_id="spase://foo", source="test")
        with pytest.raises(AttributeError):
            entry.spdf_id = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        entry = CrosswalkEntry(
            id=1,
            spdf_id="AC_H2_MFI",
            spase_id="spase://NASA/NumericalData/ACE/MAG/L2/PT16S",
            source="spdf_harvest",
        )
        assert entry.spdf_id == "AC_H2_MFI"
        assert entry.spase_id.startswith("spase://")


# ---------------------------------------------------------------------------
# Integration tests (require SCIX_TEST_DSN)
# ---------------------------------------------------------------------------


@_skip_destructive
@pytest.mark.integration
class TestCrosswalkIntegration:
    @pytest.fixture()
    def conn(self):
        if is_production_dsn(TEST_DSN):
            pytest.skip("Refuses to write test data to production. Set SCIX_TEST_DSN.")
        with psycopg.connect(TEST_DSN) as c:
            c.autocommit = False
            yield c

    @pytest.fixture(autouse=True)
    def _cleanup(self, conn):
        yield
        with conn.cursor() as cur:
            cur.execute("DELETE FROM spdf_spase_crosswalk WHERE source = 'test_crosswalk'")
        conn.commit()

    def test_upsert_creates_entry(self, conn) -> None:
        entry = upsert_crosswalk(
            conn,
            "TEST_DATASET_1",
            "spase://TEST/NumericalData/Test/1",
            source="test_crosswalk",
        )
        assert entry.spdf_id == "TEST_DATASET_1"
        assert entry.spase_id == "spase://TEST/NumericalData/Test/1"
        assert entry.source == "test_crosswalk"

    def test_upsert_idempotent(self, conn) -> None:
        e1 = upsert_crosswalk(
            conn,
            "TEST_DATASET_2",
            "spase://TEST/NumericalData/Test/2",
            source="test_crosswalk",
        )
        e2 = upsert_crosswalk(
            conn,
            "TEST_DATASET_2",
            "spase://TEST/NumericalData/Test/2",
            source="test_crosswalk",
        )
        assert e1.id == e2.id

    def test_lookup_by_spdf_id(self, conn) -> None:
        upsert_crosswalk(
            conn,
            "TEST_LOOKUP_SPDF",
            "spase://TEST/ND/Lookup/1",
            source="test_crosswalk",
        )
        results = lookup_by_spdf_id(conn, "TEST_LOOKUP_SPDF")
        assert len(results) >= 1
        assert any(r.spase_id == "spase://TEST/ND/Lookup/1" for r in results)

    def test_lookup_by_spase_id(self, conn) -> None:
        upsert_crosswalk(
            conn,
            "TEST_LOOKUP_SPASE",
            "spase://TEST/ND/Lookup/2",
            source="test_crosswalk",
        )
        results = lookup_by_spase_id(conn, "spase://TEST/ND/Lookup/2")
        assert len(results) >= 1
        assert any(r.spdf_id == "TEST_LOOKUP_SPASE" for r in results)

    def test_lookup_empty(self, conn) -> None:
        results = lookup_by_spdf_id(conn, "NONEXISTENT_DATASET_XYZ")
        assert results == []

    def test_multiple_spase_for_one_spdf(self, conn) -> None:
        upsert_crosswalk(
            conn,
            "TEST_MULTI",
            "spase://TEST/ND/Multi/1",
            source="test_crosswalk",
        )
        upsert_crosswalk(
            conn,
            "TEST_MULTI",
            "spase://TEST/ND/Multi/2",
            source="test_crosswalk",
        )
        results = lookup_by_spdf_id(conn, "TEST_MULTI")
        assert len(results) == 2
