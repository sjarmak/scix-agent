"""Tests for entity dictionary module.

Unit tests verify function signatures and argument handling.
Integration tests (marked @pytest.mark.integration) require a running scix
database with migration 013 applied.
"""

from __future__ import annotations

import psycopg
import pytest
from helpers import DSN

from scix.dictionary import bulk_load, get_stats, lookup, upsert_entry

# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "id",
    "canonical_name",
    "entity_type",
    "source",
    "external_id",
    "aliases",
    "metadata",
}


def _has_entity_dictionary(conn: psycopg.Connection) -> bool:
    """Check if entity_dictionary table exists (migration 013 applied)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'entity_dictionary'
        """)
        return cur.fetchone()[0] == 1


@pytest.fixture()
def db_conn():
    """Provide a database connection, skip if unavailable or table missing."""
    try:
        conn = psycopg.connect(DSN)
    except psycopg.OperationalError:
        pytest.skip("Database not available")
        return

    if not _has_entity_dictionary(conn):
        conn.close()
        pytest.skip("entity_dictionary table not found (migration 013 not applied)")
        return

    yield conn

    # Clean up test data
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM entity_dictionary WHERE source IN ('test', 'test-bulk')")
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestUpsertEntry:
    """Integration: upsert_entry inserts and updates correctly."""

    def test_insert_returns_dict_with_expected_keys(self, db_conn: psycopg.Connection) -> None:
        result = upsert_entry(
            db_conn,
            canonical_name="Astropy",
            entity_type="software",
            source="test",
            external_id="Q4854639",
            aliases=["astropy"],
            metadata={"language": "Python"},
        )
        assert isinstance(result, dict)
        assert EXPECTED_KEYS.issubset(result.keys())
        assert result["canonical_name"] == "Astropy"
        assert result["entity_type"] == "software"
        assert result["source"] == "test"
        assert result["external_id"] == "Q4854639"
        assert "astropy" in result["aliases"]

    def test_upsert_updates_on_conflict(self, db_conn: psycopg.Connection) -> None:
        upsert_entry(
            db_conn,
            canonical_name="Astropy",
            entity_type="software",
            source="test",
            external_id="Q4854639",
            aliases=["astropy"],
        )
        updated = upsert_entry(
            db_conn,
            canonical_name="Astropy",
            entity_type="software",
            source="test",
            external_id="Q4854639-v2",
            aliases=["astropy", "AstroPy"],
        )
        assert updated["external_id"] == "Q4854639-v2"
        assert "AstroPy" in updated["aliases"]

    def test_defaults_for_optional_fields(self, db_conn: psycopg.Connection) -> None:
        result = upsert_entry(
            db_conn,
            canonical_name="TestEntity",
            entity_type="instrument",
            source="test",
        )
        assert result["external_id"] is None
        assert result["aliases"] == []
        assert result["metadata"] == {}


@pytest.mark.integration
class TestLookup:
    """Integration: lookup finds entries by canonical name and alias."""

    def test_lookup_by_canonical_name(self, db_conn: psycopg.Connection) -> None:
        upsert_entry(
            db_conn,
            canonical_name="Astropy",
            entity_type="software",
            source="test",
            external_id="Q4854639",
            aliases=["astropy"],
            metadata={"language": "Python"},
        )
        result = lookup(db_conn, "Astropy", entity_type="software")
        assert result is not None
        assert EXPECTED_KEYS.issubset(result.keys())
        assert result["canonical_name"] == "Astropy"
        assert result["entity_type"] == "software"
        assert result["source"] == "test"
        assert result["external_id"] == "Q4854639"
        assert "astropy" in result["aliases"]
        assert result["metadata"] == {"language": "Python"}

    def test_lookup_case_insensitive(self, db_conn: psycopg.Connection) -> None:
        upsert_entry(
            db_conn,
            canonical_name="Astropy",
            entity_type="software",
            source="test",
        )
        result = lookup(db_conn, "astropy", entity_type="software")
        assert result is not None
        assert result["canonical_name"] == "Astropy"

    def test_lookup_by_alias(self, db_conn: psycopg.Connection) -> None:
        upsert_entry(
            db_conn,
            canonical_name="Hubble Space Telescope",
            entity_type="instrument",
            source="test",
            aliases=["HST", "Hubble"],
        )
        result = lookup(db_conn, "HST", entity_type="instrument")
        assert result is not None
        assert result["canonical_name"] == "Hubble Space Telescope"

    def test_lookup_not_found(self, db_conn: psycopg.Connection) -> None:
        result = lookup(db_conn, "NonexistentEntity12345", entity_type="software")
        assert result is None

    def test_lookup_without_entity_type(self, db_conn: psycopg.Connection) -> None:
        upsert_entry(
            db_conn,
            canonical_name="Astropy",
            entity_type="software",
            source="test",
        )
        result = lookup(db_conn, "Astropy")
        assert result is not None
        assert result["canonical_name"] == "Astropy"


@pytest.mark.integration
class TestBulkLoad:
    """Integration: bulk_load inserts multiple entries with ON CONFLICT."""

    def test_bulk_load_count(self, db_conn: psycopg.Connection) -> None:
        entries = [
            {
                "canonical_name": "NumPy",
                "entity_type": "software",
                "source": "test-bulk",
                "aliases": ["numpy"],
            },
            {
                "canonical_name": "SciPy",
                "entity_type": "software",
                "source": "test-bulk",
                "aliases": ["scipy"],
            },
            {
                "canonical_name": "Chandra",
                "entity_type": "mission",
                "source": "test-bulk",
                "external_id": "Q219615",
            },
        ]
        count = bulk_load(db_conn, entries)
        assert count == 3

    def test_bulk_load_upsert(self, db_conn: psycopg.Connection) -> None:
        entries = [
            {
                "canonical_name": "NumPy",
                "entity_type": "software",
                "source": "test-bulk",
                "aliases": ["numpy"],
            },
        ]
        bulk_load(db_conn, entries)
        # Load again with updated aliases
        entries[0]["aliases"] = ["numpy", "np"]
        count = bulk_load(db_conn, entries)
        assert count == 1

        result = lookup(db_conn, "NumPy", entity_type="software")
        assert result is not None
        assert "np" in result["aliases"]

    def test_bulk_load_empty(self, db_conn: psycopg.Connection) -> None:
        count = bulk_load(db_conn, [])
        assert count == 0


@pytest.mark.integration
class TestGetStats:
    """Integration: get_stats returns summary counts."""

    def test_stats_structure(self, db_conn: psycopg.Connection) -> None:
        upsert_entry(
            db_conn,
            canonical_name="Astropy",
            entity_type="software",
            source="test",
        )
        upsert_entry(
            db_conn,
            canonical_name="Chandra",
            entity_type="mission",
            source="test",
        )
        stats = get_stats(db_conn)
        assert "total" in stats
        assert "by_type" in stats
        assert isinstance(stats["total"], int)
        assert isinstance(stats["by_type"], dict)
        assert stats["total"] >= 2

    def test_stats_by_type_has_entries(self, db_conn: psycopg.Connection) -> None:
        upsert_entry(
            db_conn,
            canonical_name="Astropy",
            entity_type="software",
            source="test",
        )
        stats = get_stats(db_conn)
        assert "software" in stats["by_type"]
        assert stats["by_type"]["software"] >= 1
