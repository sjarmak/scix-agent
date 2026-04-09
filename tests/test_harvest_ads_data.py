"""Tests for scripts/harvest_ads_data_field.py.

Unit tests verify build_entries() and fetch logic.
Integration tests (marked @pytest.mark.integration) require a running scix
database with migrations 012 and 013 applied.
"""

from __future__ import annotations

import psycopg
import pytest
from helpers import DSN, is_production_dsn

import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from harvest_ads_data_field import (  # noqa: E402
    ENTITY_TYPE,
    SOURCE,
    build_entries,
    fetch_data_sources,
    main,
)
from scix.dictionary import lookup  # noqa: E402

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestBuildEntries:
    """Unit: build_entries converts source rows into entity_dictionary dicts."""

    def test_empty_input(self) -> None:
        assert build_entries([]) == []

    def test_single_entry(self) -> None:
        sources = [{"source_label": "CDS", "paper_count": 42}]
        entries = build_entries(sources)
        assert len(entries) == 1
        assert entries[0]["canonical_name"] == "CDS"
        assert entries[0]["entity_type"] == ENTITY_TYPE
        assert entries[0]["source"] == SOURCE
        assert entries[0]["metadata"] == {"paper_count": 42}

    def test_multiple_entries_preserve_order(self) -> None:
        sources = [
            {"source_label": "CDS", "paper_count": 100},
            {"source_label": "MAST", "paper_count": 50},
            {"source_label": "NExScI", "paper_count": 10},
        ]
        entries = build_entries(sources)
        assert len(entries) == 3
        assert entries[0]["canonical_name"] == "CDS"
        assert entries[1]["canonical_name"] == "MAST"
        assert entries[2]["canonical_name"] == "NExScI"

    def test_metadata_contains_paper_count(self) -> None:
        sources = [{"source_label": "HEASARC", "paper_count": 999}]
        entries = build_entries(sources)
        assert "paper_count" in entries[0]["metadata"]
        assert entries[0]["metadata"]["paper_count"] == 999


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


def _has_required_tables(conn: psycopg.Connection) -> bool:
    """Check if papers and entity_dictionary tables exist."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('papers', 'entity_dictionary')
        """)
        return cur.fetchone()[0] == 2


def _has_data_column(conn: psycopg.Connection) -> bool:
    """Check if papers.data column exists (migration 012)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'papers' AND column_name = 'data'
            )
        """)
        return cur.fetchone()[0]


TEST_BIBCODES = [
    "2025test.harvest..001A",
    "2025test.harvest..002B",
    "2025test.harvest..003C",
]


@pytest.fixture()
def db_conn():
    """Provide a database connection, skip if unavailable or tables missing."""
    if is_production_dsn(DSN):
        pytest.skip("Refuses to write test data to production. Set SCIX_TEST_DSN.")
    try:
        conn = psycopg.connect(DSN)
    except psycopg.OperationalError:
        pytest.skip("Database not available")
        return

    if not _has_required_tables(conn):
        conn.close()
        pytest.skip("Required tables not found (migrations 012+013)")
        return

    if not _has_data_column(conn):
        conn.close()
        pytest.skip("papers.data column not found (migration 012)")
        return

    yield conn

    # Clean up test data
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM entity_dictionary WHERE source = %s",
                (SOURCE + "-test",),
            )
            cur.execute(
                "DELETE FROM papers WHERE bibcode = ANY(%s)",
                (TEST_BIBCODES,),
            )
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


@pytest.fixture()
def seeded_db(db_conn: psycopg.Connection):
    """Insert test papers with data arrays, yield connection."""
    with db_conn.cursor() as cur:
        # Paper 1: CDS and MAST
        cur.execute(
            "INSERT INTO papers (bibcode, data) VALUES (%s, %s) ON CONFLICT (bibcode) DO UPDATE SET data = EXCLUDED.data",
            (TEST_BIBCODES[0], ["CDS:1", "MAST:2"]),
        )
        # Paper 2: CDS and HEASARC
        cur.execute(
            "INSERT INTO papers (bibcode, data) VALUES (%s, %s) ON CONFLICT (bibcode) DO UPDATE SET data = EXCLUDED.data",
            (TEST_BIBCODES[1], ["CDS:3", "HEASARC:1"]),
        )
        # Paper 3: MAST only
        cur.execute(
            "INSERT INTO papers (bibcode, data) VALUES (%s, %s) ON CONFLICT (bibcode) DO UPDATE SET data = EXCLUDED.data",
            (TEST_BIBCODES[2], ["MAST:5"]),
        )
    db_conn.commit()
    return db_conn


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFetchDataSources:
    """Integration: fetch_data_sources queries papers.data[] correctly."""

    def test_returns_list(self, seeded_db: psycopg.Connection) -> None:
        results = fetch_data_sources(seeded_db)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_sources_have_expected_keys(self, seeded_db: psycopg.Connection) -> None:
        results = fetch_data_sources(seeded_db)
        for r in results:
            assert "source_label" in r
            assert "paper_count" in r

    def test_cds_appears_with_count(self, seeded_db: psycopg.Connection) -> None:
        results = fetch_data_sources(seeded_db)
        cds_entries = [r for r in results if r["source_label"] == "CDS:1"]
        # CDS:1 appears in paper 1 only
        assert len(cds_entries) == 1
        assert cds_entries[0]["paper_count"] >= 1

    def test_min_count_filter(self, seeded_db: psycopg.Connection) -> None:
        results = fetch_data_sources(seeded_db, min_count=2)
        # CDS:3 and HEASARC:1 appear in only 1 paper each, should be filtered out
        labels = {r["source_label"] for r in results}
        assert "CDS:3" not in labels
        assert "HEASARC:1" not in labels

    def test_limit(self, seeded_db: psycopg.Connection) -> None:
        results = fetch_data_sources(seeded_db, limit=2)
        assert len(results) <= 2

    def test_ordered_by_count_desc(self, seeded_db: psycopg.Connection) -> None:
        results = fetch_data_sources(seeded_db)
        counts = [r["paper_count"] for r in results]
        assert counts == sorted(counts, reverse=True)


@pytest.mark.integration
class TestMainCLI:
    """Integration: main() CLI runs end-to-end."""

    def test_dry_run(self, seeded_db: psycopg.Connection) -> None:
        # dry-run should not insert into entity_dictionary
        exit_code = main(["--dsn", DSN, "--dry-run"])
        assert exit_code == 0

    def test_load_into_dictionary(self, seeded_db: psycopg.Connection) -> None:
        exit_code = main(["--dsn", DSN])
        assert exit_code == 0

        # Verify entries were loaded
        result = lookup(seeded_db, "CDS:1", entity_type=ENTITY_TYPE)
        assert result is not None
        assert result["source"] == SOURCE
        assert result["metadata"]["paper_count"] >= 1
