"""Integration tests for ``scripts/normalize_canonical_names.py``.

Seeds the test database with a small set of GLiNER ``entities`` rows
that mimic the real corpus's markup-artifact splits (``co<sub>2</sub>``
vs ``co2``, em-dash isotopes, NBSP variants), runs the backfill, and
verifies the merge invariants.

Requires ``SCIX_TEST_DSN`` and refuses to run against the production
DSN. Each test cleans up its own fixture rows on teardown.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from helpers import get_test_dsn

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the script as a module — it lives in scripts/ which is not a
# package, so we load by file path the same way other tests do.
_SCRIPT_PATH = REPO_ROOT / "scripts" / "normalize_canonical_names.py"
_spec = importlib.util.spec_from_file_location(
    "normalize_canonical_names", _SCRIPT_PATH
)
backfill = importlib.util.module_from_spec(_spec)
sys.modules["normalize_canonical_names"] = backfill
assert _spec.loader is not None
_spec.loader.exec_module(backfill)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# All test fixture canonicals are prefixed so cleanup is precise.
FIXTURE_PREFIX = "_norm_test_"


@pytest.fixture
def test_dsn() -> str:
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set or points at production")
    return dsn


@pytest.fixture
def conn(test_dsn: str) -> Iterator[psycopg.Connection]:
    c = psycopg.connect(test_dsn)
    try:
        yield c
    finally:
        c.close()


def _cleanup(conn: psycopg.Connection) -> None:
    """Drop any rows left behind by previous fixture runs."""
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM document_entities
            WHERE entity_id IN (
                SELECT id FROM entities
                WHERE source='gliner' AND canonical_name LIKE %s
            )
            """,
            (f"%{FIXTURE_PREFIX}%",),
        )
        cur.execute(
            "DELETE FROM entity_aliases WHERE alias LIKE %s",
            (f"%{FIXTURE_PREFIX}%",),
        )
        cur.execute(
            """
            DELETE FROM entity_merge_log
            WHERE old_entity_id IN (
                SELECT id FROM entities
                WHERE source='gliner' AND canonical_name LIKE %s
            )
               OR new_entity_id IN (
                SELECT id FROM entities
                WHERE source='gliner' AND canonical_name LIKE %s
            )
            """,
            (f"%{FIXTURE_PREFIX}%", f"%{FIXTURE_PREFIX}%"),
        )
        cur.execute(
            "DELETE FROM entities WHERE source='gliner' AND canonical_name LIKE %s",
            (f"%{FIXTURE_PREFIX}%",),
        )
        cur.execute("DELETE FROM papers WHERE bibcode LIKE '_NORM_TEST_%'")
    conn.commit()


@pytest.fixture
def clean_db(conn: psycopg.Connection) -> Iterator[psycopg.Connection]:
    _cleanup(conn)
    try:
        yield conn
    finally:
        _cleanup(conn)


def _insert_entity(
    conn: psycopg.Connection,
    canonical_name: str,
    entity_type: str,
    *,
    source_version: str = "gliner_test/v1",
) -> int:
    """Insert a gliner entity row, returning its id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entities
                (canonical_name, entity_type, source, source_version)
            VALUES (%s, %s, 'gliner', %s)
            RETURNING id
            """,
            (canonical_name, entity_type, source_version),
        )
        row = cur.fetchone()
    conn.commit()
    return row[0]


def _insert_paper(conn: psycopg.Connection, bibcode: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO papers (bibcode, title, abstract, year) "
            "VALUES (%s, 'fixture', 'body', 2099) "
            "ON CONFLICT (bibcode) DO NOTHING",
            (bibcode,),
        )
    conn.commit()


def _link(
    conn: psycopg.Connection,
    bibcode: str,
    entity_id: int,
    *,
    link_type: str = "abstract_match",
    tier: int = 4,
    confidence: float = 0.9,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO document_entities
                (bibcode, entity_id, link_type, confidence, match_method,
                 tier, tier_version)
            VALUES (%s, %s, %s, %s, 'gliner', %s, 1)
            ON CONFLICT (bibcode, entity_id, link_type, tier) DO NOTHING
            """,
            (bibcode, entity_id, link_type, confidence, tier),
        )
    conn.commit()


def _entity_exists(conn: psycopg.Connection, entity_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM entities WHERE id = %s", (entity_id,))
        return cur.fetchone() is not None


def _entity_canonical(conn: psycopg.Connection, entity_id: int) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (entity_id,))
        row = cur.fetchone()
    return row[0] if row else None


def _aliases_for(conn: psycopg.Connection, entity_id: int) -> list[tuple[str, str | None]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT alias, alias_source FROM entity_aliases WHERE entity_id = %s "
            "ORDER BY alias",
            (entity_id,),
        )
        return list(cur.fetchall())


def _document_entities_for(conn: psycopg.Connection, entity_id: int) -> list[Any]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bibcode, link_type, tier FROM document_entities "
            "WHERE entity_id = %s ORDER BY bibcode, link_type, tier",
            (entity_id,),
        )
        return list(cur.fetchall())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassifyRows:
    """Unit-level: row partitioning into rename-only vs merge groups."""

    def test_unchanged_rows_are_dropped(self) -> None:
        rows = [(1, "co2", "v1"), (2, "iron-59", "v1")]
        rename, merges = backfill._classify_rows(rows)
        assert rename == []
        assert merges == {}

    def test_rename_only_when_no_collision(self) -> None:
        rows = [(1, "co<sub>2</sub>", "v1")]
        rename, merges = backfill._classify_rows(rows)
        assert rename == [(1, "co2")]
        assert merges == {}

    def test_merge_group_when_collision_exists(self) -> None:
        # 'co<sub>2</sub>' and 'co2' both normalize to 'co2'.
        rows = [(1, "co2", "v1"), (2, "co<sub>2</sub>", "v1")]
        rename, merges = backfill._classify_rows(rows)
        assert rename == []
        # Lowest id is the survivor; 2 is the loser.
        assert merges == {"co2": [1, 2]}

    def test_survivor_is_lowest_id(self) -> None:
        rows = [
            (5, "co<sub>2</sub>", "v1"),
            (3, "co2", "v1"),
            (7, "co<sup>2</sup>", "v1"),
        ]
        rename, merges = backfill._classify_rows(rows)
        assert merges == {"co2": [3, 5, 7]}

    def test_unicode_dash_collision(self) -> None:
        rows = [(1, "iron-59", "v1"), (2, "iron–59", "v1")]
        _, merges = backfill._classify_rows(rows)
        assert merges == {"iron-59": [1, 2]}


class TestNormalizeEntityType:
    """End-to-end: seed dirty rows, run pass, verify merges."""

    def test_markup_split_collapses(self, clean_db: psycopg.Connection) -> None:
        bib1 = "_NORM_TEST_001"
        bib2 = "_NORM_TEST_002"
        _insert_paper(clean_db, bib1)
        _insert_paper(clean_db, bib2)

        # Two canonicals in chemical that normalize to the same value:
        # the survivor (lower id, clean name) and the loser (markup).
        survivor_canon = f"{FIXTURE_PREFIX}co2"
        loser_canon = f"{FIXTURE_PREFIX}co<sub>2</sub>"
        sid = _insert_entity(clean_db, survivor_canon, "chemical")
        lid = _insert_entity(clean_db, loser_canon, "chemical")

        _link(clean_db, bib1, sid)
        _link(clean_db, bib2, lid)

        stats = backfill.normalize_entity_type(
            clean_db, "chemical", actor="test"
        )
        clean_db.commit()

        assert stats.merge_groups == 1
        assert stats.rows_merged_away == 1
        assert stats.document_entities_repointed >= 1

        assert _entity_exists(clean_db, sid)
        assert not _entity_exists(clean_db, lid)

        # Survivor's canonical name is the clean form. (Note: prefix
        # contains ``_`` not ``<``, so canonical_name on disk is the
        # already-clean ``_norm_test_co2``.)
        assert _entity_canonical(clean_db, sid) == survivor_canon

        # The loser's mentions were re-pointed.
        de = _document_entities_for(clean_db, sid)
        bibs = {row[0] for row in de}
        assert bib1 in bibs and bib2 in bibs

        # Loser's old canonical was captured as an alias.
        aliases = _aliases_for(clean_db, sid)
        alias_strs = [a for a, _ in aliases]
        assert loser_canon in alias_strs
        # Stamped with our alias_source.
        assert any(src == backfill.ALIAS_SOURCE for _, src in aliases)

    def test_unicode_dash_merge(self, clean_db: psycopg.Connection) -> None:
        bib = "_NORM_TEST_010"
        _insert_paper(clean_db, bib)

        survivor = _insert_entity(
            clean_db, f"{FIXTURE_PREFIX}iron-59", "chemical"
        )
        # En dash – fragment.
        loser = _insert_entity(
            clean_db, f"{FIXTURE_PREFIX}iron–59", "chemical"
        )
        _link(clean_db, bib, loser)

        stats = backfill.normalize_entity_type(
            clean_db, "chemical", actor="test"
        )
        clean_db.commit()

        assert stats.merge_groups >= 1
        assert _entity_exists(clean_db, survivor)
        assert not _entity_exists(clean_db, loser)
        # Mention rolls up to survivor.
        assert any(row[0] == bib for row in _document_entities_for(clean_db, survivor))

    def test_rename_only_no_collision(self, clean_db: psycopg.Connection) -> None:
        # Single dirty entity with no clean counterpart: just rename in place.
        eid = _insert_entity(
            clean_db, f"{FIXTURE_PREFIX}lone<sub>x</sub>", "chemical"
        )
        before_aliases = _aliases_for(clean_db, eid)

        stats = backfill.normalize_entity_type(
            clean_db, "chemical", actor="test"
        )
        clean_db.commit()

        # The row still exists, with its canonical_name rewritten.
        assert _entity_exists(clean_db, eid)
        assert _entity_canonical(clean_db, eid) == f"{FIXTURE_PREFIX}lonex"

        # No merge was recorded for a bare rename.
        assert stats.merge_groups == 0
        assert stats.rows_renamed == 1

        # Aliases unchanged for a bare rename.
        assert _aliases_for(clean_db, eid) == before_aliases

    def test_already_clean_rows_untouched(
        self, clean_db: psycopg.Connection
    ) -> None:
        eid = _insert_entity(clean_db, f"{FIXTURE_PREFIX}clean", "chemical")
        stats = backfill.normalize_entity_type(
            clean_db, "chemical", actor="test"
        )
        clean_db.commit()

        assert stats.rows_unchanged == 1
        assert stats.rows_renamed == 0
        assert stats.merge_groups == 0
        assert _entity_canonical(clean_db, eid) == f"{FIXTURE_PREFIX}clean"

    def test_records_merge_log_entry(
        self, clean_db: psycopg.Connection
    ) -> None:
        survivor = _insert_entity(
            clean_db, f"{FIXTURE_PREFIX}r2", "chemical"
        )
        loser = _insert_entity(
            clean_db, f"{FIXTURE_PREFIX}r<sup>2</sup>", "chemical"
        )

        backfill.normalize_entity_type(clean_db, "chemical", actor="test")
        clean_db.commit()

        with clean_db.cursor() as cur:
            cur.execute(
                "SELECT new_entity_id, reason, merged_by "
                "FROM entity_merge_log WHERE old_entity_id = %s",
                (loser,),
            )
            log = cur.fetchone()
        assert log is not None
        assert log[0] == survivor
        assert log[1] == backfill.MERGE_REASON
        assert "test" in (log[2] or "")

    def test_de_duplication_on_survivor(
        self, clean_db: psycopg.Connection
    ) -> None:
        # Both survivor and loser have a row for the same bibcode;
        # the merge must not violate the composite PK.
        bib = "_NORM_TEST_DUP"
        _insert_paper(clean_db, bib)
        survivor = _insert_entity(
            clean_db, f"{FIXTURE_PREFIX}h2o", "chemical"
        )
        loser = _insert_entity(
            clean_db, f"{FIXTURE_PREFIX}h<sub>2</sub>o", "chemical"
        )
        _link(clean_db, bib, survivor)
        _link(clean_db, bib, loser)

        backfill.normalize_entity_type(clean_db, "chemical", actor="test")
        clean_db.commit()

        de = _document_entities_for(clean_db, survivor)
        # Exactly one row survives for that bibcode (PK-deduplicated).
        assert len([row for row in de if row[0] == bib]) == 1

    def test_dirty_survivor_clean_loser_no_collision(
        self, clean_db: psycopg.Connection
    ) -> None:
        # Regression: when the lowest-id row carries the dirty
        # canonical (e.g. survivor='cocl<sub>2</sub>') and a higher-id
        # row already has the clean canonical (e.g. loser='cocl2'),
        # renaming the survivor must not violate the unique constraint
        # on (canonical_name, entity_type, source). The fix is to
        # delete losers BEFORE updating the survivor's canonical_name.
        bib1 = "_NORM_TEST_DIRTY_SURV_1"
        bib2 = "_NORM_TEST_DIRTY_SURV_2"
        _insert_paper(clean_db, bib1)
        _insert_paper(clean_db, bib2)

        # Insert dirty first (lower id), clean second (higher id).
        dirty_canon = f"{FIXTURE_PREFIX}cocl<sub>2</sub>"
        clean_canon = f"{FIXTURE_PREFIX}cocl2"
        survivor = _insert_entity(clean_db, dirty_canon, "chemical")
        loser = _insert_entity(clean_db, clean_canon, "chemical")
        assert survivor < loser  # ordering invariant for the test

        _link(clean_db, bib1, survivor)
        _link(clean_db, bib2, loser)

        # Should not raise UniqueViolation.
        stats = backfill.normalize_entity_type(
            clean_db, "chemical", actor="test"
        )
        clean_db.commit()

        assert stats.merge_groups == 1
        assert stats.rows_merged_away == 1

        # Survivor remains with the canonical (clean) name.
        assert _entity_exists(clean_db, survivor)
        assert not _entity_exists(clean_db, loser)
        assert _entity_canonical(clean_db, survivor) == clean_canon

        # Both bibcodes rolled up onto the survivor.
        de_bibs = {row[0] for row in _document_entities_for(clean_db, survivor)}
        assert bib1 in de_bibs and bib2 in de_bibs

        # The survivor's pre-rename name was captured as an alias so
        # query-time alias lookup still finds 'cocl<sub>2</sub>'.
        # (Note: in this scenario the dirty form WAS the survivor's
        # canonical, so it does not show up via the loser-canonical
        # alias path; the rename simply rewrites it. Confirm via the
        # alias for the deleted loser's old form, which equalled the
        # new canonical and was therefore skipped.)
        aliases = {a for a, _ in _aliases_for(clean_db, survivor)}
        # The loser's old canonical equals the new canonical, so it
        # is correctly skipped (alias_source='canonical_pre_normalize'
        # only fires when canonical_name <> new_canon).
        assert clean_canon not in aliases

    def test_merge_atomicity_on_rollback(
        self, clean_db: psycopg.Connection
    ) -> None:
        # Regression: record_merge() previously called conn.commit()
        # internally, which broke atomicity for _merge_one_group. If the
        # caller rolls back the transaction (e.g. survivor rename fails
        # mid-loop), the merge log row would survive while losers stayed
        # in entities. With record_merge no longer auto-committing, a
        # rollback must wipe both sides.
        bib = "_NORM_TEST_ATOMIC"
        _insert_paper(clean_db, bib)

        survivor_canon = f"{FIXTURE_PREFIX}atomic2"
        loser_canon = f"{FIXTURE_PREFIX}atomic<sub>2</sub>"
        survivor = _insert_entity(clean_db, survivor_canon, "chemical")
        loser = _insert_entity(clean_db, loser_canon, "chemical")
        _link(clean_db, bib, loser)
        clean_db.commit()

        # Run the merge body but roll back instead of committing.
        backfill.normalize_entity_type(clean_db, "chemical", actor="test")
        clean_db.rollback()

        # Both entities must still exist (rollback restored the loser).
        assert _entity_exists(clean_db, survivor)
        assert _entity_exists(clean_db, loser)
        # No audit row should be visible after rollback.
        with clean_db.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM entity_merge_log "
                "WHERE old_entity_id = %s AND new_entity_id = %s",
                (loser, survivor),
            )
            assert cur.fetchone() is None
        # Loser's document_entities row also rolled back to the loser.
        de = _document_entities_for(clean_db, loser)
        assert any(row[0] == bib for row in de)
