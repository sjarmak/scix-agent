"""Tests for entity merge/split audit log."""

from __future__ import annotations

import os
from datetime import datetime

import psycopg
import pytest

from helpers import DSN, is_production_dsn

from scix.entity_audit import (
    MergeEntry,
    SplitEntry,
    get_audit_history,
    get_merge_history,
    get_split_history,
    record_merge,
    record_split,
)

TEST_DSN = os.environ.get("SCIX_TEST_DSN")

_skip_destructive = pytest.mark.skipif(
    TEST_DSN is None,
    reason="Entity audit tests require SCIX_TEST_DSN (writes to audit tables).",
)


# ---------------------------------------------------------------------------
# Unit tests (no database)
# ---------------------------------------------------------------------------


class TestMergeEntry:
    def test_frozen(self) -> None:
        entry = MergeEntry(
            id=1,
            old_entity_id=10,
            new_entity_id=20,
            reason="duplicate",
            merged_by="test",
            merged_at=datetime(2026, 1, 1),
        )
        with pytest.raises(AttributeError):
            entry.reason = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        entry = MergeEntry(
            id=1,
            old_entity_id=10,
            new_entity_id=20,
            reason=None,
            merged_by=None,
            merged_at=datetime(2026, 1, 1),
        )
        assert entry.old_entity_id == 10
        assert entry.new_entity_id == 20
        assert entry.reason is None


class TestSplitEntry:
    def test_frozen(self) -> None:
        entry = SplitEntry(
            id=1,
            parent_entity_id=100,
            child_entity_ids=(200, 201),
            reason="disambiguation",
            split_by="test",
            split_at=datetime(2026, 1, 1),
        )
        with pytest.raises(AttributeError):
            entry.reason = "changed"  # type: ignore[misc]

    def test_child_ids_are_tuple(self) -> None:
        entry = SplitEntry(
            id=1,
            parent_entity_id=100,
            child_entity_ids=(200, 201, 202),
            reason=None,
            split_by=None,
            split_at=datetime(2026, 1, 1),
        )
        assert isinstance(entry.child_entity_ids, tuple)
        assert len(entry.child_entity_ids) == 3


# ---------------------------------------------------------------------------
# Integration tests (require SCIX_TEST_DSN)
# ---------------------------------------------------------------------------


@_skip_destructive
@pytest.mark.integration
class TestEntityAuditIntegration:
    @pytest.fixture()
    def conn(self):
        if is_production_dsn(TEST_DSN):
            pytest.skip("Refuses to write test data to production. Set SCIX_TEST_DSN.")
        with psycopg.connect(TEST_DSN) as c:
            c.autocommit = False
            yield c

    @pytest.fixture(autouse=True)
    def _setup_entities(self, conn):
        """Create test entities and clean up audit tables after."""
        with conn.cursor() as cur:
            # Insert test entities
            cur.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, source)
                VALUES
                    ('Test Entity A', 'mission', 'test'),
                    ('Test Entity B', 'mission', 'test'),
                    ('Test Child 1', 'mission', 'test'),
                    ('Test Child 2', 'mission', 'test')
                ON CONFLICT DO NOTHING
                """
            )
            # Get IDs
            cur.execute(
                "SELECT id FROM entities WHERE canonical_name = 'Test Entity A' AND source = 'test'"
            )
            self.entity_a_id = cur.fetchone()[0]
            cur.execute(
                "SELECT id FROM entities WHERE canonical_name = 'Test Entity B' AND source = 'test'"
            )
            self.entity_b_id = cur.fetchone()[0]
            cur.execute(
                "SELECT id FROM entities WHERE canonical_name = 'Test Child 1' AND source = 'test'"
            )
            self.child1_id = cur.fetchone()[0]
            cur.execute(
                "SELECT id FROM entities WHERE canonical_name = 'Test Child 2' AND source = 'test'"
            )
            self.child2_id = cur.fetchone()[0]
        conn.commit()

        yield

        # Cleanup
        with conn.cursor() as cur:
            cur.execute("DELETE FROM entity_merge_log WHERE merged_by = 'test_audit'")
            cur.execute("DELETE FROM entity_split_log WHERE split_by = 'test_audit'")
            cur.execute("DELETE FROM entities WHERE source = 'test'")
        conn.commit()

    def test_record_merge(self, conn) -> None:
        entry = record_merge(
            conn,
            self.entity_a_id,
            self.entity_b_id,
            reason="duplicate entries",
            merged_by="test_audit",
        )
        assert entry.old_entity_id == self.entity_a_id
        assert entry.new_entity_id == self.entity_b_id
        assert entry.reason == "duplicate entries"
        assert entry.merged_by == "test_audit"
        assert isinstance(entry.merged_at, datetime)

    def test_record_split(self, conn) -> None:
        entry = record_split(
            conn,
            self.entity_a_id,
            [self.child1_id, self.child2_id],
            reason="disambiguation",
            split_by="test_audit",
        )
        assert entry.parent_entity_id == self.entity_a_id
        assert entry.child_entity_ids == (self.child1_id, self.child2_id)
        assert entry.reason == "disambiguation"

    def test_get_merge_history(self, conn) -> None:
        record_merge(
            conn,
            self.entity_a_id,
            self.entity_b_id,
            reason="dup",
            merged_by="test_audit",
        )
        history = get_merge_history(conn, self.entity_a_id)
        assert len(history) >= 1
        assert any(e.old_entity_id == self.entity_a_id for e in history)

        # Also findable via new_entity_id
        history_b = get_merge_history(conn, self.entity_b_id)
        assert len(history_b) >= 1

    def test_get_split_history(self, conn) -> None:
        record_split(
            conn,
            self.entity_a_id,
            [self.child1_id, self.child2_id],
            reason="split",
            split_by="test_audit",
        )
        # Findable via parent
        history = get_split_history(conn, self.entity_a_id)
        assert len(history) >= 1

        # Findable via child
        history_child = get_split_history(conn, self.child1_id)
        assert len(history_child) >= 1

    def test_get_audit_history(self, conn) -> None:
        record_merge(
            conn,
            self.entity_a_id,
            self.entity_b_id,
            reason="merge",
            merged_by="test_audit",
        )
        record_split(
            conn,
            self.entity_a_id,
            [self.child1_id, self.child2_id],
            reason="split",
            split_by="test_audit",
        )
        result = get_audit_history(conn, self.entity_a_id)
        assert "merges" in result
        assert "splits" in result
        assert len(result["merges"]) >= 1
        assert len(result["splits"]) >= 1
