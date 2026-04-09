"""Tests for incremental sync manager."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from helpers import DSN, is_production_dsn

from scix.sync_manager import (
    DEFAULT_CADENCE_HOURS,
    SourceStatus,
    format_sync_status,
    get_last_sync,
    needs_refresh,
    sync_status,
)

TEST_DSN = os.environ.get("SCIX_TEST_DSN")


# ---------------------------------------------------------------------------
# Unit tests (no database)
# ---------------------------------------------------------------------------


class TestSourceStatus:
    def test_frozen(self) -> None:
        s = SourceStatus(
            source="spdf",
            last_sync=datetime.now(timezone.utc),
            cadence=timedelta(hours=24),
            is_stale=False,
            next_sync=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        with pytest.raises(AttributeError):
            s.source = "other"  # type: ignore[misc]


class TestFormatSyncStatus:
    def test_empty(self) -> None:
        assert format_sync_status([]) == "No harvest sources found."

    def test_formats_table(self) -> None:
        now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
        statuses = [
            SourceStatus(
                source="spdf",
                last_sync=now,
                cadence=timedelta(hours=24),
                is_stale=False,
                next_sync=now + timedelta(hours=24),
            ),
            SourceStatus(
                source="gcmd",
                last_sync=None,
                cadence=timedelta(hours=48),
                is_stale=True,
                next_sync=None,
            ),
        ]
        output = format_sync_status(statuses)
        assert "spdf" in output
        assert "gcmd" in output
        assert "STALE" in output
        assert "OK" in output
        assert "never" in output


class TestCadenceConfig:
    def test_env_override(self) -> None:
        from scix.sync_manager import _cadence_for

        with patch.dict(os.environ, {"SCIX_SYNC_CADENCE_SPDF": "48"}):
            assert _cadence_for("spdf") == timedelta(hours=48)

    def test_default_cadence(self) -> None:
        from scix.sync_manager import _cadence_for

        # Clear any env override
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SCIX_SYNC_CADENCE_TEST_SRC", None)
            assert _cadence_for("test_src") == timedelta(hours=DEFAULT_CADENCE_HOURS)


# ---------------------------------------------------------------------------
# Integration tests (require SCIX_TEST_DSN)
# ---------------------------------------------------------------------------

_skip_destructive = pytest.mark.skipif(
    TEST_DSN is None,
    reason="Sync manager tests require SCIX_TEST_DSN.",
)


@_skip_destructive
@pytest.mark.integration
class TestSyncManagerIntegration:
    @pytest.fixture()
    def conn(self):
        if is_production_dsn(TEST_DSN):
            pytest.skip("Refuses to write test data to production.")
        with psycopg.connect(TEST_DSN) as c:
            c.autocommit = False
            yield c

    @pytest.fixture(autouse=True)
    def _cleanup(self, conn):
        yield
        with conn.cursor() as cur:
            cur.execute("DELETE FROM harvest_runs WHERE source LIKE 'test_sync_%%'")
        conn.commit()

    def test_no_runs_returns_none(self, conn) -> None:
        assert get_last_sync(conn, "test_sync_nonexistent") is None

    def test_needs_refresh_no_runs(self, conn) -> None:
        assert needs_refresh(conn, "test_sync_never") is True

    def test_completed_run_tracked(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO harvest_runs (source, status, finished_at)
                VALUES ('test_sync_src', 'completed', NOW())
                """
            )
        conn.commit()

        last = get_last_sync(conn, "test_sync_src")
        assert last is not None
        assert (datetime.now(timezone.utc) - last).total_seconds() < 60

    def test_needs_refresh_fresh(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO harvest_runs (source, status, finished_at)
                VALUES ('test_sync_fresh', 'completed', NOW())
                """
            )
        conn.commit()

        assert needs_refresh(conn, "test_sync_fresh") is False

    def test_needs_refresh_stale(self, conn) -> None:
        stale_time = datetime.now(timezone.utc) - timedelta(hours=48)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO harvest_runs (source, status, finished_at)
                VALUES ('test_sync_stale', 'completed', %s)
                """,
                (stale_time,),
            )
        conn.commit()

        assert needs_refresh(conn, "test_sync_stale") is True

    def test_sync_status_multiple_sources(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO harvest_runs (source, status, finished_at) VALUES
                ('test_sync_a', 'completed', NOW()),
                ('test_sync_b', 'completed', NOW() - INTERVAL '48 hours')
                """
            )
        conn.commit()

        statuses = sync_status(conn, sources=["test_sync_a", "test_sync_b"])
        assert len(statuses) == 2
        by_source = {s.source: s for s in statuses}
        assert by_source["test_sync_a"].is_stale is False
        assert by_source["test_sync_b"].is_stale is True

    def test_failed_runs_ignored(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO harvest_runs (source, status, finished_at)
                VALUES ('test_sync_fail', 'failed', NOW())
                """
            )
        conn.commit()

        assert get_last_sync(conn, "test_sync_fail") is None
        assert needs_refresh(conn, "test_sync_fail") is True
