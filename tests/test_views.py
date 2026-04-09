"""Tests for materialized view refresh helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import psycopg.errors
import pytest

from scix.views import AGENT_VIEWS, RefreshResult, refresh_all_views, refresh_view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn(autocommit: bool = True) -> MagicMock:
    """Return a mock psycopg connection with a cursor context manager."""
    conn = MagicMock(spec=psycopg.Connection)
    conn.autocommit = autocommit
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


# ---------------------------------------------------------------------------
# refresh_view
# ---------------------------------------------------------------------------


class TestRefreshView:
    def test_success_returns_result(self) -> None:
        conn = _make_conn()
        result = refresh_view(conn, "agent_document_context")

        assert result.success is True
        assert result.view_name == "agent_document_context"
        assert result.duration_s >= 0
        assert result.error is None

    def test_calls_set_parallel_workers(self) -> None:
        conn = _make_conn()
        refresh_view(conn, "agent_document_context")

        cur = conn.cursor().__enter__()
        executed = [str(c) for c in cur.execute.call_args_list]
        assert any("max_parallel_maintenance_workers" in s for s in executed)
        assert any("maintenance_work_mem" in s for s in executed)

    def test_undefined_table_returns_failure(self) -> None:
        conn = _make_conn()
        cur = conn.cursor().__enter__()
        cur.execute.side_effect = [
            None,  # SET max_parallel_maintenance_workers
            None,  # SET maintenance_work_mem
            psycopg.errors.UndefinedTable("relation does not exist"),
        ]

        result = refresh_view(conn, "nonexistent_view")

        assert result.success is False
        assert result.view_name == "nonexistent_view"
        assert "does not exist" in result.error

    def test_no_unique_index_returns_failure(self) -> None:
        conn = _make_conn()
        cur = conn.cursor().__enter__()
        cur.execute.side_effect = [
            None,  # SET max_parallel_maintenance_workers
            None,  # SET maintenance_work_mem
            psycopg.errors.ObjectNotInPrerequisiteState("no unique index"),
        ]

        result = refresh_view(conn, "bad_view")

        assert result.success is False
        assert "unique index" in result.error

    def test_generic_error_returns_failure(self) -> None:
        conn = _make_conn()
        cur = conn.cursor().__enter__()
        cur.execute.side_effect = [
            None,
            None,
            psycopg.Error("something broke"),
        ]

        result = refresh_view(conn, "agent_entity_context")

        assert result.success is False
        assert "something broke" in result.error


# ---------------------------------------------------------------------------
# refresh_all_views
# ---------------------------------------------------------------------------


class TestRefreshAllViews:
    def test_refreshes_all_three_views(self) -> None:
        conn = _make_conn()
        results = refresh_all_views(conn)

        assert len(results) == 3
        assert [r.view_name for r in results] == list(AGENT_VIEWS)
        assert all(r.success for r in results)

    def test_continues_on_failure(self) -> None:
        """If one view fails, the remaining views should still be attempted."""
        conn = _make_conn()
        cur = conn.cursor().__enter__()

        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail on the 5th execute (3rd call of 2nd view: SET, SET, REFRESH)
            # View 1: calls 1,2,3 — View 2: calls 4,5,6
            if call_count == 6:
                raise psycopg.errors.UndefinedTable("missing")

        cur.execute.side_effect = _side_effect

        results = refresh_all_views(conn)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True


# ---------------------------------------------------------------------------
# RefreshResult
# ---------------------------------------------------------------------------


class TestRefreshResult:
    def test_frozen(self) -> None:
        r = RefreshResult(view_name="v", duration_s=1.0, success=True)
        with pytest.raises(AttributeError):
            r.view_name = "other"  # type: ignore[misc]

    def test_default_error_is_none(self) -> None:
        r = RefreshResult(view_name="v", duration_s=0.5, success=True)
        assert r.error is None


# ---------------------------------------------------------------------------
# AGENT_VIEWS ordering
# ---------------------------------------------------------------------------


class TestAgentViews:
    def test_contains_all_three(self) -> None:
        assert "agent_document_context" in AGENT_VIEWS
        assert "agent_entity_context" in AGENT_VIEWS
        assert "agent_dataset_context" in AGENT_VIEWS

    def test_document_first(self) -> None:
        assert AGENT_VIEWS[0] == "agent_document_context"
