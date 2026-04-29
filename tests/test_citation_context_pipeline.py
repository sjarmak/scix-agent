"""Tests for ``run_pipeline`` shard + ingest_log integration (PRD 79n.1).

Covers the pipeline-level deltas: shard predicate injection into the
SELECT, and ``ingest_log`` start/finish bookkeeping.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix import citation_context as cc

# ---------------------------------------------------------------------------
# _build_papers_select — pure SQL composition
# ---------------------------------------------------------------------------


class TestBuildPapersSelect:
    def test_no_shard_no_limit_returns_base_query(self) -> None:
        sql, params = cc._build_papers_select(shard=None, limit=None)
        assert "FROM papers" in sql
        assert "mod(hashtext" not in sql
        assert "LIMIT" not in sql.upper()
        assert params == []

    def test_shard_appends_mod_hashtext_predicate(self) -> None:
        sql, params = cc._build_papers_select(shard=(2, 4), limit=None)
        # Predicate uses mod() + hashtext() per PRD spec, parameterized.
        assert "mod(hashtext(p.bibcode), %s) = %s" in sql
        # Params are (total_shards, shard_index) per the literal predicate above.
        assert params == [4, 2]

    def test_limit_appends_param(self) -> None:
        sql, params = cc._build_papers_select(shard=None, limit=1000)
        assert sql.rstrip().upper().endswith("LIMIT %S")
        assert params == [1000]

    def test_shard_and_limit_compose(self) -> None:
        sql, params = cc._build_papers_select(shard=(0, 4), limit=500)
        assert "mod(hashtext(p.bibcode), %s) = %s" in sql
        assert sql.rstrip().upper().endswith("LIMIT %S")
        # Shard params come before LIMIT param so ordering matches placeholders.
        assert params == [4, 0, 500]

    def test_rejects_invalid_shard_index(self) -> None:
        # Defence in depth: even if CLI parser accepted bad input, the SQL
        # builder should refuse.
        with pytest.raises(ValueError):
            cc._build_papers_select(shard=(5, 4), limit=None)

    def test_rejects_zero_total(self) -> None:
        with pytest.raises(ValueError):
            cc._build_papers_select(shard=(0, 0), limit=None)

    def test_rejects_negative_index(self) -> None:
        with pytest.raises(ValueError):
            cc._build_papers_select(shard=(-1, 4), limit=None)


# ---------------------------------------------------------------------------
# run_pipeline ingest_log integration
# ---------------------------------------------------------------------------


class _FakeServerCursor:
    """Cursor stand-in for ``conn.cursor(name=...)`` server-side cursors.

    Yields no rows so the pipeline does no work — we're testing the
    bookkeeping wrapper, not extraction.
    """

    def __enter__(self) -> "_FakeServerCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.last_sql = sql
        self.last_params = params

    def __iter__(self) -> "_FakeServerCursor":
        return self

    def __next__(self) -> Any:
        raise StopIteration


class _FakeConn:
    """Minimal psycopg connection double for run_pipeline."""

    def __init__(self) -> None:
        self.closed = False
        self.commit_count = 0
        self._server_cursor = _FakeServerCursor()
        self._client_cursor = MagicMock()
        # cursor() with no args returns the client-side cursor; with name=
        # returns the server-side cursor used for streaming reads.
        self._client_cursor.__enter__ = MagicMock(return_value=self._client_cursor)
        self._client_cursor.__exit__ = MagicMock(return_value=None)

    def cursor(self, name: str | None = None) -> Any:
        if name is not None:
            return self._server_cursor
        return self._client_cursor

    def commit(self) -> None:
        self.commit_count += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def fake_conns() -> tuple[_FakeConn, _FakeConn]:
    return (_FakeConn(), _FakeConn())


class TestRunPipelineIngestLog:
    def test_ingest_log_filename_records_start_and_finish(
        self, fake_conns: tuple[_FakeConn, _FakeConn]
    ) -> None:
        read_conn, write_conn = fake_conns

        log = MagicMock()

        # Two-call sequence: read connection first, then write connection.
        seq = iter([read_conn, write_conn])
        with patch.object(cc, "get_connection", lambda dsn=None: next(seq)), patch.object(
            cc, "IngestLog", return_value=log
        ):
            total = cc.run_pipeline(
                dsn="dbname=scix_test",
                batch_size=1000,
                limit=None,
                ingest_log_filename="citctx_full_backfill_2026_shard_0_of_4",
            )

        assert total == 0  # no rows in fake cursor
        log.start.assert_called_once_with("citctx_full_backfill_2026_shard_0_of_4")
        log.finish.assert_called_once_with("citctx_full_backfill_2026_shard_0_of_4")
        log.update_counts.assert_called()

    def test_no_ingest_log_when_filename_omitted(
        self, fake_conns: tuple[_FakeConn, _FakeConn]
    ) -> None:
        read_conn, write_conn = fake_conns
        seq = iter([read_conn, write_conn])
        with patch.object(cc, "get_connection", lambda dsn=None: next(seq)), patch.object(
            cc, "IngestLog"
        ) as mock_log_cls:
            cc.run_pipeline(dsn="dbname=scix_test", batch_size=1000, limit=None)

        # Backwards-compat: callers that don't ask for logging don't get any.
        mock_log_cls.assert_not_called()

    def test_ingest_log_marks_failed_on_exception(
        self, fake_conns: tuple[_FakeConn, _FakeConn]
    ) -> None:
        read_conn, write_conn = fake_conns
        log = MagicMock()

        # Force the SELECT to raise so we exercise the failure path.
        def _boom(*_: Any, **__: Any) -> None:
            raise RuntimeError("simulated DB failure")

        read_conn._server_cursor.execute = _boom  # type: ignore[method-assign]

        seq = iter([read_conn, write_conn])
        with patch.object(cc, "get_connection", lambda dsn=None: next(seq)), patch.object(
            cc, "IngestLog", return_value=log
        ):
            with pytest.raises(RuntimeError, match="simulated DB failure"):
                cc.run_pipeline(
                    dsn="dbname=scix_test",
                    batch_size=1000,
                    limit=None,
                    ingest_log_filename="citctx_full_backfill_2026",
                )

        log.start.assert_called_once_with("citctx_full_backfill_2026")
        log.mark_failed.assert_called_once_with("citctx_full_backfill_2026")
        log.finish.assert_not_called()


class TestRunPipelineShardThreaded:
    def test_shard_kwarg_propagates_to_select(
        self, fake_conns: tuple[_FakeConn, _FakeConn]
    ) -> None:
        read_conn, write_conn = fake_conns
        seq = iter([read_conn, write_conn])
        with patch.object(cc, "get_connection", lambda dsn=None: next(seq)):
            cc.run_pipeline(
                dsn="dbname=scix_test",
                batch_size=1000,
                limit=None,
                shard=(1, 4),
            )

        # The streaming SELECT should carry the shard predicate.
        sql = read_conn._server_cursor.last_sql
        params = read_conn._server_cursor.last_params
        assert "mod(hashtext(p.bibcode), %s) = %s" in sql
        assert list(params)[:2] == [4, 1]
