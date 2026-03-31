"""Tests for pgvector version detection and iterative scan configuration.

Unit tests mock the database to test logic without requiring a running PostgreSQL.
Integration tests (marked @pytest.mark.integration) require a running scix database.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scix.db import (
    configure_iterative_scan,
    get_pgvector_version,
    supports_iterative_scan,
)

# ---------------------------------------------------------------------------
# Unit tests — version parsing
# ---------------------------------------------------------------------------


class TestGetPgvectorVersion:
    def test_parses_three_part_version(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ("0.8.0",)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = get_pgvector_version(mock_conn)
        assert result == (0, 8, 0)

    def test_parses_two_part_version(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = ("0.6",)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = get_pgvector_version(mock_conn)
        assert result == (0, 6)

    def test_returns_none_when_not_installed(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = get_pgvector_version(mock_conn)
        assert result is None


class TestSupportsIterativeScan:
    def test_supports_080(self) -> None:
        with patch("scix.db.get_pgvector_version", return_value=(0, 8, 0)):
            assert supports_iterative_scan(MagicMock()) is True

    def test_supports_090(self) -> None:
        with patch("scix.db.get_pgvector_version", return_value=(0, 9, 0)):
            assert supports_iterative_scan(MagicMock()) is True

    def test_does_not_support_060(self) -> None:
        with patch("scix.db.get_pgvector_version", return_value=(0, 6, 0)):
            assert supports_iterative_scan(MagicMock()) is False

    def test_does_not_support_070(self) -> None:
        with patch("scix.db.get_pgvector_version", return_value=(0, 7, 2)):
            assert supports_iterative_scan(MagicMock()) is False

    def test_not_installed(self) -> None:
        with patch("scix.db.get_pgvector_version", return_value=None):
            assert supports_iterative_scan(MagicMock()) is False


class TestConfigureIterativeScan:
    def test_applies_setting_when_supported(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("scix.db.supports_iterative_scan", return_value=True):
            result = configure_iterative_scan(mock_conn, mode="relaxed_order")

        assert result is True
        mock_cur.execute.assert_called_once_with("SET LOCAL hnsw.iterative_scan = relaxed_order")

    def test_applies_strict_order(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("scix.db.supports_iterative_scan", return_value=True):
            result = configure_iterative_scan(mock_conn, mode="strict_order")

        assert result is True
        mock_cur.execute.assert_called_once_with("SET LOCAL hnsw.iterative_scan = strict_order")

    def test_applies_off_mode(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("scix.db.supports_iterative_scan", return_value=True):
            result = configure_iterative_scan(mock_conn, mode="off")

        assert result is True
        mock_cur.execute.assert_called_once_with("SET LOCAL hnsw.iterative_scan = off")

    def test_skips_when_not_supported(self) -> None:
        mock_conn = MagicMock()

        with patch("scix.db.supports_iterative_scan", return_value=False):
            result = configure_iterative_scan(mock_conn, mode="relaxed_order")

        assert result is False

    def test_default_mode_is_relaxed_order(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("scix.db.supports_iterative_scan", return_value=True):
            configure_iterative_scan(mock_conn)

        mock_cur.execute.assert_called_once_with("SET LOCAL hnsw.iterative_scan = relaxed_order")


# ---------------------------------------------------------------------------
# Integration tests (require running scix database with pgvector)
# ---------------------------------------------------------------------------

import os

DSN = os.environ.get("SCIX_DSN", "dbname=scix")


@pytest.mark.integration
class TestPgvectorVersionIntegration:
    @pytest.fixture()
    def conn(self):
        import psycopg

        try:
            c = psycopg.connect(DSN)
            c.autocommit = False
            yield c
            c.close()
        except psycopg.OperationalError:
            pytest.skip("scix database not available")

    def test_version_returns_tuple(self, conn) -> None:
        version = get_pgvector_version(conn)
        assert version is not None
        assert len(version) >= 2
        assert all(isinstance(v, int) for v in version)

    def test_supports_check_returns_bool(self, conn) -> None:
        result = supports_iterative_scan(conn)
        assert isinstance(result, bool)

    def test_configure_returns_bool(self, conn) -> None:
        result = configure_iterative_scan(conn, mode="relaxed_order")
        assert isinstance(result, bool)
        conn.rollback()  # clean up SET LOCAL
