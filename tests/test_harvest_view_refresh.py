"""Tests for auto-refresh of agent views after harvest completion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scix.harvest_utils import HarvestRunLog


class TestHarvestViewRefresh:
    """Verify that HarvestRunLog.complete() triggers view refresh."""

    def _make_log(self) -> HarvestRunLog:
        """Create a HarvestRunLog with a mock connection and pre-set run_id."""
        conn = MagicMock()
        conn.autocommit = False
        log = HarvestRunLog(conn, "test_source")
        log._run_id = 42
        return log

    @patch("scix.views.refresh_all_views")
    def test_complete_triggers_refresh(self, mock_refresh) -> None:
        mock_refresh.return_value = []
        log = self._make_log()

        log.complete(records_fetched=100, records_upserted=50)

        mock_refresh.assert_called_once()

    @patch("scix.views.refresh_all_views")
    def test_complete_refresh_disabled(self, mock_refresh) -> None:
        log = self._make_log()

        log.complete(records_fetched=100, records_upserted=50, refresh_views=False)

        mock_refresh.assert_not_called()

    @patch("scix.views.refresh_all_views")
    def test_refresh_failure_does_not_raise(self, mock_refresh) -> None:
        mock_refresh.side_effect = RuntimeError("view refresh exploded")
        log = self._make_log()

        # Should not raise — harvest completion must not fail due to view refresh
        log.complete(records_fetched=100, records_upserted=50)

    @patch("scix.views.refresh_all_views")
    def test_autocommit_set_during_refresh(self, mock_refresh) -> None:
        mock_refresh.return_value = []
        log = self._make_log()
        conn = log._conn

        log.complete(records_fetched=10, records_upserted=5)

        # autocommit should be restored to original value
        assert conn.autocommit is False

    @patch("scix.views.refresh_all_views")
    def test_autocommit_restored_on_refresh_error(self, mock_refresh) -> None:
        mock_refresh.side_effect = Exception("boom")
        log = self._make_log()
        conn = log._conn

        log.complete(records_fetched=10, records_upserted=5)

        # autocommit should be restored even after error
        assert conn.autocommit is False
