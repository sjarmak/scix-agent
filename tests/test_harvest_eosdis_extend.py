"""Tests for scripts/harvest_eosdis_extend.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import harvest_eosdis_extend


@pytest.mark.unit
class TestClearCmrCache:
    """Verify cache directory cleanup behavior."""

    def test_returns_zero_when_dir_missing(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cmr"
        assert not cache_dir.exists()
        count = harvest_eosdis_extend.clear_cmr_cache(cache_dir)
        assert count == 0

    def test_removes_existing_files(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cmr"
        cache_dir.mkdir(parents=True)
        for i in range(3):
            (cache_dir / f"page_{i}.json").write_text("{}")

        count = harvest_eosdis_extend.clear_cmr_cache(cache_dir)
        assert count == 3
        assert not cache_dir.exists()


@pytest.mark.unit
class TestRunExtend:
    """Verify run_extend wires up to harvest_cmr.run_harvest correctly."""

    def test_calls_harvest_cmr_after_clearing_cache(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cmr"
        cache_dir.mkdir(parents=True)
        (cache_dir / "stale.json").write_text("{}")

        with patch.object(
            harvest_eosdis_extend,
            "CMR_CACHE_DIR",
            cache_dir,
        ), patch.object(
            harvest_eosdis_extend.harvest_cmr,
            "run_harvest",
            return_value={"datasets": 100, "pages": 1},
        ) as mock_run:
            counts = harvest_eosdis_extend.run_extend(dry_run=True)

        assert counts == {"datasets": 100, "pages": 1}
        mock_run.assert_called_once()
        # Cache directory should have been removed
        assert not cache_dir.exists()

    def test_skip_cache_clear_keeps_cache(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cmr"
        cache_dir.mkdir(parents=True)
        (cache_dir / "warm.json").write_text("{}")

        with patch.object(
            harvest_eosdis_extend,
            "CMR_CACHE_DIR",
            cache_dir,
        ), patch.object(
            harvest_eosdis_extend.harvest_cmr,
            "run_harvest",
            return_value={"datasets": 1},
        ):
            harvest_eosdis_extend.run_extend(dry_run=True, skip_cache_clear=True)

        assert cache_dir.exists()
        assert (cache_dir / "warm.json").exists()

    def test_passes_dsn_through(self, tmp_path: Path) -> None:
        with patch.object(
            harvest_eosdis_extend,
            "CMR_CACHE_DIR",
            tmp_path / "missing",
        ), patch.object(
            harvest_eosdis_extend.harvest_cmr,
            "run_harvest",
            return_value={},
        ) as mock_run:
            harvest_eosdis_extend.run_extend(
                dsn="dbname=scix_test", dry_run=False
            )

        assert mock_run.call_args.kwargs["dsn"] == "dbname=scix_test"
        assert mock_run.call_args.kwargs["dry_run"] is False
