"""Tests for the extract_citation_contexts.py CLI prep gaps from PRD 79n.1.

Covers:
- ``parse_shard`` helper accepts ``"i/n"`` and rejects malformed values.
- ``enforce_prod_guard`` mirrors the pattern in
  ``backfill_part_of_inheritance.py``: refuses prod DSN without
  ``--allow-prod``, refuses ``--allow-prod`` outside a systemd scope.
- ``enforce_free_disk_guard`` (bead 6hr7 AC d) refuses to run when free
  disk is below the configured floor.
- ``ingest_log_filename_for_shard`` derives the canonical filename used
  to track progress in the ``ingest_log`` table.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import namedtuple
from pathlib import Path

import pytest

# Load the CLI script as a module so we can unit-test its helpers without
# relying on subprocess plumbing.  The script lives outside the package so
# importlib.util.spec_from_file_location is the cleanest path.
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "extract_citation_contexts.py"
_spec = importlib.util.spec_from_file_location("extract_citation_contexts_cli", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
extract_cli = importlib.util.module_from_spec(_spec)
sys.modules["extract_citation_contexts_cli"] = extract_cli
_spec.loader.exec_module(extract_cli)


# ---------------------------------------------------------------------------
# parse_shard
# ---------------------------------------------------------------------------


class TestParseShard:
    def test_zero_of_four(self) -> None:
        assert extract_cli.parse_shard("0/4") == (0, 4)

    def test_three_of_four(self) -> None:
        assert extract_cli.parse_shard("3/4") == (3, 4)

    def test_single_shard(self) -> None:
        # A single shard is a valid degenerate case; index must still be 0.
        assert extract_cli.parse_shard("0/1") == (0, 1)

    def test_rejects_missing_slash(self) -> None:
        with pytest.raises(ValueError):
            extract_cli.parse_shard("0-4")

    def test_rejects_non_integer(self) -> None:
        with pytest.raises(ValueError):
            extract_cli.parse_shard("a/4")

    def test_rejects_index_equal_to_total(self) -> None:
        # mod-arithmetic invariant: 0 <= index < total
        with pytest.raises(ValueError):
            extract_cli.parse_shard("4/4")

    def test_rejects_index_greater_than_total(self) -> None:
        with pytest.raises(ValueError):
            extract_cli.parse_shard("5/4")

    def test_rejects_negative_index(self) -> None:
        with pytest.raises(ValueError):
            extract_cli.parse_shard("-1/4")

    def test_rejects_zero_total(self) -> None:
        with pytest.raises(ValueError):
            extract_cli.parse_shard("0/0")

    def test_rejects_extra_pieces(self) -> None:
        with pytest.raises(ValueError):
            extract_cli.parse_shard("0/4/8")


# ---------------------------------------------------------------------------
# enforce_prod_guard
# ---------------------------------------------------------------------------


class TestEnforceProdGuard:
    def test_refuses_prod_dsn_without_allow_prod(self) -> None:
        with pytest.raises(SystemExit) as exc:
            extract_cli.enforce_prod_guard(
                dsn="dbname=scix",
                allow_prod=False,
                env={"INVOCATION_ID": "abc"},
            )
        assert exc.value.code == 2

    def test_refuses_allow_prod_without_systemd_scope(self) -> None:
        with pytest.raises(SystemExit) as exc:
            extract_cli.enforce_prod_guard(
                dsn="dbname=scix",
                allow_prod=True,
                env={},
            )
        assert exc.value.code == 2

    def test_allows_prod_dsn_with_allow_prod_inside_systemd(self) -> None:
        # Should not raise.
        extract_cli.enforce_prod_guard(
            dsn="dbname=scix",
            allow_prod=True,
            env={"INVOCATION_ID": "abc"},
        )

    def test_allows_test_dsn_without_allow_prod(self) -> None:
        # Non-production DSN bypasses the systemd-scope requirement entirely.
        extract_cli.enforce_prod_guard(
            dsn="dbname=scix_test",
            allow_prod=False,
            env={},
        )

    def test_allows_uri_test_dsn(self) -> None:
        extract_cli.enforce_prod_guard(
            dsn="postgresql://localhost/scix_test",
            allow_prod=False,
            env={},
        )


# ---------------------------------------------------------------------------
# enforce_free_disk_guard (bead 6hr7 AC d)
# ---------------------------------------------------------------------------


_FakeDiskUsage = namedtuple("_FakeDiskUsage", ["total", "used", "free"])


class TestEnforceFreeDiskGuard:
    """The free-disk guard runs *after* prod-guard, so its sole job is to
    refuse to start a shard run when the partition holding the Postgres
    data dir is below the configured floor.
    """

    def test_passes_when_free_above_floor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 100 GB free, 50 GB floor — should not raise.
        monkeypatch.setattr(
            extract_cli.shutil,
            "disk_usage",
            lambda path: _FakeDiskUsage(
                total=200 * 1024**3, used=100 * 1024**3, free=100 * 1024**3
            ),
        )
        extract_cli.enforce_free_disk_guard(path="/", min_free_gb=50)

    def test_passes_at_exact_floor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Exactly 50 GB free with a 50 GB floor passes (>= comparison).
        monkeypatch.setattr(
            extract_cli.shutil,
            "disk_usage",
            lambda path: _FakeDiskUsage(total=200 * 1024**3, used=150 * 1024**3, free=50 * 1024**3),
        )
        extract_cli.enforce_free_disk_guard(path="/", min_free_gb=50)

    def test_refuses_when_below_floor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 30 GB free, 50 GB floor — should raise SystemExit(3).
        monkeypatch.setattr(
            extract_cli.shutil,
            "disk_usage",
            lambda path: _FakeDiskUsage(total=200 * 1024**3, used=170 * 1024**3, free=30 * 1024**3),
        )
        with pytest.raises(SystemExit) as exc:
            extract_cli.enforce_free_disk_guard(path="/", min_free_gb=50)
        assert exc.value.code == 3

    def test_refuses_when_just_below_floor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 49.9 GB free, 50 GB floor — should raise SystemExit(3).
        monkeypatch.setattr(
            extract_cli.shutil,
            "disk_usage",
            lambda path: _FakeDiskUsage(
                total=200 * 1024**3,
                used=200 * 1024**3 - int(49.9 * 1024**3),
                free=int(49.9 * 1024**3),
            ),
        )
        with pytest.raises(SystemExit) as exc:
            extract_cli.enforce_free_disk_guard(path="/", min_free_gb=50)
        assert exc.value.code == 3

    def test_distinct_exit_code_from_prod_guard(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Operators distinguish the failure mode by exit code:
        # 2 = prod-guard, 3 = free-disk guard.
        monkeypatch.setattr(
            extract_cli.shutil,
            "disk_usage",
            lambda path: _FakeDiskUsage(total=200 * 1024**3, used=199 * 1024**3, free=1 * 1024**3),
        )
        with pytest.raises(SystemExit) as disk_exc:
            extract_cli.enforce_free_disk_guard(path="/", min_free_gb=50)
        with pytest.raises(SystemExit) as prod_exc:
            extract_cli.enforce_prod_guard(
                dsn="dbname=scix",
                allow_prod=False,
                env={"INVOCATION_ID": "abc"},
            )
        assert disk_exc.value.code == 3
        assert prod_exc.value.code == 2


# ---------------------------------------------------------------------------
# ingest_log filename derivation
# ---------------------------------------------------------------------------


class TestIngestLogFilenameForShard:
    def test_unsharded(self) -> None:
        assert extract_cli.ingest_log_filename_for_shard(None) == "citctx_full_backfill_2026"

    def test_shard_zero_of_four(self) -> None:
        assert (
            extract_cli.ingest_log_filename_for_shard((0, 4))
            == "citctx_full_backfill_2026_shard_0_of_4"
        )

    def test_shard_three_of_four(self) -> None:
        assert (
            extract_cli.ingest_log_filename_for_shard((3, 4))
            == "citctx_full_backfill_2026_shard_3_of_4"
        )


# ---------------------------------------------------------------------------
# --include-closed flag (bead 8584)
# ---------------------------------------------------------------------------


class TestIncludeClosedFlag:
    """``--include-closed`` flips ``oa_only=False`` through ``main`` into
    ``run_pipeline`` (and from there into ``_build_papers_select``).
    """

    def test_main_default_passes_oa_only_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def _spy_run(**kwargs: object) -> int:
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(extract_cli, "run_pipeline", _spy_run)
        # Use scix_test DSN so the prod guard doesn't reject the call.
        rc = extract_cli.main(["--dsn", "dbname=scix_test"])
        assert rc == 0
        assert captured.get("oa_only") is True

    def test_main_include_closed_passes_oa_only_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def _spy_run(**kwargs: object) -> int:
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(extract_cli, "run_pipeline", _spy_run)
        rc = extract_cli.main(["--dsn", "dbname=scix_test", "--include-closed"])
        assert rc == 0
        assert captured.get("oa_only") is False
