"""Tests for the extract_citation_contexts.py CLI prep gaps from PRD 79n.1.

Covers:
- ``parse_shard`` helper accepts ``"i/n"`` and rejects malformed values.
- ``enforce_prod_guard`` mirrors the pattern in
  ``backfill_part_of_inheritance.py``: refuses prod DSN without
  ``--allow-prod``, refuses ``--allow-prod`` outside a systemd scope.
- ``ingest_log_filename_for_shard`` derives the canonical filename used
  to track progress in the ``ingest_log`` table.
"""

from __future__ import annotations

import importlib.util
import sys
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
