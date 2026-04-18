"""Unit tests for scripts/build_streamingdiskann_variants.py.

Covers the acceptance criteria:
  (a) production-DSN refusal
  (b) each of v1/v2/v3 DDL contains 'diskann' keyword
  (c) V1 DDL does NOT contain SBQ/num_bits flag while V2 and V3 do
  (d) argparse wiring (flags, choices, defaults)
  (e) result entry schema (all required keys present)
  (f) merge-mode JSON write semantics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import build_streamingdiskann_variants as mod


# ---------------------------------------------------------------------------
# Production-DSN refusal
# ---------------------------------------------------------------------------


class TestAssertPilotDsn:
    def test_refuses_production_dsn(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            mod.assert_pilot_dsn("dbname=scix")
        assert excinfo.value.code == 2

    def test_refuses_production_dsn_uri(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            mod.assert_pilot_dsn("postgresql://user@localhost/scix")
        assert excinfo.value.code == 2

    def test_allows_pilot_dsn(self) -> None:
        # Does not raise.
        mod.assert_pilot_dsn("dbname=scix_pilot")
        mod.assert_pilot_dsn("dbname=scix_test")

    def test_allow_prod_override(self) -> None:
        # When --allow-prod is set, production DSN is permitted.
        mod.assert_pilot_dsn("dbname=scix", allow_prod=True)

    def test_main_exits_2_on_production_dsn(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            mod.main(["--dsn", "dbname=scix", "--variant", "v1", "--dry-run"])
        assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# DDL content checks
# ---------------------------------------------------------------------------


class TestVariantDDL:
    def test_all_three_variants_defined(self) -> None:
        assert set(mod.VARIANTS.keys()) == {"v1", "v2", "v3"}

    @pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
    def test_ddl_contains_diskann_keyword(self, variant: str) -> None:
        ddl = mod.VARIANTS[variant]["ddl"]
        assert "diskann" in ddl.lower(), f"{variant} DDL missing 'diskann': {ddl}"

    @pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
    def test_ddl_starts_with_create_index(self, variant: str) -> None:
        assert mod.VARIANTS[variant]["ddl"].startswith("CREATE INDEX")

    @pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
    def test_ddl_uses_halfvec_cosine_ops(self, variant: str) -> None:
        assert "halfvec_cosine_ops" in mod.VARIANTS[variant]["ddl"]

    @pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
    def test_ddl_filters_on_indus_model(self, variant: str) -> None:
        assert "model_name='indus'" in mod.VARIANTS[variant]["ddl"]

    def test_v1_has_no_sbq(self) -> None:
        ddl = mod.VARIANTS["v1"]["ddl"]
        assert "num_bits_per_dimension" not in ddl
        assert "num_bits" not in ddl
        assert mod.VARIANTS["v1"]["params"]["num_bits_per_dimension"] is None

    def test_v2_has_sbq(self) -> None:
        ddl = mod.VARIANTS["v2"]["ddl"]
        assert "num_bits_per_dimension" in ddl
        assert mod.VARIANTS["v2"]["params"]["num_bits_per_dimension"] == 2

    def test_v3_has_sbq_and_tuned_params(self) -> None:
        ddl = mod.VARIANTS["v3"]["ddl"]
        assert "num_bits_per_dimension" in ddl
        assert "storage_layout" in ddl
        assert "memory_optimized" in ddl
        assert "num_neighbors" in ddl
        assert mod.VARIANTS["v3"]["params"]["storage_layout"] == "memory_optimized"
        assert mod.VARIANTS["v3"]["params"]["num_neighbors"] == 64
        assert mod.VARIANTS["v3"]["params"]["num_bits_per_dimension"] == 2


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


class TestArgparseWiring:
    def test_help_includes_flags(self, capsys: pytest.CaptureFixture[str]) -> None:
        parser = mod.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        out = capsys.readouterr().out
        assert "--dsn" in out
        assert "--variant" in out
        assert "--dry-run" in out

    def test_variant_choices(self) -> None:
        parser = mod.build_parser()
        # Valid choices.
        for choice in ["v1", "v2", "v3", "all"]:
            ns = parser.parse_args(["--variant", choice])
            assert ns.variant == choice

    def test_variant_rejects_invalid_choice(self) -> None:
        parser = mod.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--variant", "v4"])

    def test_dry_run_flag(self) -> None:
        parser = mod.build_parser()
        ns = parser.parse_args(["--dry-run"])
        assert ns.dry_run is True

    def test_dsn_flag(self) -> None:
        parser = mod.build_parser()
        ns = parser.parse_args(["--dsn", "dbname=scix_pilot"])
        assert ns.dsn == "dbname=scix_pilot"

    def test_default_variant_is_all(self) -> None:
        parser = mod.build_parser()
        ns = parser.parse_args([])
        assert ns.variant == "all"


# ---------------------------------------------------------------------------
# Dry-run build_variant schema
# ---------------------------------------------------------------------------


REQUIRED_KEYS = {
    "variant",
    "ddl",
    "build_wall_seconds",
    "peak_rss_bytes",
    "index_size_bytes",
    "total_relation_size_bytes",
    "params",
    "run_id",
    "timestamp",
}


class TestBuildVariantDryRun:
    @pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
    def test_dry_run_returns_all_required_keys(self, variant: str) -> None:
        entry = mod.build_variant("dbname=scix_pilot", variant, dry_run=True)
        assert REQUIRED_KEYS.issubset(entry.keys()), (
            f"missing keys: {REQUIRED_KEYS - entry.keys()}"
        )
        assert entry["variant"] == variant
        assert entry["build_wall_seconds"] == 0.0
        assert isinstance(entry["params"], dict)
        assert entry["ddl"] == mod.VARIANTS[variant]["ddl"]
        assert entry["peak_rss_bytes"] >= 0
        assert entry["run_id"]
        assert entry["timestamp"]

    def test_dry_run_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError):
            mod.build_variant("dbname=scix_pilot", "v99", dry_run=True)


# ---------------------------------------------------------------------------
# Merge-mode JSON write
# ---------------------------------------------------------------------------


def _fake_entry(variant: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "variant": variant,
        "ddl": mod.VARIANTS[variant]["ddl"],
        "build_wall_seconds": 1.0,
        "peak_rss_bytes": 12345,
        "index_size_bytes": 100,
        "total_relation_size_bytes": 200,
        "params": mod.VARIANTS[variant]["params"],
        "run_id": "abc123",
        "timestamp": "2026-04-18T00:00:00+00:00",
    }
    base.update(overrides)
    return base


class TestMergeAndWrite:
    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "streamingdiskann_builds.json"
        mod.merge_and_write(path, [_fake_entry("v1")])
        assert path.exists()
        doc = json.loads(path.read_text())
        assert "v1" in doc["variants"]

    def test_preserves_unrelated_variants(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        mod.merge_and_write(path, [_fake_entry("v1", build_wall_seconds=10.0)])
        mod.merge_and_write(path, [_fake_entry("v2", build_wall_seconds=20.0)])
        doc = json.loads(path.read_text())
        assert set(doc["variants"].keys()) == {"v1", "v2"}
        assert doc["variants"]["v1"]["build_wall_seconds"] == 10.0
        assert doc["variants"]["v2"]["build_wall_seconds"] == 20.0

    def test_overwrites_same_variant(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        mod.merge_and_write(path, [_fake_entry("v1", build_wall_seconds=10.0)])
        mod.merge_and_write(path, [_fake_entry("v1", build_wall_seconds=50.0)])
        doc = json.loads(path.read_text())
        assert doc["variants"]["v1"]["build_wall_seconds"] == 50.0

    def test_handles_corrupted_existing_json(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("not json at all")
        mod.merge_and_write(path, [_fake_entry("v1")])
        doc = json.loads(path.read_text())
        assert "v1" in doc["variants"]


# ---------------------------------------------------------------------------
# Main entrypoint — dry-run variant=all path
# ---------------------------------------------------------------------------


class TestMainDryRun:
    def test_main_dry_run_all_writes_merged_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "streamingdiskann_builds.json"

        # Belt-and-suspenders: ensure no accidental psycopg.connect call.
        def _refuse(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("psycopg.connect should not be called in --dry-run")

        monkeypatch.setattr(mod.psycopg, "connect", _refuse)

        rc = mod.main(
            [
                "--dsn", "dbname=scix_pilot",
                "--variant", "all",
                "--dry-run",
                "--results-path", str(path),
            ]
        )
        assert rc == 0
        assert path.exists()
        doc = json.loads(path.read_text())
        assert set(doc["variants"].keys()) == {"v1", "v2", "v3"}
        for variant in ("v1", "v2", "v3"):
            entry = doc["variants"][variant]
            assert REQUIRED_KEYS.issubset(entry.keys())
            assert entry["build_wall_seconds"] == 0.0

    def test_main_dry_run_single_variant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "out.json"
        monkeypatch.setattr(
            mod.psycopg, "connect",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("psycopg.connect not expected in --dry-run")
            ),
        )
        rc = mod.main(
            [
                "--dsn", "dbname=scix_pilot",
                "--variant", "v2",
                "--dry-run",
                "--results-path", str(path),
            ]
        )
        assert rc == 0
        doc = json.loads(path.read_text())
        assert list(doc["variants"].keys()) == ["v2"]
