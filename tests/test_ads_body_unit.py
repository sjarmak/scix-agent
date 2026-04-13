"""Pure unit tests for src/scix/ads_body.py (no database required).

Covers the security-critical helpers `is_production_dsn` and `_redact_dsn`
plus the JSONL record parser. These tests are always runnable.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scix.ads_body import (  # noqa: E402
    LoaderConfig,
    _parse_entry_date,
    _parse_record,
    _redact_dsn,
    is_production_dsn,
)

# ---------------------------------------------------------------------------
# is_production_dsn — recognises key=value and URI forms
# ---------------------------------------------------------------------------


class TestIsProductionDsn:
    @pytest.mark.parametrize(
        "dsn",
        [
            "dbname=scix",
            "host=localhost dbname=scix",
            "host=localhost user=ds dbname=scix port=5432",
            "postgresql://localhost/scix",
            "postgresql://user:password@localhost/scix",
            "postgresql://user:password@localhost:5432/scix",
            "postgresql://user:password@localhost:5432/scix?sslmode=require",
            "postgres://user:password@localhost/scix",
        ],
    )
    def test_flags_production(self, dsn: str) -> None:
        assert is_production_dsn(dsn) is True

    @pytest.mark.parametrize(
        "dsn",
        [
            "",
            "dbname=scix_test",
            "dbname=scix_dev",
            "host=localhost dbname=scix_test",
            "postgresql://localhost/scix_test",
            "postgresql://user:pass@host/scix_staging?sslmode=require",
        ],
    )
    def test_allows_non_production(self, dsn: str) -> None:
        assert is_production_dsn(dsn) is False


# ---------------------------------------------------------------------------
# _redact_dsn — never leak passwords
# ---------------------------------------------------------------------------


class TestRedactDsn:
    def test_kv_strips_password(self) -> None:
        assert "secret" not in _redact_dsn("host=h dbname=scix password=secret user=ds")

    def test_kv_keeps_dbname(self) -> None:
        assert "dbname=scix" in _redact_dsn("host=h dbname=scix password=secret")

    def test_uri_masks_password(self) -> None:
        redacted = _redact_dsn("postgresql://user:supersecret@localhost:5432/scix")
        assert "supersecret" not in redacted
        assert "user:***" in redacted
        assert "localhost" in redacted
        assert "/scix" in redacted

    def test_uri_strips_query_params(self) -> None:
        redacted = _redact_dsn("postgresql://user:p@host/scix?sslmode=require&passfile=/etc/shadow")
        assert "passfile" not in redacted
        assert "/etc/shadow" not in redacted

    def test_empty_dsn(self) -> None:
        assert _redact_dsn("") == "<redacted>"


# ---------------------------------------------------------------------------
# _parse_record — JSONL record parser
# ---------------------------------------------------------------------------


class TestParseRecord:
    def test_valid_record(self) -> None:
        line = (
            '{"bibcode":"2024ApJ...1..1A","body":"Full text here",'
            '"entry_date":"2024-01-15T00:00:00Z"}'
        )
        row = _parse_record(line, filename="t.jsonl", line_no=1)
        assert row is not None
        bibcode, body, length, harvested_at = row
        assert bibcode == "2024ApJ...1..1A"
        assert body == "Full text here"
        assert length == len("Full text here")
        assert harvested_at.year == 2024

    def test_missing_bibcode_skipped(self) -> None:
        assert _parse_record('{"body":"x"}', filename="t", line_no=1) is None

    def test_missing_body_skipped(self) -> None:
        assert _parse_record('{"bibcode":"b"}', filename="t", line_no=1) is None

    def test_empty_body_skipped(self) -> None:
        assert _parse_record('{"bibcode":"b","body":""}', filename="t", line_no=1) is None

    def test_whitespace_body_skipped(self) -> None:
        assert _parse_record('{"bibcode":"b","body":"   \\n  "}', filename="t", line_no=1) is None

    def test_null_body_skipped(self) -> None:
        assert _parse_record('{"bibcode":"b","body":null}', filename="t", line_no=1) is None

    def test_invalid_json_returns_none(self) -> None:
        assert _parse_record("not valid json", filename="t", line_no=1) is None

    def test_non_object_returns_none(self) -> None:
        assert _parse_record('["list"]', filename="t", line_no=1) is None


# ---------------------------------------------------------------------------
# _parse_entry_date
# ---------------------------------------------------------------------------


class TestParseEntryDate:
    def test_iso_with_z(self) -> None:
        dt = _parse_entry_date("2024-03-15T00:00:00Z")
        assert dt.year == 2024 and dt.tzinfo is not None

    def test_iso_with_offset(self) -> None:
        dt = _parse_entry_date("2024-03-15T12:00:00+00:00")
        assert dt.year == 2024 and dt.tzinfo is not None

    def test_naive_iso_becomes_utc(self) -> None:
        dt = _parse_entry_date("2024-03-15T12:00:00")
        assert dt.tzinfo == timezone.utc

    def test_none_falls_back_to_epoch(self) -> None:
        assert _parse_entry_date(None) == datetime(1970, 1, 1, tzinfo=timezone.utc)

    def test_garbage_falls_back_to_epoch(self) -> None:
        assert _parse_entry_date("not a date") == datetime(1970, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# LoaderConfig is frozen (unit-level; no DB needed)
# ---------------------------------------------------------------------------


class TestLoaderConfig:
    def test_config_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        cfg = LoaderConfig(dsn="dbname=foo", jsonl_path=Path("/nonexistent.jsonl"))
        with pytest.raises(FrozenInstanceError):
            cfg.dsn = "mutated"  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = LoaderConfig(dsn="dbname=foo", jsonl_path=Path("/x.jsonl"))
        assert cfg.batch_size == 10_000
        assert cfg.dry_run is False
        assert cfg.yes_production is False
        assert cfg.drop_indexes is False
