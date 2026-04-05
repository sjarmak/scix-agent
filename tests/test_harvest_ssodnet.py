"""Tests for scripts/harvest_ssodnet.py — SsODNet harvester."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# Ensure scripts/ and src/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import harvest_ssodnet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PARQUET_ROW: dict[str, Any] = {
    "sso_name": "Ceres",
    "sso_number": 1,
    "other_designations": "A801 AA|1943 XB",
    "spkid": "2000001",
    "diameter": 939.4,
    "albedo": 0.09,
    "taxonomy_class": "C",
}

SAMPLE_PARQUET_ROW_MINIMAL: dict[str, Any] = {
    "sso_name": "Vesta",
    "sso_number": None,
    "other_designations": "",
    "spkid": None,
    "diameter": None,
    "albedo": None,
    "taxonomy_class": None,
}

SAMPLE_SSOCARD_RESPONSE: dict[str, Any] = {
    "Ceres": {
        "spkid": "2000001",
        "other_names": ["A801 AA", "1943 XB"],
        "parameters": {
            "physical": {
                "diameter": {"value": 939.4},
                "albedo": {"value": 0.09},
                "taxonomy": {"class": "C"},
            },
            "dynamical": {},
        },
    }
}


# ---------------------------------------------------------------------------
# Test: module is importable
# ---------------------------------------------------------------------------


def test_module_importable() -> None:
    """harvest_ssodnet module is importable."""
    assert hasattr(harvest_ssodnet, "run_harvest")
    assert hasattr(harvest_ssodnet, "main")
    assert hasattr(harvest_ssodnet, "parse_sso_record")


def test_imports_resilient_client() -> None:
    """Script imports ResilientClient from scix.http_client."""
    assert hasattr(harvest_ssodnet, "ResilientClient")


def test_imports_harvest_run_log() -> None:
    """Script imports HarvestRunLog from scix.harvest_utils."""
    assert hasattr(harvest_ssodnet, "HarvestRunLog")


# ---------------------------------------------------------------------------
# Test: parse_sso_record
# ---------------------------------------------------------------------------


def test_parse_sso_record_full() -> None:
    """parse_sso_record extracts all fields from a complete row."""
    rec = harvest_ssodnet.parse_sso_record(SAMPLE_PARQUET_ROW)
    assert rec is not None
    assert rec["canonical_name"] == "Ceres"
    assert rec["entity_type"] == "target"
    assert rec["source"] == "ssodnet"
    assert rec["discipline"] == "planetary_science"


def test_parse_sso_record_properties() -> None:
    """Properties JSONB contains diameter, albedo, taxonomy."""
    rec = harvest_ssodnet.parse_sso_record(SAMPLE_PARQUET_ROW)
    assert rec is not None
    props = rec["properties"]
    assert props["diameter"] == 939.4
    assert props["albedo"] == 0.09
    assert props["taxonomy"] == "C"
    assert props["sso_number"] == 1


def test_parse_sso_record_identifiers_ssodnet() -> None:
    """Entity identifiers include id_scheme='ssodnet'."""
    rec = harvest_ssodnet.parse_sso_record(SAMPLE_PARQUET_ROW)
    assert rec is not None
    ssodnet_ids = [i for i in rec["identifiers"] if i["id_scheme"] == "ssodnet"]
    assert len(ssodnet_ids) == 1
    assert ssodnet_ids[0]["external_id"] == "Ceres"
    assert ssodnet_ids[0]["is_primary"] is True


def test_parse_sso_record_identifiers_spkid() -> None:
    """Entity identifiers include id_scheme='sbdb_spkid'."""
    rec = harvest_ssodnet.parse_sso_record(SAMPLE_PARQUET_ROW)
    assert rec is not None
    spk_ids = [i for i in rec["identifiers"] if i["id_scheme"] == "sbdb_spkid"]
    assert len(spk_ids) == 1
    assert spk_ids[0]["external_id"] == "2000001"
    assert spk_ids[0]["is_primary"] is False


def test_parse_sso_record_aliases() -> None:
    """Entity aliases populated from other_designations."""
    rec = harvest_ssodnet.parse_sso_record(SAMPLE_PARQUET_ROW)
    assert rec is not None
    aliases = rec["aliases"]
    assert "A801 AA" in aliases
    assert "1943 XB" in aliases
    # Numbered designation alias
    assert "(1) Ceres" in aliases


def test_parse_sso_record_minimal() -> None:
    """parse_sso_record handles minimal data gracefully."""
    rec = harvest_ssodnet.parse_sso_record(SAMPLE_PARQUET_ROW_MINIMAL)
    assert rec is not None
    assert rec["canonical_name"] == "Vesta"
    # No spkid identifier
    spk_ids = [i for i in rec["identifiers"] if i["id_scheme"] == "sbdb_spkid"]
    assert len(spk_ids) == 0
    # No aliases from empty other_designations
    assert len(rec["aliases"]) == 0


def test_parse_sso_record_empty_name() -> None:
    """parse_sso_record returns None for empty sso_name."""
    rec = harvest_ssodnet.parse_sso_record({"sso_name": ""})
    assert rec is None


def test_parse_sso_record_missing_name() -> None:
    """parse_sso_record returns None for missing sso_name."""
    rec = harvest_ssodnet.parse_sso_record({})
    assert rec is None


# ---------------------------------------------------------------------------
# Test: parse_ssocard (seed mode)
# ---------------------------------------------------------------------------


def test_parse_ssocard() -> None:
    """parse_ssocard extracts data from API response."""
    rec = harvest_ssodnet.parse_ssocard(SAMPLE_SSOCARD_RESPONSE, "Ceres")
    assert rec is not None
    assert rec["canonical_name"] == "Ceres"
    assert rec["entity_type"] == "target"
    assert rec["source"] == "ssodnet"
    assert rec["properties"]["diameter"] == 939.4
    assert rec["properties"]["albedo"] == 0.09
    assert rec["properties"]["taxonomy"] == "C"

    # Identifiers
    ssodnet_ids = [i for i in rec["identifiers"] if i["id_scheme"] == "ssodnet"]
    assert len(ssodnet_ids) == 1
    spk_ids = [i for i in rec["identifiers"] if i["id_scheme"] == "sbdb_spkid"]
    assert len(spk_ids) == 1
    assert spk_ids[0]["external_id"] == "2000001"

    # Aliases
    assert "A801 AA" in rec["aliases"]
    assert "1943 XB" in rec["aliases"]


def test_parse_ssocard_empty() -> None:
    """parse_ssocard handles empty response."""
    rec = harvest_ssodnet.parse_ssocard({}, "Unknown")
    # Should still return a minimal record with the queried name
    assert rec is not None
    assert rec["canonical_name"] == "Unknown"


# ---------------------------------------------------------------------------
# Test: CLI argument parsing
# ---------------------------------------------------------------------------


def test_cli_mode_bulk() -> None:
    """CLI parses --mode bulk correctly."""
    with patch.object(harvest_ssodnet, "run_harvest", return_value=100) as mock_rh:
        harvest_ssodnet.main(["--mode", "bulk", "--dry-run"])
        mock_rh.assert_called_once_with(dsn=None, mode="bulk", dry_run=True)


def test_cli_mode_seed() -> None:
    """CLI parses --mode seed correctly."""
    with patch.object(harvest_ssodnet, "run_harvest", return_value=20) as mock_rh:
        harvest_ssodnet.main(["--mode", "seed", "--dry-run"])
        mock_rh.assert_called_once_with(dsn=None, mode="seed", dry_run=True)


def test_cli_default_mode() -> None:
    """CLI defaults to seed mode."""
    with patch.object(harvest_ssodnet, "run_harvest", return_value=20) as mock_rh:
        harvest_ssodnet.main(["--dry-run"])
        mock_rh.assert_called_once_with(dsn=None, mode="seed", dry_run=True)


def test_cli_dsn_flag() -> None:
    """CLI passes --dsn to run_harvest."""
    with patch.object(harvest_ssodnet, "run_harvest", return_value=5) as mock_rh:
        harvest_ssodnet.main(["--dsn", "postgresql://localhost/test", "--dry-run"])
        mock_rh.assert_called_once_with(
            dsn="postgresql://localhost/test", mode="seed", dry_run=True
        )


# ---------------------------------------------------------------------------
# Test: run_harvest dispatches correctly
# ---------------------------------------------------------------------------


def test_run_harvest_dispatches_bulk() -> None:
    """run_harvest dispatches to run_bulk_harvest for mode='bulk'."""
    with patch.object(harvest_ssodnet, "run_bulk_harvest", return_value=1000) as mock:
        result = harvest_ssodnet.run_harvest(dsn="test", mode="bulk", dry_run=True)
        assert result == 1000
        mock.assert_called_once_with(dsn="test", dry_run=True)


def test_run_harvest_dispatches_seed() -> None:
    """run_harvest dispatches to run_seed_harvest for mode='seed'."""
    with patch.object(harvest_ssodnet, "run_seed_harvest", return_value=20) as mock:
        result = harvest_ssodnet.run_harvest(dsn="test", mode="seed", dry_run=True)
        assert result == 20
        mock.assert_called_once_with(dsn="test", dry_run=True)


def test_run_harvest_invalid_mode() -> None:
    """run_harvest raises ValueError for unknown mode."""
    with pytest.raises(ValueError, match="Unknown mode"):
        harvest_ssodnet.run_harvest(mode="invalid")


# ---------------------------------------------------------------------------
# Test: bulk mode uses staging schema
# ---------------------------------------------------------------------------


def test_write_staging_entities_calls_copy() -> None:
    """write_staging_entities writes to staging tables via COPY."""
    records = [harvest_ssodnet.parse_sso_record(SAMPLE_PARQUET_ROW)]

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_copy_ctx = MagicMock()

    # Setup cursor context manager
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    # Setup copy context manager
    mock_cursor.copy.return_value.__enter__ = MagicMock(return_value=mock_copy_ctx)
    mock_cursor.copy.return_value.__exit__ = MagicMock(return_value=False)

    counts = harvest_ssodnet.write_staging_entities(mock_conn, records)

    assert counts["entities"] == 1
    assert counts["identifiers"] == 2  # ssodnet + sbdb_spkid
    assert counts["aliases"] == 3  # A801 AA, 1943 XB, (1) Ceres

    # Verify TRUNCATE calls happened (staging cleanup)
    execute_calls = [str(c) for c in mock_cursor.execute.call_args_list]
    truncate_calls = [c for c in execute_calls if "TRUNCATE" in c]
    assert len(truncate_calls) >= 3  # 3 staging tables truncated

    # Verify COPY calls happened for all 3 tables
    copy_calls = [str(c) for c in mock_cursor.copy.call_args_list]
    assert any("staging.entities" in c for c in copy_calls)
    assert any("staging.entity_identifiers" in c for c in copy_calls)
    assert any("staging.entity_aliases" in c for c in copy_calls)


def test_promote_staging_calls_function() -> None:
    """promote_staging calls staging.promote_entities()."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = (42,)

    result = harvest_ssodnet.promote_staging(mock_conn)

    assert result == 42
    mock_cursor.execute.assert_called_once_with("SELECT staging.promote_entities()")


# ---------------------------------------------------------------------------
# Test: seed mode uses upsert helpers
# ---------------------------------------------------------------------------


@patch("harvest_ssodnet.get_connection")
@patch("harvest_ssodnet._get_client")
def test_seed_harvest_uses_upsert_helpers(
    mock_get_client: MagicMock,
    mock_get_conn: MagicMock,
) -> None:
    """Seed mode uses upsert_entity, upsert_entity_identifier, upsert_entity_alias."""
    # Mock client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_SSOCARD_RESPONSE
    mock_client.get.return_value = mock_response
    mock_get_client.return_value = mock_client

    # Mock connection
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    # HarvestRunLog.start() returns run_id
    mock_cursor.fetchone.return_value = (1,)
    mock_get_conn.return_value = mock_conn

    with (
        patch("harvest_ssodnet.upsert_entity", return_value=100) as mock_ue,
        patch("harvest_ssodnet.upsert_entity_identifier") as mock_uei,
        patch("harvest_ssodnet.upsert_entity_alias") as mock_uea,
    ):
        result = harvest_ssodnet.run_seed_harvest(
            dsn="test://db",
            objects=["Ceres"],
            dry_run=False,
        )

    assert result == 1
    # upsert_entity called with correct params
    mock_ue.assert_called_once()
    ue_kwargs = mock_ue.call_args[1]
    assert ue_kwargs["canonical_name"] == "Ceres"
    assert ue_kwargs["entity_type"] == "target"
    assert ue_kwargs["source"] == "ssodnet"

    # upsert_entity_identifier called for ssodnet and sbdb_spkid
    uei_calls = mock_uei.call_args_list
    id_schemes = {c[1]["id_scheme"] for c in uei_calls}
    assert "ssodnet" in id_schemes
    assert "sbdb_spkid" in id_schemes

    # upsert_entity_alias called for aliases
    assert mock_uea.call_count >= 1


# ---------------------------------------------------------------------------
# Test: HarvestRunLog lifecycle
# ---------------------------------------------------------------------------


@patch("harvest_ssodnet.get_connection")
@patch("harvest_ssodnet._get_client")
def test_harvest_run_log_lifecycle(
    mock_get_client: MagicMock,
    mock_get_conn: MagicMock,
) -> None:
    """HarvestRunLog goes through start -> complete lifecycle."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_SSOCARD_RESPONSE
    mock_client.get.return_value = mock_response
    mock_get_client.return_value = mock_client

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = (1,)
    mock_get_conn.return_value = mock_conn

    with (
        patch("harvest_ssodnet.upsert_entity", return_value=100),
        patch("harvest_ssodnet.upsert_entity_identifier"),
        patch("harvest_ssodnet.upsert_entity_alias"),
    ):
        harvest_ssodnet.run_seed_harvest(
            dsn="test://db",
            objects=["Ceres"],
        )

    # Verify SQL calls include harvest_runs INSERT (start) and UPDATE (complete)
    sql_calls = [str(c) for c in mock_cursor.execute.call_args_list]
    start_calls = [c for c in sql_calls if "INSERT INTO harvest_runs" in c]
    complete_calls = [c for c in sql_calls if "status = 'completed'" in c]
    assert len(start_calls) >= 1, "HarvestRunLog.start() should INSERT into harvest_runs"
    assert len(complete_calls) >= 1, "HarvestRunLog.complete() should UPDATE status"


@patch("harvest_ssodnet.get_connection")
@patch("harvest_ssodnet._get_client")
def test_harvest_run_log_fail_on_error(
    mock_get_client: MagicMock,
    mock_get_conn: MagicMock,
) -> None:
    """HarvestRunLog.fail() is called when an error occurs during DB writes."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_SSOCARD_RESPONSE
    mock_client.get.return_value = mock_response
    mock_get_client.return_value = mock_client

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = (1,)
    mock_get_conn.return_value = mock_conn

    # Make upsert_entity raise to trigger the fail path
    with (
        patch(
            "harvest_ssodnet.upsert_entity",
            side_effect=RuntimeError("DB write error"),
        ),
        pytest.raises(RuntimeError, match="DB write error"),
    ):
        harvest_ssodnet.run_seed_harvest(
            dsn="test://db",
            objects=["Ceres"],
        )

    # Verify fail was called
    sql_calls = [str(c) for c in mock_cursor.execute.call_args_list]
    fail_calls = [c for c in sql_calls if "status = 'failed'" in c]
    assert len(fail_calls) >= 1, "HarvestRunLog.fail() should UPDATE status to failed"


# ---------------------------------------------------------------------------
# Test: dry-run mode
# ---------------------------------------------------------------------------


def test_seed_dry_run_skips_db() -> None:
    """Seed dry-run fetches but does not write to DB."""
    with (
        patch.object(harvest_ssodnet, "_get_client") as mock_gc,
        patch.object(harvest_ssodnet, "get_connection") as mock_conn,
    ):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_SSOCARD_RESPONSE
        mock_client.get.return_value = mock_response
        mock_gc.return_value = mock_client

        result = harvest_ssodnet.run_seed_harvest(objects=["Ceres"], dry_run=True)

    assert result == 1
    mock_conn.assert_not_called()


# ---------------------------------------------------------------------------
# Test: download_parquet with checksum
# ---------------------------------------------------------------------------


def test_download_parquet_computes_sha256(tmp_path: Path) -> None:
    """download_parquet saves file and returns SHA-256."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    test_content = b"fake parquet content"
    mock_response.content = test_content
    mock_client.get.return_value = mock_response

    dest = tmp_path / "test.parquet"
    path, sha256 = harvest_ssodnet.download_parquet(
        "https://example.com/test.parquet",
        dest,
        client=mock_client,
    )

    assert path == dest
    assert dest.read_bytes() == test_content
    import hashlib

    expected_sha = hashlib.sha256(test_content).hexdigest()
    assert sha256 == expected_sha


# ---------------------------------------------------------------------------
# Test: dedup in write_staging_entities
# ---------------------------------------------------------------------------


def test_write_staging_deduplicates_entities() -> None:
    """write_staging_entities deduplicates by canonical_name."""
    rec = harvest_ssodnet.parse_sso_record(SAMPLE_PARQUET_ROW)
    records = [rec, rec]  # duplicate

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_copy_ctx = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.copy.return_value.__enter__ = MagicMock(return_value=mock_copy_ctx)
    mock_cursor.copy.return_value.__exit__ = MagicMock(return_value=False)

    counts = harvest_ssodnet.write_staging_entities(mock_conn, records)
    assert counts["entities"] == 1  # deduped
