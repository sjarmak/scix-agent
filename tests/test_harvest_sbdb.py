"""Tests for scripts/harvest_sbdb.py — SBDB enrichment harvester."""

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

import harvest_sbdb

# ---------------------------------------------------------------------------
# Fixtures / sample data
# ---------------------------------------------------------------------------

SAMPLE_SBDB_RESPONSE: dict[str, Any] = {
    "object": {
        "orbit_class": {"name": "Main-belt Asteroid", "code": "MBA"},
        "neo": False,
        "pha": False,
    },
    "discovery": {
        "date": "1801-01-01",
        "site": "Palermo",
        "name": "G. Piazzi",
    },
}

SAMPLE_SBDB_RESPONSE_NEO: dict[str, Any] = {
    "object": {
        "orbit_class": {"name": "Near-Earth Asteroid", "code": "NEA"},
        "neo": True,
        "pha": True,
    },
    "discovery": {
        "date": "1898-08-13",
        "site": "Berlin",
        "name": "C. G. Witt",
    },
}

SAMPLE_SBDB_RESPONSE_EMPTY: dict[str, Any] = {
    "object": {},
    "discovery": {},
}

SAMPLE_SBDB_ERROR_RESPONSE: dict[str, Any] = {
    "code": "300",
    "message": "specified object was not found",
}


# ---------------------------------------------------------------------------
# Test: module is importable
# ---------------------------------------------------------------------------


def test_module_importable() -> None:
    """harvest_sbdb module is importable."""
    assert hasattr(harvest_sbdb, "run_harvest")
    assert hasattr(harvest_sbdb, "main")
    assert hasattr(harvest_sbdb, "parse_sbdb_response")
    assert hasattr(harvest_sbdb, "fetch_sbdb_record")


def test_imports_resilient_client() -> None:
    """Script imports ResilientClient from scix.http_client."""
    assert hasattr(harvest_sbdb, "ResilientClient")


def test_imports_harvest_run_log() -> None:
    """Script imports HarvestRunLog from scix.harvest_utils."""
    assert hasattr(harvest_sbdb, "HarvestRunLog")


# ---------------------------------------------------------------------------
# Test: ResilientClient instantiation with rate_limit=1.0
# ---------------------------------------------------------------------------


def test_client_rate_limit() -> None:
    """_get_client creates ResilientClient with rate_limit=1.0 for JPL SBDB."""
    # Reset module-level client
    harvest_sbdb._client = None
    with patch.object(harvest_sbdb, "ResilientClient") as mock_cls:
        mock_cls.return_value = MagicMock()
        client = harvest_sbdb._get_client()
        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args[1]
        assert kwargs["rate_limit"] == 1.0
    # Reset so other tests are not affected
    harvest_sbdb._client = None


# ---------------------------------------------------------------------------
# Test: parse_sbdb_response
# ---------------------------------------------------------------------------


def test_parse_sbdb_response_full() -> None:
    """parse_sbdb_response extracts all enrichment fields."""
    result = harvest_sbdb.parse_sbdb_response(SAMPLE_SBDB_RESPONSE)
    assert result["orbital_class"] == "Main-belt Asteroid"
    assert result["neo"] is False
    assert result["pha"] is False
    assert result["discovery_date"] == "1801-01-01"
    assert result["discovery_site"] == "Palermo"
    assert result["discoverer"] == "G. Piazzi"


def test_parse_sbdb_response_neo_pha() -> None:
    """parse_sbdb_response correctly identifies NEO/PHA objects."""
    result = harvest_sbdb.parse_sbdb_response(SAMPLE_SBDB_RESPONSE_NEO)
    assert result["orbital_class"] == "Near-Earth Asteroid"
    assert result["neo"] is True
    assert result["pha"] is True
    assert result["discovery_date"] == "1898-08-13"
    assert result["discovery_site"] == "Berlin"


def test_parse_sbdb_response_empty() -> None:
    """parse_sbdb_response returns empty dict for missing fields."""
    result = harvest_sbdb.parse_sbdb_response(SAMPLE_SBDB_RESPONSE_EMPTY)
    assert result == {}


def test_parse_sbdb_response_no_orbit_class_name() -> None:
    """parse_sbdb_response skips orbital_class if no name."""
    data: dict[str, Any] = {
        "object": {"orbit_class": {"code": "MBA"}},
    }
    result = harvest_sbdb.parse_sbdb_response(data)
    assert "orbital_class" not in result


def test_parse_sbdb_response_partial_discovery() -> None:
    """parse_sbdb_response handles partial discovery info."""
    data: dict[str, Any] = {
        "object": {},
        "discovery": {"date": "2020-01-01"},
    }
    result = harvest_sbdb.parse_sbdb_response(data)
    assert result["discovery_date"] == "2020-01-01"
    assert "discovery_site" not in result
    assert "discoverer" not in result


# ---------------------------------------------------------------------------
# Test: fetch_sbdb_record
# ---------------------------------------------------------------------------


def test_fetch_sbdb_record_success() -> None:
    """fetch_sbdb_record returns enrichment dict on success."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_SBDB_RESPONSE
    mock_client.get.return_value = mock_response

    result = harvest_sbdb.fetch_sbdb_record(mock_client, "Ceres")

    assert result is not None
    assert result["orbital_class"] == "Main-belt Asteroid"
    mock_client.get.assert_called_once()
    # Verify params include designation
    call_kwargs = mock_client.get.call_args
    assert call_kwargs[1]["params"]["des"] == "Ceres"


def test_fetch_sbdb_record_error_code() -> None:
    """fetch_sbdb_record returns None on SBDB error code."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_SBDB_ERROR_RESPONSE
    mock_client.get.return_value = mock_response

    result = harvest_sbdb.fetch_sbdb_record(mock_client, "NotAnObject")
    assert result is None


def test_fetch_sbdb_record_exception() -> None:
    """fetch_sbdb_record returns None on network exception."""
    mock_client = MagicMock()
    mock_client.get.side_effect = ConnectionError("timeout")

    result = harvest_sbdb.fetch_sbdb_record(mock_client, "Ceres")
    assert result is None


def test_fetch_sbdb_record_empty_enrichment() -> None:
    """fetch_sbdb_record returns None when no enrichment fields found."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_SBDB_RESPONSE_EMPTY
    mock_client.get.return_value = mock_response

    result = harvest_sbdb.fetch_sbdb_record(mock_client, "Unknown")
    assert result is None


# ---------------------------------------------------------------------------
# Test: entity querying from entities table WHERE source='ssodnet'
# ---------------------------------------------------------------------------


def test_fetch_ssodnet_entities_no_cursor() -> None:
    """fetch_ssodnet_entities queries all ssodnet entities when no cursor."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchall.return_value = [(1, "Ceres"), (2, "Vesta")]

    result = harvest_sbdb.fetch_ssodnet_entities(mock_conn)

    assert result == [(1, "Ceres"), (2, "Vesta")]
    # Verify query filters by source='ssodnet'
    sql = mock_cursor.execute.call_args[0][0]
    assert "source" in sql
    params = mock_cursor.execute.call_args[0][1]
    assert "ssodnet" in params


def test_fetch_ssodnet_entities_with_cursor() -> None:
    """fetch_ssodnet_entities filters by id > after_id for resumption."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchall.return_value = [(3, "Pallas")]

    result = harvest_sbdb.fetch_ssodnet_entities(mock_conn, after_id=2)

    assert result == [(3, "Pallas")]
    sql = mock_cursor.execute.call_args[0][0]
    assert "id >" in sql
    params = mock_cursor.execute.call_args[0][1]
    assert "ssodnet" in params
    assert 2 in params


# ---------------------------------------------------------------------------
# Test: properties update with orbital_class, neo, pha, discovery_date, discovery_site
# ---------------------------------------------------------------------------


def test_update_entity_properties() -> None:
    """update_entity_properties merges enrichment into properties JSONB."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    enrichment = {
        "orbital_class": "Main-belt Asteroid",
        "neo": False,
        "pha": False,
        "discovery_date": "1801-01-01",
        "discovery_site": "Palermo",
    }
    harvest_sbdb.update_entity_properties(mock_conn, 42, enrichment)

    mock_cursor.execute.assert_called_once()
    sql = mock_cursor.execute.call_args[0][0]
    assert "properties" in sql
    assert "||" in sql  # JSONB merge operator
    params = mock_cursor.execute.call_args[0][1]
    written_json = json.loads(params[0])
    assert written_json["orbital_class"] == "Main-belt Asteroid"
    assert written_json["neo"] is False
    assert written_json["pha"] is False
    assert written_json["discovery_date"] == "1801-01-01"
    assert written_json["discovery_site"] == "Palermo"
    assert params[1] == 42


# ---------------------------------------------------------------------------
# Test: cursor-based resumption logic
# ---------------------------------------------------------------------------


def test_get_last_cursor_no_prior_run() -> None:
    """get_last_cursor returns None when no completed SBDB runs exist."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = None

    result = harvest_sbdb.get_last_cursor(mock_conn)
    assert result is None


def test_get_last_cursor_with_prior_run() -> None:
    """get_last_cursor returns last_entity_id from completed run."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = (json.dumps({"last_entity_id": 99}),)

    result = harvest_sbdb.get_last_cursor(mock_conn)
    assert result == 99


def test_get_last_cursor_with_dict_cursor_state() -> None:
    """get_last_cursor handles cursor_state already parsed as dict."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = ({"last_entity_id": 50},)

    result = harvest_sbdb.get_last_cursor(mock_conn)
    assert result == 50


def test_save_cursor() -> None:
    """save_cursor persists entity_id to harvest_runs.cursor_state."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    harvest_sbdb.save_cursor(mock_conn, run_id=5, entity_id=42)

    mock_cursor.execute.assert_called_once()
    sql = mock_cursor.execute.call_args[0][0]
    assert "cursor_state" in sql
    assert "harvest_runs" in sql
    params = mock_cursor.execute.call_args[0][1]
    state = json.loads(params[0])
    assert state["last_entity_id"] == 42
    assert params[1] == 5
    mock_conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Test: HarvestRunLog lifecycle (start/complete/fail)
# ---------------------------------------------------------------------------


@patch("harvest_sbdb.get_connection")
@patch("harvest_sbdb._get_client")
def test_harvest_run_log_start_complete(
    mock_get_client: MagicMock,
    mock_get_conn: MagicMock,
) -> None:
    """HarvestRunLog goes through start -> complete lifecycle."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_SBDB_RESPONSE
    mock_client.get.return_value = mock_response
    mock_get_client.return_value = mock_client

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    # resume=False skips get_last_cursor; fetchone is only called by HarvestRunLog.start()
    mock_cursor.fetchone.return_value = (1,)
    mock_cursor.fetchall.return_value = [(10, "Ceres")]
    mock_get_conn.return_value = mock_conn

    result = harvest_sbdb.run_harvest(dsn="test://db", dry_run=False, resume=False, limit=1)

    assert result == 1

    # Verify SQL calls include harvest_runs INSERT (start) and UPDATE (complete)
    sql_calls = [str(c) for c in mock_cursor.execute.call_args_list]
    start_calls = [c for c in sql_calls if "INSERT INTO harvest_runs" in c]
    complete_calls = [c for c in sql_calls if "status = 'completed'" in c]
    assert len(start_calls) >= 1, "HarvestRunLog.start() should INSERT into harvest_runs"
    assert len(complete_calls) >= 1, "HarvestRunLog.complete() should UPDATE status"


@patch("harvest_sbdb.get_connection")
@patch("harvest_sbdb._get_client")
def test_harvest_run_log_fail_on_error(
    mock_get_client: MagicMock,
    mock_get_conn: MagicMock,
) -> None:
    """HarvestRunLog.fail() is called when an error occurs."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_SBDB_RESPONSE
    mock_client.get.return_value = mock_response
    mock_get_client.return_value = mock_client

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    # resume=False skips get_last_cursor; fetchone is only called by HarvestRunLog.start()
    mock_cursor.fetchone.return_value = (1,)
    mock_cursor.fetchall.return_value = [(10, "Ceres")]
    mock_get_conn.return_value = mock_conn

    # Make update_entity_properties raise to trigger fail path
    with (
        patch.object(
            harvest_sbdb,
            "update_entity_properties",
            side_effect=RuntimeError("DB error"),
        ),
        pytest.raises(RuntimeError, match="DB error"),
    ):
        harvest_sbdb.run_harvest(dsn="test://db", dry_run=False, resume=False, limit=1)

    sql_calls = [str(c) for c in mock_cursor.execute.call_args_list]
    fail_calls = [c for c in sql_calls if "status = 'failed'" in c]
    assert len(fail_calls) >= 1, "HarvestRunLog.fail() should UPDATE status to failed"


# ---------------------------------------------------------------------------
# Test: full run_harvest pipeline
# ---------------------------------------------------------------------------


@patch("harvest_sbdb.get_connection")
@patch("harvest_sbdb._get_client")
def test_run_harvest_enriches_entities(
    mock_get_client: MagicMock,
    mock_get_conn: MagicMock,
) -> None:
    """run_harvest fetches ssodnet entities, queries SBDB, and updates properties."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_SBDB_RESPONSE
    mock_client.get.return_value = mock_response
    mock_get_client.return_value = mock_client

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    # get_last_cursor returns None (no resume), HarvestRunLog.start returns run_id=1
    mock_cursor.fetchone.side_effect = [None, (1,)]
    mock_cursor.fetchall.return_value = [(10, "Ceres"), (20, "Eros")]
    mock_get_conn.return_value = mock_conn

    result = harvest_sbdb.run_harvest(dsn="test://db", dry_run=False, resume=True)

    assert result == 2
    # Verify API was called for both entities
    assert mock_client.get.call_count == 2


@patch("harvest_sbdb.get_connection")
@patch("harvest_sbdb._get_client")
def test_run_harvest_resume_from_cursor(
    mock_get_client: MagicMock,
    mock_get_conn: MagicMock,
) -> None:
    """run_harvest resumes from last cursor position."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_SBDB_RESPONSE
    mock_client.get.return_value = mock_response
    mock_get_client.return_value = mock_client

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    # get_last_cursor returns 50, then HarvestRunLog.start returns run_id=2
    mock_cursor.fetchone.side_effect = [
        (json.dumps({"last_entity_id": 50}),),  # get_last_cursor
        (2,),  # HarvestRunLog.start
    ]
    mock_cursor.fetchall.return_value = [(51, "Pallas")]
    mock_get_conn.return_value = mock_conn

    result = harvest_sbdb.run_harvest(dsn="test://db", dry_run=False, resume=True)

    assert result == 1
    # Verify fetch_ssodnet_entities was called with after_id
    fetchall_sql_calls = [c for c in mock_cursor.execute.call_args_list if "entities" in str(c)]
    # At least one call should have the ssodnet source and id > 50
    entity_query_found = False
    for c in mock_cursor.execute.call_args_list:
        sql_str = str(c)
        if "ssodnet" in sql_str and "id >" in sql_str:
            entity_query_found = True
            break
    assert entity_query_found, "Should query entities with id > cursor"


@patch("harvest_sbdb.get_connection")
@patch("harvest_sbdb._get_client")
def test_run_harvest_no_entities(
    mock_get_client: MagicMock,
    mock_get_conn: MagicMock,
) -> None:
    """run_harvest returns 0 when no ssodnet entities exist."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_get_conn.return_value = mock_conn

    result = harvest_sbdb.run_harvest(dsn="test://db", dry_run=False, resume=False)

    assert result == 0
    # No API calls should be made
    mock_client.get.assert_not_called()


# ---------------------------------------------------------------------------
# Test: dry-run mode
# ---------------------------------------------------------------------------


@patch("harvest_sbdb.get_connection")
@patch("harvest_sbdb._get_client")
def test_dry_run_skips_db_writes(
    mock_get_client: MagicMock,
    mock_get_conn: MagicMock,
) -> None:
    """Dry run queries API but does not update entity properties."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = SAMPLE_SBDB_RESPONSE
    mock_client.get.return_value = mock_response
    mock_get_client.return_value = mock_client

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = None
    mock_cursor.fetchall.return_value = [(10, "Ceres")]
    mock_get_conn.return_value = mock_conn

    result = harvest_sbdb.run_harvest(dsn="test://db", dry_run=True, resume=False)

    assert result == 1
    # API was called
    mock_client.get.assert_called_once()
    # No UPDATE to entities properties (no harvest_runs INSERT either)
    sql_calls = [str(c) for c in mock_cursor.execute.call_args_list]
    update_calls = [c for c in sql_calls if "UPDATE entities" in c]
    assert len(update_calls) == 0, "Dry run should not UPDATE entities"


# ---------------------------------------------------------------------------
# Test: CLI argument parsing
# ---------------------------------------------------------------------------


def test_cli_dry_run() -> None:
    """CLI parses --dry-run correctly."""
    with patch.object(harvest_sbdb, "run_harvest", return_value=5) as mock_rh:
        harvest_sbdb.main(["--dry-run"])
        mock_rh.assert_called_once_with(dsn=None, dry_run=True, resume=True, limit=None)


def test_cli_no_resume() -> None:
    """CLI parses --no-resume correctly."""
    with patch.object(harvest_sbdb, "run_harvest", return_value=0) as mock_rh:
        harvest_sbdb.main(["--no-resume", "--dry-run"])
        mock_rh.assert_called_once_with(dsn=None, dry_run=True, resume=False, limit=None)


def test_cli_limit() -> None:
    """CLI parses --limit correctly."""
    with patch.object(harvest_sbdb, "run_harvest", return_value=10) as mock_rh:
        harvest_sbdb.main(["--limit", "100", "--dry-run"])
        mock_rh.assert_called_once_with(dsn=None, dry_run=True, resume=True, limit=100)


def test_cli_dsn_flag() -> None:
    """CLI passes --dsn to run_harvest."""
    with patch.object(harvest_sbdb, "run_harvest", return_value=0) as mock_rh:
        harvest_sbdb.main(["--dsn", "postgresql://localhost/test", "--dry-run"])
        mock_rh.assert_called_once_with(
            dsn="postgresql://localhost/test", dry_run=True, resume=True, limit=None
        )
