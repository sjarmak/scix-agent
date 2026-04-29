"""Tests for scripts/harvest_cmr.py — NASA CMR Collection Harvester."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import harvest_cmr

# ---------------------------------------------------------------------------
# Fixtures — realistic CMR UMM-JSON responses
# ---------------------------------------------------------------------------

MOCK_COLLECTION_ITEM_1: dict[str, Any] = {
    "meta": {
        "concept-id": "C1234-PROVIDER",
        "revision-id": "5",
    },
    "umm": {
        "ShortName": "MOD09GA",
        "EntryTitle": "MODIS/Terra Surface Reflectance Daily L2G Global 1km",
        "Abstract": "Daily surface reflectance from MODIS Terra.",
        "Platforms": [
            {
                "ShortName": "Terra",
                "Instruments": [
                    {"ShortName": "MODIS"},
                ],
            },
        ],
        "ScienceKeywords": [
            {
                "Category": "EARTH SCIENCE",
                "Topic": "LAND SURFACE",
                "Term": "SURFACE RADIATIVE PROPERTIES",
                "VariableLevel1": "REFLECTANCE",
            },
        ],
        "TemporalExtents": [
            {
                "RangeDateTimes": [
                    {
                        "BeginningDateTime": "2000-02-24T00:00:00.000Z",
                        "EndingDateTime": "2025-01-01T23:59:59.000Z",
                    }
                ]
            }
        ],
    },
}

MOCK_COLLECTION_ITEM_2: dict[str, Any] = {
    "meta": {
        "concept-id": "C5678-PROVIDER",
    },
    "umm": {
        "ShortName": "GPM_3IMERGHH",
        "EntryTitle": "GPM IMERG Final Precipitation L3 Half Hourly",
        "Abstract": "Global precipitation from GPM.",
        "Platforms": [
            {
                "ShortName": "GPM",
                "Instruments": [
                    {"ShortName": "GMI"},
                    {"ShortName": "DPR"},
                ],
            },
            {
                "ShortName": "TRMM",
                "Instruments": [
                    {"ShortName": "TMI"},
                ],
            },
        ],
        "ScienceKeywords": [
            {
                "Category": "EARTH SCIENCE",
                "Topic": "ATMOSPHERE",
                "Term": "PRECIPITATION",
            },
        ],
    },
}

MOCK_COLLECTION_ITEM_NO_ID: dict[str, Any] = {
    "meta": {},
    "umm": {"ShortName": "ORPHAN"},
}


def _make_page_response(
    items: list[dict[str, Any]],
    hits: int,
    search_after: str | None = None,
) -> MagicMock:
    """Build a mock response for a CMR page."""
    resp = MagicMock()
    resp.json.return_value = {"hits": hits, "items": items}
    resp.headers = {}
    if search_after is not None:
        resp.headers["CMR-Search-After"] = search_after
    resp.status_code = 200
    return resp


# ---------------------------------------------------------------------------
# Test: ResilientClient is used
# ---------------------------------------------------------------------------


class TestResilientClientUsed:
    """Verify that ResilientClient is used for all HTTP calls."""

    def test_make_client_returns_resilient_client(self) -> None:
        client = harvest_cmr._make_client()
        assert isinstance(client, harvest_cmr.ResilientClient)


# ---------------------------------------------------------------------------
# Test: Search-After pagination
# ---------------------------------------------------------------------------


class TestSearchAfterPagination:
    """Verify Search-After header pagination pattern."""

    def test_single_page(self) -> None:
        """When all results fit in one page, no Search-After needed."""
        client = MagicMock()
        client.get.return_value = _make_page_response([MOCK_COLLECTION_ITEM_1], hits=1)

        items, page_count = harvest_cmr.fetch_collections(client, page_size=2000)

        assert len(items) == 1
        assert page_count == 1
        # First call should NOT have Search-After header
        call_kwargs = client.get.call_args_list[0]
        headers = call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))
        assert "Search-After" not in headers

    def test_multi_page_uses_search_after(self) -> None:
        """Verify Search-After header is sent on subsequent pages."""
        client = MagicMock()

        page1 = _make_page_response([MOCK_COLLECTION_ITEM_1], hits=2, search_after="token-abc")
        page2 = _make_page_response([MOCK_COLLECTION_ITEM_2], hits=2)
        client.get.side_effect = [page1, page2]

        items, page_count = harvest_cmr.fetch_collections(client, page_size=1)

        assert len(items) == 2
        assert page_count == 2

        # Second call should have CMR-Search-After header (CMR's mandated
        # request header — the bare "Search-After" form is ignored).
        second_call = client.get.call_args_list[1]
        headers = second_call.kwargs.get("headers", second_call[1].get("headers", {}))
        assert headers.get("CMR-Search-After") == "token-abc"
        assert "Search-After" not in headers

    def test_stops_when_no_search_after_header(self) -> None:
        """Pagination stops if CMR-Search-After header is missing."""
        client = MagicMock()
        # Return items but no Search-After header, hits > len(items)
        page1 = _make_page_response([MOCK_COLLECTION_ITEM_1], hits=100)
        client.get.return_value = page1

        items, page_count = harvest_cmr.fetch_collections(client, page_size=1)

        assert len(items) == 1
        assert page_count == 1


# ---------------------------------------------------------------------------
# Test: Accept header is UMM-JSON
# ---------------------------------------------------------------------------


class TestUmmJsonAcceptHeader:
    """Verify the Accept header is set to UMM-JSON format."""

    def test_accept_header_set(self) -> None:
        client = MagicMock()
        client.get.return_value = _make_page_response([], hits=0)

        harvest_cmr.fetch_collections(client)

        call_kwargs = client.get.call_args_list[0]
        headers = call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))
        assert headers.get("Accept") == harvest_cmr.UMM_JSON_ACCEPT
        assert "umm_results" in headers["Accept"]


# ---------------------------------------------------------------------------
# Test: Parsing
# ---------------------------------------------------------------------------


class TestParseCollection:
    """Verify extraction of instruments, platforms, keywords from UMM-JSON."""

    def test_basic_parse(self) -> None:
        result = harvest_cmr.parse_collection(MOCK_COLLECTION_ITEM_1)
        assert result is not None
        assert result["concept_id"] == "C1234-PROVIDER"
        assert result["name"] == "MODIS/Terra Surface Reflectance Daily L2G Global 1km"
        assert result["short_name"] == "MOD09GA"
        assert "abstract" in result

    def test_extract_platforms(self) -> None:
        result = harvest_cmr.parse_collection(MOCK_COLLECTION_ITEM_2)
        assert result is not None
        assert "GPM" in result["platforms"]
        assert "TRMM" in result["platforms"]
        assert len(result["platforms"]) == 2

    def test_extract_instruments(self) -> None:
        result = harvest_cmr.parse_collection(MOCK_COLLECTION_ITEM_2)
        assert result is not None
        assert "GMI" in result["instruments"]
        assert "DPR" in result["instruments"]
        assert "TMI" in result["instruments"]
        assert len(result["instruments"]) == 3

    def test_extract_science_keywords(self) -> None:
        result = harvest_cmr.parse_collection(MOCK_COLLECTION_ITEM_1)
        assert result is not None
        kws = result["science_keywords"]
        assert len(kws) == 1
        assert kws[0]["Category"] == "EARTH SCIENCE"
        assert kws[0]["Topic"] == "LAND SURFACE"
        assert kws[0]["Term"] == "SURFACE RADIATIVE PROPERTIES"
        assert kws[0]["VariableLevel1"] == "REFLECTANCE"

    def test_temporal_extent(self) -> None:
        result = harvest_cmr.parse_collection(MOCK_COLLECTION_ITEM_1)
        assert result is not None
        assert result["temporal_start"] == "2000-02-24"
        assert result["temporal_end"] == "2025-01-01"

    def test_missing_concept_id_returns_none(self) -> None:
        result = harvest_cmr.parse_collection(MOCK_COLLECTION_ITEM_NO_ID)
        assert result is None

    def test_dedup_instruments(self) -> None:
        """Duplicate instrument names across platforms are deduplicated."""
        item: dict[str, Any] = {
            "meta": {"concept-id": "C999-TEST"},
            "umm": {
                "ShortName": "TEST",
                "EntryTitle": "Test",
                "Platforms": [
                    {"ShortName": "P1", "Instruments": [{"ShortName": "MODIS"}]},
                    {"ShortName": "P2", "Instruments": [{"ShortName": "MODIS"}]},
                ],
            },
        }
        result = harvest_cmr.parse_collection(item)
        assert result is not None
        assert result["instruments"] == ["MODIS"]

    def test_parse_collections_skips_invalid(self) -> None:
        items = [MOCK_COLLECTION_ITEM_1, MOCK_COLLECTION_ITEM_NO_ID, MOCK_COLLECTION_ITEM_2]
        results = harvest_cmr.parse_collections(items)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Test: Dataset storage with source='cmr'
# ---------------------------------------------------------------------------


class TestDatasetStorage:
    """Verify datasets are stored with source='cmr' and correct fields."""

    def test_upsert_dataset_source_cmr(self) -> None:
        """Verify _upsert_dataset uses source='cmr' and ON CONFLICT works."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (42,)

        ds_id = harvest_cmr._upsert_dataset(
            mock_conn,
            name="Test Collection",
            discipline="earth_science",
            source="cmr",
            canonical_id="C1234-PROVIDER",
            description="A test",
            properties={"science_keywords": []},
            harvest_run_id=1,
        )

        assert ds_id == 42
        # Verify SQL includes ON CONFLICT
        sql = mock_cursor.execute.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "source" in sql
        assert "canonical_id" in sql

    def test_upsert_dataset_entity(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        harvest_cmr._upsert_dataset_entity(
            mock_conn,
            dataset_id=42,
            entity_id=7,
            relationship="has_instrument",
        )

        sql = mock_cursor.execute.call_args[0][0]
        assert "dataset_entities" in sql
        assert "ON CONFLICT" in sql


# ---------------------------------------------------------------------------
# Test: GCMD cross-referencing
# ---------------------------------------------------------------------------


class TestGcmdCrossReference:
    """Verify GCMD entity lookup and linking via entity_identifiers."""

    def test_lookup_gcmd_entities_by_name(self) -> None:
        """Verify the query uses gcmd_uuid id_scheme."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [("MODIS", 10), ("AIRS", 20)]

        result = harvest_cmr._lookup_gcmd_entities_by_name(
            mock_conn, ["MODIS", "AIRS", "UNKNOWN"], "instrument"
        )

        assert result == {"MODIS": 10, "AIRS": 20}
        sql = mock_cursor.execute.call_args[0][0]
        assert "gcmd_uuid" in sql
        assert "entity_identifiers" in sql

    def test_lookup_empty_names(self) -> None:
        mock_conn = MagicMock()
        result = harvest_cmr._lookup_gcmd_entities_by_name(mock_conn, [], "instrument")
        assert result == {}

    def test_store_collections_links_gcmd_entities(self) -> None:
        """Full store_collections test verifying GCMD cross-reference linking."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # First two calls are GCMD lookups (instruments, platforms)
        # Third+ calls are dataset upserts and entity links
        call_count = [0]

        def fetchall_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                # Instrument lookup
                return [("MODIS", 10)]
            elif call_count[0] == 2:
                # Platform lookup
                return [("Terra", 20)]
            return []

        def fetchone_side_effect():
            return (42,)  # dataset id

        mock_cursor.fetchall.side_effect = fetchall_side_effect
        mock_cursor.fetchone.side_effect = fetchone_side_effect

        collections = [
            {
                "concept_id": "C1234-PROVIDER",
                "name": "Test Collection",
                "short_name": "TEST",
                "instruments": ["MODIS"],
                "platforms": ["Terra"],
                "science_keywords": [
                    {"Category": "EARTH SCIENCE", "Topic": "LAND SURFACE", "Term": "REFLECTANCE"}
                ],
            },
        ]

        counts = harvest_cmr.store_collections(mock_conn, collections, run_id=1)

        assert counts["datasets"] == 1
        assert counts["instrument_links"] == 1
        assert counts["platform_links"] == 1
        assert counts["gcmd_instruments_matched"] == 1
        assert counts["gcmd_platforms_matched"] == 1


# ---------------------------------------------------------------------------
# Test: Concept-ID deduplication
# ---------------------------------------------------------------------------


class TestConceptIdDeduplication:
    """Verify deduplication on concept-id via ON CONFLICT."""

    def test_duplicate_concept_id_handled(self) -> None:
        """Second insert with same concept-id should update, not error."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        # Both calls return same id (upsert behavior)
        mock_cursor.fetchone.return_value = (42,)

        for _ in range(2):
            ds_id = harvest_cmr._upsert_dataset(
                mock_conn,
                name="Same Collection",
                discipline="earth_science",
                source="cmr",
                canonical_id="C1234-PROVIDER",
                harvest_run_id=1,
            )
            assert ds_id == 42

        # SQL uses ON CONFLICT DO UPDATE, not error
        sql = mock_cursor.execute.call_args[0][0]
        assert "ON CONFLICT (source, canonical_id) DO UPDATE" in sql


# ---------------------------------------------------------------------------
# Test: HarvestRunLog lifecycle
# ---------------------------------------------------------------------------


class TestHarvestRunLogLifecycle:
    """Verify HarvestRunLog is used correctly."""

    @patch("harvest_cmr._make_client")
    @patch("harvest_cmr.get_connection")
    def test_harvest_run_log_complete(
        self, mock_get_conn: MagicMock, mock_make_client: MagicMock
    ) -> None:
        """run_harvest calls HarvestRunLog.start and .complete on success."""
        # Mock client
        mock_client = MagicMock()
        mock_client.get.return_value = _make_page_response([MOCK_COLLECTION_ITEM_1], hits=1)
        mock_make_client.return_value = mock_client

        # Mock DB connection and HarvestRunLog
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        with patch("harvest_cmr.HarvestRunLog") as MockRunLog:
            mock_log = MagicMock()
            mock_log.run_id = 99
            MockRunLog.return_value = mock_log
            mock_log.start.return_value = 99

            # Mock store_collections to avoid DB interaction
            with patch("harvest_cmr.store_collections") as mock_store:
                mock_store.return_value = {
                    "datasets": 1,
                    "instrument_links": 0,
                    "platform_links": 0,
                    "gcmd_instruments_matched": 0,
                    "gcmd_platforms_matched": 0,
                }

                counts = harvest_cmr.run_harvest(dsn="test-dsn")

            MockRunLog.assert_called_once_with(mock_conn, "cmr")
            mock_log.start.assert_called_once()
            mock_log.complete.assert_called_once()
            mock_log.fail.assert_not_called()

    @patch("harvest_cmr._make_client")
    @patch("harvest_cmr.get_connection")
    def test_harvest_run_log_fail_on_error(
        self, mock_get_conn: MagicMock, mock_make_client: MagicMock
    ) -> None:
        """run_harvest calls HarvestRunLog.fail on exception."""
        mock_client = MagicMock()
        mock_client.get.return_value = _make_page_response([MOCK_COLLECTION_ITEM_1], hits=1)
        mock_make_client.return_value = mock_client

        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        with patch("harvest_cmr.HarvestRunLog") as MockRunLog:
            mock_log = MagicMock()
            mock_log.run_id = 99
            MockRunLog.return_value = mock_log
            mock_log.start.return_value = 99

            with patch("harvest_cmr.store_collections") as mock_store:
                mock_store.side_effect = RuntimeError("DB error")

                with pytest.raises(RuntimeError, match="DB error"):
                    harvest_cmr.run_harvest(dsn="test-dsn")

            mock_log.fail.assert_called_once()

    @patch("harvest_cmr._make_client")
    def test_dry_run_skips_db(self, mock_make_client: MagicMock) -> None:
        """Dry run fetches and parses but does not write to DB."""
        mock_client = MagicMock()
        mock_client.get.return_value = _make_page_response(
            [MOCK_COLLECTION_ITEM_1, MOCK_COLLECTION_ITEM_2], hits=2
        )
        mock_make_client.return_value = mock_client

        counts = harvest_cmr.run_harvest(dry_run=True)

        assert counts["collections"] == 2
        assert counts["pages"] == 1


# ---------------------------------------------------------------------------
# Test: CLI (argparse)
# ---------------------------------------------------------------------------


class TestCLI:
    """Verify argparse configuration."""

    @patch("harvest_cmr.run_harvest")
    def test_main_dry_run(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"collections": 5, "pages": 1}
        result = harvest_cmr.main(["--dry-run"])
        assert result == 0
        mock_run.assert_called_once_with(dsn=None, dry_run=True)

    @patch("harvest_cmr.run_harvest")
    def test_main_with_dsn(self, mock_run: MagicMock) -> None:
        mock_run.return_value = {"datasets": 10}
        result = harvest_cmr.main(["--dsn", "postgres://localhost/test"])
        assert result == 0
        mock_run.assert_called_once_with(dsn="postgres://localhost/test", dry_run=False)


# ---------------------------------------------------------------------------
# Test: Module is importable
# ---------------------------------------------------------------------------


class TestImportable:
    """Verify the module is importable and has expected attributes."""

    def test_imports(self) -> None:
        assert hasattr(harvest_cmr, "ResilientClient")
        assert hasattr(harvest_cmr, "HarvestRunLog")
        assert hasattr(harvest_cmr, "fetch_collections")
        assert hasattr(harvest_cmr, "parse_collection")
        assert hasattr(harvest_cmr, "run_harvest")
        assert harvest_cmr.SOURCE == "cmr"
