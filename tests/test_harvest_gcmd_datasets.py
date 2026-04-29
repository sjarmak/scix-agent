"""Tests for scripts/harvest_gcmd_datasets.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import harvest_gcmd_datasets


def _make_mock_cursor(fetchall_return: list[Any]) -> MagicMock:
    """Build a mock cursor whose fetchall returns ``fetchall_return``."""
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return
    cursor.description = [
        MagicMock(name=name)
        for name in (
            "id",
            "name",
            "canonical_id",
            "description",
            "discipline",
            "temporal_start",
            "temporal_end",
            "properties",
        )
    ]
    # MagicMock(name=...) doesn't actually set .name — workaround.
    for col, attr in zip(
        cursor.description,
        [
            "id",
            "name",
            "canonical_id",
            "description",
            "discipline",
            "temporal_start",
            "temporal_end",
            "properties",
        ],
        strict=True,
    ):
        col.name = attr
    return cursor


def _make_mock_conn(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.mark.unit
class TestRelationshipMapping:
    def test_known_mappings(self) -> None:
        assert harvest_gcmd_datasets.RELATIONSHIP_PREDICATE["on_platform"] == "observed_by"
        assert harvest_gcmd_datasets.RELATIONSHIP_PREDICATE["has_instrument"] == "measured_with"

    def test_unknown_relationship_excluded(self) -> None:
        assert "uses_other" not in harvest_gcmd_datasets.RELATIONSHIP_PREDICATE


@pytest.mark.unit
class TestSelectGcmdLinkedDatasets:
    def test_uses_gcmd_uuid_filter(self) -> None:
        cursor = _make_mock_cursor([])
        conn = _make_mock_conn(cursor)

        rows = harvest_gcmd_datasets._select_gcmd_linked_datasets(conn)
        assert rows == []
        sql = cursor.execute.call_args[0][0]
        assert "datasets" in sql
        assert "id_scheme = 'gcmd_uuid'" in sql

    def test_returns_dicts(self) -> None:
        sample_row = (
            7,
            "MODIS Surface Reflectance",
            "C123-PROVIDER",
            "Daily reflectance.",
            "earth_science",
            "2000-02-24",
            "2025-01-01",
            {"short_name": "MOD09GA", "platforms": ["Terra"], "instruments": ["MODIS"]},
        )
        cursor = _make_mock_cursor([sample_row])
        conn = _make_mock_conn(cursor)

        rows = harvest_gcmd_datasets._select_gcmd_linked_datasets(conn)
        assert len(rows) == 1
        assert rows[0]["id"] == 7
        assert rows[0]["canonical_id"] == "C123-PROVIDER"
        assert rows[0]["properties"]["short_name"] == "MOD09GA"

    def test_limit_appended(self) -> None:
        cursor = _make_mock_cursor([])
        conn = _make_mock_conn(cursor)

        harvest_gcmd_datasets._select_gcmd_linked_datasets(conn, limit=10)
        sql = cursor.execute.call_args[0][0]
        assert "LIMIT" in sql
        params = cursor.execute.call_args[0][1]
        assert params == (10,)


@pytest.mark.unit
class TestIngestDataset:
    def _sample_dataset(self) -> dict[str, Any]:
        return {
            "id": 7,
            "name": "MODIS Surface Reflectance",
            "canonical_id": "C123-PROVIDER",
            "description": "Daily reflectance",
            "discipline": "earth_science",
            "temporal_start": "2000-02-24",
            "temporal_end": "2025-01-01",
            "properties": {
                "short_name": "MOD09GA",
                "platforms": ["Terra"],
                "instruments": ["MODIS"],
                "science_keywords": [],
            },
        }

    def test_creates_entity_and_identifier_and_relationships(self) -> None:
        dataset = self._sample_dataset()
        # Two GCMD-linked entities returned by the relationships query
        gcmd_links = [(101, "on_platform"), (202, "has_instrument")]
        cursor = _make_mock_cursor(gcmd_links)
        conn = _make_mock_conn(cursor)

        with patch.object(
            harvest_gcmd_datasets, "upsert_entity", return_value=999
        ) as mock_ue, patch.object(
            harvest_gcmd_datasets, "upsert_entity_identifier"
        ) as mock_uei, patch.object(
            harvest_gcmd_datasets, "upsert_entity_alias"
        ) as mock_uea, patch.object(
            harvest_gcmd_datasets, "upsert_entity_relationship"
        ) as mock_uer:
            entity_id, alias_count, rel_count = harvest_gcmd_datasets._ingest_dataset(
                conn, dataset, harvest_run_id=42
            )

        assert entity_id == 999
        assert alias_count == 1  # short_name MOD09GA != name
        assert rel_count == 2

        mock_ue.assert_called_once()
        ue_kwargs = mock_ue.call_args.kwargs
        assert ue_kwargs["entity_type"] == "dataset"
        assert ue_kwargs["source"] == "gcmd"
        assert ue_kwargs["properties"]["cmr_concept_id"] == "C123-PROVIDER"

        mock_uei.assert_called_once()
        uei_kwargs = mock_uei.call_args.kwargs
        assert uei_kwargs["id_scheme"] == "cmr_concept_id"
        assert uei_kwargs["external_id"] == "C123-PROVIDER"

        mock_uea.assert_called_once()
        uea_kwargs = mock_uea.call_args.kwargs
        assert uea_kwargs["alias"] == "MOD09GA"

        # Relationships use the predicate map
        predicates = [c.kwargs["predicate"] for c in mock_uer.call_args_list]
        assert "observed_by" in predicates
        assert "measured_with" in predicates

    def test_unknown_relationship_skipped(self) -> None:
        dataset = self._sample_dataset()
        cursor = _make_mock_cursor([(303, "weird_link")])
        conn = _make_mock_conn(cursor)

        with patch.object(
            harvest_gcmd_datasets, "upsert_entity", return_value=999
        ), patch.object(harvest_gcmd_datasets, "upsert_entity_identifier"), patch.object(
            harvest_gcmd_datasets, "upsert_entity_alias"
        ), patch.object(
            harvest_gcmd_datasets, "upsert_entity_relationship"
        ) as mock_uer:
            _, _, rel_count = harvest_gcmd_datasets._ingest_dataset(
                conn, dataset, harvest_run_id=1
            )

        assert rel_count == 0
        mock_uer.assert_not_called()

    def test_no_alias_when_short_name_missing(self) -> None:
        dataset = self._sample_dataset()
        dataset["properties"] = {**dataset["properties"], "short_name": ""}
        cursor = _make_mock_cursor([])  # no relationships
        conn = _make_mock_conn(cursor)

        with patch.object(
            harvest_gcmd_datasets, "upsert_entity", return_value=999
        ), patch.object(harvest_gcmd_datasets, "upsert_entity_identifier"), patch.object(
            harvest_gcmd_datasets, "upsert_entity_alias"
        ) as mock_uea, patch.object(
            harvest_gcmd_datasets, "upsert_entity_relationship"
        ):
            _, alias_count, _ = harvest_gcmd_datasets._ingest_dataset(
                conn, dataset, harvest_run_id=1
            )

        assert alias_count == 0
        mock_uea.assert_not_called()


@pytest.mark.unit
class TestRunHarvestDryRun:
    def test_dry_run_does_not_call_run_log(self) -> None:
        cursor = _make_mock_cursor([])
        conn = _make_mock_conn(cursor)

        with patch.object(
            harvest_gcmd_datasets, "get_connection", return_value=conn
        ), patch.object(
            harvest_gcmd_datasets, "HarvestRunLog"
        ) as mock_log_cls:
            counts = harvest_gcmd_datasets.run_harvest(dry_run=True)

        assert "candidate_datasets" in counts
        mock_log_cls.assert_not_called()
