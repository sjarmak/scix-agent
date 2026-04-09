"""Tests for src/scix/link_datasets.py — document-dataset linking pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scix.link_datasets import (
    DatasetResolver,
    DatasetResolverMatch,
    _extract_dataset_mentions,
    _parse_ads_data_field,
    get_dataset_linking_progress,
    link_datasets_batch,
)


# ---------------------------------------------------------------------------
# _parse_ads_data_field
# ---------------------------------------------------------------------------


class TestParseAdsDataField:
    def test_typical_values(self) -> None:
        data = ["CDS:1", "IRSA:2", "NED:4", "SIMBAD:492903"]
        result = _parse_ads_data_field(data)
        assert result == {"CDS": 1, "IRSA": 2, "NED": 4, "SIMBAD": 492903}

    def test_empty_list(self) -> None:
        assert _parse_ads_data_field([]) == {}

    def test_none_input(self) -> None:
        assert _parse_ads_data_field(None) == {}

    def test_malformed_entries_skipped(self) -> None:
        data = ["CDS:1", "badvalue", "", "IRSA:abc", "NED:3"]
        result = _parse_ads_data_field(data)
        assert result == {"CDS": 1, "NED": 3}

    def test_colon_in_name(self) -> None:
        # Some rare entries might have colons in names
        data = ["ESA:1"]
        result = _parse_ads_data_field(data)
        assert result == {"ESA": 1}


# ---------------------------------------------------------------------------
# _extract_dataset_mentions
# ---------------------------------------------------------------------------


class TestExtractDatasetMentions:
    def test_per_type_payload(self) -> None:
        payload = {"entities": ["SDSS DR16", "ImageNet"]}
        mentions = _extract_dataset_mentions(payload, "datasets")
        assert len(mentions) == 2
        assert ("SDSS DR16", "extraction") in mentions
        assert ("ImageNet", "extraction") in mentions

    def test_combined_payload_datasets_key(self) -> None:
        payload = {
            "instruments": ["HST"],
            "datasets": ["SDSS DR16", "2MASS"],
        }
        mentions = _extract_dataset_mentions(payload, "entity_extraction_v3")
        # Only extracts from "datasets" key
        assert len(mentions) == 2
        assert ("SDSS DR16", "extraction") in mentions
        assert ("2MASS", "extraction") in mentions

    def test_empty_payload(self) -> None:
        assert _extract_dataset_mentions({}, "datasets") == []

    def test_whitespace_stripped(self) -> None:
        payload = {"entities": ["  SDSS  ", ""]}
        mentions = _extract_dataset_mentions(payload, "datasets")
        assert len(mentions) == 1
        assert mentions[0][0] == "SDSS"

    def test_non_string_items_skipped(self) -> None:
        payload = {"entities": [123, None, "COCO"]}
        mentions = _extract_dataset_mentions(payload, "datasets")
        assert len(mentions) == 1
        assert mentions[0][0] == "COCO"


# ---------------------------------------------------------------------------
# DatasetResolver
# ---------------------------------------------------------------------------


class TestDatasetResolver:
    def _make_conn(
        self,
        datasets: list[tuple[int, str]],
    ) -> MagicMock:
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cursor.fetchall.return_value = datasets
        return conn

    def test_exact_match(self) -> None:
        conn = self._make_conn(datasets=[(1, "Sloan Digital Sky Survey")])
        resolver = DatasetResolver(conn)
        match = resolver.resolve("sloan digital sky survey")
        assert match is not None
        assert match.dataset_id == 1
        assert match.confidence == 1.0
        assert match.match_method == "name_exact"

    def test_no_match(self) -> None:
        conn = self._make_conn(datasets=[])
        resolver = DatasetResolver(conn)
        assert resolver.resolve("nonexistent") is None

    def test_cache_built_once(self) -> None:
        conn = self._make_conn(datasets=[(1, "Test")])
        resolver = DatasetResolver(conn)
        resolver.resolve("test")
        resolver.resolve("test")
        assert conn.cursor.call_count == 1

    def test_case_insensitive(self) -> None:
        conn = self._make_conn(datasets=[(1, "ImageNet")])
        resolver = DatasetResolver(conn)
        match = resolver.resolve("IMAGENET")
        assert match is not None
        assert match.dataset_id == 1


# ---------------------------------------------------------------------------
# link_datasets_batch
# ---------------------------------------------------------------------------


def _mock_conn_for_linking(
    extraction_rows: list[tuple[str, str, dict]],
    linked_bibcodes: list[str] | None = None,
    datasets: list[tuple[int, str]] | None = None,
) -> MagicMock:
    """Build a mock connection for link_datasets_batch tests."""
    conn = MagicMock()
    linked_bibcodes = linked_bibcodes or []
    datasets = datasets or []

    all_bibcodes = list({r[0] for r in extraction_rows})
    unlinked = [b for b in all_bibcodes if b not in linked_bibcodes]

    call_results: list[list] = []

    # Call 1: SELECT DISTINCT bibcode (with NOT EXISTS filtering)
    call_results.append([(b,) for b in unlinked])

    # Call 2: SELECT bibcode, extraction_type, payload (batch)
    call_results.append([r for r in extraction_rows if r[0] in unlinked])

    # Call 3: Dataset cache build
    call_results.append(datasets)

    cursor_mock = MagicMock()
    cursor_mock.fetchall = MagicMock(side_effect=call_results)
    cursor_mock.fetchone = MagicMock(return_value=None)

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=cursor_mock)
    ctx.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = ctx

    return conn


class TestLinkDatasetsBatch:
    def test_basic_linking(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "datasets", {"entities": ["SDSS DR16"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            datasets=[(10, "SDSS DR16")],
        )

        result = link_datasets_batch(conn, batch_size=100, resume=True)

        assert result["bibcodes_processed"] == 1
        assert result["links_created"] == 1
        assert result["skipped_no_match"] == 0

    def test_no_match_counted(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "datasets", {"entities": ["Unknown Dataset"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            datasets=[],
        )

        result = link_datasets_batch(conn, batch_size=100, resume=True)

        assert result["skipped_no_match"] == 1
        assert result["links_created"] == 0

    def test_resume_skips_linked(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "datasets", {"entities": ["SDSS DR16"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            linked_bibcodes=["2024ApJ...1A"],
            datasets=[(10, "SDSS DR16")],
        )

        result = link_datasets_batch(conn, batch_size=100, resume=True)

        assert result["bibcodes_processed"] == 0

    def test_commit_called_per_batch(self) -> None:
        conn = MagicMock()
        cursor_mock = MagicMock()

        call_idx = {"i": 0}
        results_sequence = [
            # Call 1: SELECT DISTINCT bibcode
            [("2024A",), ("2024B",)],
            # Batch 1: extraction payload for 2024A
            [("2024A", "datasets", {"entities": ["SDSS"]})],
            # DatasetResolver cache
            [(10, "SDSS")],
            # Batch 2: extraction payload for 2024B
            [("2024B", "datasets", {"entities": ["SDSS"]})],
        ]

        def fetchall_side_effect() -> list:
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx < len(results_sequence):
                return results_sequence[idx]
            return []

        cursor_mock.fetchall = MagicMock(side_effect=fetchall_side_effect)
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=cursor_mock)
        ctx.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = ctx

        result = link_datasets_batch(conn, batch_size=1, resume=True)

        assert result["bibcodes_processed"] == 2
        assert conn.commit.call_count == 2

    def test_dry_run_no_writes(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "datasets", {"entities": ["SDSS"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            datasets=[(10, "SDSS")],
        )

        result = link_datasets_batch(conn, batch_size=100, resume=True, dry_run=True)

        assert result["links_created"] == 1
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.executemany.assert_not_called()
        conn.commit.assert_not_called()

    def test_empty_extractions(self) -> None:
        conn = MagicMock()
        cursor_mock = MagicMock()
        cursor_mock.fetchall.return_value = []
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=cursor_mock)
        ctx.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = ctx

        result = link_datasets_batch(conn, batch_size=100)

        assert result["bibcodes_processed"] == 0
        assert result["links_created"] == 0

    def test_inserts_to_document_datasets(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "datasets", {"entities": ["SDSS"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            datasets=[(10, "SDSS")],
        )

        link_datasets_batch(conn, batch_size=100, resume=True)

        cursor = conn.cursor.return_value.__enter__.return_value
        assert cursor.executemany.call_count > 0
        sql_text = cursor.executemany.call_args[0][0]
        assert "INSERT INTO document_datasets" in sql_text
        assert "ON CONFLICT" in sql_text
        assert "DO NOTHING" in sql_text

    def test_link_type_is_analyzes_dataset(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "datasets", {"entities": ["SDSS"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            datasets=[(10, "SDSS")],
        )

        link_datasets_batch(conn, batch_size=100, resume=True)

        cursor = conn.cursor.return_value.__enter__.return_value
        params_list = cursor.executemany.call_args[0][1]
        # Param tuple: (bibcode, dataset_id, link_type, confidence, match_method)
        assert params_list[0][2] == "analyzes_dataset"

    def test_match_stats_in_summary(self) -> None:
        extraction_rows = [
            ("2024ApJ...1A", "datasets", {"entities": ["SDSS", "Unknown"]}),
        ]
        conn = _mock_conn_for_linking(
            extraction_rows=extraction_rows,
            datasets=[(10, "SDSS")],
        )

        result = link_datasets_batch(conn, batch_size=100, resume=True)

        assert result["links_created"] == 1
        assert result["skipped_no_match"] == 1
        assert "by_method" in result
        assert result["by_method"]["extraction"] == 1


# ---------------------------------------------------------------------------
# get_dataset_linking_progress
# ---------------------------------------------------------------------------


class TestGetDatasetLinkingProgress:
    def test_returns_progress(self) -> None:
        conn = MagicMock()
        cursor_mock = MagicMock()
        cursor_mock.fetchone = MagicMock(side_effect=[(50,), (20,)])
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=cursor_mock)
        ctx.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = ctx

        result = get_dataset_linking_progress(conn)
        assert result == {
            "total_bibcodes": 50,
            "linked_bibcodes": 20,
            "pending_bibcodes": 30,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_default_args(self) -> None:
        from scripts.link_datasets import main

        with (
            patch("scripts.link_datasets.get_connection") as mock_conn,
            patch("scripts.link_datasets.link_datasets_batch") as mock_link,
            patch(
                "scripts.link_datasets.get_dataset_linking_progress"
            ) as mock_progress,
        ):
            mock_link.return_value = {
                "bibcodes_processed": 0,
                "links_created": 0,
                "skipped_no_match": 0,
                "by_method": {},
            }
            mock_progress.return_value = {
                "total_bibcodes": 0,
                "linked_bibcodes": 0,
                "pending_bibcodes": 0,
            }

            main([])

            mock_conn.assert_called_once_with(None)
            mock_link.assert_called_once()
            kwargs = mock_link.call_args[1]
            assert kwargs["batch_size"] == 1000
            assert kwargs["resume"] is False
            assert kwargs["dry_run"] is False

    def test_custom_args(self) -> None:
        from scripts.link_datasets import main

        with (
            patch("scripts.link_datasets.get_connection") as mock_conn,
            patch("scripts.link_datasets.link_datasets_batch") as mock_link,
            patch(
                "scripts.link_datasets.get_dataset_linking_progress"
            ) as mock_progress,
        ):
            mock_link.return_value = {
                "bibcodes_processed": 0,
                "links_created": 0,
                "skipped_no_match": 0,
                "by_method": {},
            }
            mock_progress.return_value = {
                "total_bibcodes": 0,
                "linked_bibcodes": 0,
                "pending_bibcodes": 0,
            }

            main(
                [
                    "--batch-size",
                    "500",
                    "--resume",
                    "--db-url",
                    "dbname=test",
                    "--dry-run",
                ]
            )

            mock_conn.assert_called_once_with("dbname=test")
            kwargs = mock_link.call_args[1]
            assert kwargs["batch_size"] == 500
            assert kwargs["resume"] is True
            assert kwargs["dry_run"] is True
