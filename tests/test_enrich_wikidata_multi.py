"""Tests for the multi-discipline Wikidata enrichment script.

Unit tests use mocked HTTP responses and DB connections for deterministic
testing without network or database access.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enrich_wikidata_multi import (
    MAX_BATCH_SIZE,
    MIN_DELAY,
    apply_enrichments,
    apply_enrichments_entities,
    build_batch_sparql_query,
    cache_path,
    chunk_list,
    execute_sparql,
    fetch_entities_from_graph,
    load_cache,
    merge_aliases,
    parse_batch_results,
    run_enrich,
    save_cache,
)

# ---------------------------------------------------------------------------
# Sample SPARQL JSON responses
# ---------------------------------------------------------------------------

SAMPLE_BATCH_RESPONSE: dict[str, Any] = {
    "results": {
        "bindings": [
            {
                "item": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q186447",
                },
                "name": {"type": "literal", "value": "MODIS", "xml:lang": "en"},
                "altLabel": {
                    "type": "literal",
                    "value": "Moderate Resolution Imaging Spectroradiometer",
                    "xml:lang": "en",
                },
            },
            {
                "item": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q186447",
                },
                "name": {"type": "literal", "value": "MODIS", "xml:lang": "en"},
                "altLabel": {
                    "type": "literal",
                    "value": "MODIS sensor",
                    "xml:lang": "en",
                },
            },
            {
                "item": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q54321",
                },
                "name": {
                    "type": "literal",
                    "value": "Mars Reconnaissance Orbiter",
                    "xml:lang": "en",
                },
                "altLabel": {
                    "type": "literal",
                    "value": "MRO",
                    "xml:lang": "en",
                },
            },
        ]
    }
}

SAMPLE_EMPTY_RESPONSE: dict[str, Any] = {"results": {"bindings": []}}


def _make_urlopen_response(data: dict[str, Any]) -> MagicMock:
    """Create a mock urllib response returning JSON-encoded data."""
    encoded = json.dumps(data).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = encoded
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: build_batch_sparql_query
# ---------------------------------------------------------------------------


class TestBuildBatchSparqlQuery:
    """Tests for batched SPARQL query construction."""

    def test_single_name_in_values(self) -> None:
        query = build_batch_sparql_query(["MODIS"])
        assert '"MODIS"@en' in query
        assert "VALUES ?name" in query

    def test_multiple_names_in_values(self) -> None:
        query = build_batch_sparql_query(["MODIS", "AIRS", "VIIRS"])
        assert '"MODIS"@en' in query
        assert '"AIRS"@en' in query
        assert '"VIIRS"@en' in query

    def test_contains_alt_label_clause(self) -> None:
        query = build_batch_sparql_query(["MODIS"])
        assert "skos:altLabel" in query

    def test_escapes_double_quotes(self) -> None:
        query = build_batch_sparql_query(['The "Great" Sensor'])
        assert r"The \"Great\" Sensor" in query

    def test_returns_string(self) -> None:
        result = build_batch_sparql_query(["Test"])
        assert isinstance(result, str)

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build_batch_sparql_query([])

    def test_raises_on_exceeding_max_batch(self) -> None:
        names = [f"name_{i}" for i in range(MAX_BATCH_SIZE + 1)]
        with pytest.raises(ValueError, match="exceeds maximum"):
            build_batch_sparql_query(names)

    def test_accepts_max_batch_size(self) -> None:
        names = [f"name_{i}" for i in range(MAX_BATCH_SIZE)]
        query = build_batch_sparql_query(names)
        assert '"name_0"@en' in query
        assert '"name_49"@en' in query


# ---------------------------------------------------------------------------
# Tests: parse_batch_results
# ---------------------------------------------------------------------------


class TestParseBatchResults:
    """Tests for parsing batched SPARQL results."""

    def test_extracts_qid_per_name(self) -> None:
        result = parse_batch_results(SAMPLE_BATCH_RESPONSE)
        assert "MODIS" in result
        assert result["MODIS"][0] == "Q186447"

    def test_extracts_aliases_per_name(self) -> None:
        result = parse_batch_results(SAMPLE_BATCH_RESPONSE)
        _, aliases = result["MODIS"]
        assert "Moderate Resolution Imaging Spectroradiometer" in aliases
        assert "MODIS sensor" in aliases

    def test_multiple_names_parsed(self) -> None:
        result = parse_batch_results(SAMPLE_BATCH_RESPONSE)
        assert "Mars Reconnaissance Orbiter" in result
        assert result["Mars Reconnaissance Orbiter"][0] == "Q54321"
        assert "MRO" in result["Mars Reconnaissance Orbiter"][1]

    def test_empty_results(self) -> None:
        result = parse_batch_results(SAMPLE_EMPTY_RESPONSE)
        assert result == {}

    def test_malformed_structure(self) -> None:
        result = parse_batch_results({"unexpected": "structure"})
        assert result == {}

    def test_aliases_are_sorted(self) -> None:
        result = parse_batch_results(SAMPLE_BATCH_RESPONSE)
        _, aliases = result["MODIS"]
        assert aliases == sorted(aliases)

    def test_deduplicates_aliases(self) -> None:
        response: dict[str, Any] = {
            "results": {
                "bindings": [
                    {
                        "item": {
                            "type": "uri",
                            "value": "http://www.wikidata.org/entity/Q100",
                        },
                        "name": {"type": "literal", "value": "TestItem"},
                        "altLabel": {"type": "literal", "value": "Alias1"},
                    },
                    {
                        "item": {
                            "type": "uri",
                            "value": "http://www.wikidata.org/entity/Q100",
                        },
                        "name": {"type": "literal", "value": "TestItem"},
                        "altLabel": {"type": "literal", "value": "Alias1"},
                    },
                ]
            }
        }
        result = parse_batch_results(response)
        assert result["TestItem"][1] == ["Alias1"]


# ---------------------------------------------------------------------------
# Tests: merge_aliases
# ---------------------------------------------------------------------------


class TestMergeAliases:
    """Tests for alias deduplication and merging."""

    def test_merges_new_aliases(self) -> None:
        result = merge_aliases(["MRO"], ["Mars Reconnaissance Orbiter"])
        assert "MRO" in result
        assert "Mars Reconnaissance Orbiter" in result

    def test_deduplicates_case_insensitive(self) -> None:
        result = merge_aliases(["MRO", "mro2"], ["Mro", "MRO2"])
        assert len(result) == 2

    def test_empty_existing(self) -> None:
        result = merge_aliases([], ["A", "B"])
        assert result == ["A", "B"]

    def test_empty_new(self) -> None:
        result = merge_aliases(["A"], [])
        assert result == ["A"]

    def test_both_empty(self) -> None:
        assert merge_aliases([], []) == []


# ---------------------------------------------------------------------------
# Tests: chunk_list
# ---------------------------------------------------------------------------


class TestChunkList:
    """Tests for the list chunking helper."""

    def test_exact_division(self) -> None:
        result = chunk_list([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_remainder(self) -> None:
        result = chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_chunk(self) -> None:
        result = chunk_list([1, 2, 3], 10)
        assert result == [[1, 2, 3]]

    def test_empty_list(self) -> None:
        result = chunk_list([], 5)
        assert result == []

    def test_chunk_size_one(self) -> None:
        result = chunk_list([1, 2, 3], 1)
        assert result == [[1], [2], [3]]


# ---------------------------------------------------------------------------
# Tests: execute_sparql
# ---------------------------------------------------------------------------


class TestExecuteSparql:
    """Tests for SPARQL endpoint communication."""

    @patch("enrich_wikidata_multi.urllib.request.urlopen")
    def test_returns_parsed_json(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_response(SAMPLE_BATCH_RESPONSE)
        result = execute_sparql("SELECT ?item WHERE { }")
        assert "results" in result

    @patch("enrich_wikidata_multi.urllib.request.urlopen")
    def test_retries_on_network_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [
            urllib.error.URLError("timeout"),
            _make_urlopen_response(SAMPLE_BATCH_RESPONSE),
        ]
        with patch("enrich_wikidata_multi.time.sleep"):
            result = execute_sparql("SELECT ?item WHERE { }")
        assert "results" in result

    @patch("enrich_wikidata_multi.urllib.request.urlopen")
    def test_raises_after_max_retries(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("persistent failure")
        with patch("enrich_wikidata_multi.time.sleep"):
            with pytest.raises(urllib.error.URLError):
                execute_sparql("SELECT ?item WHERE { }")

    @patch("enrich_wikidata_multi.urllib.request.urlopen")
    def test_raises_on_malformed_json(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        with pytest.raises(ValueError, match="Malformed"):
            execute_sparql("SELECT ?item WHERE { }")


# ---------------------------------------------------------------------------
# Tests: cache read/write
# ---------------------------------------------------------------------------


class TestCache:
    """Tests for disk caching of SPARQL responses."""

    def test_cache_path_format(self) -> None:
        p = cache_path(Path("/tmp/cache"), "gcmd", "instrument", 3)
        assert p == Path("/tmp/cache/gcmd_instrument_3.json")

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        cp = tmp_path / "test_cache.json"
        save_cache(cp, SAMPLE_BATCH_RESPONSE)
        loaded = load_cache(cp)
        assert loaded is not None
        assert loaded["results"]["bindings"][0]["name"]["value"] == "MODIS"

    def test_load_missing_file_returns_none(self, tmp_path: Path) -> None:
        cp = tmp_path / "nonexistent.json"
        assert load_cache(cp) is None

    def test_load_invalid_json_returns_none(self, tmp_path: Path) -> None:
        cp = tmp_path / "bad.json"
        cp.write_text("not json", encoding="utf-8")
        assert load_cache(cp) is None

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        cp = tmp_path / "sub" / "dir" / "cache.json"
        save_cache(cp, {"test": True})
        assert cp.exists()


# ---------------------------------------------------------------------------
# Tests: apply_enrichments
# ---------------------------------------------------------------------------


class TestApplyEnrichments:
    """Tests for DB update logic."""

    @patch("enrich_wikidata_multi.upsert_entry")
    def test_updates_matching_entries(self, mock_upsert: MagicMock) -> None:
        mock_upsert.return_value = {}
        entries = [
            {
                "canonical_name": "MODIS",
                "entity_type": "instrument",
                "source": "gcmd",
                "external_id": "GCMD-123",
                "aliases": ["Terra MODIS"],
                "metadata": {"some_key": "some_value"},
            },
        ]
        matches = {"MODIS": ("Q186447", ["Moderate Resolution Imaging Spectroradiometer"])}
        conn = MagicMock()

        enriched = apply_enrichments(conn, entries, matches)

        assert enriched == 1
        mock_upsert.assert_called_once()
        kwargs = mock_upsert.call_args.kwargs
        assert kwargs["metadata"]["wikidata_qid"] == "Q186447"
        assert "Terra MODIS" in kwargs["aliases"]
        assert "Moderate Resolution Imaging Spectroradiometer" in kwargs["aliases"]

    @patch("enrich_wikidata_multi.upsert_entry")
    def test_skips_non_matching_entries(self, mock_upsert: MagicMock) -> None:
        entries = [
            {
                "canonical_name": "Unknown Sensor",
                "entity_type": "instrument",
                "source": "gcmd",
                "aliases": [],
                "metadata": {},
            },
        ]
        matches: dict[str, tuple[str, list[str]]] = {}
        conn = MagicMock()

        enriched = apply_enrichments(conn, entries, matches)

        assert enriched == 0
        mock_upsert.assert_not_called()

    @patch("enrich_wikidata_multi.upsert_entry")
    def test_dry_run_skips_db_write(self, mock_upsert: MagicMock) -> None:
        entries = [
            {
                "canonical_name": "MODIS",
                "entity_type": "instrument",
                "source": "gcmd",
                "external_id": None,
                "aliases": [],
                "metadata": {},
            },
        ]
        matches = {"MODIS": ("Q186447", [])}
        conn = MagicMock()

        enriched = apply_enrichments(conn, entries, matches, dry_run=True)

        assert enriched == 1
        mock_upsert.assert_not_called()

    @patch("enrich_wikidata_multi.upsert_entry")
    def test_preserves_existing_metadata(self, mock_upsert: MagicMock) -> None:
        mock_upsert.return_value = {}
        entries = [
            {
                "canonical_name": "MODIS",
                "entity_type": "instrument",
                "source": "gcmd",
                "external_id": None,
                "aliases": [],
                "metadata": {"original_key": "original_value"},
            },
        ]
        matches = {"MODIS": ("Q186447", [])}
        conn = MagicMock()

        apply_enrichments(conn, entries, matches)

        kwargs = mock_upsert.call_args.kwargs
        assert kwargs["metadata"]["original_key"] == "original_value"
        assert kwargs["metadata"]["wikidata_qid"] == "Q186447"


# ---------------------------------------------------------------------------
# Tests: batch size enforcement
# ---------------------------------------------------------------------------


class TestBatchSizeEnforcement:
    """Tests that batching respects the 50-item limit."""

    def test_chunk_list_respects_max_batch(self) -> None:
        items = list(range(120))
        batches = chunk_list(items, MAX_BATCH_SIZE)
        for batch in batches:
            assert len(batch) <= MAX_BATCH_SIZE
        assert len(batches) == 3  # 50+50+20

    def test_build_query_rejects_oversized(self) -> None:
        names = [f"name_{i}" for i in range(51)]
        with pytest.raises(ValueError, match="exceeds maximum"):
            build_batch_sparql_query(names)


# ---------------------------------------------------------------------------
# Tests: sleep between batches
# ---------------------------------------------------------------------------


class TestSleepBetweenBatches:
    """Tests that rate limiting is enforced between SPARQL batch requests."""

    @patch("enrich_wikidata_multi.apply_enrichments", return_value=0)
    @patch("enrich_wikidata_multi.execute_sparql")
    @patch("enrich_wikidata_multi.fetch_entries")
    @patch("enrich_wikidata_multi.get_connection")
    @patch("enrich_wikidata_multi.time.sleep")
    def test_sleep_called_between_batches(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_sparql: MagicMock,
        mock_apply: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        # 3 entries with batch_size=1 => 3 batches => 2 sleeps
        mock_fetch.return_value = [
            {
                "canonical_name": "A",
                "entity_type": "instrument",
                "source": "gcmd",
                "aliases": [],
                "metadata": {},
            },
            {
                "canonical_name": "B",
                "entity_type": "instrument",
                "source": "gcmd",
                "aliases": [],
                "metadata": {},
            },
            {
                "canonical_name": "C",
                "entity_type": "instrument",
                "source": "gcmd",
                "aliases": [],
                "metadata": {},
            },
        ]
        mock_sparql.return_value = SAMPLE_EMPTY_RESPONSE

        run_enrich(
            dsn="dbname=test",
            source_filter="gcmd",
            entity_type_filter="instrument",
            batch_size=1,
            delay=2.0,
            use_cache=False,
        )

        # Sleep should be called between batches (not before first)
        sleep_calls = [c for c in mock_sleep.call_args_list if c == call(2.0)]
        assert len(sleep_calls) == 2

    @patch("enrich_wikidata_multi.apply_enrichments", return_value=0)
    @patch("enrich_wikidata_multi.execute_sparql")
    @patch("enrich_wikidata_multi.fetch_entries")
    @patch("enrich_wikidata_multi.get_connection")
    @patch("enrich_wikidata_multi.time.sleep")
    def test_delay_enforced_minimum_2s(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_sparql: MagicMock,
        mock_apply: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch.return_value = [
            {
                "canonical_name": "A",
                "entity_type": "instrument",
                "source": "gcmd",
                "aliases": [],
                "metadata": {},
            },
            {
                "canonical_name": "B",
                "entity_type": "instrument",
                "source": "gcmd",
                "aliases": [],
                "metadata": {},
            },
        ]
        mock_sparql.return_value = SAMPLE_EMPTY_RESPONSE

        # Pass delay < 2.0 — should be clamped to 2.0
        run_enrich(
            dsn="dbname=test",
            source_filter="gcmd",
            entity_type_filter="instrument",
            batch_size=1,
            delay=0.5,
            use_cache=False,
        )

        sleep_calls = [c for c in mock_sleep.call_args_list if c == call(MIN_DELAY)]
        assert len(sleep_calls) >= 1


# ---------------------------------------------------------------------------
# Tests: run_enrich pipeline
# ---------------------------------------------------------------------------


class TestRunEnrich:
    """Tests for the full enrichment pipeline with mocked components."""

    @patch("enrich_wikidata_multi.apply_enrichments", return_value=5)
    @patch("enrich_wikidata_multi.execute_sparql")
    @patch("enrich_wikidata_multi.fetch_entries")
    @patch("enrich_wikidata_multi.get_connection")
    @patch("enrich_wikidata_multi.time.sleep")
    def test_processes_single_config(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_sparql: MagicMock,
        mock_apply: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch.return_value = [
            {
                "canonical_name": f"item_{i}",
                "entity_type": "instrument",
                "source": "gcmd",
                "aliases": [],
                "metadata": {},
            }
            for i in range(10)
        ]
        mock_sparql.return_value = SAMPLE_EMPTY_RESPONSE

        total, enriched = run_enrich(
            dsn="dbname=test",
            source_filter="gcmd",
            entity_type_filter="instrument",
            use_cache=False,
        )

        assert total == 10
        assert enriched == 5

    @patch("enrich_wikidata_multi.apply_enrichments", return_value=0)
    @patch("enrich_wikidata_multi.execute_sparql")
    @patch("enrich_wikidata_multi.fetch_entries")
    @patch("enrich_wikidata_multi.get_connection")
    @patch("enrich_wikidata_multi.time.sleep")
    def test_closes_connection(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_sparql: MagicMock,
        mock_apply: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch.return_value = []

        run_enrich(
            dsn="dbname=test",
            source_filter="gcmd",
            entity_type_filter="instrument",
            use_cache=False,
        )

        mock_conn.close.assert_called_once()

    @patch("enrich_wikidata_multi.fetch_entries")
    @patch("enrich_wikidata_multi.get_connection")
    @patch("enrich_wikidata_multi.time.sleep")
    def test_closes_connection_on_error(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch.side_effect = RuntimeError("DB error")

        with pytest.raises(RuntimeError):
            run_enrich(
                dsn="dbname=test",
                source_filter="gcmd",
                entity_type_filter="instrument",
                use_cache=False,
            )

        mock_conn.close.assert_called_once()

    @patch("enrich_wikidata_multi.apply_enrichments", return_value=0)
    @patch("enrich_wikidata_multi.execute_sparql")
    @patch("enrich_wikidata_multi.fetch_entries")
    @patch("enrich_wikidata_multi.get_connection")
    @patch("enrich_wikidata_multi.time.sleep")
    def test_no_configs_returns_zero(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_sparql: MagicMock,
        mock_apply: MagicMock,
    ) -> None:
        total, enriched = run_enrich(
            dsn="dbname=test",
            source_filter="nonexistent",
            use_cache=False,
        )
        assert total == 0
        assert enriched == 0

    @patch("enrich_wikidata_multi.apply_enrichments", return_value=0)
    @patch("enrich_wikidata_multi.execute_sparql")
    @patch("enrich_wikidata_multi.fetch_entries")
    @patch("enrich_wikidata_multi.get_connection")
    @patch("enrich_wikidata_multi.time.sleep")
    def test_uses_cache_when_available(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_sparql: MagicMock,
        mock_apply: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch.return_value = [
            {
                "canonical_name": "MODIS",
                "entity_type": "instrument",
                "source": "gcmd",
                "aliases": [],
                "metadata": {},
            },
        ]

        # Pre-populate cache
        cp = tmp_path / "gcmd_instrument_0.json"
        cp.write_text(json.dumps(SAMPLE_BATCH_RESPONSE), encoding="utf-8")

        run_enrich(
            dsn="dbname=test",
            source_filter="gcmd",
            entity_type_filter="instrument",
            use_cache=True,
            cache_dir=tmp_path,
        )

        # SPARQL should NOT be called since cache was used
        mock_sparql.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: CLI --help
# ---------------------------------------------------------------------------


class TestCli:
    """Tests for the CLI interface."""

    def test_help_exits_zero(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parent.parent / "scripts" / "enrich_wikidata_multi.py"
                ),
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Wikidata" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--source" in result.stdout
        assert "--entity-type" in result.stdout
        assert "--no-cache" in result.stdout
        assert "--entity-source" in result.stdout


# ---------------------------------------------------------------------------
# Tests: fetch_entities_from_graph
# ---------------------------------------------------------------------------


class TestFetchEntitiesFromGraph:
    """Tests for querying the entities table."""

    def test_returns_entities_rows(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"id": 1, "canonical_name": "MODIS", "entity_type": "instrument", "source": "gcmd"},
            {"id": 2, "canonical_name": "AIRS", "entity_type": "instrument", "source": "gcmd"},
        ]
        mock_cursor.__enter__ = lambda self: self
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        result = fetch_entities_from_graph(mock_conn, "gcmd", "instrument")

        assert len(result) == 2
        assert result[0]["canonical_name"] == "MODIS"
        assert result[0]["id"] == 1
        assert result[1]["canonical_name"] == "AIRS"

    def test_passes_correct_params(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.__enter__ = lambda self: self
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        fetch_entities_from_graph(mock_conn, "pds4", "mission")

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "entities" in executed_sql
        params = mock_cursor.execute.call_args[0][1]
        assert params["source"] == "pds4"
        assert params["entity_type"] == "mission"

    def test_returns_empty_for_no_rows(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.__enter__ = lambda self: self
        mock_cursor.__exit__ = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        result = fetch_entities_from_graph(mock_conn, "gcmd", "instrument")
        assert result == []


# ---------------------------------------------------------------------------
# Tests: apply_enrichments_entities
# ---------------------------------------------------------------------------


class TestApplyEnrichmentsEntities:
    """Tests for entity graph enrichment logic."""

    @patch("enrich_wikidata_multi.upsert_entity_alias")
    @patch("enrich_wikidata_multi.upsert_entity_identifier")
    def test_creates_identifier_and_aliases(
        self,
        mock_upsert_id: MagicMock,
        mock_upsert_alias: MagicMock,
    ) -> None:
        entries = [
            {"id": 42, "canonical_name": "MODIS", "entity_type": "instrument", "source": "gcmd"},
        ]
        matches = {"MODIS": ("Q186447", ["Moderate Resolution Imaging Spectroradiometer"])}
        mock_conn = MagicMock()

        enriched = apply_enrichments_entities(mock_conn, entries, matches)

        assert enriched == 1
        mock_upsert_id.assert_called_once_with(
            mock_conn,
            entity_id=42,
            id_scheme="wikidata",
            external_id="Q186447",
            is_primary=False,
        )
        mock_upsert_alias.assert_called_once_with(
            mock_conn,
            entity_id=42,
            alias="Moderate Resolution Imaging Spectroradiometer",
            alias_source="wikidata",
        )
        mock_conn.commit.assert_called_once()

    @patch("enrich_wikidata_multi.upsert_entity_alias")
    @patch("enrich_wikidata_multi.upsert_entity_identifier")
    def test_skips_non_matching_entries(
        self,
        mock_upsert_id: MagicMock,
        mock_upsert_alias: MagicMock,
    ) -> None:
        entries = [
            {"id": 99, "canonical_name": "Unknown", "entity_type": "instrument", "source": "gcmd"},
        ]
        matches: dict[str, tuple[str, list[str]]] = {}
        mock_conn = MagicMock()

        enriched = apply_enrichments_entities(mock_conn, entries, matches)

        assert enriched == 0
        mock_upsert_id.assert_not_called()
        mock_upsert_alias.assert_not_called()

    @patch("enrich_wikidata_multi.upsert_entity_alias")
    @patch("enrich_wikidata_multi.upsert_entity_identifier")
    def test_dry_run_skips_db_writes(
        self,
        mock_upsert_id: MagicMock,
        mock_upsert_alias: MagicMock,
    ) -> None:
        entries = [
            {"id": 42, "canonical_name": "MODIS", "entity_type": "instrument", "source": "gcmd"},
        ]
        matches = {"MODIS": ("Q186447", ["Alias1"])}
        mock_conn = MagicMock()

        enriched = apply_enrichments_entities(mock_conn, entries, matches, dry_run=True)

        assert enriched == 1
        mock_upsert_id.assert_not_called()
        mock_upsert_alias.assert_not_called()
        mock_conn.commit.assert_not_called()

    @patch("enrich_wikidata_multi.upsert_entity_alias")
    @patch("enrich_wikidata_multi.upsert_entity_identifier")
    def test_multiple_aliases_upserted(
        self,
        mock_upsert_id: MagicMock,
        mock_upsert_alias: MagicMock,
    ) -> None:
        entries = [
            {"id": 10, "canonical_name": "MRO", "entity_type": "mission", "source": "pds4"},
        ]
        matches = {"MRO": ("Q54321", ["Mars Reconnaissance Orbiter", "Mars Recon"])}
        mock_conn = MagicMock()

        apply_enrichments_entities(mock_conn, entries, matches)

        assert mock_upsert_alias.call_count == 2
        alias_calls = [c.kwargs["alias"] for c in mock_upsert_alias.call_args_list]
        assert "Mars Reconnaissance Orbiter" in alias_calls
        assert "Mars Recon" in alias_calls

    @patch("enrich_wikidata_multi.upsert_entity_alias")
    @patch("enrich_wikidata_multi.upsert_entity_identifier")
    def test_no_aliases_still_creates_identifier(
        self,
        mock_upsert_id: MagicMock,
        mock_upsert_alias: MagicMock,
    ) -> None:
        entries = [
            {"id": 5, "canonical_name": "TestItem", "entity_type": "target", "source": "pds4"},
        ]
        matches = {"TestItem": ("Q999", [])}
        mock_conn = MagicMock()

        enriched = apply_enrichments_entities(mock_conn, entries, matches)

        assert enriched == 1
        mock_upsert_id.assert_called_once()
        mock_upsert_alias.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: run_enrich with entity_source="entities"
# ---------------------------------------------------------------------------


class TestRunEnrichEntitiesMode:
    """Tests for the enrichment pipeline in entities table mode."""

    @patch("enrich_wikidata_multi.apply_enrichments_entities", return_value=3)
    @patch("enrich_wikidata_multi.execute_sparql")
    @patch("enrich_wikidata_multi.fetch_entities_from_graph")
    @patch("enrich_wikidata_multi.get_connection")
    @patch("enrich_wikidata_multi.time.sleep")
    def test_uses_entities_table_fetch(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch_graph: MagicMock,
        mock_sparql: MagicMock,
        mock_apply_entities: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch_graph.return_value = [
            {"id": 1, "canonical_name": "MODIS", "entity_type": "instrument", "source": "gcmd"},
            {"id": 2, "canonical_name": "AIRS", "entity_type": "instrument", "source": "gcmd"},
        ]
        mock_sparql.return_value = SAMPLE_EMPTY_RESPONSE

        total, enriched = run_enrich(
            dsn="dbname=test",
            source_filter="gcmd",
            entity_type_filter="instrument",
            use_cache=False,
            entity_source="entities",
        )

        assert total == 2
        assert enriched == 3
        mock_fetch_graph.assert_called_once()
        mock_apply_entities.assert_called_once()

    @patch("enrich_wikidata_multi.apply_enrichments_entities", return_value=0)
    @patch("enrich_wikidata_multi.apply_enrichments", return_value=0)
    @patch("enrich_wikidata_multi.execute_sparql")
    @patch("enrich_wikidata_multi.fetch_entries")
    @patch("enrich_wikidata_multi.get_connection")
    @patch("enrich_wikidata_multi.time.sleep")
    def test_dictionary_mode_unchanged(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch_dict: MagicMock,
        mock_sparql: MagicMock,
        mock_apply_dict: MagicMock,
        mock_apply_entities: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch_dict.return_value = [
            {
                "canonical_name": "MODIS",
                "entity_type": "instrument",
                "source": "gcmd",
                "aliases": [],
                "metadata": {},
            },
        ]
        mock_sparql.return_value = SAMPLE_EMPTY_RESPONSE

        run_enrich(
            dsn="dbname=test",
            source_filter="gcmd",
            entity_type_filter="instrument",
            use_cache=False,
            entity_source="dictionary",
        )

        mock_fetch_dict.assert_called_once()
        mock_apply_dict.assert_called_once()
        mock_apply_entities.assert_not_called()

    @patch("enrich_wikidata_multi.apply_enrichments_entities", return_value=0)
    @patch("enrich_wikidata_multi.execute_sparql")
    @patch("enrich_wikidata_multi.fetch_entities_from_graph")
    @patch("enrich_wikidata_multi.get_connection")
    @patch("enrich_wikidata_multi.time.sleep")
    def test_entities_mode_closes_connection(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch_graph: MagicMock,
        mock_sparql: MagicMock,
        mock_apply_entities: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch_graph.return_value = []

        run_enrich(
            dsn="dbname=test",
            source_filter="gcmd",
            entity_type_filter="instrument",
            use_cache=False,
            entity_source="entities",
        )

        mock_conn.close.assert_called_once()
