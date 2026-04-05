"""Tests for the Wikidata instrument enrichment script.

Unit tests use mocked HTTP responses and DB connections for deterministic
testing without network or database access.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enrich_wikidata_instruments import (
    build_sparql_query,
    enrich_entry,
    execute_sparql,
    merge_aliases,
    parse_sparql_results,
    run_enrich,
)

# ---------------------------------------------------------------------------
# Sample SPARQL JSON responses
# ---------------------------------------------------------------------------

SAMPLE_SPARQL_RESPONSE_HST: dict[str, Any] = {
    "results": {
        "bindings": [
            {
                "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q2513"},
                "altLabel": {"type": "literal", "value": "HST", "xml:lang": "en"},
            },
            {
                "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q2513"},
                "altLabel": {
                    "type": "literal",
                    "value": "Hubble",
                    "xml:lang": "en",
                },
            },
            {
                "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q2513"},
                "partLabel": {
                    "type": "literal",
                    "value": "Wide Field Camera 3",
                    "xml:lang": "en",
                },
            },
            {
                "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q2513"},
                "partLabel": {
                    "type": "literal",
                    "value": "Cosmic Origins Spectrograph",
                    "xml:lang": "en",
                },
            },
        ]
    }
}

SAMPLE_SPARQL_RESPONSE_EMPTY: dict[str, Any] = {"results": {"bindings": []}}

SAMPLE_SPARQL_RESPONSE_NO_ALIASES: dict[str, Any] = {
    "results": {
        "bindings": [
            {
                "item": {
                    "type": "uri",
                    "value": "http://www.wikidata.org/entity/Q12345",
                },
            },
        ]
    }
}


def _make_urlopen_response(data: dict[str, Any]) -> MagicMock:
    """Create a mock urllib response returning JSON-encoded data."""
    encoded = json.dumps(data).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = encoded
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Unit tests: build_sparql_query
# ---------------------------------------------------------------------------


class TestBuildSparqlQuery:
    """Tests for SPARQL query construction."""

    def test_contains_facility_name(self) -> None:
        query = build_sparql_query("Hubble Space Telescope")
        assert '"Hubble Space Telescope"@en' in query

    def test_contains_alt_label_clause(self) -> None:
        query = build_sparql_query("Chandra")
        assert "skos:altLabel" in query

    def test_contains_has_part_property(self) -> None:
        query = build_sparql_query("Chandra")
        assert "wdt:P527" in query

    def test_escapes_double_quotes_in_name(self) -> None:
        query = build_sparql_query('The "Great" Observatory')
        assert r"The \"Great\" Observatory" in query

    def test_returns_string(self) -> None:
        result = build_sparql_query("JWST")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Unit tests: parse_sparql_results
# ---------------------------------------------------------------------------


class TestParseSparqlResults:
    """Tests for parsing SPARQL JSON into QID, aliases, and parts."""

    def test_extracts_qid(self) -> None:
        qid, _, _ = parse_sparql_results(SAMPLE_SPARQL_RESPONSE_HST)
        assert qid == "Q2513"

    def test_extracts_aliases(self) -> None:
        _, aliases, _ = parse_sparql_results(SAMPLE_SPARQL_RESPONSE_HST)
        assert "HST" in aliases
        assert "Hubble" in aliases

    def test_extracts_sub_components(self) -> None:
        _, _, parts = parse_sparql_results(SAMPLE_SPARQL_RESPONSE_HST)
        assert "Wide Field Camera 3" in parts
        assert "Cosmic Origins Spectrograph" in parts

    def test_empty_results_returns_none_qid(self) -> None:
        qid, aliases, parts = parse_sparql_results(SAMPLE_SPARQL_RESPONSE_EMPTY)
        assert qid is None
        assert aliases == []
        assert parts == []

    def test_no_aliases_returns_empty_list(self) -> None:
        _, aliases, _ = parse_sparql_results(SAMPLE_SPARQL_RESPONSE_NO_ALIASES)
        assert aliases == []

    def test_aliases_are_sorted(self) -> None:
        _, aliases, _ = parse_sparql_results(SAMPLE_SPARQL_RESPONSE_HST)
        assert aliases == sorted(aliases)

    def test_parts_are_sorted(self) -> None:
        _, _, parts = parse_sparql_results(SAMPLE_SPARQL_RESPONSE_HST)
        assert parts == sorted(parts)

    def test_malformed_results_key(self) -> None:
        qid, aliases, parts = parse_sparql_results({"unexpected": "structure"})
        assert qid is None
        assert aliases == []
        assert parts == []

    def test_deduplicates_aliases(self) -> None:
        """Duplicate altLabel values should appear only once."""
        response: dict[str, Any] = {
            "results": {
                "bindings": [
                    {
                        "item": {
                            "type": "uri",
                            "value": "http://www.wikidata.org/entity/Q100",
                        },
                        "altLabel": {"type": "literal", "value": "Alias1"},
                    },
                    {
                        "item": {
                            "type": "uri",
                            "value": "http://www.wikidata.org/entity/Q100",
                        },
                        "altLabel": {"type": "literal", "value": "Alias1"},
                    },
                ]
            }
        }
        _, aliases, _ = parse_sparql_results(response)
        assert aliases == ["Alias1"]


# ---------------------------------------------------------------------------
# Unit tests: merge_aliases
# ---------------------------------------------------------------------------


class TestMergeAliases:
    """Tests for alias deduplication and merging."""

    def test_merges_new_aliases(self) -> None:
        result = merge_aliases(["HST"], ["Hubble", "Space Telescope"])
        assert "HST" in result
        assert "Hubble" in result
        assert "Space Telescope" in result

    def test_deduplicates_case_insensitive(self) -> None:
        result = merge_aliases(["HST", "hubble"], ["Hubble", "hst"])
        assert len(result) == 2
        # Original casing preserved
        assert "HST" in result
        assert "hubble" in result

    def test_empty_existing(self) -> None:
        result = merge_aliases([], ["A", "B"])
        assert result == ["A", "B"]

    def test_empty_new(self) -> None:
        result = merge_aliases(["A", "B"], [])
        assert result == ["A", "B"]

    def test_both_empty(self) -> None:
        result = merge_aliases([], [])
        assert result == []

    def test_preserves_order_of_existing(self) -> None:
        result = merge_aliases(["C", "A", "B"], ["D"])
        assert result[:3] == ["C", "A", "B"]
        assert result[3] == "D"


# ---------------------------------------------------------------------------
# Unit tests: execute_sparql
# ---------------------------------------------------------------------------


class TestExecuteSparql:
    """Tests for SPARQL endpoint communication."""

    @patch("enrich_wikidata_instruments.urllib.request.urlopen")
    def test_returns_parsed_json(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_response(SAMPLE_SPARQL_RESPONSE_HST)
        result = execute_sparql("SELECT ?item WHERE { ?item rdfs:label 'test'@en }")
        assert "results" in result
        assert len(result["results"]["bindings"]) == 4

    @patch("enrich_wikidata_instruments.urllib.request.urlopen")
    def test_retries_on_network_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [
            urllib.error.URLError("timeout"),
            _make_urlopen_response(SAMPLE_SPARQL_RESPONSE_HST),
        ]
        with patch("enrich_wikidata_instruments.time.sleep"):
            result = execute_sparql("SELECT ?item WHERE { }")
        assert "results" in result

    @patch("enrich_wikidata_instruments.urllib.request.urlopen")
    def test_raises_after_max_retries(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("persistent failure")
        with patch("enrich_wikidata_instruments.time.sleep"):
            with pytest.raises(urllib.error.URLError):
                execute_sparql("SELECT ?item WHERE { }")

    @patch("enrich_wikidata_instruments.urllib.request.urlopen")
    def test_raises_on_malformed_json(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        with pytest.raises(ValueError, match="Malformed"):
            execute_sparql("SELECT ?item WHERE { }")


# ---------------------------------------------------------------------------
# Unit tests: enrich_entry
# ---------------------------------------------------------------------------


class TestEnrichEntry:
    """Tests for single entry enrichment with mocked SPARQL."""

    @patch("enrich_wikidata_instruments.upsert_entry")
    @patch("enrich_wikidata_instruments.execute_sparql")
    def test_enriches_matching_entry(self, mock_sparql: MagicMock, mock_upsert: MagicMock) -> None:
        mock_sparql.return_value = SAMPLE_SPARQL_RESPONSE_HST
        mock_upsert.return_value = {}

        entry = {
            "canonical_name": "Hubble Space Telescope",
            "entity_type": "instrument",
            "source": "aas",
            "external_id": None,
            "aliases": [],
            "metadata": {"wavelength_regimes": ["optical", "ultraviolet"]},
        }
        conn = MagicMock()
        result = enrich_entry(conn, entry)

        assert result is True
        mock_upsert.assert_called_once()
        kwargs = mock_upsert.call_args
        assert "Q2513" in kwargs.kwargs.get("metadata", {}).get("wikidata_qid", "")

    @patch("enrich_wikidata_instruments.execute_sparql")
    def test_returns_false_on_no_match(self, mock_sparql: MagicMock) -> None:
        mock_sparql.return_value = SAMPLE_SPARQL_RESPONSE_EMPTY

        entry = {
            "canonical_name": "Nonexistent Telescope",
            "entity_type": "instrument",
            "source": "aas",
            "aliases": [],
            "metadata": {},
        }
        conn = MagicMock()
        result = enrich_entry(conn, entry)
        assert result is False

    @patch("enrich_wikidata_instruments.execute_sparql")
    def test_returns_false_on_sparql_error(self, mock_sparql: MagicMock) -> None:
        mock_sparql.side_effect = urllib.error.URLError("network down")

        entry = {
            "canonical_name": "Some Telescope",
            "entity_type": "instrument",
            "source": "aas",
            "aliases": [],
            "metadata": {},
        }
        conn = MagicMock()
        result = enrich_entry(conn, entry)
        assert result is False

    @patch("enrich_wikidata_instruments.upsert_entry")
    @patch("enrich_wikidata_instruments.execute_sparql")
    def test_merges_existing_aliases(self, mock_sparql: MagicMock, mock_upsert: MagicMock) -> None:
        mock_sparql.return_value = SAMPLE_SPARQL_RESPONSE_HST
        mock_upsert.return_value = {}

        entry = {
            "canonical_name": "Hubble Space Telescope",
            "entity_type": "instrument",
            "source": "aas",
            "external_id": None,
            "aliases": ["HST"],
            "metadata": {},
        }
        conn = MagicMock()
        enrich_entry(conn, entry)

        kwargs = mock_upsert.call_args.kwargs
        aliases = kwargs["aliases"]
        # HST already existed, should not be duplicated
        assert aliases.count("HST") == 1
        # Hubble should be added
        assert "Hubble" in aliases

    @patch("enrich_wikidata_instruments.upsert_entry")
    @patch("enrich_wikidata_instruments.execute_sparql")
    def test_stores_sub_components_in_metadata(
        self, mock_sparql: MagicMock, mock_upsert: MagicMock
    ) -> None:
        mock_sparql.return_value = SAMPLE_SPARQL_RESPONSE_HST
        mock_upsert.return_value = {}

        entry = {
            "canonical_name": "Hubble Space Telescope",
            "entity_type": "instrument",
            "source": "aas",
            "external_id": None,
            "aliases": [],
            "metadata": {},
        }
        conn = MagicMock()
        enrich_entry(conn, entry)

        kwargs = mock_upsert.call_args.kwargs
        metadata = kwargs["metadata"]
        assert "wikidata_sub_components" in metadata
        assert "Wide Field Camera 3" in metadata["wikidata_sub_components"]


# ---------------------------------------------------------------------------
# Unit tests: run_enrich (end-to-end with mocks)
# ---------------------------------------------------------------------------


class TestRunEnrich:
    """Tests for the full enrichment pipeline with mocked DB and SPARQL."""

    @patch("enrich_wikidata_instruments.enrich_entry")
    @patch("enrich_wikidata_instruments.fetch_instrument_entries")
    @patch("enrich_wikidata_instruments.get_connection")
    @patch("enrich_wikidata_instruments.time.sleep")
    def test_processes_all_entries(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_enrich: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch.return_value = [
            {"canonical_name": "HST", "entity_type": "instrument", "source": "aas"},
            {"canonical_name": "JWST", "entity_type": "instrument", "source": "aas"},
        ]
        mock_enrich.return_value = True

        total, enriched = run_enrich(dsn="dbname=test", delay=0.5)

        assert total == 2
        assert enriched == 2
        assert mock_enrich.call_count == 2

    @patch("enrich_wikidata_instruments.enrich_entry")
    @patch("enrich_wikidata_instruments.fetch_instrument_entries")
    @patch("enrich_wikidata_instruments.get_connection")
    @patch("enrich_wikidata_instruments.time.sleep")
    def test_counts_only_enriched(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_enrich: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch.return_value = [
            {"canonical_name": "HST", "entity_type": "instrument", "source": "aas"},
            {"canonical_name": "Unknown", "entity_type": "instrument", "source": "aas"},
        ]
        mock_enrich.side_effect = [True, False]

        total, enriched = run_enrich(dsn="dbname=test")

        assert total == 2
        assert enriched == 1

    @patch("enrich_wikidata_instruments.enrich_entry")
    @patch("enrich_wikidata_instruments.fetch_instrument_entries")
    @patch("enrich_wikidata_instruments.get_connection")
    @patch("enrich_wikidata_instruments.time.sleep")
    def test_closes_connection(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_enrich: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch.return_value = []
        mock_enrich.return_value = False

        run_enrich()

        mock_conn.close.assert_called_once()

    @patch("enrich_wikidata_instruments.enrich_entry")
    @patch("enrich_wikidata_instruments.fetch_instrument_entries")
    @patch("enrich_wikidata_instruments.get_connection")
    @patch("enrich_wikidata_instruments.time.sleep")
    def test_closes_connection_on_error(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_enrich: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch.side_effect = RuntimeError("DB error")

        with pytest.raises(RuntimeError):
            run_enrich()

        mock_conn.close.assert_called_once()

    @patch("enrich_wikidata_instruments.enrich_entry")
    @patch("enrich_wikidata_instruments.fetch_instrument_entries")
    @patch("enrich_wikidata_instruments.get_connection")
    @patch("enrich_wikidata_instruments.time.sleep")
    def test_rate_limiting_between_requests(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_enrich: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch.return_value = [
            {"canonical_name": "A", "entity_type": "instrument", "source": "aas"},
            {"canonical_name": "B", "entity_type": "instrument", "source": "aas"},
            {"canonical_name": "C", "entity_type": "instrument", "source": "aas"},
        ]
        mock_enrich.return_value = True

        run_enrich(dsn="dbname=test", delay=2.0)

        # Sleep called between entries (not before first)
        sleep_calls = [c for c in mock_sleep.call_args_list if c == call(2.0)]
        assert len(sleep_calls) == 2

    @patch("enrich_wikidata_instruments.enrich_entry")
    @patch("enrich_wikidata_instruments.fetch_instrument_entries")
    @patch("enrich_wikidata_instruments.get_connection")
    @patch("enrich_wikidata_instruments.time.sleep")
    def test_empty_entries_returns_zero(
        self,
        mock_sleep: MagicMock,
        mock_get_conn: MagicMock,
        mock_fetch: MagicMock,
        mock_enrich: MagicMock,
    ) -> None:
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_fetch.return_value = []

        total, enriched = run_enrich()

        assert total == 0
        assert enriched == 0
        mock_enrich.assert_not_called()
