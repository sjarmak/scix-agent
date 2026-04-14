"""Tests for src/scix/sources/s2_datasets.py — Semantic Scholar Datasets API.

Covers:
    - S2 Datasets API client (mocked HTTP)
    - Presigned S3 URL handling
    - S2ORC body_text normalization
    - Citation intent parsing
    - S2AG metadata pruning
    - DOI/arXiv join logic for citation merging
    - Production DSN guard
    - API key from env var (never hardcoded)
    - Health ping logic

Unit tests (no DB required) are always runnable.
Integration tests require SCIX_TEST_DSN.
"""

from __future__ import annotations

import os
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scix.db import is_production_dsn  # noqa: E402
from scix.sources.s2_datasets import (  # noqa: E402
    S2ClientConfig,
    S2DatasetsClient,
    S2HealthPing,
    S2OrcBodyNormalizer,
    CitationIntentMerger,
    ProductionGuardError,
    normalize_s2orc_body,
    parse_citation_intent,
    prune_s2ag_metadata,
)

# ---------------------------------------------------------------------------
# S2ClientConfig — frozen, env-var API key
# ---------------------------------------------------------------------------


class TestS2ClientConfig:
    def test_config_is_frozen(self) -> None:
        cfg = S2ClientConfig(api_key="test-key")
        with pytest.raises(FrozenInstanceError):
            cfg.api_key = "mutated"  # type: ignore[misc]

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "env-key-123")
        cfg = S2ClientConfig.from_env()
        assert cfg.api_key == "env-key-123"

    def test_api_key_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        with pytest.raises(KeyError, match="SEMANTIC_SCHOLAR_API_KEY"):
            S2ClientConfig.from_env()

    def test_api_key_never_hardcoded(self) -> None:
        """Ensure no default API key is baked into the config."""
        source_file = (
            Path(__file__).resolve().parent.parent / "src" / "scix" / "sources" / "s2_datasets.py"
        )
        content = source_file.read_text()
        # Should not contain any literal API key patterns
        # No hardcoded API key values (header name "x-api-key" is fine, but
        # a string that looks like a real key value must not appear)
        assert 'api_key = "' not in content and "api_key = '" not in content
        # The real check: there must be no hardcoded default for api_key
        # Config should require explicit key or env var
        cfg_lines = [
            line for line in content.splitlines() if "api_key" in line and "default" in line.lower()
        ]
        for line in cfg_lines:
            # Allow None as default (forces explicit setting), disallow string literals
            assert (
                '""' not in line and "'" not in line.split("#")[0].split("default")[1]
                if "default" in line.lower()
                else True
            )

    def test_defaults(self) -> None:
        cfg = S2ClientConfig(api_key="k")
        assert cfg.base_url == "https://api.semanticscholar.org"
        assert cfg.datasets_api_path == "/datasets/v1/release"
        assert cfg.graph_api_path == "/graph/v1"


# ---------------------------------------------------------------------------
# S2DatasetsClient — API client with presigned URL handling
# ---------------------------------------------------------------------------


class TestS2DatasetsClient:
    def _make_client(self, api_key: str = "test-key") -> S2DatasetsClient:
        return S2DatasetsClient(S2ClientConfig(api_key=api_key))

    def test_get_latest_release(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = {"release_id": "2026-04-07"}
        mock_response.status_code = 200

        with patch.object(client._http, "get", return_value=mock_response) as mock_get:
            result = client.get_latest_release()
            assert result == {"release_id": "2026-04-07"}
            call_args = mock_get.call_args
            assert "datasets/v1/release/latest" in call_args[0][0]

    def test_get_dataset_partitions(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "files": [
                "https://s3.amazonaws.com/bucket/s2orc/part-00000.gz",
                "https://s3.amazonaws.com/bucket/s2orc/part-00001.gz",
            ]
        }
        mock_response.status_code = 200

        with patch.object(client._http, "get", return_value=mock_response) as mock_get:
            urls = client.get_dataset_partitions("s2orc", release_id="2026-04-07")
            assert len(urls) == 2
            assert all(u.startswith("https://") for u in urls)

    def test_presigned_urls_are_s3_signed(self) -> None:
        """Verify client handles presigned S3 URLs (with query params)."""
        client = self._make_client()
        mock_response = MagicMock()
        presigned_url = (
            "https://s3-us-west-2.amazonaws.com/ai2-s2-datasets/"
            "s2orc/part-00000.jsonl.gz?"
            "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA..."
            "&X-Amz-Date=20260407T000000Z&X-Amz-Expires=3600"
            "&X-Amz-SignedHeaders=host&X-Amz-Signature=abc123"
        )
        mock_response.json.return_value = {"files": [presigned_url]}
        mock_response.status_code = 200

        with patch.object(client._http, "get", return_value=mock_response):
            urls = client.get_dataset_partitions("s2orc")
            assert len(urls) == 1
            assert "X-Amz-Signature" in urls[0]

    def test_api_key_in_headers(self) -> None:
        client = self._make_client(api_key="my-secret-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.status_code = 200

        with patch.object(client._http, "get", return_value=mock_response) as mock_get:
            client.get_latest_release()
            call_kwargs = mock_get.call_args
            headers = call_kwargs[1].get("headers", {}) if call_kwargs[1] else {}
            assert headers.get("x-api-key") == "my-secret-key"

    def test_get_diff_partitions(self) -> None:
        """Test incremental diff support between two releases."""
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "files": ["https://s3.amazonaws.com/bucket/diff/part-00000.gz"]
        }
        mock_response.status_code = 200

        with patch.object(client._http, "get", return_value=mock_response) as mock_get:
            urls = client.get_diff_partitions(
                "s2orc", from_release="2026-03-01", to_release="2026-04-07"
            )
            assert len(urls) == 1
            call_url = mock_get.call_args[0][0]
            assert "diff" in call_url


# ---------------------------------------------------------------------------
# S2ORC body_text normalization
# ---------------------------------------------------------------------------


class TestNormalizeS2orcBody:
    def test_basic_body_text(self) -> None:
        s2orc_record = {
            "corpusid": 12345,
            "externalids": {"DOI": "10.1234/test", "ArXiv": "2024.12345"},
            "content": {
                "source": {"pdfurls": ["https://example.com/paper.pdf"]},
                "text": [
                    {"section": "Introduction", "text": "This paper studies X."},
                    {"section": "Methods", "text": "We use method Y."},
                    {"section": "Results", "text": "We found Z."},
                ],
            },
        }
        result = normalize_s2orc_body(s2orc_record)
        assert result is not None
        corpus_id, body_text, sections = result
        assert corpus_id == 12345
        assert "Introduction" in body_text
        assert "This paper studies X." in body_text
        assert "Methods" in body_text
        assert "We use method Y." in body_text
        assert len(sections) == 3

    def test_missing_content_returns_none(self) -> None:
        result = normalize_s2orc_body({"corpusid": 1})
        assert result is None

    def test_empty_text_returns_none(self) -> None:
        result = normalize_s2orc_body({"corpusid": 1, "content": {"text": []}})
        assert result is None

    def test_null_text_returns_none(self) -> None:
        result = normalize_s2orc_body({"corpusid": 1, "content": {"text": None}})
        assert result is None

    def test_missing_corpusid_returns_none(self) -> None:
        result = normalize_s2orc_body(
            {"content": {"text": [{"section": "Intro", "text": "hello"}]}}
        )
        assert result is None

    def test_cite_spans_preserved_in_section_metadata(self) -> None:
        s2orc_record = {
            "corpusid": 99,
            "content": {
                "text": [
                    {
                        "section": "Related Work",
                        "text": "As shown in [1], this is relevant.",
                        "cite_spans": [
                            {"start": 12, "end": 15, "text": "[1]", "ref_id": "BIBREF0"}
                        ],
                    },
                ],
            },
        }
        result = normalize_s2orc_body(s2orc_record)
        assert result is not None
        _, _, sections = result
        assert sections[0]["cite_spans"] is not None
        assert len(sections[0]["cite_spans"]) == 1


# ---------------------------------------------------------------------------
# Citation intent parsing
# ---------------------------------------------------------------------------


class TestParseCitationIntent:
    def test_basic_intent(self) -> None:
        s2_citation = {
            "citingcorpusid": 100,
            "citedcorpusid": 200,
            "isinfluential": True,
            "intents": ["methodology", "background"],
        }
        result = parse_citation_intent(s2_citation)
        assert result is not None
        citing, cited, is_influential, intents = result
        assert citing == 100
        assert cited == 200
        assert is_influential is True
        assert intents == ["methodology", "background"]

    def test_empty_intents(self) -> None:
        s2_citation = {
            "citingcorpusid": 100,
            "citedcorpusid": 200,
            "isinfluential": False,
            "intents": [],
        }
        result = parse_citation_intent(s2_citation)
        assert result is not None
        _, _, is_influential, intents = result
        assert is_influential is False
        assert intents == []

    def test_null_intents(self) -> None:
        s2_citation = {
            "citingcorpusid": 100,
            "citedcorpusid": 200,
            "isinfluential": False,
        }
        result = parse_citation_intent(s2_citation)
        assert result is not None
        _, _, _, intents = result
        assert intents == []

    def test_missing_corpus_ids_returns_none(self) -> None:
        assert parse_citation_intent({"isinfluential": True}) is None
        assert parse_citation_intent({"citingcorpusid": 1}) is None
        assert parse_citation_intent({"citedcorpusid": 2}) is None


# ---------------------------------------------------------------------------
# S2AG metadata pruning
# ---------------------------------------------------------------------------


class TestPruneS2agMetadata:
    def test_prunes_to_essential_fields(self) -> None:
        full_record = {
            "corpusid": 42,
            "externalids": {"DOI": "10.1234/test", "ArXiv": "2024.12345"},
            "title": "A Test Paper",
            "authors": [{"authorId": "1", "name": "Alice"}],
            "year": 2024,
            "venue": "NeurIPS",
            "publicationvenueid": "abc-123",
            "referencecount": 30,
            "citationcount": 5,
            "influentialcitationcount": 2,
            "isopenaccess": True,
            "s2fieldsofstudy": [{"category": "Computer Science", "source": "s2"}],
            "publicationtypes": ["JournalArticle"],
            "publicationdate": "2024-06-15",
            "journal": {"name": "Test Journal", "volume": "1"},
            "extra_field_to_drop": "should not appear",
            "another_extra": [1, 2, 3],
        }
        pruned = prune_s2ag_metadata(full_record)
        assert pruned is not None
        assert pruned["corpusid"] == 42
        assert pruned["title"] == "A Test Paper"
        assert "extra_field_to_drop" not in pruned
        assert "another_extra" not in pruned

    def test_missing_corpusid_returns_none(self) -> None:
        assert prune_s2ag_metadata({"title": "No ID"}) is None

    def test_minimal_record(self) -> None:
        pruned = prune_s2ag_metadata({"corpusid": 1})
        assert pruned is not None
        assert pruned["corpusid"] == 1


# ---------------------------------------------------------------------------
# Production DSN guard
# ---------------------------------------------------------------------------


class TestProductionGuard:
    def test_production_dsn_blocked(self) -> None:
        merger = CitationIntentMerger(dsn="dbname=scix")
        with pytest.raises(ProductionGuardError, match="production"):
            merger._check_production_guard()

    def test_test_dsn_allowed(self) -> None:
        merger = CitationIntentMerger(dsn="dbname=scix_test")
        merger._check_production_guard()  # should not raise

    def test_none_dsn_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scix.sources.s2_datasets as s2_mod

        monkeypatch.setattr(s2_mod, "DEFAULT_DSN", "dbname=scix")
        merger = CitationIntentMerger(dsn=None)
        with pytest.raises(ProductionGuardError, match="production"):
            merger._check_production_guard()

    def test_uri_production_dsn_blocked(self) -> None:
        merger = CitationIntentMerger(dsn="postgresql://user:pw@host:5432/scix")
        with pytest.raises(ProductionGuardError):
            merger._check_production_guard()

    def test_yes_production_overrides(self) -> None:
        merger = CitationIntentMerger(dsn="dbname=scix", yes_production=True)
        merger._check_production_guard()  # should not raise

    def test_normalizer_production_guard(self) -> None:
        normalizer = S2OrcBodyNormalizer(dsn="dbname=scix")
        with pytest.raises(ProductionGuardError, match="production"):
            normalizer._check_production_guard()


# ---------------------------------------------------------------------------
# Health ping
# ---------------------------------------------------------------------------


class TestHealthPing:
    def test_ping_success(self) -> None:
        ping = S2HealthPing(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(ping._http, "get", return_value=mock_response):
            result = ping.check()
            assert result is True

    def test_ping_failure(self) -> None:
        ping = S2HealthPing(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch.object(ping._http, "get", return_value=mock_response):
            result = ping.check()
            assert result is False

    def test_ping_uses_correct_endpoint(self) -> None:
        ping = S2HealthPing(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(ping._http, "get", return_value=mock_response) as mock_get:
            ping.check()
            call_url = mock_get.call_args[0][0]
            assert "/graph/v1/paper/search" in call_url

    def test_ping_sends_api_key(self) -> None:
        ping = S2HealthPing(api_key="my-key")
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(ping._http, "get", return_value=mock_response) as mock_get:
            ping.check()
            call_kwargs = mock_get.call_args
            headers = call_kwargs[1].get("headers", {}) if call_kwargs[1] else {}
            assert headers.get("x-api-key") == "my-key"

    def test_ping_connection_error_returns_false(self) -> None:
        import requests

        ping = S2HealthPing(api_key="test-key")

        with patch.object(ping._http, "get", side_effect=requests.ConnectionError("no route")):
            result = ping.check()
            assert result is False


# ---------------------------------------------------------------------------
# DOI/arXiv join logic (unit level — just parsing logic)
# ---------------------------------------------------------------------------


class TestCitationIntentMerger:
    def test_build_edge_attrs_jsonb(self) -> None:
        """Test that edge_attrs JSONB is built correctly from S2 citation data."""
        attrs = CitationIntentMerger.build_edge_attrs(
            intents=["methodology", "background"],
            is_influential=True,
        )
        assert attrs["s2_intents"] == ["methodology", "background"]
        assert attrs["s2_is_influential"] is True

    def test_build_edge_attrs_empty_intents(self) -> None:
        attrs = CitationIntentMerger.build_edge_attrs(intents=[], is_influential=False)
        assert attrs["s2_intents"] == []
        assert attrs["s2_is_influential"] is False


# ---------------------------------------------------------------------------
# Integration tests (require SCIX_TEST_DSN)
# ---------------------------------------------------------------------------

_TEST_DSN = os.environ.get("SCIX_TEST_DSN")
needs_test_db = pytest.mark.skipif(
    _TEST_DSN is None or is_production_dsn(_TEST_DSN),
    reason="SCIX_TEST_DSN not set or points at production",
)


@needs_test_db
class TestS2DatasetsIntegration:
    """Integration tests that run against scix_test database."""

    @pytest.fixture(autouse=True)
    def _guard_not_production(self) -> None:
        assert _TEST_DSN is not None
        assert not is_production_dsn(_TEST_DSN), "SCIX_TEST_DSN must not point at production"

    def test_migration_042_creates_tables(self) -> None:
        """Verify the migration created the expected tables."""
        import psycopg

        assert _TEST_DSN is not None
        with psycopg.connect(_TEST_DSN) as conn:
            with conn.cursor() as cur:
                for table in ("papers_s2orc_raw", "papers_s2ag", "s2_citations"):
                    cur.execute(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_name = %s AND table_schema = 'public'",
                        (table,),
                    )
                    row = cur.fetchone()
                    assert row is not None, f"Table {table} not found"

    def test_tables_are_logged(self) -> None:
        """Verify all S2 tables are LOGGED (not UNLOGGED)."""
        import psycopg

        assert _TEST_DSN is not None
        with psycopg.connect(_TEST_DSN) as conn:
            with conn.cursor() as cur:
                for table in ("papers_s2orc_raw", "papers_s2ag", "s2_citations"):
                    cur.execute(
                        "SELECT relpersistence FROM pg_class "
                        "WHERE relname = %s AND relnamespace = 'public'::regnamespace",
                        (table,),
                    )
                    row = cur.fetchone()
                    assert row is not None, f"Table {table} not in pg_class"
                    assert (
                        row[0] == "p"
                    ), f"Table {table} must be LOGGED (relpersistence='p'), got '{row[0]}'"

    def test_citation_edges_has_edge_attrs(self) -> None:
        """Verify edge_attrs column was added to citation_edges."""
        import psycopg

        assert _TEST_DSN is not None
        with psycopg.connect(_TEST_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'citation_edges' AND column_name = 'edge_attrs'",
                )
                row = cur.fetchone()
                assert row is not None, "edge_attrs column not found on citation_edges"
