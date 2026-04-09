"""Tests for Papers With Code methods harvester.

Unit tests verify parsing logic with mock data (no network, no DB).
Integration tests (marked @pytest.mark.integration) require a running scix
database with migration 013 applied.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from helpers import DSN, is_production_dsn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_pwc_methods import download_methods, load_methods, parse_methods, run_pipeline

# ---------------------------------------------------------------------------
# Sample data factory
# ---------------------------------------------------------------------------


def _make_method(
    *,
    name: str = "ResNet",
    full_name: str = "Residual Network",
    description: str = "A deep residual learning framework.",
    introduced_year: int | None = 2015,
    source_url: str = "https://paperswithcode.com/method/resnet",
    collection: str | dict[str, Any] | None = "Computer Vision Models",
    paper: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a single PWC method object for testing."""
    method: dict[str, Any] = {
        "name": name,
        "full_name": full_name,
        "description": description,
    }
    if introduced_year is not None:
        method["introduced_year"] = introduced_year
    if source_url:
        method["source_url"] = source_url
    if collection is not None:
        method["main_collection"] = collection
    if paper is not None:
        method["paper"] = paper
    return method


def _write_methods_gzip(methods: list[dict[str, Any]], dest: Path) -> Path:
    """Write a list of method dicts as gzip-compressed JSON to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt", encoding="utf-8") as fh:
        json.dump(methods, fh)
    return dest


# ---------------------------------------------------------------------------
# Unit tests: parse_methods
# ---------------------------------------------------------------------------


class TestParseMethods:
    """Unit tests for parse_methods — no network or DB required."""

    def test_basic_parse(self, tmp_path: Path) -> None:
        """Parses a standard method entry correctly."""
        methods = [_make_method()]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")

        entries = parse_methods(data_path)
        assert len(entries) == 1

        entry = entries[0]
        assert entry["canonical_name"] == "Residual Network"
        assert entry["entity_type"] == "method"
        assert entry["source"] == "pwc"
        assert "ResNet" in entry["aliases"]
        assert entry["metadata"]["description"] == "A deep residual learning framework."
        assert entry["metadata"]["introduced_year"] == 2015

    def test_full_name_preferred_over_name(self, tmp_path: Path) -> None:
        """canonical_name uses full_name when available."""
        methods = [_make_method(name="BERT", full_name="Bidirectional Encoder Representations")]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")

        entries = parse_methods(data_path)
        assert entries[0]["canonical_name"] == "Bidirectional Encoder Representations"
        assert "BERT" in entries[0]["aliases"]

    def test_name_used_when_no_full_name(self, tmp_path: Path) -> None:
        """Falls back to name when full_name is empty."""
        methods = [_make_method(name="Dropout", full_name="")]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")

        entries = parse_methods(data_path)
        assert entries[0]["canonical_name"] == "Dropout"
        assert entries[0]["aliases"] == []

    def test_no_alias_when_name_equals_full_name(self, tmp_path: Path) -> None:
        """No alias added when name and full_name are identical."""
        methods = [_make_method(name="Attention", full_name="Attention")]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")

        entries = parse_methods(data_path)
        assert entries[0]["aliases"] == []

    def test_skips_methods_with_no_name(self, tmp_path: Path) -> None:
        """Methods with neither name nor full_name are skipped."""
        methods = [
            _make_method(name="", full_name=""),
            _make_method(name="Valid", full_name="Valid Method"),
        ]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")

        entries = parse_methods(data_path)
        assert len(entries) == 1
        assert entries[0]["canonical_name"] == "Valid Method"

    def test_collection_as_dict(self, tmp_path: Path) -> None:
        """Handles main_collection as a dict with 'name' key."""
        methods = [_make_method(collection={"name": "NLP Models", "id": 42})]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")

        entries = parse_methods(data_path)
        assert entries[0]["metadata"]["collection"] == "NLP Models"

    def test_collection_as_string(self, tmp_path: Path) -> None:
        """Handles main_collection as a plain string."""
        methods = [_make_method(collection="Generative Models")]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")

        entries = parse_methods(data_path)
        assert entries[0]["metadata"]["collection"] == "Generative Models"

    def test_paper_metadata_extracted(self, tmp_path: Path) -> None:
        """Paper info is included in metadata when present."""
        methods = [
            _make_method(
                paper={
                    "title": "Deep Residual Learning",
                    "url": "https://arxiv.org/abs/1512.03385",
                    "arxiv_id": "1512.03385",
                }
            )
        ]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")

        entries = parse_methods(data_path)
        paper = entries[0]["metadata"]["paper"]
        assert paper["title"] == "Deep Residual Learning"
        assert paper["arxiv_id"] == "1512.03385"

    def test_empty_description_omitted(self, tmp_path: Path) -> None:
        """Empty description is not included in metadata."""
        methods = [_make_method(description="")]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")

        entries = parse_methods(data_path)
        assert "description" not in entries[0]["metadata"]

    def test_many_methods_parsed(self, tmp_path: Path) -> None:
        """Parsing works with a large number of methods."""
        methods = [
            _make_method(name=f"Method{i}", full_name=f"Full Method {i}") for i in range(1500)
        ]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")

        entries = parse_methods(data_path)
        assert len(entries) == 1500


# ---------------------------------------------------------------------------
# Unit tests: download_methods
# ---------------------------------------------------------------------------


class TestDownloadMethods:
    """Unit tests for download_methods — mocked network."""

    def test_skips_existing_file(self, tmp_path: Path) -> None:
        """Does not download if file already exists."""
        dest = tmp_path / "methods.json.gz"
        dest.write_bytes(b"existing data")

        result = download_methods(dest=dest)
        assert result == dest
        assert dest.read_bytes() == b"existing data"

    def test_downloads_when_missing(self, tmp_path: Path) -> None:
        """Downloads file when it does not exist."""
        dest = tmp_path / "sub" / "methods.json.gz"
        fake_data = gzip.compress(json.dumps([]).encode())

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = fake_data
        mock_client.get.return_value = mock_resp

        with patch("harvest_pwc_methods._get_client", return_value=mock_client):
            result = download_methods(dest=dest)

        assert result == dest
        assert dest.exists()
        assert dest.stat().st_size > 0


# ---------------------------------------------------------------------------
# Unit tests: run_pipeline (mocked DB)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Unit tests for the full pipeline with mocked DB."""

    @patch("harvest_pwc_methods.HarvestRunLog")
    @patch("harvest_pwc_methods.get_connection")
    @patch("harvest_pwc_methods.bulk_load")
    def test_pipeline_with_local_file(
        self,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline parses local file and calls bulk_load."""
        methods = [_make_method(name=f"M{i}", full_name=f"Method {i}") for i in range(5)]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 5
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        count = run_pipeline(data_path=data_path, dsn="fake")

        assert count == 5
        mock_bulk_load.assert_called_once()
        loaded_entries = mock_bulk_load.call_args[0][1]
        assert len(loaded_entries) == 5

    def test_pipeline_file_not_found(self, tmp_path: Path) -> None:
        """Pipeline raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            run_pipeline(data_path=tmp_path / "nonexistent.json.gz")

    @patch("harvest_pwc_methods.HarvestRunLog")
    @patch("harvest_pwc_methods.get_connection")
    @patch("harvest_pwc_methods.bulk_load")
    def test_pipeline_creates_harvest_run(
        self,
        mock_bulk_load: MagicMock,
        mock_get_conn: MagicMock,
        mock_run_log_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify harvest_runs record is created and completed."""
        methods = [_make_method(name=f"M{i}", full_name=f"Method {i}") for i in range(5)]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_bulk_load.return_value = 5
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 1
        mock_run_log.run_id = 1
        mock_run_log_cls.return_value = mock_run_log

        run_pipeline(data_path=data_path, dsn="fake")

        mock_run_log_cls.assert_called_once_with(mock_conn, "pwc")
        mock_run_log.start.assert_called_once()
        mock_run_log.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Integration tests (require DB)
# ---------------------------------------------------------------------------


def _has_entity_dictionary(conn: psycopg.Connection) -> bool:
    """Check if entity_dictionary table exists."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'entity_dictionary'
        """)
        return cur.fetchone()[0] == 1


@pytest.fixture()
def db_conn():
    """Provide a database connection, skip if unavailable or table missing."""
    if is_production_dsn(DSN):
        pytest.skip("Refuses to write test data to production. Set SCIX_TEST_DSN.")
    try:
        conn = psycopg.connect(DSN)
    except psycopg.OperationalError:
        pytest.skip("Database not available")
        return

    if not _has_entity_dictionary(conn):
        conn.close()
        pytest.skip("entity_dictionary table not found (migration 013 not applied)")
        return

    yield conn

    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM entity_dictionary WHERE source = 'pwc-test'")
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


@pytest.mark.integration
class TestLoadMethodsIntegration:
    """Integration: loading parsed entries into the database."""

    def test_bulk_load_methods(self, db_conn: psycopg.Connection, tmp_path: Path) -> None:
        """Parsed method entries load successfully via bulk_load."""
        from scix.dictionary import bulk_load as dict_bulk_load

        methods = [
            _make_method(name=f"TestMethod{i}", full_name=f"Test Method {i}") for i in range(10)
        ]
        data_path = _write_methods_gzip(methods, tmp_path / "methods.json.gz")
        entries = parse_methods(data_path)

        # Override source to 'pwc-test' for cleanup
        for entry in entries:
            entry["source"] = "pwc-test"

        count = dict_bulk_load(db_conn, entries)
        assert count == 10

        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entity_dictionary WHERE source = 'pwc-test'")
            db_count = cur.fetchone()[0]
        assert db_count == 10
