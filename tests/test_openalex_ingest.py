"""Unit tests for src/scix/sources/openalex.py — OpenAlex S3 ingest loader.

Covers:
    - Inverted-abstract reconstruction
    - Work record pruning/normalization
    - DOI/arXiv ID extraction from Work records
    - Partition manifest validation
    - Production DSN guard (regression: None-dsn bypass)
    - LoaderConfig immutability
    - LoaderStats immutability
    - ArXiv ID extraction from external_ids and IDs

No database required — all tests use pure functions or mocked connections.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scix.sources.openalex import (  # noqa: E402
    LoaderConfig,
    LoaderStats,
    OpenAlexLoader,
    ProductionGuardError,
    extract_arxiv_id,
    normalize_openalex_id,
    prune_work_record,
    reconstruct_abstract,
    validate_manifest,
)

# ---------------------------------------------------------------------------
# reconstruct_abstract — inverted abstract index reassembly
# ---------------------------------------------------------------------------


class TestReconstructAbstract:
    """Test the inverted-abstract reconstruction algorithm."""

    def test_simple_abstract(self) -> None:
        inverted = {"The": [0], "quick": [1], "brown": [2], "fox": [3]}
        assert reconstruct_abstract(inverted) == "The quick brown fox"

    def test_word_at_multiple_positions(self) -> None:
        inverted = {"the": [0, 4], "cat": [1], "sat": [2], "on": [3], "mat": [5]}
        assert reconstruct_abstract(inverted) == "the cat sat on the mat"

    def test_none_returns_none(self) -> None:
        assert reconstruct_abstract(None) is None

    def test_empty_dict_returns_none(self) -> None:
        assert reconstruct_abstract({}) is None

    def test_preserves_punctuation_in_tokens(self) -> None:
        inverted = {"Hello,": [0], "world.": [1]}
        assert reconstruct_abstract(inverted) == "Hello, world."

    def test_sparse_positions_with_gaps(self) -> None:
        """Gaps in positions produce empty-string tokens that become spaces."""
        inverted = {"A": [0], "B": [5]}
        result = reconstruct_abstract(inverted)
        # Positions 1-4 are missing — filled with empty strings
        assert result is not None
        assert "A" in result
        assert "B" in result

    def test_large_abstract(self) -> None:
        """Stress test with 1000 words."""
        words = [f"word{i}" for i in range(1000)]
        inverted = {w: [i] for i, w in enumerate(words)}
        result = reconstruct_abstract(inverted)
        assert result == " ".join(words)


# ---------------------------------------------------------------------------
# prune_work_record — Work record normalization
# ---------------------------------------------------------------------------


class TestPruneWorkRecord:
    """Test pruning and normalization of OpenAlex Work records."""

    def _make_work(self, **overrides: object) -> dict:
        """Create a minimal valid Work record with optional overrides."""
        base = {
            "id": "https://openalex.org/W2741809807",
            "doi": "https://doi.org/10.1038/s41586-020-2649-2",
            "title": "Array programming with NumPy",
            "publication_year": 2020,
            "abstract_inverted_index": {"Array": [0], "programming": [1]},
            "topics": [{"id": "T1", "display_name": "Comp Sci", "score": 0.9}],
            "open_access": {"is_oa": True},
            "best_oa_location": {"pdf_url": "https://example.com/paper.pdf"},
            "cited_by_count": 5000,
            "referenced_works_count": 42,
            "type": "article",
            "updated_date": "2024-01-15",
            "created_date": "2020-09-21",
            "referenced_works": [
                "https://openalex.org/W100",
                "https://openalex.org/W200",
            ],
        }
        base.update(overrides)
        return base

    def test_extracts_core_fields(self) -> None:
        work = self._make_work()
        row, refs = prune_work_record(work)
        assert row["openalex_id"] == "W2741809807"
        assert row["doi"] == "10.1038/s41586-020-2649-2"
        assert row["title"] == "Array programming with NumPy"
        assert row["publication_year"] == 2020
        assert row["abstract"] == "Array programming"
        assert row["cited_by_count"] == 5000
        assert row["referenced_works_count"] == 42
        assert row["type"] == "article"

    def test_strips_doi_prefix(self) -> None:
        row, _ = prune_work_record(self._make_work(doi="https://doi.org/10.1234/test"))
        assert row["doi"] == "10.1234/test"

    def test_normalizes_openalex_id(self) -> None:
        row, _ = prune_work_record(self._make_work(id="https://openalex.org/W999"))
        assert row["openalex_id"] == "W999"

    def test_extracts_references(self) -> None:
        work = self._make_work(
            referenced_works=[
                "https://openalex.org/W100",
                "https://openalex.org/W200",
            ]
        )
        _, refs = prune_work_record(work)
        assert refs == [("W2741809807", "W100"), ("W2741809807", "W200")]

    def test_no_references(self) -> None:
        work = self._make_work(referenced_works=[])
        _, refs = prune_work_record(work)
        assert refs == []

    def test_missing_references_key(self) -> None:
        work = self._make_work()
        del work["referenced_works"]
        _, refs = prune_work_record(work)
        assert refs == []

    def test_null_doi(self) -> None:
        row, _ = prune_work_record(self._make_work(doi=None))
        assert row["doi"] is None

    def test_missing_abstract(self) -> None:
        work = self._make_work()
        del work["abstract_inverted_index"]
        row, _ = prune_work_record(work)
        assert row["abstract"] is None

    def test_topics_preserved_as_json(self) -> None:
        topics = [{"id": "T1", "display_name": "CS", "score": 0.9}]
        row, _ = prune_work_record(self._make_work(topics=topics))
        assert json.loads(row["topics"]) == topics

    def test_open_access_preserved(self) -> None:
        oa = {"is_oa": True, "oa_status": "gold"}
        row, _ = prune_work_record(self._make_work(open_access=oa))
        assert json.loads(row["open_access"]) == oa

    def test_dates_as_strings(self) -> None:
        row, _ = prune_work_record(
            self._make_work(updated_date="2024-01-15", created_date="2020-09-21")
        )
        assert row["updated_date"] == "2024-01-15"
        assert row["created_date"] == "2020-09-21"

    def test_missing_id_raises(self) -> None:
        work = self._make_work()
        del work["id"]
        with pytest.raises(ValueError, match="missing id"):
            prune_work_record(work)

    def test_null_bytes_stripped_from_title(self) -> None:
        row, _ = prune_work_record(self._make_work(title="Hello\x00World"))
        assert row["title"] == "HelloWorld"
        assert "\x00" not in row["title"]

    def test_null_bytes_stripped_from_abstract(self) -> None:
        inverted = {"Hello\x00World": [0], "test": [1]}
        row, _ = prune_work_record(self._make_work(abstract_inverted_index=inverted))
        assert "\x00" not in (row["abstract"] or "")


# ---------------------------------------------------------------------------
# normalize_openalex_id
# ---------------------------------------------------------------------------


class TestNormalizeOpenAlexId:
    def test_strips_url_prefix(self) -> None:
        assert normalize_openalex_id("https://openalex.org/W123") == "W123"

    def test_already_short(self) -> None:
        assert normalize_openalex_id("W123") == "W123"

    def test_none_returns_none(self) -> None:
        assert normalize_openalex_id(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert normalize_openalex_id("") is None


# ---------------------------------------------------------------------------
# extract_arxiv_id — from OpenAlex external_ids / ids
# ---------------------------------------------------------------------------


class TestExtractArxivId:
    def test_from_ids_list(self) -> None:
        work = {
            "ids": {
                "openalex": "https://openalex.org/W123",
                "doi": "https://doi.org/10.1234/test",
                "pmid": "https://pubmed.ncbi.nlm.nih.gov/12345",
            }
        }
        # No arXiv ID present
        assert extract_arxiv_id(work) is None

    def test_from_ids_dict_with_arxiv(self) -> None:
        work = {
            "ids": {
                "openalex": "https://openalex.org/W123",
                "doi": "https://doi.org/10.1234/test",
            },
            "primary_location": {
                "source": {
                    "display_name": "arXiv (Cornell University)",
                    "host_organization_name": "Cornell University",
                },
                "landing_page_url": "https://arxiv.org/abs/2301.12345",
            },
        }
        assert extract_arxiv_id(work) == "2301.12345"

    def test_from_landing_page_url_v_suffix(self) -> None:
        work = {
            "primary_location": {
                "landing_page_url": "https://arxiv.org/abs/2301.12345v2",
            },
        }
        assert extract_arxiv_id(work) == "2301.12345"

    def test_no_arxiv_info(self) -> None:
        work = {
            "primary_location": {
                "landing_page_url": "https://journals.aps.org/prd/abstract/10.1103/PhysRevD.1.1",
            },
        }
        assert extract_arxiv_id(work) is None

    def test_empty_work(self) -> None:
        assert extract_arxiv_id({}) is None


# ---------------------------------------------------------------------------
# validate_manifest — partition manifest checking
# ---------------------------------------------------------------------------


class TestValidateManifest:
    def test_valid_manifest(self, tmp_path: Path) -> None:
        partition_dir = tmp_path / "updated_date=2024-01-15"
        partition_dir.mkdir()
        (partition_dir / "part_000.parquet").write_bytes(b"PAR1fake")
        manifest = partition_dir / "_SUCCESS"
        manifest.write_text("")
        assert validate_manifest(partition_dir) is True

    def test_missing_manifest(self, tmp_path: Path) -> None:
        partition_dir = tmp_path / "updated_date=2024-01-15"
        partition_dir.mkdir()
        (partition_dir / "part_000.parquet").write_bytes(b"PAR1fake")
        assert validate_manifest(partition_dir) is False

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert validate_manifest(tmp_path / "nonexistent") is False


# ---------------------------------------------------------------------------
# Production DSN guard
# ---------------------------------------------------------------------------


class TestProductionGuard:
    """Regression tests for the None-dsn bypass (mirrors test_ads_body_unit.py)."""

    def test_none_dsn_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import scix.sources.openalex as oa_module

        monkeypatch.setattr(oa_module, "DEFAULT_DSN", "dbname=scix")
        cfg = LoaderConfig(dsn=None, data_dir=tmp_path)
        loader = OpenAlexLoader(cfg)
        with pytest.raises(ProductionGuardError, match="production"):
            loader._check_production_guard()

    def test_none_dsn_non_prod_default_allowed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import scix.sources.openalex as oa_module

        monkeypatch.setattr(oa_module, "DEFAULT_DSN", "dbname=scix_test")
        cfg = LoaderConfig(dsn=None, data_dir=tmp_path)
        loader = OpenAlexLoader(cfg)
        loader._check_production_guard()  # Should not raise

    def test_yes_production_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import scix.sources.openalex as oa_module

        monkeypatch.setattr(oa_module, "DEFAULT_DSN", "dbname=scix")
        cfg = LoaderConfig(dsn=None, data_dir=tmp_path, yes_production=True)
        loader = OpenAlexLoader(cfg)
        loader._check_production_guard()  # Should not raise

    def test_explicit_production_dsn_blocked(self, tmp_path: Path) -> None:
        cfg = LoaderConfig(dsn="dbname=scix", data_dir=tmp_path)
        with pytest.raises(ProductionGuardError):
            OpenAlexLoader(cfg)._check_production_guard()

    def test_uri_production_dsn_blocked(self, tmp_path: Path) -> None:
        cfg = LoaderConfig(dsn="postgresql://user:pw@host/scix", data_dir=tmp_path)
        with pytest.raises(ProductionGuardError):
            OpenAlexLoader(cfg)._check_production_guard()


# ---------------------------------------------------------------------------
# LoaderConfig — frozen dataclass
# ---------------------------------------------------------------------------


class TestLoaderConfig:
    def test_config_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        cfg = LoaderConfig(dsn="dbname=foo", data_dir=Path("/nonexistent"))
        with pytest.raises(FrozenInstanceError):
            cfg.dsn = "mutated"  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = LoaderConfig(dsn="dbname=foo", data_dir=Path("/x"))
        assert cfg.batch_size == 50_000
        assert cfg.dry_run is False
        assert cfg.yes_production is False
        assert cfg.drop_indexes is False
        assert cfg.works_only is False


# ---------------------------------------------------------------------------
# LoaderStats — frozen dataclass
# ---------------------------------------------------------------------------


class TestLoaderStats:
    def test_stats_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        stats = LoaderStats(
            partitions_processed=10,
            works_loaded=1000,
            references_loaded=5000,
            works_skipped=5,
            crosswalk_updated=800,
            elapsed_seconds=42.0,
            dry_run=False,
        )
        with pytest.raises(FrozenInstanceError):
            stats.works_loaded = 0  # type: ignore[misc]
