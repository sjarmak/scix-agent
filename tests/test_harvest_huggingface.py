"""Tests for HuggingFace Hub harvester.

Unit tests verify classification, parsing, and pipeline flow with mocked
HF API and DB. No network or DB required for unit tests; an integration
class is included but skipped without ``SCIX_TEST_DSN``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvest_huggingface import (
    SCIENTIFIC_TAGS,
    classify_model,
    extract_arxiv_id_from_tag,
    extract_arxiv_ids_from_identifier,
    harvest_models,
    model_to_entry,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal stand-in for ``huggingface_hub.ModelInfo``."""

    def __init__(
        self,
        model_id: str,
        tags: list[str] | None = None,
        downloads: int | None = 0,
        pipeline_tag: str | None = None,
        library_name: str | None = None,
    ) -> None:
        self.id = model_id
        self.tags = tags or []
        self.downloads = downloads
        self.pipeline_tag = pipeline_tag
        self.library_name = library_name


def _make_api(models: list[_FakeModel]) -> MagicMock:
    """Build a mock HfApi whose ``list_models`` returns ``models``.

    Honours the ``filter`` keyword by returning only models whose tags
    contain that filter value — this lets ``iter_models_tags_only`` be
    exercised without a real Hub.
    """
    api = MagicMock()

    def _list(*, filter=None, sort=None):  # noqa: A002 — match real signature
        if filter is None:
            yield from models
            return
        wanted = {filter} if isinstance(filter, str) else set(filter)
        for m in models:
            if wanted & set(m.tags or ()):
                yield m

    api.list_models.side_effect = _list
    return api


# ---------------------------------------------------------------------------
# Arxiv-id extraction
# ---------------------------------------------------------------------------


class TestExtractArxivIdFromTag:
    def test_modern_format(self) -> None:
        assert extract_arxiv_id_from_tag("arxiv:2112.00590") == "2112.00590"

    def test_modern_format_long_decimal(self) -> None:
        assert extract_arxiv_id_from_tag("arxiv:2604.06628") == "2604.06628"

    def test_old_format(self) -> None:
        assert extract_arxiv_id_from_tag("arxiv:cs/0101001") == "cs/0101001"

    def test_uppercase_prefix(self) -> None:
        assert extract_arxiv_id_from_tag("arXiv:1903.10676") == "1903.10676"

    def test_abs_prefix(self) -> None:
        assert extract_arxiv_id_from_tag("arxiv:abs/1903.10676") == "1903.10676"

    def test_non_arxiv_tag(self) -> None:
        assert extract_arxiv_id_from_tag("biology") is None

    def test_malformed(self) -> None:
        assert extract_arxiv_id_from_tag("arxiv:nope") is None

    def test_empty(self) -> None:
        assert extract_arxiv_id_from_tag("") is None


class TestExtractArxivIdsFromIdentifier:
    def test_arxiv_prefix(self) -> None:
        ids = extract_arxiv_ids_from_identifier(["arXiv:2108.03126"])
        assert ids == {"2108.03126"}

    def test_doi_form(self) -> None:
        ids = extract_arxiv_ids_from_identifier(["10.48550/arXiv.2108.03126"])
        assert ids == {"2108.03126"}

    def test_bibcode_form_ignored(self) -> None:
        # The bibcode 2021arXiv210803126L is intentionally not matched —
        # it has no separator before the digits.
        ids = extract_arxiv_ids_from_identifier(["2021arXiv210803126L"])
        assert ids == set()

    def test_mixed_array(self) -> None:
        ids = extract_arxiv_ids_from_identifier(
            [
                "2021JFM...928R...3L",
                "10.48550/arXiv.2108.03126",
                "arXiv:2108.03126",
                "10.1080/01468037708240527",
            ]
        )
        assert ids == {"2108.03126"}

    def test_empty_input(self) -> None:
        assert extract_arxiv_ids_from_identifier(None) == set()
        assert extract_arxiv_ids_from_identifier([]) == set()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassifyModel:
    def test_via_scientific_tag(self) -> None:
        matched, sci, arx = classify_model(
            ["transformers", "biology", "bert"],
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids=set(),
        )
        assert matched is True
        assert sci == ["biology"]
        assert arx == []

    def test_via_arxiv_corpus_match(self) -> None:
        matched, sci, arx = classify_model(
            ["transformers", "arxiv:1903.10676", "bert"],
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids={"1903.10676"},
        )
        assert matched is True
        assert sci == []
        assert arx == ["1903.10676"]

    def test_arxiv_not_in_corpus_skipped(self) -> None:
        matched, sci, arx = classify_model(
            ["transformers", "arxiv:9999.99999"],
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids={"1903.10676"},
        )
        assert matched is False
        assert sci == []
        assert arx == []

    def test_both_signals(self) -> None:
        matched, sci, arx = classify_model(
            ["physics", "arxiv:2112.00590"],
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids={"2112.00590"},
        )
        assert matched is True
        assert sci == ["physics"]
        assert arx == ["2112.00590"]

    def test_no_signals(self) -> None:
        matched, sci, arx = classify_model(
            ["transformers", "bert", "en"],
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids=set(),
        )
        assert matched is False

    def test_dedup_arxiv_ids(self) -> None:
        # Same arxiv id appearing twice in tags should yield one entry.
        matched, sci, arx = classify_model(
            ["arxiv:1903.10676", "arxiv:1903.10676"],
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids={"1903.10676"},
        )
        assert matched is True
        assert arx == ["1903.10676"]

    def test_empty_tags(self) -> None:
        matched, sci, arx = classify_model(
            [],
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids={"1903.10676"},
        )
        assert matched is False


# ---------------------------------------------------------------------------
# model_to_entry
# ---------------------------------------------------------------------------


class TestModelToEntry:
    def test_basic_scientific_tag(self) -> None:
        m = _FakeModel(
            "ScientaLab/eva-rna",
            tags=["transformers", "biology", "rna"],
            downloads=1234,
            pipeline_tag="feature-extraction",
            library_name="transformers",
        )
        entry = model_to_entry(
            m,
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids=set(),
        )
        assert entry is not None
        assert entry["canonical_name"] == "ScientaLab/eva-rna"
        assert entry["entity_type"] == "software"
        assert entry["source"] == "huggingface"
        assert entry["external_id"] == "ScientaLab/eva-rna"
        assert "eva-rna" in entry["aliases"]
        meta = entry["metadata"]
        assert meta["downloads"] == 1234
        assert meta["pipeline_tag"] == "feature-extraction"
        assert meta["library_name"] == "transformers"
        assert "biology" in meta["matched_scientific_tags"]
        assert meta["arxiv_id"] == []
        assert meta["arxiv_id_in_corpus"] == []
        assert "biology" in meta["tags"]

    def test_via_arxiv_only(self) -> None:
        m = _FakeModel(
            "adsabs/astroBERT",
            tags=["transformers", "bert", "arxiv:2112.00590", "en"],
            downloads=42,
        )
        entry = model_to_entry(
            m,
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids={"2112.00590"},
        )
        assert entry is not None
        meta = entry["metadata"]
        assert meta["arxiv_id"] == ["2112.00590"]
        assert meta["arxiv_id_in_corpus"] == ["2112.00590"]
        assert meta["matched_scientific_tags"] == []

    def test_filtered_out(self) -> None:
        m = _FakeModel("foo/bar", tags=["transformers", "en"])
        assert (
            model_to_entry(
                m,
                scientific_tags=SCIENTIFIC_TAGS,
                corpus_arxiv_ids=set(),
            )
            is None
        )

    def test_no_id(self) -> None:
        m = _FakeModel("", tags=["biology"])
        assert (
            model_to_entry(
                m,
                scientific_tags=SCIENTIFIC_TAGS,
                corpus_arxiv_ids=set(),
            )
            is None
        )

    def test_unscoped_model_id_no_alias(self) -> None:
        # A model id without an "/" — no alias to add.
        m = _FakeModel("standalone-model", tags=["biology"])
        entry = model_to_entry(
            m,
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids=set(),
        )
        assert entry is not None
        assert entry["aliases"] == []

    def test_arxiv_id_collected_even_when_not_in_corpus(self) -> None:
        # The full arxiv list is preserved in metadata so a later linker
        # can expand the corpus; only ``arxiv_id_in_corpus`` is filtered.
        m = _FakeModel(
            "foo/bar",
            tags=["biology", "arxiv:9999.99999", "arxiv:2112.00590"],
        )
        entry = model_to_entry(
            m,
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids={"2112.00590"},
        )
        assert entry is not None
        meta = entry["metadata"]
        assert set(meta["arxiv_id"]) == {"9999.99999", "2112.00590"}
        assert meta["arxiv_id_in_corpus"] == ["2112.00590"]


# ---------------------------------------------------------------------------
# harvest_models
# ---------------------------------------------------------------------------


class TestHarvestModels:
    def _models(self) -> list[_FakeModel]:
        return [
            _FakeModel("a/scientific", tags=["biology", "transformers"]),
            _FakeModel("b/arxiv-only", tags=["arxiv:2112.00590", "transformers"]),
            _FakeModel("c/skipped", tags=["transformers", "en"]),
            _FakeModel(
                "d/both",
                tags=["physics", "arxiv:1903.10676", "transformers"],
            ),
        ]

    def test_tags_only_mode(self) -> None:
        api = _make_api(self._models())
        entries, counts = harvest_models(
            api,
            mode="tags-only",
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids=set(),
        )
        ids = {e["canonical_name"] for e in entries}
        assert ids == {"a/scientific", "d/both"}
        assert counts["matched"] == 2
        assert counts["via_tag"] == 2
        assert counts["via_arxiv"] == 0

    def test_full_mode(self) -> None:
        api = _make_api(self._models())
        entries, counts = harvest_models(
            api,
            mode="full",
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids={"2112.00590", "1903.10676"},
        )
        ids = {e["canonical_name"] for e in entries}
        assert ids == {"a/scientific", "b/arxiv-only", "d/both"}
        assert counts["matched"] == 3
        assert counts["scanned"] == 4
        assert counts["via_tag"] == 1  # a/scientific
        assert counts["via_arxiv"] == 1  # b/arxiv-only
        assert counts["via_both"] == 1  # d/both

    def test_full_mode_with_limit(self) -> None:
        api = _make_api(self._models())
        entries, counts = harvest_models(
            api,
            mode="full",
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids={"2112.00590"},
            limit=1,
        )
        assert counts["scanned"] == 2  # limit is exclusive break
        assert len(entries) == 1
        assert entries[0]["canonical_name"] == "a/scientific"

    def test_unknown_mode(self) -> None:
        api = _make_api([])
        with pytest.raises(ValueError, match="Unknown mode"):
            harvest_models(
                api,
                mode="bogus",
                scientific_tags=SCIENTIFIC_TAGS,
                corpus_arxiv_ids=set(),
            )

    def test_dedup(self) -> None:
        # Tags-only iteration over multiple tags can surface the same model
        # twice; harvest_models must dedup by canonical_name.
        models = [_FakeModel("a/multi", tags=["biology", "physics"])]
        api = _make_api(models)
        entries, counts = harvest_models(
            api,
            mode="tags-only",
            scientific_tags=SCIENTIFIC_TAGS,
            corpus_arxiv_ids=set(),
        )
        assert len(entries) == 1
        assert counts["matched"] == 1


# ---------------------------------------------------------------------------
# run_pipeline (mocked DB + HF API)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    @patch("harvest_huggingface.HarvestRunLog")
    @patch("harvest_huggingface.bulk_load")
    @patch("harvest_huggingface.fetch_corpus_arxiv_ids")
    @patch("harvest_huggingface._get_hf_api")
    @patch("harvest_huggingface.get_connection")
    def test_full_mode_pipeline(
        self,
        mock_get_conn: MagicMock,
        mock_get_api: MagicMock,
        mock_fetch_arxiv: MagicMock,
        mock_bulk_load: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        mock_get_conn.return_value = MagicMock()
        mock_fetch_arxiv.return_value = {"2112.00590"}
        mock_get_api.return_value = _make_api(
            [
                _FakeModel("a/scientific", tags=["biology"]),
                _FakeModel("b/skipped", tags=["en"]),
                _FakeModel(
                    "c/arxiv", tags=["arxiv:2112.00590", "transformers"]
                ),
            ]
        )
        mock_bulk_load.return_value = 2
        mock_run_log = MagicMock()
        mock_run_log.start.return_value = 7
        mock_run_log.run_id = 7
        mock_run_log_cls.return_value = mock_run_log

        count = run_pipeline(mode="full", dsn="fake")

        assert count == 2
        mock_fetch_arxiv.assert_called_once()
        mock_bulk_load.assert_called_once()
        loaded_entries = mock_bulk_load.call_args.args[1]
        ids = {e["canonical_name"] for e in loaded_entries}
        assert ids == {"a/scientific", "c/arxiv"}
        mock_run_log.complete.assert_called_once()

    @patch("harvest_huggingface.HarvestRunLog")
    @patch("harvest_huggingface.bulk_load")
    @patch("harvest_huggingface.fetch_corpus_arxiv_ids")
    @patch("harvest_huggingface._get_hf_api")
    @patch("harvest_huggingface.get_connection")
    def test_tags_only_skips_corpus_load(
        self,
        mock_get_conn: MagicMock,
        mock_get_api: MagicMock,
        mock_fetch_arxiv: MagicMock,
        mock_bulk_load: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        mock_get_conn.return_value = MagicMock()
        mock_get_api.return_value = _make_api(
            [_FakeModel("a/sci", tags=["biology"])]
        )
        mock_bulk_load.return_value = 1
        mock_run_log = MagicMock()
        mock_run_log_cls.return_value = mock_run_log

        run_pipeline(mode="tags-only", dsn="fake")

        mock_fetch_arxiv.assert_not_called()

    @patch("harvest_huggingface.HarvestRunLog")
    @patch("harvest_huggingface.bulk_load")
    @patch("harvest_huggingface.fetch_corpus_arxiv_ids")
    @patch("harvest_huggingface._get_hf_api")
    @patch("harvest_huggingface.get_connection")
    def test_no_matches_records_zero(
        self,
        mock_get_conn: MagicMock,
        mock_get_api: MagicMock,
        mock_fetch_arxiv: MagicMock,
        mock_bulk_load: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        mock_get_conn.return_value = MagicMock()
        mock_fetch_arxiv.return_value = set()
        mock_get_api.return_value = _make_api(
            [_FakeModel("only/skipped", tags=["en"])]
        )
        mock_run_log = MagicMock()
        mock_run_log_cls.return_value = mock_run_log

        count = run_pipeline(mode="full", dsn="fake")

        assert count == 0
        mock_bulk_load.assert_not_called()
        mock_run_log.complete.assert_called_once()

    @patch("harvest_huggingface.HarvestRunLog")
    @patch("harvest_huggingface.bulk_load")
    @patch("harvest_huggingface.fetch_corpus_arxiv_ids")
    @patch("harvest_huggingface._get_hf_api")
    @patch("harvest_huggingface.get_connection")
    def test_failure_marks_run_failed(
        self,
        mock_get_conn: MagicMock,
        mock_get_api: MagicMock,
        mock_fetch_arxiv: MagicMock,
        mock_bulk_load: MagicMock,
        mock_run_log_cls: MagicMock,
    ) -> None:
        mock_get_conn.return_value = MagicMock()
        mock_fetch_arxiv.side_effect = RuntimeError("boom")
        mock_run_log = MagicMock()
        mock_run_log_cls.return_value = mock_run_log

        with pytest.raises(RuntimeError, match="boom"):
            run_pipeline(mode="full", dsn="fake")

        mock_run_log.fail.assert_called_once()


# ---------------------------------------------------------------------------
# Integration (skipped without SCIX_TEST_DSN)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    def test_corpus_arxiv_extraction(self) -> None:
        """Verify ``fetch_corpus_arxiv_ids`` against a live test DB."""
        test_dsn = os.environ.get("SCIX_TEST_DSN")
        if not test_dsn:
            pytest.skip("SCIX_TEST_DSN not set")

        import psycopg

        from harvest_huggingface import fetch_corpus_arxiv_ids

        conn = psycopg.connect(test_dsn)
        try:
            ids = fetch_corpus_arxiv_ids(conn)
        finally:
            conn.close()
        # Empty corpus is acceptable; format must be lowercase strings.
        assert all(s == s.lower() for s in ids)
