"""Unit tests for src/scix/extract.py.

Tests prompt construction, JSONL checkpoint I/O, chunked DB loading,
and idempotent upserts. No external services required.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from scix.extract import (
    EXTRACTION_TYPES,
    EXTRACTION_VERSION,
    ExtractionRequest,
    ExtractionRow,
    _SYSTEM_PROMPT,
    _TOOL_SCHEMA,
    _checkpoint_path,
    _get_existing_bibcodes,
    _load_checkpoint,
    _parse_extraction_rows,
    _save_checkpoint,
    build_extraction_prompt,
    load_results_to_db,
    save_results_jsonl,
    select_pilot_cohort,
    submit_batch,
)

# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------


class TestBuildExtractionPrompt:
    def test_returns_required_keys(self) -> None:
        result = build_extraction_prompt("Test Title", "Test abstract text.")
        assert "system" in result
        assert "messages" in result
        assert "tools" in result
        assert "tool_choice" in result

    def test_system_prompt_present(self) -> None:
        result = build_extraction_prompt("Title", "Abstract")
        assert result["system"] == _SYSTEM_PROMPT
        assert "entity extraction" in result["system"].lower()

    def test_tool_schema_included(self) -> None:
        result = build_extraction_prompt("Title", "Abstract")
        assert result["tools"] == [_TOOL_SCHEMA]
        tool = result["tools"][0]
        props = tool["input_schema"]["properties"]
        for etype in EXTRACTION_TYPES:
            assert etype in props

    def test_tool_choice_forces_extract_entities(self) -> None:
        result = build_extraction_prompt("Title", "Abstract")
        assert result["tool_choice"] == {"type": "tool", "name": "extract_entities"}

    def test_few_shot_examples_present(self) -> None:
        """At least 3 few-shot exchanges (user+assistant pairs) should appear."""
        result = build_extraction_prompt("Title", "Abstract")
        messages = result["messages"]
        # Count assistant messages with tool_use content
        assistant_tool_msgs = [
            m
            for m in messages
            if m["role"] == "assistant"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        ]
        assert len(assistant_tool_msgs) >= 3

    def test_user_message_at_end(self) -> None:
        result = build_extraction_prompt("My Title", "My abstract about stars.")
        last_msg = result["messages"][-1]
        assert last_msg["role"] == "user"
        # Content may be a list of blocks (tool_result + text) or a string
        content = last_msg["content"]
        if isinstance(content, list):
            text_parts = [b.get("text", "") for b in content if isinstance(b, dict)]
            joined = " ".join(text_parts)
        else:
            joined = content
        assert "My Title" in joined
        assert "My abstract about stars." in joined

    def test_few_shot_covers_multiple_subfields(self) -> None:
        """Few-shot examples should cover different astronomy subfields."""
        result = build_extraction_prompt("Title", "Abstract")
        all_text = json.dumps(result["messages"])
        # Cosmology, exoplanets, planetary science
        assert "Cosmological" in all_text or "CMB" in all_text
        assert "Exoplanet" in all_text or "JWST" in all_text
        assert "Mars" in all_text or "Solar Wind" in all_text


# ---------------------------------------------------------------------------
# select_pilot_cohort tests
# ---------------------------------------------------------------------------


class TestSelectPilotCohort:
    def test_returns_extraction_requests(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = [
            ("2024ApJ...1A", "Title One", "Abstract one " * 20),
            ("2024ApJ...2B", "Title Two", "Abstract two " * 20),
        ]

        results = select_pilot_cohort(mock_conn, limit=100)

        assert len(results) == 2
        assert all(isinstance(r, ExtractionRequest) for r in results)
        assert results[0].bibcode == "2024ApJ...1A"

    def test_query_orders_by_citation_count(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = []

        select_pilot_cohort(mock_conn, limit=5)

        sql = mock_cur.execute.call_args[0][0]
        assert "citation_count DESC" in sql
        assert "length(abstract) > 100" in sql


# ---------------------------------------------------------------------------
# JSONL checkpoint tests
# ---------------------------------------------------------------------------


class TestSaveResultsJsonl:
    def test_writes_jsonl_file(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        result1 = MagicMock(spec=["custom_id", "result"])
        result1.custom_id = "bibcode_1"
        # Plain dict — no model_dump, triggers hasattr fallback
        result1.result = {"type": "succeeded"}

        result2 = MagicMock()
        result2.custom_id = "bibcode_2"
        result2.result = MagicMock()
        result2.result.model_dump.return_value = {"type": "succeeded", "message": {}}

        mock_client.messages.batches.results.return_value = [result1, result2]

        output = tmp_path / "results.jsonl"
        returned_path = save_results_jsonl(mock_client, "batch_123", output)

        assert returned_path == output
        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2

        parsed = json.loads(lines[0])
        assert parsed["custom_id"] == "bibcode_1"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.messages.batches.results.return_value = []

        output = tmp_path / "nested" / "dir" / "results.jsonl"
        save_results_jsonl(mock_client, "batch_456", output)

        assert output.parent.exists()


# ---------------------------------------------------------------------------
# Parsing extraction rows
# ---------------------------------------------------------------------------


class TestParseExtractionRows:
    def test_parses_tool_use_result(self) -> None:
        line = {
            "custom_id": "2024ApJ...1A",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "extract_entities",
                            "input": {
                                "methods": ["MCMC"],
                                "datasets": ["Gaia DR3"],
                                "instruments": [],
                                "materials": [],
                            },
                        }
                    ]
                },
            },
        }
        rows = _parse_extraction_rows(line, "v1")
        assert len(rows) == 2  # methods and datasets (empty lists skipped)
        types = {r.extraction_type for r in rows}
        assert types == {"methods", "datasets"}

    def test_skips_errored_results(self) -> None:
        line = {
            "custom_id": "bibcode_err",
            "result": {"type": "errored"},
        }
        rows = _parse_extraction_rows(line, "v1")
        assert rows == []

    def test_skips_expired_results(self) -> None:
        line = {
            "custom_id": "bibcode_exp",
            "result": {"type": "expired"},
        }
        rows = _parse_extraction_rows(line, "v1")
        assert rows == []


# ---------------------------------------------------------------------------
# DB loading tests
# ---------------------------------------------------------------------------


class TestLoadResultsToDb:
    def _make_jsonl(self, tmp_path: Path, lines: list[dict[str, Any]]) -> Path:
        p = tmp_path / "test_results.jsonl"
        with open(p, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        return p

    def _sample_result_line(self, bibcode: str = "2024ApJ...1A") -> dict[str, Any]:
        return {
            "custom_id": bibcode,
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "extract_entities",
                            "input": {
                                "methods": ["PCA", "Random Forest"],
                                "datasets": ["SDSS DR17"],
                                "instruments": ["Hubble"],
                                "materials": [],
                            },
                        }
                    ]
                },
            },
        }

    def test_loads_rows_and_returns_count(self, tmp_path: Path) -> None:
        jsonl = self._make_jsonl(tmp_path, [self._sample_result_line()])

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        count = load_results_to_db(mock_conn, jsonl)

        # 3 non-empty extraction types: methods, datasets, instruments
        assert count == 3
        assert mock_cur.execute.call_count == 3
        mock_conn.commit.assert_called()

    def test_upsert_sql_uses_on_conflict(self, tmp_path: Path) -> None:
        jsonl = self._make_jsonl(tmp_path, [self._sample_result_line()])

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        load_results_to_db(mock_conn, jsonl)

        sql = mock_cur.execute.call_args_list[0][0][0]
        assert "ON CONFLICT (bibcode, extraction_type, extraction_version)" in sql
        assert "DO UPDATE SET" in sql

    def test_chunked_commit(self, tmp_path: Path) -> None:
        """With chunk_size=2, 5 rows should produce 3 commits."""
        # Create enough data for 5 extraction rows
        lines = [
            self._sample_result_line("bib_1"),  # 3 rows
            self._sample_result_line("bib_2"),  # 3 rows -> total 6
        ]
        jsonl = self._make_jsonl(tmp_path, lines)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        count = load_results_to_db(mock_conn, jsonl, chunk_size=2)

        # 6 rows total, chunk_size=2 -> 3 commits
        assert count == 6
        assert mock_conn.commit.call_count == 3

    def test_empty_jsonl_returns_zero(self, tmp_path: Path) -> None:
        jsonl = self._make_jsonl(tmp_path, [])

        mock_conn = MagicMock()
        count = load_results_to_db(mock_conn, jsonl)

        assert count == 0
        mock_conn.commit.assert_not_called()

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "sparse.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps(self._sample_result_line()) + "\n")
            f.write("\n")
            f.write("   \n")

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        count = load_results_to_db(mock_conn, p)
        assert count == 3  # 3 non-empty extraction types


# ---------------------------------------------------------------------------
# submit_batch tests
# ---------------------------------------------------------------------------


class TestSubmitBatch:
    def test_submits_batch_and_returns_id(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.batches.create.return_value = MagicMock(id="batch_abc")

        reqs = [
            ExtractionRequest("bib1", "Title 1", "Abstract " * 20),
            ExtractionRequest("bib2", "Title 2", "Abstract " * 20),
        ]
        batch_id, id_to_bibcode = submit_batch(mock_client, reqs)

        assert batch_id == "batch_abc"
        assert isinstance(id_to_bibcode, dict)
        assert "bib1" in id_to_bibcode.values()
        create_call = mock_client.messages.batches.create.call_args
        assert len(create_call.kwargs["requests"]) == 2
        assert create_call.kwargs["requests"][0]["custom_id"] == "bib1"


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestDataStructures:
    def test_extraction_request_is_frozen(self) -> None:
        req = ExtractionRequest("bib", "title", "abstract")
        with pytest.raises(AttributeError):
            req.bibcode = "other"  # type: ignore[misc]

    def test_extraction_row_is_frozen(self) -> None:
        row = ExtractionRow("bib", "methods", "v1", {"entities": ["PCA"]})
        with pytest.raises(AttributeError):
            row.bibcode = "other"  # type: ignore[misc]

    def test_extraction_types_constant(self) -> None:
        assert "methods" in EXTRACTION_TYPES
        assert "datasets" in EXTRACTION_TYPES
        assert "instruments" in EXTRACTION_TYPES
        assert "materials" in EXTRACTION_TYPES


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_checkpoint_path_includes_version(self, tmp_path: Path) -> None:
        path = _checkpoint_path(str(tmp_path), "v1")
        assert path.name == ".checkpoint_v1.json"
        assert path.parent == tmp_path

    def test_load_checkpoint_returns_defaults_when_missing(self, tmp_path: Path) -> None:
        cp = _load_checkpoint(str(tmp_path), "v1")
        assert cp["version"] == "v1"
        assert cp["processed_bibcodes"] == set()
        assert cp["cumulative_cost_usd"] == 0.0

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        bibcodes = {"bib_A", "bib_B", "bib_C"}
        _save_checkpoint(str(tmp_path), "v1", bibcodes, 1.23)

        cp = _load_checkpoint(str(tmp_path), "v1")
        assert cp["version"] == "v1"
        assert cp["processed_bibcodes"] == bibcodes
        assert abs(cp["cumulative_cost_usd"] - 1.23) < 1e-6

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        _save_checkpoint(str(nested), "v2", set(), 0.0)
        assert _checkpoint_path(str(nested), "v2").exists()

    def test_checkpoint_overwrites_on_update(self, tmp_path: Path) -> None:
        _save_checkpoint(str(tmp_path), "v1", {"bib_1"}, 0.5)
        _save_checkpoint(str(tmp_path), "v1", {"bib_1", "bib_2"}, 1.0)

        cp = _load_checkpoint(str(tmp_path), "v1")
        assert cp["processed_bibcodes"] == {"bib_1", "bib_2"}
        assert abs(cp["cumulative_cost_usd"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# DB idempotency tests
# ---------------------------------------------------------------------------


class TestGetExistingBibcodes:
    def test_returns_set_of_bibcodes(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = [("bib_1",), ("bib_2",)]

        result = _get_existing_bibcodes(mock_conn, "v1")

        assert result == {"bib_1", "bib_2"}
        sql = mock_cur.execute.call_args[0][0]
        assert "extractions" in sql
        assert "extraction_version" in sql

    def test_returns_empty_set_when_no_results(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = []

        result = _get_existing_bibcodes(mock_conn, "v1")
        assert result == set()


# ---------------------------------------------------------------------------
# Pipeline checkpointing + idempotency + budget integration tests
# ---------------------------------------------------------------------------


class TestPipelineCheckpointing:
    """Tests for checkpoint resume, idempotency, and budget accumulator."""

    def _make_mock_batch_result(self, bibcode: str) -> MagicMock:
        """Create a mock batch result for a single bibcode."""
        result = MagicMock()
        result.custom_id = bibcode
        result.result = MagicMock()
        result.result.model_dump.return_value = {
            "type": "succeeded",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "extract_entities",
                        "input": {
                            "methods": ["PCA"],
                            "datasets": [],
                            "instruments": [],
                            "materials": [],
                        },
                    }
                ]
            },
        }
        return result

    def _setup_pipeline_mocks(
        self,
        cohort_bibcodes: list[str],
        existing_db_bibcodes: list[str] | None = None,
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        """Set up mocks for the pipeline: anthropic client, DB conn, cohort."""
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Batch create returns a batch object
        mock_client.messages.batches.create.return_value = MagicMock(id="batch_001")
        # Batch retrieve returns ended status
        mock_client.messages.batches.retrieve.return_value = MagicMock(processing_status="ended")

        # Batch results returns mock results for each bibcode in the request
        def results_side_effect(batch_id: str) -> list[MagicMock]:
            # Return results for all cohort bibcodes not in existing_db
            return [self._make_mock_batch_result(b) for b in cohort_bibcodes]

        mock_client.messages.batches.results.side_effect = results_side_effect

        # Mock DB connection
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # select_pilot_cohort query
        mock_cur.fetchall.return_value = [
            (bib, f"Title {bib}", "Abstract text " * 20) for bib in cohort_bibcodes
        ]

        return mock_anthropic, mock_client, mock_conn

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.extract.get_connection")
    @patch("scix.extract._get_existing_bibcodes")
    def test_checkpoint_skips_processed_bibcodes(
        self, mock_get_existing: MagicMock, mock_get_conn: MagicMock, tmp_path: Path
    ) -> None:
        """Pipeline should skip bibcodes found in checkpoint file."""
        output_dir = str(tmp_path / "extractions")

        # Pre-seed checkpoint with bib_1 already processed
        _save_checkpoint(output_dir, "v1", {"bib_1"}, 0.001)

        mock_anthropic, mock_client, mock_conn = self._setup_pipeline_mocks(
            ["bib_1", "bib_2", "bib_3"]
        )
        mock_get_conn.return_value = mock_conn
        mock_get_existing.return_value = set()

        # Make results return only bib_2, bib_3 (since bib_1 is skipped)
        mock_client.messages.batches.results.side_effect = lambda bid: [
            self._make_mock_batch_result(b) for b in ["bib_2", "bib_3"]
        ]

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from scix.extract import run_extraction_pipeline

            result = run_extraction_pipeline(
                dsn="test",
                pilot_size=10,
                model="claude-haiku-4-5-20251001",
                output_dir=output_dir,
                extraction_version="v1",
                batch_size=100,
                poll_interval=0,
                budget_usd=100.0,
            )

        # Should have submitted a batch (bib_2 and bib_3 only)
        create_call = mock_client.messages.batches.create.call_args
        submitted_ids = [r["custom_id"] for r in create_call.kwargs["requests"]]
        assert "bib_1" not in submitted_ids
        assert len(submitted_ids) == 2

        # Checkpoint should now contain all 3
        cp = _load_checkpoint(output_dir, "v1")
        assert "bib_1" in cp["processed_bibcodes"]
        assert "bib_2" in cp["processed_bibcodes"]
        assert "bib_3" in cp["processed_bibcodes"]

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.extract.get_connection")
    @patch("scix.extract._get_existing_bibcodes")
    def test_db_idempotency_skips_existing(
        self, mock_get_existing: MagicMock, mock_get_conn: MagicMock, tmp_path: Path
    ) -> None:
        """Pipeline should skip bibcodes already in DB for current version."""
        output_dir = str(tmp_path / "extractions")

        mock_anthropic, mock_client, mock_conn = self._setup_pipeline_mocks(["bib_1", "bib_2"])
        mock_get_conn.return_value = mock_conn
        # bib_1 already exists in DB
        mock_get_existing.return_value = {"bib_1"}

        mock_client.messages.batches.results.side_effect = lambda bid: [
            self._make_mock_batch_result("bib_2")
        ]

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from scix.extract import run_extraction_pipeline

            run_extraction_pipeline(
                dsn="test",
                pilot_size=10,
                model="claude-haiku-4-5-20251001",
                output_dir=output_dir,
                extraction_version="v1",
                batch_size=100,
                poll_interval=0,
                budget_usd=100.0,
            )

        # Only bib_2 should have been submitted
        create_call = mock_client.messages.batches.create.call_args
        submitted_ids = [r["custom_id"] for r in create_call.kwargs["requests"]]
        assert "bib_1" not in submitted_ids
        assert "bib_2" in submitted_ids

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.extract.get_connection")
    @patch("scix.extract._get_existing_bibcodes")
    def test_duplicate_submission_no_duplicates(
        self, mock_get_existing: MagicMock, mock_get_conn: MagicMock, tmp_path: Path
    ) -> None:
        """Submitting the same cohort twice should not create duplicate extractions."""
        output_dir = str(tmp_path / "extractions")

        mock_anthropic, mock_client, mock_conn = self._setup_pipeline_mocks(["bib_1"])
        mock_get_conn.return_value = mock_conn
        mock_get_existing.return_value = set()

        mock_client.messages.batches.results.side_effect = lambda bid: [
            self._make_mock_batch_result("bib_1")
        ]

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from scix.extract import run_extraction_pipeline

            # First run
            run_extraction_pipeline(
                dsn="test",
                pilot_size=10,
                model="claude-haiku-4-5-20251001",
                output_dir=output_dir,
                extraction_version="v1",
                batch_size=100,
                poll_interval=0,
                budget_usd=100.0,
            )

        assert mock_client.messages.batches.create.call_count == 1

        # Second run — bib_1 is now in checkpoint
        mock_client.messages.batches.create.reset_mock()
        mock_get_conn.return_value = mock_conn

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from scix.extract import run_extraction_pipeline

            result = run_extraction_pipeline(
                dsn="test",
                pilot_size=10,
                model="claude-haiku-4-5-20251001",
                output_dir=output_dir,
                extraction_version="v1",
                batch_size=100,
                poll_interval=0,
                budget_usd=100.0,
            )

        # Second run should not have submitted any batch
        mock_client.messages.batches.create.assert_not_called()
        assert result == 0

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.extract.get_connection")
    @patch("scix.extract._get_existing_bibcodes")
    def test_budget_halt_at_80_percent(
        self, mock_get_existing: MagicMock, mock_get_conn: MagicMock, tmp_path: Path
    ) -> None:
        """Pipeline halts with 'budget' message when cumulative cost reaches 80%."""
        output_dir = str(tmp_path / "extractions")

        # Pre-seed checkpoint with cost already at 80% of budget
        # With budget_usd=1.0, 80% = 0.80
        _save_checkpoint(output_dir, "v1", set(), 0.85)

        mock_anthropic, mock_client, mock_conn = self._setup_pipeline_mocks(["bib_1"])
        mock_get_conn.return_value = mock_conn
        mock_get_existing.return_value = set()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from scix.extract import run_extraction_pipeline

            with pytest.raises(ValueError, match="budget"):
                run_extraction_pipeline(
                    dsn="test",
                    pilot_size=10,
                    model="claude-haiku-4-5-20251001",
                    output_dir=output_dir,
                    extraction_version="v1",
                    batch_size=100,
                    poll_interval=0,
                    budget_usd=1.0,
                )

        # No batch should have been submitted
        mock_client.messages.batches.create.assert_not_called()

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.extract.get_connection")
    @patch("scix.extract._get_existing_bibcodes")
    def test_cost_accumulates_across_batches(
        self, mock_get_existing: MagicMock, mock_get_conn: MagicMock, tmp_path: Path
    ) -> None:
        """Cumulative cost should increase after each batch and persist in checkpoint."""
        output_dir = str(tmp_path / "extractions")

        mock_anthropic, mock_client, mock_conn = self._setup_pipeline_mocks(["bib_1", "bib_2"])
        mock_get_conn.return_value = mock_conn
        mock_get_existing.return_value = set()

        mock_client.messages.batches.results.side_effect = lambda bid: [
            self._make_mock_batch_result("bib_1"),
            self._make_mock_batch_result("bib_2"),
        ]

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from scix.extract import run_extraction_pipeline

            run_extraction_pipeline(
                dsn="test",
                pilot_size=10,
                model="claude-haiku-4-5-20251001",
                output_dir=output_dir,
                extraction_version="v1",
                batch_size=100,
                poll_interval=0,
                budget_usd=100.0,
            )

        cp = _load_checkpoint(output_dir, "v1")
        assert cp["cumulative_cost_usd"] > 0.0
        assert "bib_1" in cp["processed_bibcodes"]
        assert "bib_2" in cp["processed_bibcodes"]
