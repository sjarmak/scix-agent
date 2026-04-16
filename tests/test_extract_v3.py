"""Unit tests for v3 entity extraction pipeline (extract.py).

Tests the expanded 6-category taxonomy, combined-payload format,
body-text support, and compatibility with link_entities.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scix.extract import (
    _SYSTEM_PROMPT_V3,
    _TOOL_SCHEMA_V3,
    _V3_MAX_BODY_CHARS,
    EXTRACTION_TYPE_V3,
    EXTRACTION_TYPES_V3,
    EXTRACTION_VERSION_V3,
    ExtractionRequest,
    _parse_v3_extraction_rows,
    build_extraction_prompt_v3,
    load_results_to_db_v3,
    select_cohort_v3,
    submit_batch_v3,
)

# ---------------------------------------------------------------------------
# Constants and taxonomy tests
# ---------------------------------------------------------------------------


class TestV3Constants:
    def test_extraction_types_v3_has_six_categories(self) -> None:
        assert len(EXTRACTION_TYPES_V3) == 6

    def test_extraction_types_v3_includes_new_categories(self) -> None:
        assert "observables" in EXTRACTION_TYPES_V3
        assert "software" in EXTRACTION_TYPES_V3

    def test_extraction_types_v3_includes_original_categories(self) -> None:
        assert "methods" in EXTRACTION_TYPES_V3
        assert "datasets" in EXTRACTION_TYPES_V3
        assert "instruments" in EXTRACTION_TYPES_V3
        assert "materials" in EXTRACTION_TYPES_V3

    def test_extraction_type_v3_matches_link_entities(self) -> None:
        """extraction_type must be 'entity_extraction_v3' to match link_entities.py."""
        assert EXTRACTION_TYPE_V3 == "entity_extraction_v3"

    def test_extraction_version_v3(self) -> None:
        assert EXTRACTION_VERSION_V3 == "v3"

    def test_v3_types_align_with_link_entities_payload_keys(self) -> None:
        """V3 types must be a subset of link_entities._PAYLOAD_KEYS."""
        from scix.link_entities import _PAYLOAD_KEYS

        v3_set = set(EXTRACTION_TYPES_V3)
        assert (
            v3_set <= _PAYLOAD_KEYS
        ), f"V3 types {v3_set - _PAYLOAD_KEYS} not in link_entities._PAYLOAD_KEYS"


# ---------------------------------------------------------------------------
# v3 prompt construction tests
# ---------------------------------------------------------------------------


class TestBuildExtractionPromptV3:
    def test_returns_required_keys(self) -> None:
        result = build_extraction_prompt_v3("Title", "Abstract text here.")
        assert "system" in result
        assert "messages" in result
        assert "tools" in result
        assert "tool_choice" in result

    def test_system_prompt_v3_used(self) -> None:
        result = build_extraction_prompt_v3("Title", "Abstract")
        assert result["system"] == _SYSTEM_PROMPT_V3

    def test_system_prompt_mentions_all_categories(self) -> None:
        for category in EXTRACTION_TYPES_V3:
            assert (
                category in _SYSTEM_PROMPT_V3.lower()
            ), f"Category '{category}' not mentioned in v3 system prompt"

    def test_tool_schema_v3_has_all_categories(self) -> None:
        result = build_extraction_prompt_v3("Title", "Abstract")
        tool = result["tools"][0]
        props = tool["input_schema"]["properties"]
        for etype in EXTRACTION_TYPES_V3:
            assert etype in props, f"Category '{etype}' missing from v3 tool schema"

    def test_tool_schema_v3_requires_all_categories(self) -> None:
        required = _TOOL_SCHEMA_V3["input_schema"]["required"]
        for etype in EXTRACTION_TYPES_V3:
            assert etype in required

    def test_tool_choice_forces_extract_entities_v3(self) -> None:
        result = build_extraction_prompt_v3("Title", "Abstract")
        assert result["tool_choice"] == {
            "type": "tool",
            "name": "extract_entities_v3",
        }

    def test_few_shot_examples_present(self) -> None:
        """At least 3 few-shot exchanges should appear."""
        result = build_extraction_prompt_v3("Title", "Abstract")
        messages = result["messages"]
        assistant_tool_msgs = [
            m
            for m in messages
            if m["role"] == "assistant"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        ]
        assert len(assistant_tool_msgs) >= 3

    def test_few_shot_covers_new_categories(self) -> None:
        """Few-shot examples should demonstrate observables and software."""
        result = build_extraction_prompt_v3("Title", "Abstract")
        all_text = json.dumps(result["messages"])
        assert "observables" in all_text.lower() or "matter density" in all_text
        assert "software" in all_text.lower() or "emcee" in all_text

    def test_user_message_at_end(self) -> None:
        result = build_extraction_prompt_v3("My Title", "My abstract about galaxies.")
        last_msg = result["messages"][-1]
        assert last_msg["role"] == "user"
        content = last_msg["content"]
        if isinstance(content, list):
            text_parts = [b.get("text", "") for b in content if isinstance(b, dict)]
            joined = " ".join(text_parts)
        else:
            joined = content
        assert "My Title" in joined
        assert "My abstract about galaxies." in joined

    def test_body_text_included_when_provided(self) -> None:
        result = build_extraction_prompt_v3(
            "Title", "Abstract", body="This is the full text body section."
        )
        last_msg = result["messages"][-1]
        content = last_msg["content"]
        text_parts = [b.get("text", "") for b in content if isinstance(b, dict)]
        joined = " ".join(text_parts)
        assert "Body (excerpt)" in joined
        assert "full text body section" in joined

    def test_body_text_truncated(self) -> None:
        long_body = "x" * (_V3_MAX_BODY_CHARS + 5000)
        result = build_extraction_prompt_v3("Title", "Abstract", body=long_body)
        last_msg = result["messages"][-1]
        content = last_msg["content"]
        text_parts = [b.get("text", "") for b in content if isinstance(b, dict)]
        joined = " ".join(text_parts)
        # Body should be truncated to _V3_MAX_BODY_CHARS
        assert len(long_body) > _V3_MAX_BODY_CHARS
        # The joined text should not contain the full long body
        assert "x" * (_V3_MAX_BODY_CHARS + 1) not in joined

    def test_no_body_section_when_body_is_none(self) -> None:
        result = build_extraction_prompt_v3("Title", "Abstract", body=None)
        last_msg = result["messages"][-1]
        content = last_msg["content"]
        text_parts = [b.get("text", "") for b in content if isinstance(b, dict)]
        joined = " ".join(text_parts)
        assert "Body (excerpt)" not in joined

    def test_no_body_section_when_body_is_empty(self) -> None:
        result = build_extraction_prompt_v3("Title", "Abstract", body="")
        last_msg = result["messages"][-1]
        content = last_msg["content"]
        text_parts = [b.get("text", "") for b in content if isinstance(b, dict)]
        joined = " ".join(text_parts)
        assert "Body (excerpt)" not in joined


# ---------------------------------------------------------------------------
# ExtractionRequest body field tests
# ---------------------------------------------------------------------------


class TestExtractionRequestBody:
    def test_body_field_defaults_to_none(self) -> None:
        req = ExtractionRequest("bib", "title", "abstract")
        assert req.body is None

    def test_body_field_can_be_set(self) -> None:
        req = ExtractionRequest("bib", "title", "abstract", body="full text")
        assert req.body == "full text"

    def test_frozen_with_body(self) -> None:
        req = ExtractionRequest("bib", "title", "abstract", body="body text")
        with pytest.raises(AttributeError):
            req.body = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# v3 parsing tests
# ---------------------------------------------------------------------------


def _make_v3_result_line(
    bibcode: str = "2024ApJ...1A",
    instruments: list[str] | None = None,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    observables: list[str] | None = None,
    materials: list[str] | None = None,
    software: list[str] | None = None,
) -> dict[str, Any]:
    """Helper to build a mock v3 JSONL result line."""
    return {
        "custom_id": bibcode,
        "result": {
            "type": "succeeded",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "extract_entities_v3",
                        "input": {
                            "instruments": instruments or [],
                            "datasets": datasets or [],
                            "methods": methods or [],
                            "observables": observables or [],
                            "materials": materials or [],
                            "software": software or [],
                        },
                    }
                ]
            },
        },
    }


class TestParseV3ExtractionRows:
    def test_parses_combined_payload(self) -> None:
        line = _make_v3_result_line(
            instruments=["ALMA", "VLT"],
            methods=["PCA"],
            observables=["redshift"],
            software=["astropy"],
        )
        rows = _parse_v3_extraction_rows(line)
        assert len(rows) == 1
        row = rows[0]
        assert row.extraction_type == EXTRACTION_TYPE_V3
        assert row.extraction_version == EXTRACTION_VERSION_V3
        assert row.payload["instruments"] == ["ALMA", "VLT"]
        assert row.payload["methods"] == ["PCA"]
        assert row.payload["observables"] == ["redshift"]
        assert row.payload["software"] == ["astropy"]
        assert row.payload["datasets"] == []
        assert row.payload["materials"] == []

    def test_single_row_per_paper(self) -> None:
        """v3 produces exactly one row per paper, not per-type rows."""
        line = _make_v3_result_line(
            instruments=["HST"],
            datasets=["Gaia DR3"],
            methods=["MCMC"],
            observables=["stellar mass"],
            materials=["silicate dust"],
            software=["emcee"],
        )
        rows = _parse_v3_extraction_rows(line)
        assert len(rows) == 1

    def test_skips_errored_results(self) -> None:
        line = {"custom_id": "bib_err", "result": {"type": "errored"}}
        rows = _parse_v3_extraction_rows(line)
        assert rows == []

    def test_skips_expired_results(self) -> None:
        line = {"custom_id": "bib_exp", "result": {"type": "expired"}}
        rows = _parse_v3_extraction_rows(line)
        assert rows == []

    def test_skips_all_empty_categories(self) -> None:
        """If no entities are extracted at all, skip the row."""
        line = _make_v3_result_line()  # all empty lists
        rows = _parse_v3_extraction_rows(line)
        assert rows == []

    def test_strips_whitespace_in_entities(self) -> None:
        line = _make_v3_result_line(instruments=["  ALMA  ", " VLT"])
        rows = _parse_v3_extraction_rows(line)
        assert rows[0].payload["instruments"] == ["ALMA", "VLT"]

    def test_filters_non_string_entities(self) -> None:
        line = _make_v3_result_line(instruments=["ALMA", 42, None, "VLT"])  # type: ignore[list-item]
        rows = _parse_v3_extraction_rows(line)
        assert rows[0].payload["instruments"] == ["ALMA", "VLT"]

    def test_filters_empty_string_entities(self) -> None:
        line = _make_v3_result_line(instruments=["ALMA", "", "  ", "VLT"])
        rows = _parse_v3_extraction_rows(line)
        assert rows[0].payload["instruments"] == ["ALMA", "VLT"]

    def test_source_and_confidence_tier(self) -> None:
        line = _make_v3_result_line(methods=["PCA"])
        rows = _parse_v3_extraction_rows(line)
        assert rows[0].source == "llm"
        assert rows[0].confidence_tier == "medium"

    def test_custom_version(self) -> None:
        line = _make_v3_result_line(methods=["PCA"])
        rows = _parse_v3_extraction_rows(line, extraction_version="v3.1")
        assert rows[0].extraction_version == "v3.1"

    def test_wrong_tool_name_returns_empty(self) -> None:
        """If the tool name is 'extract_entities' (v1), v3 parser ignores it."""
        line = {
            "custom_id": "bib_v1",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "extract_entities",  # v1 name
                            "input": {
                                "methods": ["PCA"],
                                "datasets": [],
                                "instruments": [],
                                "materials": [],
                            },
                        }
                    ]
                },
            },
        }
        rows = _parse_v3_extraction_rows(line)
        assert rows == []


# ---------------------------------------------------------------------------
# v3 DB loading tests
# ---------------------------------------------------------------------------


class TestLoadResultsToDbV3:
    def _make_jsonl(self, tmp_path: Path, lines: list[dict[str, Any]]) -> Path:
        p = tmp_path / "test_v3_results.jsonl"
        with open(p, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        return p

    def test_loads_combined_rows(self, tmp_path: Path) -> None:
        line = _make_v3_result_line(
            instruments=["ALMA"],
            methods=["PCA"],
            software=["astropy"],
        )
        jsonl = self._make_jsonl(tmp_path, [line])

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        count = load_results_to_db_v3(mock_conn, jsonl)

        # One combined row per paper
        assert count == 1
        assert mock_cur.execute.call_count == 1
        mock_conn.commit.assert_called()

        # Verify the extraction_type in the INSERT args
        args = mock_cur.execute.call_args[0][1]
        assert args[1] == EXTRACTION_TYPE_V3  # extraction_type

    def test_payload_is_combined_json(self, tmp_path: Path) -> None:
        line = _make_v3_result_line(
            instruments=["HST"],
            observables=["redshift"],
        )
        jsonl = self._make_jsonl(tmp_path, [line])

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        load_results_to_db_v3(mock_conn, jsonl)

        # The payload should be a JSON string with all 6 categories
        args = mock_cur.execute.call_args[0][1]
        payload = json.loads(args[3])  # payload is 4th arg
        assert "instruments" in payload
        assert "observables" in payload
        assert "software" in payload
        assert payload["instruments"] == ["HST"]
        assert payload["observables"] == ["redshift"]

    def test_multiple_papers(self, tmp_path: Path) -> None:
        lines = [
            _make_v3_result_line("bib_1", instruments=["ALMA"]),
            _make_v3_result_line("bib_2", software=["emcee"]),
        ]
        jsonl = self._make_jsonl(tmp_path, lines)

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        count = load_results_to_db_v3(mock_conn, jsonl)
        assert count == 2

    def test_empty_jsonl_returns_zero(self, tmp_path: Path) -> None:
        jsonl = self._make_jsonl(tmp_path, [])
        mock_conn = MagicMock()
        count = load_results_to_db_v3(mock_conn, jsonl)
        assert count == 0
        mock_conn.commit.assert_not_called()

    def test_upsert_sql_uses_on_conflict(self, tmp_path: Path) -> None:
        line = _make_v3_result_line(methods=["MCMC"])
        jsonl = self._make_jsonl(tmp_path, [line])

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        load_results_to_db_v3(mock_conn, jsonl)

        sql = mock_cur.execute.call_args[0][0]
        assert "ON CONFLICT (bibcode, extraction_type, extraction_version)" in sql
        assert "DO UPDATE SET" in sql


# ---------------------------------------------------------------------------
# v3 cohort selection tests
# ---------------------------------------------------------------------------


class TestSelectCohortV3:
    def _mock_conn(self, rows: list[tuple[Any, ...]]) -> MagicMock:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = rows
        return mock_conn

    def test_returns_extraction_requests(self) -> None:
        mock_conn = self._mock_conn(
            [
                ("2024ApJ...1A", "Title One", "Abstract one " * 20),
                ("2024ApJ...2B", "Title Two", "Abstract two " * 20),
            ]
        )
        results = select_cohort_v3(mock_conn, limit=100)
        assert len(results) == 2
        assert all(isinstance(r, ExtractionRequest) for r in results)
        assert results[0].body is None  # No body when with_body=False

    def test_with_body_includes_body_text(self) -> None:
        mock_conn = self._mock_conn([("bib1", "Title", "Abstract " * 20, "Full text body here.")])
        results = select_cohort_v3(mock_conn, limit=100, with_body=True)
        assert len(results) == 1
        assert results[0].body == "Full text body here."

    def test_with_body_none_body(self) -> None:
        """Papers without body text should have body=None."""
        mock_conn = self._mock_conn([("bib1", "Title", "Abstract " * 20, None)])
        results = select_cohort_v3(mock_conn, limit=100, with_body=True)
        assert results[0].body is None

    def test_query_orders_by_citation_count(self) -> None:
        mock_conn = self._mock_conn([])
        select_cohort_v3(mock_conn, limit=5)
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        sql = mock_cur.execute.call_args[0][0]
        assert "citation_count DESC" in sql

    def test_with_body_selects_papers_body(self) -> None:
        mock_conn = self._mock_conn([])
        select_cohort_v3(mock_conn, limit=5, with_body=True)
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value
        sql = mock_cur.execute.call_args[0][0]
        assert "body" in sql
        assert "papers_ads_body" not in sql


# ---------------------------------------------------------------------------
# v3 batch submission tests
# ---------------------------------------------------------------------------


class TestSubmitBatchV3:
    def test_submits_batch_with_v3_prompt(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.batches.create.return_value = MagicMock(id="batch_v3_001")

        reqs = [
            ExtractionRequest("bib1", "Title 1", "Abstract " * 20),
            ExtractionRequest("bib2", "Title 2", "Abstract " * 20, body="Body text"),
        ]
        batch_id, id_to_bibcode = submit_batch_v3(mock_client, reqs)

        assert batch_id == "batch_v3_001"
        assert isinstance(id_to_bibcode, dict)
        create_call = mock_client.messages.batches.create.call_args
        batch_reqs = create_call.kwargs["requests"]
        assert len(batch_reqs) == 2

        # Verify v3 tool is used
        tools = batch_reqs[0]["params"]["tools"]
        assert tools[0]["name"] == "extract_entities_v3"

    def test_uses_haiku_by_default(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.batches.create.return_value = MagicMock(id="batch_v3")

        reqs = [ExtractionRequest("bib1", "Title", "Abstract " * 20)]
        submit_batch_v3(mock_client, reqs)

        create_call = mock_client.messages.batches.create.call_args
        batch_reqs = create_call.kwargs["requests"]
        assert batch_reqs[0]["params"]["model"] == "claude-haiku-4-5-20251001"

    def test_body_included_in_prompt(self) -> None:
        mock_client = MagicMock()
        mock_client.messages.batches.create.return_value = MagicMock(id="batch_v3")

        reqs = [ExtractionRequest("bib1", "Title", "Abstract " * 20, body="Methods: We used CASA.")]
        submit_batch_v3(mock_client, reqs)

        create_call = mock_client.messages.batches.create.call_args
        batch_reqs = create_call.kwargs["requests"]
        messages = batch_reqs[0]["params"]["messages"]
        all_text = json.dumps(messages)
        assert "Body (excerpt)" in all_text
        assert "CASA" in all_text


# ---------------------------------------------------------------------------
# v3 payload compatibility with link_entities
# ---------------------------------------------------------------------------


class TestV3PayloadCompatibility:
    """Verify that v3 extraction output is compatible with link_entities.py."""

    def test_combined_payload_parseable_by_link_entities(self) -> None:
        """link_entities._extract_mentions_from_payload should parse v3 payloads."""
        from scix.link_entities import _extract_mentions_from_payload

        payload = {
            "instruments": ["ALMA", "VLT"],
            "datasets": ["Gaia DR3"],
            "methods": ["MCMC"],
            "observables": ["redshift"],
            "materials": ["silicate dust"],
            "software": ["astropy"],
        }
        mentions = _extract_mentions_from_payload(payload, EXTRACTION_TYPE_V3)

        # Should find all 7 mentions across 6 categories
        mention_texts = [m[0] for m in mentions]
        assert "ALMA" in mention_texts
        assert "VLT" in mention_texts
        assert "Gaia DR3" in mention_texts
        assert "MCMC" in mention_texts
        assert "redshift" in mention_texts
        assert "silicate dust" in mention_texts
        assert "astropy" in mention_texts

    def test_empty_categories_produce_no_mentions(self) -> None:
        from scix.link_entities import _extract_mentions_from_payload

        payload = {
            "instruments": [],
            "datasets": [],
            "methods": ["PCA"],
            "observables": [],
            "materials": [],
            "software": [],
        }
        mentions = _extract_mentions_from_payload(payload, EXTRACTION_TYPE_V3)
        assert len(mentions) == 1
        assert mentions[0] == ("PCA", "methods")

    def test_payload_key_matches_extraction_type(self) -> None:
        """The payload_key in mentions should match the v3 category names."""
        from scix.link_entities import _extract_mentions_from_payload

        payload = {
            "instruments": ["HST"],
            "datasets": [],
            "methods": [],
            "observables": ["luminosity"],
            "materials": [],
            "software": ["emcee"],
        }
        mentions = _extract_mentions_from_payload(payload, EXTRACTION_TYPE_V3)
        keys = {m[1] for m in mentions}
        assert keys == {"instruments", "observables", "software"}
