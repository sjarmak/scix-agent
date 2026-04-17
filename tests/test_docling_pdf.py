"""Tests for scix.sources.docling_pdf.

Covers:
- DoclingResult normalization to papers_fulltext schema
- PDF URL resolution from OpenAlex metadata
- Failure record creation with R15 exponential backoff
- Production guard enforcement
- Input validation

All tests use mocks — no actual Docling installation or network access required.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from scix.sources.docling_pdf import (
    PARSER_VERSION,
    SOURCE_TAG,
    DoclingConfig,
    DoclingResult,
    FailureRecord,
    compute_retry_after,
    normalize_to_fulltext_row,
    resolve_pdf_url,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


_DEFAULT_SECTIONS: list[dict[str, Any]] = [
    {"heading": "Introduction", "level": 1, "text": "We study galaxies.", "offset": 0},
    {"heading": "Methods", "level": 1, "text": "We used a telescope.", "offset": 19},
]


def _make_docling_result(
    *,
    sections: list[dict[str, Any]] | None = None,
    figures: list[dict[str, Any]] | None = None,
    tables: list[dict[str, Any]] | None = None,
    equations: list[dict[str, Any]] | None = None,
) -> DoclingResult:
    """Build a DoclingResult with sensible defaults."""
    return DoclingResult(
        sections=_DEFAULT_SECTIONS if sections is None else sections,
        figures=figures if figures is not None else [],
        tables=tables if tables is not None else [],
        equations=equations if equations is not None else [],
        parser_version=PARSER_VERSION,
        page_count=10,
        source_pdf_url="https://example.com/paper.pdf",
    )


# ---------------------------------------------------------------------------
# DoclingResult
# ---------------------------------------------------------------------------


class TestDoclingResult:
    def test_is_frozen(self) -> None:
        result = _make_docling_result()
        with pytest.raises(Exception):
            result.sections = []  # type: ignore[misc]

    def test_source_tag_is_docling(self) -> None:
        assert SOURCE_TAG == "docling"

    def test_parser_version_starts_with_docling(self) -> None:
        assert PARSER_VERSION.startswith("docling@")


# ---------------------------------------------------------------------------
# normalize_to_fulltext_row
# ---------------------------------------------------------------------------


class TestNormalizeToFulltextRow:
    def test_produces_correct_keys(self) -> None:
        result = _make_docling_result()
        row = normalize_to_fulltext_row("2023ApJ...100..123X", result)
        assert row["bibcode"] == "2023ApJ...100..123X"
        assert row["source"] == "docling"
        assert row["parser_version"] == PARSER_VERSION
        assert isinstance(row["sections"], str)  # JSON string
        assert isinstance(row["inline_cites"], str)  # JSON string
        assert isinstance(row["figures"], str)
        assert isinstance(row["tables"], str)
        assert isinstance(row["equations"], str)

    def test_sections_round_trip_as_json(self) -> None:
        import json

        result = _make_docling_result()
        row = normalize_to_fulltext_row("2023ApJ...100..123X", result)
        sections = json.loads(row["sections"])
        assert len(sections) == 2
        assert sections[0]["heading"] == "Introduction"

    def test_inline_cites_is_empty_list_json(self) -> None:
        import json

        result = _make_docling_result()
        row = normalize_to_fulltext_row("2023ApJ...100..123X", result)
        assert json.loads(row["inline_cites"]) == []

    def test_empty_sections_produces_empty_json_array(self) -> None:
        import json

        result = _make_docling_result(sections=[])
        row = normalize_to_fulltext_row("2023ApJ...100..123X", result)
        assert json.loads(row["sections"]) == []


# ---------------------------------------------------------------------------
# resolve_pdf_url
# ---------------------------------------------------------------------------


class TestResolvePdfUrl:
    def test_returns_pdf_url_from_openalex_location(self) -> None:
        openalex_meta = {
            "best_oa_location": {"pdf_url": "https://example.com/paper.pdf"},
        }
        assert resolve_pdf_url(openalex_meta) == "https://example.com/paper.pdf"

    def test_returns_none_when_no_best_oa_location(self) -> None:
        assert resolve_pdf_url({}) is None

    def test_returns_none_when_pdf_url_missing(self) -> None:
        openalex_meta = {"best_oa_location": {"landing_page_url": "https://example.com"}}
        assert resolve_pdf_url(openalex_meta) is None

    def test_returns_none_when_best_oa_location_is_none(self) -> None:
        assert resolve_pdf_url({"best_oa_location": None}) is None

    def test_rejects_non_https_urls(self) -> None:
        openalex_meta = {
            "best_oa_location": {"pdf_url": "ftp://example.com/paper.pdf"},
        }
        assert resolve_pdf_url(openalex_meta) is None

    def test_accepts_http_urls(self) -> None:
        openalex_meta = {
            "best_oa_location": {"pdf_url": "http://example.com/paper.pdf"},
        }
        assert resolve_pdf_url(openalex_meta) == "http://example.com/paper.pdf"


# ---------------------------------------------------------------------------
# FailureRecord + R15 backoff
# ---------------------------------------------------------------------------


class TestFailureRecord:
    def test_is_frozen(self) -> None:
        rec = FailureRecord(
            bibcode="2023ApJ...100..123X",
            parser_version=PARSER_VERSION,
            failure_reason="pdf_download_failed",
            attempts=1,
            retry_after=dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=24),
        )
        with pytest.raises(Exception):
            rec.bibcode = "x"  # type: ignore[misc]


class TestComputeRetryAfter:
    def test_attempt_1_is_24_hours(self) -> None:
        now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        retry = compute_retry_after(attempts=1, now=now)
        assert retry == now + dt.timedelta(hours=24)

    def test_attempt_2_is_3_days(self) -> None:
        now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        retry = compute_retry_after(attempts=2, now=now)
        assert retry == now + dt.timedelta(days=3)

    def test_attempt_3_is_7_days(self) -> None:
        now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        retry = compute_retry_after(attempts=3, now=now)
        assert retry == now + dt.timedelta(days=7)

    def test_attempt_4_plus_is_30_days(self) -> None:
        now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        for attempt in (4, 5, 10):
            retry = compute_retry_after(attempts=attempt, now=now)
            assert retry == now + dt.timedelta(days=30)


# ---------------------------------------------------------------------------
# DoclingConfig
# ---------------------------------------------------------------------------


class TestDoclingConfig:
    def test_is_frozen(self) -> None:
        cfg = DoclingConfig(dsn="dbname=scix_test")
        with pytest.raises(Exception):
            cfg.dsn = "x"  # type: ignore[misc]

    def test_defaults(self) -> None:
        cfg = DoclingConfig(dsn="dbname=scix_test")
        assert cfg.yes_production is False
        assert cfg.batch_size == 100
        assert cfg.dry_run is False
