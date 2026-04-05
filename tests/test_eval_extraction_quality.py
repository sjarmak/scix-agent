"""Tests for extraction quality evaluation script.

Unit tests use mocked database connections and EntityResolver to verify
sampling, mention extraction, evaluation metrics, and report formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_extraction_quality import (
    EvalResult,
    evaluate_mentions,
    format_report,
    get_mentions,
    sample_papers,
    write_report,
)
from scix.entity_resolver import EntityCandidate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    entity_id: int = 1,
    canonical_name: str = "Test Entity",
    entity_type: str = "instrument",
    source: str = "test",
    discipline: str | None = None,
    confidence: float = 1.0,
    match_method: str = "exact_canonical",
) -> EntityCandidate:
    """Create an EntityCandidate for testing."""
    return EntityCandidate(
        entity_id=entity_id,
        canonical_name=canonical_name,
        entity_type=entity_type,
        source=source,
        discipline=discipline,
        confidence=confidence,
        match_method=match_method,
    )


# ---------------------------------------------------------------------------
# Tests: script importability
# ---------------------------------------------------------------------------


class TestImportability:
    """Verify the script can be imported without side effects."""

    def test_script_is_importable(self) -> None:
        import eval_extraction_quality

        assert hasattr(eval_extraction_quality, "main")
        assert hasattr(eval_extraction_quality, "sample_papers")
        assert hasattr(eval_extraction_quality, "evaluate_mentions")


# ---------------------------------------------------------------------------
# Tests: sample_papers
# ---------------------------------------------------------------------------


class TestSamplePapers:
    """Tests for sample_papers function."""

    def test_queries_extractions_table(self) -> None:
        """Verify that sample_papers queries the extractions table."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda self: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("2024ApJ...1A",),
            ("2024ApJ...2B",),
        ]

        result = sample_papers(mock_conn, n=50)

        # Check that execute was called with a query mentioning extractions
        call_args = mock_cursor.execute.call_args
        query = call_args[0][0]
        assert "extractions" in query.lower()
        assert len(result) == 2
        assert "2024ApJ...1A" in result

    def test_returns_empty_when_no_papers(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda self: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        result = sample_papers(mock_conn, n=50)
        assert result == []

    def test_passes_sample_size_to_query(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda self: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        sample_papers(mock_conn, n=25)

        call_args = mock_cursor.execute.call_args
        params = call_args[0][1]
        assert params == (25,)


# ---------------------------------------------------------------------------
# Tests: get_mentions
# ---------------------------------------------------------------------------


class TestGetMentions:
    """Tests for get_mentions function."""

    def test_parses_payload_entities(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda self: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("bib1", "instruments", {"entities": ["Hubble", "ALMA"]}),
            ("bib1", "methods", {"entities": ["MCMC"]}),
            ("bib2", "datasets", {"entities": ["Gaia DR3"]}),
        ]

        result = get_mentions(mock_conn, ["bib1", "bib2"])

        assert set(result["bib1"]) == {"Hubble", "ALMA", "MCMC"}
        assert result["bib2"] == ["Gaia DR3"]

    def test_handles_string_payload(self) -> None:
        """Payload may come as JSON string instead of dict."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda self: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("bib1", "instruments", '{"entities": ["HST"]}'),
        ]

        result = get_mentions(mock_conn, ["bib1"])
        assert result["bib1"] == ["HST"]

    def test_empty_bibcodes_returns_empty(self) -> None:
        mock_conn = MagicMock()
        result = get_mentions(mock_conn, [])
        assert result == {}

    def test_handles_missing_entities_key(self) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda self: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("bib1", "instruments", {"other_key": "value"}),
        ]

        result = get_mentions(mock_conn, ["bib1"])
        assert result["bib1"] == []


# ---------------------------------------------------------------------------
# Tests: evaluate_mentions
# ---------------------------------------------------------------------------


class TestEvaluateMentions:
    """Tests for evaluate_mentions function."""

    def test_computes_resolution_rate(self) -> None:
        mock_resolver = MagicMock()
        # First mention resolves, second does not
        mock_resolver.resolve.side_effect = [
            [_make_candidate(match_method="exact_canonical")],
            [],
        ]

        mentions = {"bib1": ["Hubble", "UnknownThing"]}
        result = evaluate_mentions(mock_resolver, mentions)

        assert result.total_mentions == 2
        assert result.resolved_mentions == 1
        assert result.resolution_rate == pytest.approx(0.5)
        assert result.recall_proxy == pytest.approx(0.5)

    def test_tracks_match_method_distribution(self) -> None:
        mock_resolver = MagicMock()
        mock_resolver.resolve.side_effect = [
            [_make_candidate(match_method="exact_canonical")],
            [_make_candidate(match_method="alias")],
            [_make_candidate(match_method="exact_canonical")],
            [_make_candidate(match_method="identifier")],
        ]

        mentions = {"bib1": ["A", "B", "C", "D"]}
        result = evaluate_mentions(mock_resolver, mentions)

        assert result.match_method_counts["exact_canonical"] == 2
        assert result.match_method_counts["alias"] == 1
        assert result.match_method_counts["identifier"] == 1

    def test_collects_unmatched_mentions(self) -> None:
        mock_resolver = MagicMock()
        mock_resolver.resolve.side_effect = [
            [_make_candidate()],
            [],
            [],
        ]

        mentions = {"bib1": ["Resolved", "Unmatched1", "Unmatched2"]}
        result = evaluate_mentions(mock_resolver, mentions)

        assert len(result.unmatched_mentions) == 2
        assert "Unmatched1" in result.unmatched_mentions
        assert "Unmatched2" in result.unmatched_mentions

    def test_empty_mentions_returns_zeros(self) -> None:
        mock_resolver = MagicMock()
        result = evaluate_mentions(mock_resolver, {})

        assert result.total_mentions == 0
        assert result.resolved_mentions == 0
        assert result.resolution_rate == 0.0
        assert result.unmatched_mentions == ()

    def test_all_resolved(self) -> None:
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = [_make_candidate(match_method="exact_canonical")]

        mentions = {"bib1": ["A", "B"]}
        result = evaluate_mentions(mock_resolver, mentions)

        assert result.resolution_rate == pytest.approx(1.0)
        assert len(result.unmatched_mentions) == 0

    def test_passes_fuzzy_flag(self) -> None:
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = []

        mentions = {"bib1": ["A"]}
        evaluate_mentions(mock_resolver, mentions, fuzzy=True)

        mock_resolver.resolve.assert_called_once_with("A", fuzzy=True)


# ---------------------------------------------------------------------------
# Tests: format_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    """Tests for format_report function."""

    def _make_result(self) -> EvalResult:
        return EvalResult(
            papers_sampled=50,
            total_mentions=100,
            resolved_mentions=75,
            unmatched_mentions=("UnknownA", "UnknownB", "UnknownC"),
            match_method_counts={
                "exact_canonical": 50,
                "alias": 20,
                "identifier": 5,
            },
        )

    def test_includes_precision_and_recall(self) -> None:
        report = format_report(self._make_result())
        assert "precision proxy" in report.lower()
        assert "recall proxy" in report.lower()
        assert "75.0%" in report

    def test_includes_match_method_distribution(self) -> None:
        report = format_report(self._make_result())
        assert "exact_canonical" in report
        assert "alias" in report
        assert "identifier" in report
        assert "Match Method Distribution" in report

    def test_includes_unmatched_examples(self) -> None:
        report = format_report(self._make_result())
        assert "UnknownA" in report
        assert "UnknownB" in report
        assert "Unmatched Mention Examples" in report

    def test_truncates_unmatched_at_20(self) -> None:
        many_unmatched = tuple(f"Mention_{i}" for i in range(30))
        result = EvalResult(
            papers_sampled=10,
            total_mentions=30,
            resolved_mentions=0,
            unmatched_mentions=many_unmatched,
            match_method_counts={},
        )
        report = format_report(result)
        assert "Mention_19" in report
        assert "Mention_20" not in report
        assert "10 more" in report

    def test_handles_zero_mentions(self) -> None:
        result = EvalResult(
            papers_sampled=0,
            total_mentions=0,
            resolved_mentions=0,
            unmatched_mentions=(),
            match_method_counts={},
        )
        report = format_report(result)
        assert "0.0%" in report
        assert "All mentions resolved" in report


# ---------------------------------------------------------------------------
# Tests: write_report
# ---------------------------------------------------------------------------


class TestWriteReport:
    """Tests for write_report function."""

    def test_writes_to_file(self, tmp_path: Path) -> None:
        output = tmp_path / "subdir" / "report.md"
        write_report("# Test Report\n", str(output))

        assert output.exists()
        assert output.read_text() == "# Test Report\n"


# ---------------------------------------------------------------------------
# Tests: EvalResult properties
# ---------------------------------------------------------------------------


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_frozen(self) -> None:
        result = EvalResult(
            papers_sampled=1,
            total_mentions=2,
            resolved_mentions=1,
            unmatched_mentions=("x",),
            match_method_counts={"exact_canonical": 1},
        )
        with pytest.raises(AttributeError):
            result.total_mentions = 5  # type: ignore[misc]

    def test_resolution_rate_property(self) -> None:
        result = EvalResult(
            papers_sampled=1,
            total_mentions=4,
            resolved_mentions=3,
            unmatched_mentions=("x",),
            match_method_counts={},
        )
        assert result.resolution_rate == pytest.approx(0.75)
        assert result.recall_proxy == pytest.approx(0.75)
