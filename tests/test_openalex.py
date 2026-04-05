"""Tests for OpenAlex DOI linking (mocked API, no real HTTP calls)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scix.openalex import (
    _extract_topics,
    fetch_openalex_by_doi,
    link_papers_batch,
)

# ---------------------------------------------------------------------------
# Sample OpenAlex API response fixture
# ---------------------------------------------------------------------------

SAMPLE_WORK = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.1038/s41586-020-2649-2",
    "title": "Array programming with NumPy",
    "topics": [
        {
            "id": "https://openalex.org/T12345",
            "display_name": "Scientific Computing",
            "score": 0.95,
            "subfield": {"display_name": "Computational Science"},
            "field": {"display_name": "Computer Science"},
            "domain": {"display_name": "Physical Sciences"},
        },
        {
            "id": "https://openalex.org/T67890",
            "display_name": "Numerical Methods",
            "score": 0.82,
            "subfield": {"display_name": "Applied Mathematics"},
            "field": {"display_name": "Mathematics"},
            "domain": {"display_name": "Physical Sciences"},
        },
    ],
}


# ---------------------------------------------------------------------------
# fetch_openalex_by_doi
# ---------------------------------------------------------------------------


class TestFetchOpenalexByDoi:
    """Tests for fetch_openalex_by_doi."""

    @patch("scix.openalex.urllib.request.urlopen")
    def test_successful_fetch(self, mock_urlopen: MagicMock) -> None:
        """A successful API call returns the parsed work dict."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(SAMPLE_WORK).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_openalex_by_doi("10.1038/s41586-020-2649-2", "test@example.com")

        assert result is not None
        assert result["id"] == "https://openalex.org/W2741809807"
        assert len(result["topics"]) == 2

    @patch("scix.openalex.urllib.request.urlopen")
    def test_404_returns_none(self, mock_urlopen: MagicMock) -> None:
        """A 404 response returns None (DOI not found in OpenAlex)."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.openalex.org/works/doi:10.0000/nonexistent",
            code=404,
            msg="Not Found",
            hdrs={},  # type: ignore[arg-type]
            fp=None,
        )

        result = fetch_openalex_by_doi("10.0000/nonexistent", "test@example.com")
        assert result is None

    @patch("scix.openalex.urllib.request.urlopen")
    def test_server_error_returns_none(self, mock_urlopen: MagicMock) -> None:
        """A 500 error returns None gracefully."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.openalex.org/works/doi:10.1234/broken",
            code=500,
            msg="Internal Server Error",
            hdrs={},  # type: ignore[arg-type]
            fp=None,
        )

        result = fetch_openalex_by_doi("10.1234/broken", "test@example.com")
        assert result is None

    @patch("scix.openalex.urllib.request.urlopen")
    def test_strips_doi_url_prefix(self, mock_urlopen: MagicMock) -> None:
        """DOIs with https://doi.org/ prefix are cleaned before use."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(SAMPLE_WORK).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        fetch_openalex_by_doi("https://doi.org/10.1038/s41586-020-2649-2", "test@example.com")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "doi:10.1038" in req.full_url
        assert "doi:https" not in req.full_url


# ---------------------------------------------------------------------------
# _extract_topics
# ---------------------------------------------------------------------------


class TestExtractTopics:
    """Tests for topic extraction from work objects."""

    def test_extracts_all_fields(self) -> None:
        topics = _extract_topics(SAMPLE_WORK)
        assert len(topics) == 2
        first = topics[0]
        assert first["display_name"] == "Scientific Computing"
        assert first["score"] == 0.95
        assert first["subfield"] == "Computational Science"
        assert first["field"] == "Computer Science"
        assert first["domain"] == "Physical Sciences"

    def test_empty_topics(self) -> None:
        work: dict = {"id": "W1", "topics": []}
        assert _extract_topics(work) == []

    def test_missing_topics_key(self) -> None:
        work: dict = {"id": "W1"}
        assert _extract_topics(work) == []

    def test_handles_missing_subfield(self) -> None:
        work: dict = {
            "topics": [
                {
                    "id": "T1",
                    "display_name": "Test",
                    "score": 0.5,
                    "subfield": None,
                    "field": None,
                    "domain": None,
                }
            ]
        }
        topics = _extract_topics(work)
        assert topics[0]["subfield"] is None
        assert topics[0]["field"] is None
        assert topics[0]["domain"] is None


# ---------------------------------------------------------------------------
# link_papers_batch
# ---------------------------------------------------------------------------


class TestLinkPapersBatch:
    """Tests for batch linking (mocked DB + API)."""

    @patch("scix.openalex.time.sleep")
    @patch("scix.openalex.fetch_openalex_by_doi")
    def test_links_papers_and_returns_count(
        self, mock_fetch: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Successfully links papers and returns the count."""
        mock_fetch.return_value = SAMPLE_WORK

        # Mock psycopg connection
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        dois_bibcodes = [
            ("10.1038/s41586-020-2649-2", "2020Natur.585..357H"),
            ("10.1234/test", "2021ApJ...900..100T"),
        ]

        result = link_papers_batch(mock_conn, dois_bibcodes, mailto="test@example.com")

        assert result == 2
        assert mock_fetch.call_count == 2
        assert mock_cursor.execute.call_count == 2
        assert mock_conn.commit.call_count == 2
        # Rate limiting sleep was called
        assert mock_sleep.call_count == 2

    @patch("scix.openalex.time.sleep")
    @patch("scix.openalex.fetch_openalex_by_doi")
    def test_skips_not_found_dois(self, mock_fetch: MagicMock, mock_sleep: MagicMock) -> None:
        """DOIs not found in OpenAlex are skipped (no DB update)."""
        mock_fetch.return_value = None

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        dois_bibcodes = [("10.0000/missing", "2020XXXX...000...0X")]

        result = link_papers_batch(mock_conn, dois_bibcodes, mailto="test@example.com")

        assert result == 0
        mock_cursor.execute.assert_not_called()
        mock_conn.commit.assert_not_called()

    @patch("scix.openalex.time.sleep")
    @patch("scix.openalex.fetch_openalex_by_doi")
    def test_empty_batch(self, mock_fetch: MagicMock, mock_sleep: MagicMock) -> None:
        """An empty batch returns 0."""
        mock_conn = MagicMock()
        result = link_papers_batch(mock_conn, [], mailto="test@example.com")
        assert result == 0
        mock_fetch.assert_not_called()

    @patch("scix.openalex.time.sleep")
    @patch("scix.openalex.fetch_openalex_by_doi")
    def test_rate_limiting_interval(self, mock_fetch: MagicMock, mock_sleep: MagicMock) -> None:
        """Each request sleeps at least 0.1s for polite pool compliance."""
        mock_fetch.return_value = SAMPLE_WORK

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        link_papers_batch(
            mock_conn,
            [("10.1038/s41586-020-2649-2", "2020Natur.585..357H")],
            mailto="test@example.com",
        )

        mock_sleep.assert_called_with(0.1)
