"""Tests for read_paper_section and search_within_paper MCP tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scix.mcp_server import _dispatch_tool
from scix.search import (
    SearchResult,
    get_document_context,
    get_entity_context,
    read_paper_section,
    search_within_paper,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BODY = """Introduction
This paper studies dark matter halos in galaxy clusters.
We present new observations from the Hubble Space Telescope.

Methods
We used spectroscopic analysis of 500 galaxies.
The data was reduced using standard IRAF pipelines.

Results
We find a strong correlation between halo mass and cluster richness.
The best-fit relation has a slope of 1.3 +/- 0.2.

Conclusions
Our results confirm previous findings and extend them to higher redshifts.
"""

SAMPLE_ABSTRACT = "We study dark matter halos in galaxy clusters using HST observations."


def _mock_cursor_with_row(row, extra_rows=None):
    """Create a mock connection whose cursor returns rows in sequence.

    The first ``fetchone()`` returns *row* (the papers table lookup). If
    *extra_rows* is provided, subsequent ``fetchone()`` calls return those
    values in order. If *extra_rows* is ``None``, subsequent calls return
    ``None`` indefinitely — matching the typical case where no
    papers_fulltext row exists for the bibcode (which short-circuits both
    the structured-section path and the ADR-006 guard query).

    The "indefinitely" matters: the read_paper_section flow can issue
    multiple secondary fetchones (papers_fulltext.sections lookup, then
    papers_fulltext.source for the latex-provenance guard), and the count
    has grown over time as new guards landed. Returning a generator that
    keeps yielding None is more resilient than the older fixed
    [row, None] sequence which exhausted on the third call.
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    if extra_rows is not None:
        # Caller is being explicit — give them exactly what they asked for.
        mock_cursor.fetchone.side_effect = [row] + extra_rows
    else:
        def _fetchone_sequence():
            yield row
            while True:
                yield None
        mock_cursor.fetchone.side_effect = _fetchone_sequence()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


# ---------------------------------------------------------------------------
# read_paper_section — unit tests
# ---------------------------------------------------------------------------


class TestReadPaperSection:
    def test_paper_with_body_full(self) -> None:
        """Reading full body returns the entire text with has_body=True."""
        row = {"body": SAMPLE_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is True
        assert paper["section_name"] == "full"
        assert paper["bibcode"] == "2024ApJ...001A"
        assert "dark matter" in paper["section_text"]
        assert paper["total_chars"] == len(SAMPLE_BODY)

    def test_paper_with_body_specific_section(self) -> None:
        """Reading a specific section returns only that section's text."""
        row = {"body": SAMPLE_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A", section="introduction")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is True
        assert paper["section_name"] == "introduction"
        assert "dark matter halos" in paper["section_text"]
        # Should NOT contain methods content
        assert "spectroscopic analysis" not in paper["section_text"]

    def test_paper_with_body_section_not_found(self) -> None:
        """Requesting a non-existent section returns empty with available sections."""
        row = {"body": SAMPLE_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A", section="acknowledgments")

        assert result.total == 0
        assert "available_sections" in result.metadata
        assert result.metadata["has_body"] is True

    def test_fallback_to_abstract(self) -> None:
        """Paper without body falls back to abstract with has_body=False."""
        row = {"body": None, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is False
        assert paper["section_name"] == "abstract"
        assert "dark matter" in paper["section_text"]
        assert result.metadata["has_body"] is False

    def test_fallback_to_abstract_empty_body(self) -> None:
        """Paper with empty string body falls back to abstract."""
        row = {"body": "", "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is False

    def test_pagination_char_offset(self) -> None:
        """Pagination with char_offset and limit works correctly."""
        row = {"body": SAMPLE_BODY, "abstract": SAMPLE_ABSTRACT, "title": "Test Paper"}
        conn = _mock_cursor_with_row(row)

        result = read_paper_section(conn, "2024ApJ...001A", char_offset=10, limit=50)

        assert result.total == 1
        paper = result.papers[0]
        assert paper["char_offset"] == 10
        assert len(paper["section_text"]) <= 50
        assert paper["total_chars"] == len(SAMPLE_BODY)

    def test_paper_not_found(self) -> None:
        """Non-existent bibcode returns empty result with error."""
        conn = _mock_cursor_with_row(None)

        result = read_paper_section(conn, "NONEXISTENT")

        assert result.total == 0
        assert "error" in result.metadata


# ---------------------------------------------------------------------------
# search_within_paper — unit tests
# ---------------------------------------------------------------------------


class TestSearchWithinPaper:
    def test_matching_query(self) -> None:
        """Search with matching terms returns headline with has_body=True."""
        row = {
            "bibcode": "2024ApJ...001A",
            "title": "Test Paper",
            "body": SAMPLE_BODY,
            "headline": "...studies <b>dark matter</b> halos in galaxy clusters...",
        }
        conn = _mock_cursor_with_row(row)

        result = search_within_paper(conn, "2024ApJ...001A", "dark matter")

        assert result.total == 1
        paper = result.papers[0]
        assert paper["has_body"] is True
        assert "dark matter" in paper["headline"]
        assert paper["bibcode"] == "2024ApJ...001A"

    def test_no_body_text(self) -> None:
        """Paper without body returns empty with has_body=False metadata."""
        # First query (body search) returns None, second (existence check) returns paper
        mock_conn = MagicMock()

        mock_cursor_1 = MagicMock()
        mock_cursor_1.fetchone.return_value = None
        mock_cursor_1.__enter__ = MagicMock(return_value=mock_cursor_1)
        mock_cursor_1.__exit__ = MagicMock(return_value=False)

        mock_cursor_2 = MagicMock()
        mock_cursor_2.fetchone.return_value = {
            "bibcode": "2024ApJ...001A",
            "title": "Test Paper",
        }
        mock_cursor_2.__enter__ = MagicMock(return_value=mock_cursor_2)
        mock_cursor_2.__exit__ = MagicMock(return_value=False)

        mock_conn.cursor.side_effect = [mock_cursor_1, mock_cursor_2]

        result = search_within_paper(mock_conn, "2024ApJ...001A", "dark matter")

        assert result.total == 0
        assert result.metadata["has_body"] is False

    def test_paper_not_found(self) -> None:
        """Non-existent paper returns empty with error metadata."""
        mock_conn = MagicMock()

        mock_cursor_1 = MagicMock()
        mock_cursor_1.fetchone.return_value = None
        mock_cursor_1.__enter__ = MagicMock(return_value=mock_cursor_1)
        mock_cursor_1.__exit__ = MagicMock(return_value=False)

        mock_cursor_2 = MagicMock()
        mock_cursor_2.fetchone.return_value = None
        mock_cursor_2.__enter__ = MagicMock(return_value=mock_cursor_2)
        mock_cursor_2.__exit__ = MagicMock(return_value=False)

        mock_conn.cursor.side_effect = [mock_cursor_1, mock_cursor_2]

        result = search_within_paper(mock_conn, "NONEXISTENT", "dark matter")

        assert result.total == 0
        assert "error" in result.metadata


# ---------------------------------------------------------------------------
# MCP dispatch tests
# ---------------------------------------------------------------------------


class TestDispatchPaperTools:
    @patch("scix.search.read_paper_section")
    def test_read_paper_section_dispatches(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = SearchResult(
            papers=[
                {
                    "bibcode": "2024ApJ...001A",
                    "section_name": "full",
                    "section_text": "body text",
                    "has_body": True,
                    "char_offset": 0,
                    "total_chars": 100,
                }
            ],
            total=1,
            timing_ms={"query_ms": 2.0},
            metadata={"has_body": True},
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "read_paper_section",
                {
                    "bibcode": "2024ApJ...001A",
                    "section": "introduction",
                    "char_offset": 100,
                    "limit": 3000,
                },
            )
        )
        assert result["total"] == 1
        mock_fn.assert_called_once_with(
            mock_conn,
            "2024ApJ...001A",
            section="introduction",
            char_offset=100,
            limit=3000,
            role=None,
        )

    @patch("scix.search.read_paper_section")
    def test_read_paper_section_defaults(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = SearchResult(papers=[], total=0, timing_ms={"query_ms": 1.0})
        mock_conn = MagicMock()
        _dispatch_tool(
            mock_conn,
            "read_paper_section",
            {"bibcode": "2024ApJ...001A"},
        )
        mock_fn.assert_called_once_with(
            mock_conn,
            "2024ApJ...001A",
            section="full",
            char_offset=0,
            limit=5000,
            role=None,
        )

    @patch("scix.search.search_within_paper")
    def test_search_within_paper_dispatches(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = SearchResult(
            papers=[
                {
                    "bibcode": "2024ApJ...001A",
                    "headline": "<b>dark matter</b> halos",
                    "has_body": True,
                }
            ],
            total=1,
            timing_ms={"query_ms": 5.0},
            metadata={"has_body": True},
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "search_within_paper",
                {"bibcode": "2024ApJ...001A", "query": "dark matter"},
            )
        )
        assert result["total"] == 1
        assert "dark matter" in result["papers"][0]["headline"]
        mock_fn.assert_called_once_with(
            mock_conn,
            "2024ApJ...001A",
            "dark matter",
        )


# ---------------------------------------------------------------------------
# get_document_context — unit tests
# ---------------------------------------------------------------------------


class TestGetDocumentContext:
    """Tests for search.get_document_context() — queries agent_document_context matview."""

    def _make_conn_with_row(self, row: dict | None) -> MagicMock:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = row
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        return mock_conn

    def test_existing_paper_returns_context(self) -> None:
        """Known bibcode returns paper metadata + linked entities."""
        row = {
            "bibcode": "2019Natur.568...55L",
            "title": "Bennu surface composition",
            "abstract": "OSIRIS-REx observations of asteroid Bennu...",
            "year": 2019,
            "citation_count": 427,
            "reference_count": 31,
            "linked_entities": [
                {
                    "entity_id": 42,
                    "name": "OSIRIS-REx",
                    "type": "mission",
                    "link_type": "extraction",
                    "confidence": 0.95,
                },
                {
                    "entity_id": 99,
                    "name": "Bennu",
                    "type": "object",
                    "link_type": "extraction",
                    "confidence": 0.99,
                },
            ],
        }
        conn = self._make_conn_with_row(row)

        result = get_document_context(conn, "2019Natur.568...55L")

        assert isinstance(result, SearchResult)
        assert result.total == 1
        assert len(result.papers) == 1
        paper = result.papers[0]
        assert paper["bibcode"] == "2019Natur.568...55L"
        assert paper["title"] == "Bennu surface composition"
        assert paper["citation_count"] == 427
        assert paper["reference_count"] == 31
        assert len(paper["linked_entities"]) == 2
        assert paper["linked_entities"][0]["name"] == "OSIRIS-REx"
        assert "query_ms" in result.timing_ms

    def test_paper_with_no_linked_entities(self) -> None:
        """Paper with empty linked_entities returns empty list, not error."""
        row = {
            "bibcode": "2024ApJ...001A",
            "title": "Test Paper",
            "abstract": "Abstract",
            "year": 2024,
            "citation_count": 0,
            "reference_count": 0,
            "linked_entities": [],
        }
        conn = self._make_conn_with_row(row)

        result = get_document_context(conn, "2024ApJ...001A")

        assert result.total == 1
        assert result.papers[0]["linked_entities"] == []

    def test_nonexistent_bibcode_returns_empty(self) -> None:
        """Unknown bibcode returns empty result with error metadata, not exception."""
        conn = self._make_conn_with_row(None)

        result = get_document_context(conn, "9999XXX...000Z")

        assert result.total == 0
        assert result.papers == []
        assert "error" in result.metadata
        assert "9999XXX...000Z" in result.metadata["error"]

    def test_query_uses_matview(self) -> None:
        """Query targets the agent_document_context matview, not papers table."""
        conn = self._make_conn_with_row(None)

        get_document_context(conn, "2024ApJ...001A")

        cur = conn.cursor.return_value
        sql = cur.execute.call_args[0][0]
        assert "agent_document_context" in sql
        params = cur.execute.call_args[0][1]
        assert params == ("2024ApJ...001A",)


class TestDispatchDocumentContext:
    """Tests for MCP _dispatch_tool routing to get_document_context."""

    @patch("scix.search.get_document_context")
    def test_dispatch_routes_to_search(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = SearchResult(
            papers=[
                {
                    "bibcode": "2024ApJ...001A",
                    "title": "Test",
                    "abstract": "Abstract",
                    "year": 2024,
                    "citation_count": 5,
                    "reference_count": 10,
                    "linked_entities": [],
                }
            ],
            total=1,
            timing_ms={"query_ms": 1.5},
        )
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "document_context",
                {"bibcode": "2024ApJ...001A"},
            )
        )
        assert result["total"] == 1
        assert result["papers"][0]["bibcode"] == "2024ApJ...001A"
        mock_fn.assert_called_once_with(mock_conn, "2024ApJ...001A")

    def test_empty_bibcode_returns_error(self) -> None:
        """Empty bibcode is rejected with a clear error before any DB call."""
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "document_context",
                {"bibcode": ""},
            )
        )
        assert "error" in result
        assert "bibcode" in result["error"].lower()

    def test_whitespace_bibcode_returns_error(self) -> None:
        """Whitespace-only bibcode is also rejected."""
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(
                mock_conn,
                "document_context",
                {"bibcode": "   "},
            )
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# get_entity_context — unit tests
# ---------------------------------------------------------------------------


class TestGetEntityContext:
    """Tests for search.get_entity_context() — queries agent_entity_context
    matview for aggregates, joins through entity_relationships for the
    relationships block, and reads entities.properties directly so
    agents see the full JSONB payload without waiting for a matview
    refresh.
    """

    def _make_conn(
        self,
        *,
        matview_row: dict | None,
        fallback_row: dict | None = None,
        properties_row: dict | None = None,
        relationship_rows: list[dict] | None = None,
    ) -> MagicMock:
        """Build a mock connection whose cursor replays the expected sequence.

        The real ``get_entity_context`` issues up to three queries when the
        matview has a hit (matview, properties, relationships) and two when
        it falls back to the ``entities`` table (fallback, relationships).
        """
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        fetchone_seq: list[dict | None]
        if matview_row is not None:
            fetchone_seq = [matview_row, properties_row]
        else:
            fetchone_seq = [None, fallback_row]
        mock_cursor.fetchone.side_effect = fetchone_seq
        mock_cursor.fetchall.return_value = relationship_rows or []
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        return mock_conn

    def test_existing_entity_returns_full_card(self) -> None:
        """Known entity_id returns full entity card with identifiers, aliases, relationships."""
        matview_row = {
            "entity_id": 42,
            "canonical_name": "OSIRIS-REx",
            "entity_type": "mission",
            "discipline": "astronomy",
            "source": "metadata",
            "identifiers": [{"scheme": "nasa_mission", "id": "osiris-rex"}],
            "aliases": ["Origins Spectral Interpretation", "OSIRIS REx"],
            "citing_paper_count": 1523,
        }
        properties_row = {"properties": {"launch_year": 2016}}
        relationship_rows = [
            {
                "direction": "out",
                "predicate": "observes",
                "object_id": 99,
                "confidence": 0.95,
                "relationship_source": "curated",
                "object_name": "Bennu",
                "object_entity_type": "target",
                "object_properties": {"diameter_m": 492},
            }
        ]
        conn = self._make_conn(
            matview_row=matview_row,
            properties_row=properties_row,
            relationship_rows=relationship_rows,
        )

        result = get_entity_context(conn, 42)

        assert isinstance(result, SearchResult)
        assert result.total == 1
        entity = result.papers[0]
        assert entity["entity_id"] == 42
        assert entity["canonical_name"] == "OSIRIS-REx"
        assert entity["entity_type"] == "mission"
        assert entity["citing_paper_count"] == 1523
        assert entity["properties"] == {"launch_year": 2016}
        assert len(entity["identifiers"]) == 1
        assert len(entity["aliases"]) == 2
        assert len(entity["relationships"]) == 1
        rel = entity["relationships"][0]
        assert rel["object_name"] == "Bennu"
        assert rel["object_properties"] == {"diameter_m": 492}
        assert "query_ms" in result.timing_ms

    def test_entity_with_no_relationships(self) -> None:
        """Entity with empty relationships/aliases returns empty lists."""
        matview_row = {
            "entity_id": 7,
            "canonical_name": "Dark matter",
            "entity_type": "concept",
            "discipline": "physics",
            "source": "uat",
            "identifiers": [],
            "aliases": [],
            "citing_paper_count": 50000,
        }
        conn = self._make_conn(
            matview_row=matview_row,
            properties_row={"properties": {}},
            relationship_rows=[],
        )

        result = get_entity_context(conn, 7)

        assert result.total == 1
        entity = result.papers[0]
        assert entity["aliases"] == []
        assert entity["relationships"] == []
        assert entity["properties"] == {}

    def test_nonexistent_entity_returns_empty(self) -> None:
        """Unknown entity_id returns empty result with error metadata."""
        conn = self._make_conn(matview_row=None, fallback_row=None)

        result = get_entity_context(conn, 999999)

        assert result.total == 0
        assert result.papers == []
        assert "error" in result.metadata
        assert "999999" in result.metadata["error"]

    def test_hub_entity_surfaces_in_edges(self) -> None:
        """Hub entity (mostly OBJECT of edges) returns in-edges with direction='in'.

        Surfaces the bug filed as scix_experiments-1fi: JWST(1588866) had
        five `part_of` in-edges from NIRSpec/NIRCam/MIRI/NIRISS/FGS but
        ``entity_context`` returned ``relationships: []`` because the
        original SQL only queried ``WHERE subject_entity_id = %s``.
        """
        matview_row = {
            "entity_id": 1588866,
            "canonical_name": "James Webb Space Telescope",
            "entity_type": "mission",
            "discipline": "astrophysics",
            "source": "curated_flagship_v1",
            "identifiers": [],
            "aliases": [],
            "citing_paper_count": 5159,
        }
        # Two in-edges (NIRSpec, NIRCam → part_of → JWST). The other end
        # (the SUBJECT in the DB row) is exposed under object_* fields
        # so consumers always read "the OTHER entity" from object_*.
        relationship_rows = [
            {
                "direction": "in",
                "predicate": "part_of",
                "object_id": 1679247,
                "confidence": 1.0,
                "relationship_source": "flagship_seed",
                "object_name": "NIRSpec",
                "object_entity_type": "instrument",
                "object_properties": {},
            },
            {
                "direction": "in",
                "predicate": "part_of",
                "object_id": 1679248,
                "confidence": 1.0,
                "relationship_source": "flagship_seed",
                "object_name": "NIRCam",
                "object_entity_type": "instrument",
                "object_properties": {},
            },
        ]
        conn = self._make_conn(
            matview_row=matview_row,
            properties_row={"properties": {}},
            relationship_rows=relationship_rows,
        )

        result = get_entity_context(conn, 1588866)

        assert result.total == 1
        entity = result.papers[0]
        assert len(entity["relationships"]) == 2
        names = sorted(r["object_name"] for r in entity["relationships"])
        assert names == ["NIRCam", "NIRSpec"]
        assert all(r["direction"] == "in" for r in entity["relationships"])
        assert all(r["predicate"] == "part_of" for r in entity["relationships"])

    def test_relationships_query_unions_in_and_out_edges(self) -> None:
        """The relationships SQL must search both subject_entity_id and object_entity_id."""
        conn = self._make_conn(matview_row=None, fallback_row=None)

        get_entity_context(conn, 42)

        # The third execute call is the relationships query (after matview-miss + fallback miss
        # exit early, only the two early calls fire — so verify on a hit path).
        # Use an entity that exists in the matview to reach the relationships query.
        matview_row = {
            "entity_id": 42,
            "canonical_name": "X",
            "entity_type": "mission",
            "discipline": "x",
            "source": "x",
            "identifiers": [],
            "aliases": [],
            "citing_paper_count": 0,
        }
        conn2 = self._make_conn(
            matview_row=matview_row,
            properties_row={"properties": {}},
            relationship_rows=[],
        )
        get_entity_context(conn2, 42)

        executed_sqls = [c.args[0] for c in conn2.cursor.return_value.execute.call_args_list]
        rel_sql = next(s for s in executed_sqls if "entity_relationships" in s)
        assert "subject_entity_id = %s" in rel_sql
        assert "object_entity_id = %s" in rel_sql
        assert "UNION ALL" in rel_sql
        # Direction column distinguishes which side the queried entity sits on.
        assert "'out'" in rel_sql
        assert "'in'" in rel_sql

    def test_query_uses_matview(self) -> None:
        """First query targets agent_entity_context matview."""
        conn = self._make_conn(matview_row=None, fallback_row=None)

        get_entity_context(conn, 42)

        cur = conn.cursor.return_value
        # First execute is the matview lookup
        first_sql = cur.execute.call_args_list[0][0][0]
        assert "agent_entity_context" in first_sql
        first_params = cur.execute.call_args_list[0][0][1]
        assert first_params == (42,)


class TestDispatchEntityContext:
    """Tests for MCP _dispatch_tool routing to entity_context."""

    @patch("scix.search.get_entity_context")
    def test_dispatch_routes_to_search(self, mock_fn: MagicMock) -> None:
        mock_fn.return_value = SearchResult(
            papers=[
                {
                    "entity_id": 42,
                    "canonical_name": "OSIRIS-REx",
                    "entity_type": "mission",
                    "discipline": "astronomy",
                    "source": "metadata",
                    "identifiers": [],
                    "aliases": [],
                    "relationships": [],
                    "citing_paper_count": 1523,
                }
            ],
            total=1,
            timing_ms={"query_ms": 0.8},
        )
        mock_conn = MagicMock()
        result = json.loads(_dispatch_tool(mock_conn, "entity_context", {"entity_id": 42}))
        assert result["total"] == 1
        assert result["papers"][0]["canonical_name"] == "OSIRIS-REx"
        mock_fn.assert_called_once_with(mock_conn, 42)

    def test_missing_entity_id_returns_error(self) -> None:
        """Missing entity_id returns clear error."""
        mock_conn = MagicMock()
        result = json.loads(_dispatch_tool(mock_conn, "entity_context", {}))
        assert "error" in result

    def test_invalid_entity_id_returns_error(self) -> None:
        """Non-integer entity_id returns clear error."""
        mock_conn = MagicMock()
        result = json.loads(_dispatch_tool(mock_conn, "entity_context", {"entity_id": "abc"}))
        assert "error" in result


# ---------------------------------------------------------------------------
# resolve_entity — MCP dispatch tests
# ---------------------------------------------------------------------------


class TestDispatchResolveEntity:
    """Tests for MCP _dispatch_tool routing to resolve_entity."""

    @patch("scix.mcp_server.EntityResolver")
    def test_dispatch_exact_match(self, MockResolver: MagicMock) -> None:
        from scix.entity_resolver import EntityCandidate

        mock_instance = MockResolver.return_value
        mock_instance.resolve.return_value = [
            EntityCandidate(
                entity_id=42,
                canonical_name="OSIRIS-REx",
                entity_type="mission",
                source="metadata",
                discipline="astronomy",
                confidence=1.0,
                match_method="exact_canonical",
            ),
        ]
        mock_conn = MagicMock()
        result = json.loads(_dispatch_tool(mock_conn, "resolve_entity", {"query": "OSIRIS-REx"}))
        assert result["total"] == 1
        assert result["candidates"][0]["canonical_name"] == "OSIRIS-REx"
        assert result["candidates"][0]["confidence"] == 1.0
        assert result["candidates"][0]["match_method"] == "exact_canonical"
        mock_instance.resolve.assert_called_once_with("OSIRIS-REx", discipline=None, fuzzy=False)

    @patch("scix.mcp_server.EntityResolver")
    def test_dispatch_with_discipline(self, MockResolver: MagicMock) -> None:
        mock_instance = MockResolver.return_value
        mock_instance.resolve.return_value = []
        mock_conn = MagicMock()
        _dispatch_tool(
            mock_conn,
            "resolve_entity",
            {"query": "Mars", "discipline": "astronomy", "fuzzy": True},
        )
        mock_instance.resolve.assert_called_once_with("Mars", discipline="astronomy", fuzzy=True)

    @patch("scix.mcp_server.EntityResolver")
    def test_dispatch_no_match(self, MockResolver: MagicMock) -> None:
        mock_instance = MockResolver.return_value
        mock_instance.resolve.return_value = []
        mock_conn = MagicMock()
        result = json.loads(
            _dispatch_tool(mock_conn, "resolve_entity", {"query": "xyznonexistent"})
        )
        assert result["total"] == 0
        assert result["candidates"] == []

    def test_empty_query_returns_error(self) -> None:
        """Empty query is rejected before DB call."""
        mock_conn = MagicMock()
        result = json.loads(_dispatch_tool(mock_conn, "resolve_entity", {"query": ""}))
        assert "error" in result

    def test_whitespace_query_returns_error(self) -> None:
        """Whitespace-only query is rejected."""
        mock_conn = MagicMock()
        result = json.loads(_dispatch_tool(mock_conn, "resolve_entity", {"query": "   "}))
        assert "error" in result
