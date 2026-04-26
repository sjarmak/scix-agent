"""Unit tests for scix.qdrant_tools feature-flagging and MCP integration.

The find_similar_by_examples MCP tool was retired on 2026-04-25 (Qdrant
backend not in active use). This test suite was rewritten on the same
date to validate the retirement contract:

  1. is_enabled() correctly reads the QDRANT_URL env var (the helper is
     still used by other code paths).
  2. bibcode_to_point_id is deterministic and within the int64 range Qdrant
     accepts (kept in case the tool returns; fits the eventual NAS-Qdrant
     migration).
  3. The MCP server does NOT register find_similar_by_examples regardless
     of QDRANT_URL, and the self-test reports exactly 15 tools either way.
  4. The dispatch path for the retired name returns a clear
     "tool_removed" error.
"""
from __future__ import annotations

import dataclasses
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scix import qdrant_tools
from scix import mcp_server


@pytest.fixture
def no_qdrant(monkeypatch: pytest.MonkeyPatch):
    """Ensure QDRANT_URL is unset for disabled-path tests."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    yield


@pytest.fixture
def fake_qdrant(monkeypatch: pytest.MonkeyPatch):
    """Set QDRANT_URL to a non-routable value — enabled flag on, but no real server."""
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:59999")
    yield


class TestIsEnabled:
    def test_disabled_when_env_unset(self, no_qdrant):
        assert qdrant_tools.is_enabled() is False
        assert mcp_server._qdrant_enabled() is False

    def test_enabled_when_env_set(self, fake_qdrant):
        assert qdrant_tools.is_enabled() is True
        assert mcp_server._qdrant_enabled() is True


class TestBibcodeToPointId:
    def test_deterministic(self):
        a = qdrant_tools.bibcode_to_point_id("2019A&A...622A...2P")
        b = qdrant_tools.bibcode_to_point_id("2019A&A...622A...2P")
        assert a == b

    def test_positive_int64(self):
        # Qdrant numeric point IDs must fit in unsigned int64.
        pid = qdrant_tools.bibcode_to_point_id("2019A&A...622A...2P")
        assert 0 <= pid < 2**63

    def test_distinct_bibcodes_distinct_ids(self):
        ids = {qdrant_tools.bibcode_to_point_id(f"fake{i}") for i in range(100)}
        assert len(ids) == 100  # no collisions in small sample


class TestExpectedToolSet:
    """Post-2026-04-25: the active tool set excludes find_similar_by_examples
    regardless of Qdrant state. We assert the set matches EXPECTED_TOOLS (the
    canonical source) rather than a hardcoded count, so this stays correct as
    new tools are added to EXPECTED_TOOLS."""

    def test_active_set_when_qdrant_disabled(self, no_qdrant):
        tools = mcp_server._expected_tool_set()
        assert tools == set(mcp_server.EXPECTED_TOOLS)
        assert "find_similar_by_examples" not in tools

    def test_active_set_when_qdrant_enabled(self, fake_qdrant):
        # The retired tool is gone from the active surface even when the
        # Qdrant feature flag is on; the gating is now hardcoded to no-op.
        tools = mcp_server._expected_tool_set()
        assert tools == set(mcp_server.EXPECTED_TOOLS)
        assert "find_similar_by_examples" not in tools


class TestRetiredToolDispatch:
    """The retired tool name must return a structured tool_removed error."""

    def test_dispatch_returns_tool_removed(self, no_qdrant):
        from scix.mcp_server import _dispatch_consolidated

        # Use _dispatch_consolidated directly — _dispatch_tool runs the
        # alias layer, but find_similar_by_examples is NOT in
        # _DEPRECATED_ALIASES (it was hard-removed, not renamed).
        out = _dispatch_consolidated(
            None,  # conn unused for the retired-tool branch
            "find_similar_by_examples",
            {"positive_bibcodes": ["2019A&A...622A...2P"]},
        )
        payload = json.loads(out)
        assert payload["error"] == "tool_removed"
        assert payload["removed_in"] == "2026-04-25"
        assert "find_similar_by_examples" in payload["message"]


class TestMCPSelfTest:
    """The server self-test must pass at exactly EXPECTED_TOOLS-many tools
    regardless of QDRANT_URL.  Count is read from EXPECTED_TOOLS rather than
    hardcoded so adding/removing tools updates only one place."""

    def test_self_test_passes_without_qdrant(self, no_qdrant):
        status = mcp_server.startup_self_test()
        assert status["ok"] is True
        assert status["tool_count"] == len(mcp_server.EXPECTED_TOOLS)
        assert "find_similar_by_examples" not in status["tool_names"]

    def test_self_test_passes_with_qdrant(self, fake_qdrant):
        status = mcp_server.startup_self_test()
        assert status["ok"] is True
        assert status["tool_count"] == len(mcp_server.EXPECTED_TOOLS)
        assert "find_similar_by_examples" not in status["tool_names"]


# ---------------------------------------------------------------------------
# Chunk-level helpers (chunk_search_by_text + fetch_chunk_snippets)
# ---------------------------------------------------------------------------


def _scored_point(payload: dict, score: float, point_id: int = 0):
    """Build a qdrant ScoredPoint-shaped object for the mocked client."""
    return SimpleNamespace(payload=payload, score=score, id=point_id)


def _stub_payload(
    *,
    bibcode: str,
    chunk_id: int,
    section_idx: int,
    section_heading_norm: str | None = "introduction",
    section_heading: str | None = None,
    char_offset: int | None = 0,
    n_tokens: int | None = 200,
    year: int = 2023,
    arxiv_class: list[str] | None = None,
    community_id_med: int = 7,
    doctype: str = "article",
) -> dict:
    return {
        "bibcode": bibcode,
        "chunk_id": chunk_id,
        "section_idx": section_idx,
        "section_heading_norm": section_heading_norm,
        "section_heading": section_heading,
        "char_offset": char_offset,
        "n_tokens": n_tokens,
        "year": year,
        "arxiv_class": arxiv_class or ["astro-ph.EP"],
        "community_id_med": community_id_med,
        "doctype": doctype,
    }


class TestChunkHitDataclass:
    """Acceptance criterion 1: ChunkHit is frozen with the correct fields."""

    def test_is_frozen_dataclass(self):
        hit = qdrant_tools.ChunkHit(
            bibcode="2023ApJ...999..123A",
            chunk_id=1,
            section_idx=0,
            section_heading_norm="introduction",
            section_heading=None,
            score=0.42,
            snippet=None,
            char_offset=0,
            n_tokens=200,
        )
        assert dataclasses.is_dataclass(hit)
        # frozen → attribute assignment raises FrozenInstanceError
        with pytest.raises(dataclasses.FrozenInstanceError):
            hit.snippet = "oops"  # type: ignore[misc]

    def test_field_set_matches_spec(self):
        names = {f.name for f in dataclasses.fields(qdrant_tools.ChunkHit)}
        assert names == {
            "bibcode",
            "chunk_id",
            "section_idx",
            "section_heading_norm",
            "section_heading",
            "score",
            "snippet",
            "char_offset",
            "n_tokens",
        }


class TestChunksCollectionConstant:
    """Acceptance criterion 2: CHUNKS_COLLECTION constant."""

    def test_constant_value(self):
        assert qdrant_tools.CHUNKS_COLLECTION == "scix_chunks_v1"


class TestChunksFilterFromKwargs:
    """Acceptance criterion 4: filter composition rules."""

    def test_no_filters_returns_none(self):
        assert qdrant_tools._chunks_filter_from_kwargs() is None

    def test_year_range(self):
        flt = qdrant_tools._chunks_filter_from_kwargs(year_min=2020, year_max=2024)
        assert flt is not None
        assert len(flt.must) == 1
        cond = flt.must[0]
        assert cond.key == "year"
        assert cond.range.gte == 2020
        assert cond.range.lte == 2024

    def test_year_min_only(self):
        flt = qdrant_tools._chunks_filter_from_kwargs(year_min=2015)
        assert flt is not None
        cond = flt.must[0]
        assert cond.range.gte == 2015
        assert cond.range.lte is None

    def test_list_filters_use_match_any(self):
        from qdrant_client.http import models as qm

        flt = qdrant_tools._chunks_filter_from_kwargs(
            arxiv_class=["astro-ph.EP", "astro-ph.SR"],
            community_id_med=[3, 7],
            section_heading_norm=["introduction", "methods"],
            bibcode=["2023ApJ...1A", "2023ApJ...2A"],
        )
        assert flt is not None
        # 4 list filters → 4 must conditions
        assert len(flt.must) == 4
        keys = [c.key for c in flt.must]
        assert keys == [
            "arxiv_class",
            "community_id_med",
            "section_heading_norm",
            "bibcode",
        ]
        for cond in flt.must:
            assert isinstance(cond.match, qm.MatchAny)

    def test_all_filters_and_combined(self):
        flt = qdrant_tools._chunks_filter_from_kwargs(
            year_min=2020,
            year_max=2025,
            arxiv_class=["astro-ph.EP"],
            community_id_med=[1],
            section_heading_norm=["abstract"],
            bibcode=["2023ApJ...1A"],
        )
        assert flt is not None
        assert flt.must_not is None or flt.must_not == []
        assert flt.should is None or flt.should == []
        # year + 4 list filters
        assert len(flt.must) == 5


class TestChunkSearchByText:
    """Acceptance criteria 3, 5, 6, 10."""

    def _patch_client(self, monkeypatch, response_points):
        client = MagicMock()
        client.query_points.return_value = SimpleNamespace(points=response_points)
        monkeypatch.setattr(qdrant_tools, "_client", lambda timeout=10.0: client)
        return client

    def test_returns_chunk_hits_with_snippet_none(self, monkeypatch):
        points = [
            _scored_point(
                _stub_payload(
                    bibcode="2023ApJ...111A", chunk_id=10, section_idx=2
                ),
                score=0.91,
            ),
            _scored_point(
                _stub_payload(
                    bibcode="2023ApJ...222B", chunk_id=20, section_idx=0
                ),
                score=0.83,
            ),
        ]
        self._patch_client(monkeypatch, points)
        hits = qdrant_tools.chunk_search_by_text([0.1] * 4)
        assert len(hits) == 2
        assert all(isinstance(h, qdrant_tools.ChunkHit) for h in hits)
        assert all(h.snippet is None for h in hits)
        assert hits[0].bibcode == "2023ApJ...111A"
        assert hits[0].chunk_id == 10
        assert hits[0].section_idx == 2
        assert hits[0].score == pytest.approx(0.91)
        assert hits[0].section_heading_norm == "introduction"
        assert hits[0].section_heading is None

    def test_default_limit_is_20(self, monkeypatch):
        client = self._patch_client(monkeypatch, [])
        qdrant_tools.chunk_search_by_text([0.1] * 4)
        kwargs = client.query_points.call_args.kwargs
        assert kwargs["limit"] == 20

    def test_explicit_limit_passed_through(self, monkeypatch):
        client = self._patch_client(monkeypatch, [])
        qdrant_tools.chunk_search_by_text([0.1] * 4, limit=5)
        kwargs = client.query_points.call_args.kwargs
        assert kwargs["limit"] == 5

    def test_uses_chunks_collection_and_no_named_vector(self, monkeypatch):
        client = self._patch_client(monkeypatch, [])
        qdrant_tools.chunk_search_by_text([0.1] * 4)
        kwargs = client.query_points.call_args.kwargs
        assert kwargs["collection_name"] == "scix_chunks_v1"
        # chunks collection is single-vector → no `using=`
        assert "using" not in kwargs
        assert kwargs["with_payload"] is True

    def test_filter_passed_to_query_points(self, monkeypatch):
        client = self._patch_client(monkeypatch, [])
        qdrant_tools.chunk_search_by_text(
            [0.1] * 4,
            year_min=2020,
            year_max=2024,
            community_id_med=[3, 7],
        )
        kwargs = client.query_points.call_args.kwargs
        flt = kwargs["query_filter"]
        assert flt is not None
        keys = [c.key for c in flt.must]
        assert "year" in keys
        assert "community_id_med" in keys

    def test_no_filters_passes_none(self, monkeypatch):
        client = self._patch_client(monkeypatch, [])
        qdrant_tools.chunk_search_by_text([0.1] * 4)
        kwargs = client.query_points.call_args.kwargs
        assert kwargs["query_filter"] is None


class TestFetchChunkSnippets:
    """Acceptance criteria 7, 8, 9."""

    def _stub_conn(self, rows):
        cur = MagicMock()
        cur.fetchall.return_value = rows
        cur.__enter__ = lambda self_: self_
        cur.__exit__ = lambda *a: None

        conn = MagicMock()
        conn.cursor.return_value = cur
        return conn, cur

    def _hit(self, bibcode: str, section_idx: int, chunk_id: int = 1) -> qdrant_tools.ChunkHit:
        return qdrant_tools.ChunkHit(
            bibcode=bibcode,
            chunk_id=chunk_id,
            section_idx=section_idx,
            section_heading_norm="introduction",
            section_heading=None,
            score=0.5,
            snippet=None,
            char_offset=0,
            n_tokens=100,
        )

    def test_empty_hits_returns_empty(self):
        conn = MagicMock()
        out = qdrant_tools.fetch_chunk_snippets(conn, [])
        assert out == []
        conn.cursor.assert_not_called()

    def test_populates_snippet_from_sections(self):
        sections = [
            {"heading": "Abstract", "text": "abstract text"},
            {"heading": "Introduction", "text": "intro text body"},
        ]
        conn, _cur = self._stub_conn([("2023ApJ...111A", sections)])
        hits = [self._hit("2023ApJ...111A", section_idx=1)]
        out = qdrant_tools.fetch_chunk_snippets(conn, hits)
        assert len(out) == 1
        assert out[0].snippet == "intro text body"
        # original is not mutated (frozen dataclass)
        assert hits[0].snippet is None
        # new instance, not the same object
        assert out[0] is not hits[0]

    def test_truncation_appends_ellipsis(self):
        big_text = "x" * 5000
        sections = [{"heading": "Abstract", "text": big_text}]
        conn, _cur = self._stub_conn([("2023ApJ...111A", sections)])
        hits = [self._hit("2023ApJ...111A", section_idx=0)]
        out = qdrant_tools.fetch_chunk_snippets(conn, hits, max_snippet_chars=100)
        assert out[0].snippet is not None
        assert out[0].snippet.endswith("...")
        # 100 chars + "..." suffix
        assert len(out[0].snippet) == 103
        assert out[0].snippet[:100] == "x" * 100

    def test_no_truncation_when_under_limit(self):
        sections = [{"heading": "Abstract", "text": "short"}]
        conn, _cur = self._stub_conn([("2023ApJ...111A", sections)])
        hits = [self._hit("2023ApJ...111A", section_idx=0)]
        out = qdrant_tools.fetch_chunk_snippets(conn, hits, max_snippet_chars=1500)
        assert out[0].snippet == "short"
        assert not out[0].snippet.endswith("...")

    def test_missing_bibcode_preserves_hit_with_none_snippet(self):
        # Postgres returns no row for the bibcode → snippet stays None,
        # but the hit must still appear in the output.
        conn, _cur = self._stub_conn([])
        hits = [self._hit("2023ApJ...999Z", section_idx=0)]
        out = qdrant_tools.fetch_chunk_snippets(conn, hits)
        assert len(out) == 1
        assert out[0].snippet is None
        assert out[0].bibcode == "2023ApJ...999Z"

    def test_out_of_range_section_idx_keeps_snippet_none(self):
        sections = [{"heading": "Abstract", "text": "only one section"}]
        conn, _cur = self._stub_conn([("2023ApJ...111A", sections)])
        hits = [self._hit("2023ApJ...111A", section_idx=5)]
        out = qdrant_tools.fetch_chunk_snippets(conn, hits)
        assert len(out) == 1
        assert out[0].snippet is None

    def test_handles_jsonb_decoded_as_string(self):
        # psycopg sometimes hands back raw JSON strings if the column type is
        # text or json (rather than jsonb) — fetch_chunk_snippets should cope.
        sections_json = json.dumps([{"heading": "Intro", "text": "decoded text"}])
        conn, _cur = self._stub_conn([("2023ApJ...111A", sections_json)])
        hits = [self._hit("2023ApJ...111A", section_idx=0)]
        out = qdrant_tools.fetch_chunk_snippets(conn, hits)
        assert out[0].snippet == "decoded text"

    def test_single_batch_query_no_n_plus_one(self):
        sections_a = [{"heading": "Intro", "text": "AAA"}]
        sections_b = [{"heading": "Intro", "text": "BBB"}]
        conn, cur = self._stub_conn(
            [("2023ApJ...111A", sections_a), ("2023ApJ...222B", sections_b)]
        )
        hits = [
            self._hit("2023ApJ...111A", section_idx=0, chunk_id=1),
            self._hit("2023ApJ...222B", section_idx=0, chunk_id=2),
            self._hit("2023ApJ...111A", section_idx=0, chunk_id=3),  # repeat bibcode
        ]
        out = qdrant_tools.fetch_chunk_snippets(conn, hits)
        # exactly one execute call for any number of hits
        assert cur.execute.call_count == 1
        sql, params = cur.execute.call_args.args
        assert "WHERE bibcode = ANY(%s)" in sql
        # deduped & sorted bibcode list
        assert params[0] == ["2023ApJ...111A", "2023ApJ...222B"]
        # all three hits returned, in input order, with snippets resolved
        assert [h.snippet for h in out] == ["AAA", "BBB", "AAA"]

    def test_preserves_other_fields_via_dataclasses_replace(self):
        sections = [{"heading": "Abstract", "text": "text"}]
        conn, _cur = self._stub_conn([("2023ApJ...111A", sections)])
        hit = qdrant_tools.ChunkHit(
            bibcode="2023ApJ...111A",
            chunk_id=42,
            section_idx=0,
            section_heading_norm="abstract",
            section_heading="Abstract",
            score=0.77,
            snippet=None,
            char_offset=128,
            n_tokens=300,
        )
        out = qdrant_tools.fetch_chunk_snippets(conn, [hit])
        new = out[0]
        assert new.bibcode == hit.bibcode
        assert new.chunk_id == hit.chunk_id
        assert new.section_idx == hit.section_idx
        assert new.section_heading_norm == hit.section_heading_norm
        assert new.section_heading == hit.section_heading
        assert new.score == hit.score
        assert new.char_offset == hit.char_offset
        assert new.n_tokens == hit.n_tokens
        assert new.snippet == "text"
