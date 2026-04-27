"""Tests for the MCP extraction wiring (PRD prd_full_text_applications_v2 — mcp-extraction-wiring).

Covers:

1. The MCP ``entity`` tool no longer advertises ``negative_result`` /
   ``quant_claim`` as ``entity_type`` values — they were removed under
   bead ``scix_experiments-mh14`` because they are claim/finding
   extractions, not entities. The schema must keep the four real entity
   types (methods/datasets/instruments/materials).
2. ``entity({entity_type='negative_result'})`` and
   ``entity({entity_type='quant_claim'})`` now return a structured
   error pointing at the follow-up bead (``scix_experiments-c996``).
3. ``read_paper`` / ``entity`` / ``search_within_paper`` (the latter via
   the deprecated alias path that resolves to ``read_paper`` with
   ``search_query``) all carry a top-level ``coverage_note`` field on
   the still-supported call shapes.
4. The ``coverage_note`` string mentions ``full-text coverage``, a
   percentage, and cites ``docs/full_text_coverage_analysis.md``.
5. When the coverage-bias JSON is missing the fallback note still
   cites the docs path so the link is always present.

DB and the JSON file read are mocked — no live DB and no real disk read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix import mcp_server
from scix.mcp_server import _dispatch_tool, _reset_coverage_note_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cov_cache() -> None:
    """Drop the cached coverage note between tests so monkeypatched paths apply."""
    _reset_coverage_note_cache()
    yield
    _reset_coverage_note_cache()


def _make_cursor_with_rows(rows: list[tuple[Any, ...]]) -> MagicMock:
    """Build a MagicMock that mimics a psycopg cursor returning ``rows``."""
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    return cursor


def _make_conn(rows: list[tuple[Any, ...]]) -> MagicMock:
    """Build a MagicMock psycopg connection whose .cursor() returns ``rows``."""
    cursor = _make_cursor_with_rows(rows)
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


# Sample M3 negative-result extraction row (matches scix.negative_results
# payload shape: {spans, n_spans, tier_counts, extractor}).
_NEG_RESULT_ROW: tuple[Any, ...] = (
    "2024ApJ...001A",
    "negative_result",
    "neg_results_v1",
    {
        "spans": [
            {
                "section": "results",
                "pattern_id": "no_significant",
                "confidence_tier": 3,
                "confidence_label": "high",
                "match_text": "no significant",
                "start_char": 100,
                "end_char": 114,
                "evidence_span": "We found no significant excess at 5 sigma.",
            }
        ],
        "n_spans": 1,
        "tier_counts": {"high": 1, "medium": 0, "low": 0},
        "extractor": "neg_results_v1",
    },
    "Search for dark matter in the Galactic halo",
)

# ---------------------------------------------------------------------------
# AC1 (mh14): the entity tool's entity_type enum no longer advertises the
# claim/finding extraction kinds — those are not entities and have been
# removed under bead scix_experiments-mh14.
# ---------------------------------------------------------------------------


def _get_entity_tool_schema() -> dict[str, Any]:
    """Pull the live ``entity`` tool inputSchema via the MCP tools/list handler."""
    import asyncio

    from mcp.types import ListToolsRequest

    from scix.mcp_server import create_server

    server = create_server(_run_self_test=False)
    handler = server.request_handlers[ListToolsRequest]
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(handler(ListToolsRequest(method="tools/list")))
    finally:
        loop.close()
    tools = result.root.tools if hasattr(result, "root") else result.tools
    entity_tool = next(t for t in tools if t.name == "entity")
    return entity_tool.inputSchema  # type: ignore[no-any-return]


def test_schema_drops_negative_result_and_quant_claim() -> None:
    """AC1 (mh14): tools/list response no longer advertises the legacy
    extraction-row kinds under entity_type."""
    try:
        schema = _get_entity_tool_schema()
    except (ImportError, AttributeError):
        pytest.skip("mcp SDK not installed or server API changed")

    enum_values = schema["properties"]["entity_type"]["enum"]
    assert "negative_result" not in enum_values
    assert "quant_claim" not in enum_values
    # Backwards compatibility — the four real entity-containment types
    # must still be advertised.
    for legacy in ("methods", "datasets", "instruments", "materials"):
        assert legacy in enum_values, f"{legacy} dropped from entity_type enum"


# ---------------------------------------------------------------------------
# AC2 (mh14): the runtime handler returns a structured error for the
# legacy extraction-row kinds instead of dispatching them through the
# entity-search code path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("legacy_type", ["negative_result", "quant_claim"])
def test_entity_legacy_extraction_type_returns_error(legacy_type: str) -> None:
    """AC2 (mh14): the entity tool refuses claim/finding extraction kinds
    with a structured error that points at the follow-up bead so an agent
    can recover in one turn."""
    conn = _make_conn([_NEG_RESULT_ROW])  # row contents are irrelevant — DB never queried

    out = _dispatch_tool(conn, "entity", {"entity_type": legacy_type})
    data = json.loads(out)

    assert "error" in data
    assert legacy_type in data["error"]
    # Surface the follow-up bead reference so an agent / human reading
    # the error can find the long-term home for this functionality.
    assert "scix_experiments-c996" in data["error"]


@pytest.mark.parametrize("legacy_type", ["negative_result", "quant_claim"])
def test_entity_legacy_extraction_type_does_not_query_db(legacy_type: str) -> None:
    """The rejection happens at the front door — the handler must not
    open a cursor for the legacy types."""
    conn = _make_conn([_NEG_RESULT_ROW])

    _dispatch_tool(
        conn,
        "entity",
        {"entity_type": legacy_type, "entity_name": "anything"},
    )

    # No SQL execution attempted.
    assert not conn.cursor.called


# ---------------------------------------------------------------------------
# AC4 + AC5: coverage_note appears on extraction-surface tools
# ---------------------------------------------------------------------------


def _assert_coverage_note(note: str) -> None:
    """AC5 helper: enforce the coverage_note string contract."""
    assert "full-text coverage" in note, note
    assert "docs/full_text_coverage_analysis.md" in note, note
    # When the JSON file is present (test default) we expect a percentage.
    assert "%" in note, note


def test_entity_response_includes_coverage_note() -> None:
    """AC4: entity responses on the still-supported call shape carry
    top-level coverage_note. Uses ``methods`` (a real entity-containment
    type) since the legacy ``negative_result`` / ``quant_claim`` kinds
    were removed under bead scix_experiments-mh14 and now early-error
    without dispatching through _inject_coverage_note."""
    methods_row: tuple[Any, ...] = (
        "2024ApJ...003C",
        "methods",
        "v1",
        {"methods": ["JWST"]},
        "Some methods paper",
    )
    conn = _make_conn([methods_row])
    out = _dispatch_tool(
        conn,
        "entity",
        {"action": "search", "entity_type": "methods", "query": "JWST"},
    )
    data = json.loads(out)
    assert "coverage_note" in data
    _assert_coverage_note(data["coverage_note"])


def test_entity_resolve_response_includes_coverage_note() -> None:
    """The resolve action also surfaces full-text-derived signals → coverage_note required."""
    conn = MagicMock()
    # Patch the resolver so we can run without DB-backed entity tables.
    with patch("scix.mcp_server.EntityResolver") as resolver_cls:
        resolver_cls.return_value.resolve.return_value = []
        out = _dispatch_tool(conn, "entity", {"action": "resolve", "query": "H0"})
    data = json.loads(out)
    assert "coverage_note" in data
    _assert_coverage_note(data["coverage_note"])


def test_read_paper_response_includes_coverage_note() -> None:
    """AC4: read_paper carries top-level coverage_note (read mode)."""
    conn = MagicMock()
    fake_result = MagicMock()
    # _result_to_json passes through non-SearchResult via json.dumps(default=str).
    # Use a plain dict to keep the test robust to the underlying implementation.
    with patch(
        "scix.mcp_server.search.read_paper_section",
        return_value={"bibcode": "2024ApJ...001A", "section": "full", "text": "abstract..."},
    ):
        out = _dispatch_tool(conn, "read_paper", {"bibcode": "2024ApJ...001A"})
    data = json.loads(out)
    assert "coverage_note" in data
    _assert_coverage_note(data["coverage_note"])


def test_search_within_paper_response_includes_coverage_note() -> None:
    """AC4: search_within_paper (via read_paper search mode + deprecated alias) carries coverage_note."""
    conn = MagicMock()
    with patch(
        "scix.mcp_server.search.search_within_paper",
        return_value={"bibcode": "2024ApJ...001A", "matches": []},
    ):
        out = _dispatch_tool(
            conn,
            "read_paper",
            {"bibcode": "2024ApJ...001A", "search_query": "dark matter"},
        )
    data = json.loads(out)
    assert "coverage_note" in data
    _assert_coverage_note(data["coverage_note"])


def test_search_within_paper_alias_carries_coverage_note() -> None:
    """The deprecated ``search_within_paper`` alias still carries coverage_note."""
    conn = MagicMock()
    with patch(
        "scix.mcp_server.search.search_within_paper",
        return_value={"bibcode": "2024ApJ...001A", "matches": []},
    ):
        out = _dispatch_tool(
            conn,
            "search_within_paper",
            {"bibcode": "2024ApJ...001A", "search_query": "dark matter"},
        )
    data = json.loads(out)
    # The deprecated-alias wrapper nests the original payload under ``data``;
    # accept either shape so we don't couple the test to that envelope.
    if "coverage_note" in data:
        note = data["coverage_note"]
    else:
        inner = data.get("data") or data.get("result") or {}
        assert isinstance(inner, dict), data
        assert "coverage_note" in inner, data
        note = inner["coverage_note"]
    _assert_coverage_note(note)


# ---------------------------------------------------------------------------
# AC6: fallback note when coverage JSON is missing
# ---------------------------------------------------------------------------


def test_coverage_note_fallback_when_json_missing(tmp_path: Path) -> None:
    """If results/full_text_coverage_bias.json is missing, fallback note still cites the docs."""
    missing = tmp_path / "no_such_file.json"
    assert not missing.exists()

    with patch(
        "scix.mcp_server._coverage_note_path",
        return_value=missing,
    ):
        _reset_coverage_note_cache()
        note = mcp_server._coverage_note()

    # Always cites the analysis doc, even on failure.
    assert "docs/full_text_coverage_analysis.md" in note
    # Fallback wording (no percentage available).
    assert "stats unavailable" in note or "coverage" in note.lower()


def test_coverage_note_uses_fulltext_pct_field(tmp_path: Path) -> None:
    """When fulltext_pct is present, the percentage is taken verbatim."""
    sample = tmp_path / "cov.json"
    sample.write_text(
        json.dumps({"fulltext_pct": 47.0, "fulltext_total": 1, "corpus_total": 1}),
        encoding="utf-8",
    )
    with patch("scix.mcp_server._coverage_note_path", return_value=sample):
        _reset_coverage_note_cache()
        note = mcp_server._coverage_note()
    assert "47.0%" in note
    assert "full-text coverage" in note
    assert "docs/full_text_coverage_analysis.md" in note


def test_coverage_note_falls_back_to_ratio_when_pct_missing(tmp_path: Path) -> None:
    """If fulltext_pct is absent, ratio of fulltext_total / corpus_total is used."""
    sample = tmp_path / "cov.json"
    sample.write_text(
        json.dumps({"fulltext_total": 50, "corpus_total": 200}),
        encoding="utf-8",
    )
    with patch("scix.mcp_server._coverage_note_path", return_value=sample):
        _reset_coverage_note_cache()
        note = mcp_server._coverage_note()
    assert "25.0%" in note
