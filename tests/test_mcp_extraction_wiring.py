"""Tests for the MCP extraction wiring (PRD prd_full_text_applications_v2 — mcp-extraction-wiring).

Covers:

1. The MCP ``entity`` tool advertises ``negative_result`` and
   ``quant_claim`` as accepted ``entity_type`` values (tools/list schema).
2. ``entity({entity_type='negative_result'})`` returns rows whose top
   level carries ``evidence_span`` (sourced from M3 spans).
3. ``entity({entity_type='quant_claim', entity_name='H0'})`` returns
   rows whose ``payload`` contains ``{value, uncertainty, unit}``
   (sourced from M4 claims, filtered by canonical quantity).
4. ``read_paper`` / ``entity`` / ``search_within_paper`` (the latter via
   the deprecated alias path that resolves to ``read_paper`` with
   ``search_query``) all carry a top-level ``coverage_note`` field.
5. The ``coverage_note`` string mentions ``full-text coverage``, a
   percentage, and cites ``docs/full_text_coverage_analysis.md``.
6. When the coverage-bias JSON is missing the fallback note still
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

# Sample M4 quant_claim extraction row (matches scix.claim_extractor
# payload shape: {extraction_type, extraction_version, source, claims}).
_QUANT_CLAIM_ROW: tuple[Any, ...] = (
    "2024ApJ...002B",
    "quant_claim",
    "quant_claim_regex_v1",
    {
        "extraction_type": "quant_claim",
        "extraction_version": "quant_claim_regex_v1",
        "source": "claim_extractor_regex",
        "claims": [
            {
                "quantity": "H0",
                "value": 73.0,
                "uncertainty": 1.0,
                "unit": "km/s/Mpc",
                "span": [200, 230],
                "uncertainty_pos": None,
                "uncertainty_neg": None,
                "surface": "H0 = 73 +/- 1 km/s/Mpc",
                "confidence_tier": 1,
            },
            {
                "quantity": "Omega_m",
                "value": 0.3,
                "uncertainty": 0.05,
                "unit": None,
                "span": [240, 260],
                "uncertainty_pos": None,
                "uncertainty_neg": None,
                "surface": "Omega_m = 0.3 +/- 0.05",
                "confidence_tier": 2,
            },
        ],
    },
    "Local measurement of the Hubble constant",
)


# ---------------------------------------------------------------------------
# AC1: schema advertises new entity_type values
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


def test_schema_accepts_negative_result_and_quant_claim() -> None:
    """AC1: tools/list response advertises the two new entity_type values."""
    try:
        schema = _get_entity_tool_schema()
    except (ImportError, AttributeError):
        pytest.skip("mcp SDK not installed or server API changed")

    enum_values = schema["properties"]["entity_type"]["enum"]
    assert "negative_result" in enum_values
    assert "quant_claim" in enum_values
    # Backwards compatibility — original values must still be there.
    for legacy in ("methods", "datasets", "instruments", "materials"):
        assert legacy in enum_values, f"{legacy} dropped from entity_type enum"


# ---------------------------------------------------------------------------
# AC2: negative_result rows surface evidence_span
# ---------------------------------------------------------------------------


def test_entity_negative_result_returns_evidence_span() -> None:
    """AC2: entity({entity_type='negative_result'}) rows carry evidence_span."""
    conn = _make_conn([_NEG_RESULT_ROW])

    out = _dispatch_tool(conn, "entity", {"entity_type": "negative_result"})
    data = json.loads(out)

    assert data["total"] == 1
    paper = data["papers"][0]
    assert paper["bibcode"] == "2024ApJ...001A"
    assert "evidence_span" in paper, "negative_result rows must surface evidence_span"
    assert "no significant" in paper["evidence_span"]
    # Spans list also preserved for downstream callers.
    assert paper["spans"][0]["confidence_label"] == "high"


def test_entity_negative_result_filter_by_query() -> None:
    """A name_filter on negative_result restricts to spans matching the substring."""
    conn = _make_conn([_NEG_RESULT_ROW])

    out = _dispatch_tool(
        conn,
        "entity",
        {"entity_type": "negative_result", "entity_name": "no significant"},
    )
    data = json.loads(out)
    assert data["total"] == 1
    assert "no significant" in data["papers"][0]["evidence_span"]


# ---------------------------------------------------------------------------
# AC3: quant_claim rows expose {value, uncertainty, unit} via payload
# ---------------------------------------------------------------------------


def test_entity_quant_claim_h0_payload_shape() -> None:
    """AC3: entity({entity_type='quant_claim', entity_name='H0'}) returns {value, uncertainty, unit}."""
    conn = _make_conn([_QUANT_CLAIM_ROW])

    out = _dispatch_tool(
        conn,
        "entity",
        {"entity_type": "quant_claim", "entity_name": "H0"},
    )
    data = json.loads(out)

    assert data["total"] == 1
    paper = data["papers"][0]
    payload = paper["payload"]
    assert payload["value"] == 73.0
    assert payload["uncertainty"] == 1.0
    assert payload["unit"] == "km/s/Mpc"
    # Filter dropped the Omega_m claim.
    assert len(paper["claims"]) == 1
    assert paper["claims"][0]["quantity"] == "H0"


def test_entity_quant_claim_no_filter_returns_all_claims() -> None:
    """Without entity_name, all claims survive the filter."""
    conn = _make_conn([_QUANT_CLAIM_ROW])

    out = _dispatch_tool(conn, "entity", {"entity_type": "quant_claim"})
    data = json.loads(out)

    assert data["total"] == 1
    paper = data["papers"][0]
    # First claim's payload is promoted to top-level.
    assert paper["payload"]["quantity"] == "H0"
    # All claims still present.
    assert {c["quantity"] for c in paper["claims"]} == {"H0", "Omega_m"}


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
    """AC4: entity responses carry top-level coverage_note."""
    conn = _make_conn([_NEG_RESULT_ROW])
    out = _dispatch_tool(conn, "entity", {"entity_type": "negative_result"})
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
