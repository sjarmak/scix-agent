"""Tests for the ``claim_search`` MCP tool (bead ``scix_experiments-c996``).

The tool surfaces ``staging.extractions`` rows for the two claim/finding
extraction kinds — ``negative_result`` and ``quant_claim`` — that were
removed from the ``entity`` tool's ``entity_type`` enum under bead
``scix_experiments-mh14`` because they are not entities.

Coverage:

* AC1 — schema: tool registers an ``action`` enum with exactly
  ``negative_result`` and ``quant_claim``.
* AC2 — happy path for ``action='negative_result'``: row is dispatched
  through ``_handle_entity_extraction_search`` and the response carries
  the flattened ``evidence_span`` field plus the standard
  ``coverage_note``.
* AC3 — happy path for ``action='quant_claim'``: claim is promoted to
  the top-level ``payload`` shape and the response carries ``claims``.
* AC4 — invalid action returns a structured error naming the bad value
  and listing the valid choices.
* AC5 — empty result set returns ``{papers: [], total: 0}`` with
  coverage_note.
* AC6 — ``query`` is forwarded as the helper's ``name_filter``.
* AC7 — entity tool's legacy-type rejection message points at
  ``claim_search`` by name (not just the bead id).

DB is mocked — no live DB.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from scix.mcp_server import _dispatch_tool, _reset_coverage_note_cache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cov_cache() -> None:
    """Drop the cached coverage note between tests."""
    _reset_coverage_note_cache()
    yield
    _reset_coverage_note_cache()


def _make_cursor_with_rows(rows: list[tuple[Any, ...]]) -> MagicMock:
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    return cursor


def _make_conn(rows: list[tuple[Any, ...]]) -> MagicMock:
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


# Sample M4 quant-claim row (payload {claims: [{quantity, value, ...}]}).
_QUANT_CLAIM_ROW: tuple[Any, ...] = (
    "2024ApJ...002B",
    "quant_claim",
    "quant_v1",
    {
        "claims": [
            {
                "quantity": "H0",
                "value": 67.4,
                "uncertainty": 0.5,
                "unit": "km/s/Mpc",
            },
            {
                "quantity": "Omega_m",
                "value": 0.315,
                "uncertainty": 0.007,
                "unit": "",
            },
        ]
    },
    "Cosmological parameters from Planck",
)


# ---------------------------------------------------------------------------
# AC1: schema
# ---------------------------------------------------------------------------


def _get_claim_search_tool_schema(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Pull the live ``claim_search`` tool inputSchema via tools/list.

    The tool is in ``_HIDDEN_TOOLS`` by default (extractions table is empty
    on prod). Monkeypatch the module-level frozenset so tools/list returns
    the full registered set.
    """
    import asyncio

    from mcp.types import ListToolsRequest

    from scix import mcp_server as ms
    from scix.mcp_server import create_server

    monkeypatch.setattr(ms, "_HIDDEN_TOOLS", frozenset())

    server = create_server(_run_self_test=False)
    handler = server.request_handlers[ListToolsRequest]
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(handler(ListToolsRequest(method="tools/list")))
    finally:
        loop.close()
    tools = result.root.tools if hasattr(result, "root") else result.tools
    claim_tool = next(t for t in tools if t.name == "claim_search")
    return claim_tool.inputSchema  # type: ignore[no-any-return]


def test_claim_search_schema_advertises_action_enum(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC1: the action enum exposes negative_result and quant_claim."""
    try:
        schema = _get_claim_search_tool_schema(monkeypatch)
    except (ImportError, AttributeError, StopIteration):
        pytest.skip("mcp SDK not installed or claim_search not registered")

    assert "action" in schema["properties"]
    enum_values = schema["properties"]["action"]["enum"]
    assert set(enum_values) == {"negative_result", "quant_claim"}, enum_values
    assert "action" in schema["required"]


# ---------------------------------------------------------------------------
# AC2 + AC3: happy paths
# ---------------------------------------------------------------------------


def test_claim_search_negative_result_happy_path() -> None:
    """AC2: action='negative_result' returns a flattened evidence_span row."""
    conn = _make_conn([_NEG_RESULT_ROW])

    out = _dispatch_tool(conn, "claim_search", {"action": "negative_result"})
    data = json.loads(out)

    assert "papers" in data
    assert data["total"] == 1
    paper = data["papers"][0]
    assert paper["bibcode"] == "2024ApJ...001A"
    assert paper["extraction_type"] == "negative_result"
    # Helper flattens the first span's evidence_span to top level.
    assert paper["evidence_span"] == "We found no significant excess at 5 sigma."
    # Coverage note must be present (consistency with entity tool surface).
    assert "coverage_note" in data


def test_claim_search_quant_claim_happy_path() -> None:
    """AC3: action='quant_claim' promotes the first claim to top-level payload."""
    conn = _make_conn([_QUANT_CLAIM_ROW])

    out = _dispatch_tool(conn, "claim_search", {"action": "quant_claim"})
    data = json.loads(out)

    assert data["total"] == 1
    paper = data["papers"][0]
    assert paper["bibcode"] == "2024ApJ...002B"
    assert paper["extraction_type"] == "quant_claim"
    # First claim promoted to top-level payload.
    assert paper["payload"]["quantity"] == "H0"
    assert paper["payload"]["value"] == 67.4
    assert paper["payload"]["unit"] == "km/s/Mpc"
    # Full claim list preserved.
    assert len(paper["claims"]) == 2
    assert "coverage_note" in data


# ---------------------------------------------------------------------------
# AC4: invalid action error path
# ---------------------------------------------------------------------------


def test_claim_search_invalid_action_returns_structured_error() -> None:
    """AC4: unknown action -> structured error naming bad value and valid choices."""
    conn = _make_conn([])

    out = _dispatch_tool(conn, "claim_search", {"action": "not_a_real_action"})
    data = json.loads(out)

    assert "error" in data
    assert "not_a_real_action" in data["error"]
    assert "negative_result" in data["error"]
    assert "quant_claim" in data["error"]
    # No SQL was issued for invalid input.
    assert not conn.cursor.called


def test_claim_search_missing_action_returns_structured_error() -> None:
    """Missing action is rejected just like an invalid one."""
    conn = _make_conn([])

    out = _dispatch_tool(conn, "claim_search", {})
    data = json.loads(out)

    assert "error" in data
    assert not conn.cursor.called


# ---------------------------------------------------------------------------
# AC5: empty result set
# ---------------------------------------------------------------------------


def test_claim_search_empty_result_returns_zero_papers() -> None:
    """AC5: zero rows -> {papers: [], total: 0} with coverage_note."""
    conn = _make_conn([])

    out = _dispatch_tool(conn, "claim_search", {"action": "negative_result"})
    data = json.loads(out)

    assert data["papers"] == []
    assert data["total"] == 0
    assert "coverage_note" in data


# ---------------------------------------------------------------------------
# AC6: query forwarded as name_filter
# ---------------------------------------------------------------------------


def test_claim_search_query_filters_quant_claim_rows() -> None:
    """AC6: query='Omega_m' filters quant_claim claims to that quantity."""
    conn = _make_conn([_QUANT_CLAIM_ROW])

    out = _dispatch_tool(
        conn,
        "claim_search",
        {"action": "quant_claim", "query": "Omega_m"},
    )
    data = json.loads(out)

    assert data["total"] == 1
    paper = data["papers"][0]
    # Only the Omega_m claim survived the filter.
    assert len(paper["claims"]) == 1
    assert paper["claims"][0]["quantity"] == "Omega_m"
    # Top-level payload also reflects the filtered claim.
    assert paper["payload"]["quantity"] == "Omega_m"
    assert paper["payload"]["value"] == 0.315


def test_claim_search_query_drops_negative_result_with_no_match() -> None:
    """A query that matches no spans drops the row entirely."""
    conn = _make_conn([_NEG_RESULT_ROW])

    out = _dispatch_tool(
        conn,
        "claim_search",
        {"action": "negative_result", "query": "nonexistent-needle-xyz"},
    )
    data = json.loads(out)

    assert data["papers"] == []
    assert data["total"] == 0


# ---------------------------------------------------------------------------
# AC7: entity tool's legacy-type rejection points at claim_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("legacy_type", ["negative_result", "quant_claim"])
def test_entity_legacy_type_error_points_at_claim_search(legacy_type: str) -> None:
    """AC7: agent rediscovering the old contract is told the new tool name."""
    conn = _make_conn([_NEG_RESULT_ROW])

    out = _dispatch_tool(conn, "entity", {"entity_type": legacy_type})
    data = json.loads(out)

    assert "error" in data
    # Bead reference preserved for traceability.
    assert "scix_experiments-c996" in data["error"]
    # And the new tool name is mentioned so the agent can recover in one turn.
    assert "claim_search" in data["error"]
