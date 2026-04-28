"""Pin the structured-error envelope convention across MCP tools.

Bead: ``scix_experiments-x5jg``.

Every named structured-error response carries:

* ``error`` — human-readable message (free-form string, may include the bad
  input value or remediation hints).
* ``error_code`` — stable machine-readable identifier the agent can branch on.

Generic exception wrappers (``json.dumps({"error": str(exc)})`` in
``call_tool`` and the lazy-import paths) are out of scope — those are
last-resort surfaces and agents are expected to retry or escalate, not
branch on them.

The mapping below is the documented public contract; adding a new
structured error should add an assertion here AND a row in
``docs/mcp_tool_audit_2026-04.md``.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scix.mcp_server import (
    _dispatch_tool,
    _reset_coverage_note_cache,
    _unscoped_broad_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn() -> MagicMock:
    """Connection mock — none of these error paths reach the DB."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=None)
    conn.cursor = MagicMock(return_value=cursor)
    return conn


@pytest.fixture(autouse=True)
def _reset_cov_cache() -> Generator[None, None, None]:
    _reset_coverage_note_cache()
    yield
    _reset_coverage_note_cache()


def _assert_envelope(payload: dict[str, Any], expected_code: str) -> None:
    """Assert the documented envelope contract for a structured error."""
    assert "error" in payload, f"missing 'error' field: {payload}"
    assert "error_code" in payload, f"missing 'error_code' field: {payload}"
    assert isinstance(payload["error"], str) and payload["error"].strip(), (
        f"'error' must be a non-empty string: {payload}"
    )
    assert payload["error_code"] == expected_code, (
        f"expected error_code={expected_code!r}, got {payload.get('error_code')!r}"
    )


# ---------------------------------------------------------------------------
# unscoped_broad_query — search tool
# ---------------------------------------------------------------------------


def test_unscoped_broad_query_envelope_has_stable_error_code() -> None:
    """``_unscoped_broad_response`` carries error_code='unscoped_broad_query'."""
    out = _unscoped_broad_response("astrophysics dark matter survey results 2025")
    data = json.loads(out)
    _assert_envelope(data, "unscoped_broad_query")
    # Telemetry contract preserved — _detect_unscoped_broad_block keys on
    # this flag, not on the error / error_code field.
    assert data["unscoped_broad_blocked"] is True
    # error must now be a human-readable message, not the bare tag.
    assert data["error"] != "unscoped_broad_query", (
        "after x5jg, 'error' should be a human-readable message; "
        "the stable tag lives in 'error_code'"
    )


# ---------------------------------------------------------------------------
# entity tool — legacy-type rejection (mh14) + invalid action
# ---------------------------------------------------------------------------


def test_entity_legacy_extraction_type_envelope() -> None:
    """entity tool rejecting negative_result/quant_claim carries error_code."""
    conn = _make_conn()
    out = _dispatch_tool(
        conn, "entity", {"action": "search", "query": "x", "entity_type": "negative_result"}
    )
    data = json.loads(out)
    _assert_envelope(data, "entity_legacy_extraction_type")
    # Human message still mentions the tool to use.
    assert "claim_search" in data["error"]


def test_entity_invalid_action_envelope() -> None:
    """entity tool with an unknown action returns error_code='invalid_action'.

    A non-empty ``query`` is supplied so the handler reaches the action
    enum check; the unrelated ``query must be a non-empty string`` guard
    is out of scope for this bead.
    """
    conn = _make_conn()
    out = _dispatch_tool(
        conn, "entity", {"action": "not_a_real_action", "query": "x"}
    )
    data = json.loads(out)
    _assert_envelope(data, "invalid_action")


# ---------------------------------------------------------------------------
# claim_search — invalid action + invalid limit (added in c996 wave)
# ---------------------------------------------------------------------------


def test_claim_search_invalid_action_envelope() -> None:
    conn = _make_conn()
    out = _dispatch_tool(conn, "claim_search", {"action": "not_a_real_action"})
    data = json.loads(out)
    _assert_envelope(data, "invalid_action")


def test_claim_search_invalid_limit_envelope() -> None:
    """claim_search rejects non-integer limit with error_code='invalid_limit'."""
    conn = _make_conn()
    out = _dispatch_tool(
        conn, "claim_search", {"action": "negative_result", "limit": "not_a_number"}
    )
    data = json.loads(out)
    _assert_envelope(data, "invalid_limit")


# ---------------------------------------------------------------------------
# citation_traverse — invalid mode + invalid direction
# ---------------------------------------------------------------------------


def test_citation_traverse_invalid_mode_envelope() -> None:
    """citation_traverse with an unknown mode carries error_code='invalid_mode'."""
    conn = _make_conn()
    out = _dispatch_tool(
        conn,
        "citation_traverse",
        {"mode": "not_a_real_mode", "bibcode": "2024ApJ...001A"},
    )
    data = json.loads(out)
    _assert_envelope(data, "invalid_mode")


def test_citation_traverse_invalid_direction_envelope() -> None:
    """citation_traverse graph mode with an unknown direction carries
    error_code='invalid_direction'."""
    conn = _make_conn()
    # Patch the underlying search functions so the handler reaches the
    # direction validation without touching the DB.
    with (
        patch("scix.search.get_citations", return_value=MagicMock(papers=[], total=0)),
        patch("scix.search.get_references", return_value=MagicMock(papers=[], total=0)),
    ):
        out = _dispatch_tool(
            conn,
            "citation_traverse",
            {
                "mode": "graph",
                "bibcode": "2024ApJ...001A",
                "direction": "sideways",
            },
        )
    data = json.loads(out)
    _assert_envelope(data, "invalid_direction")


# ---------------------------------------------------------------------------
# citation_similarity — invalid method (co_citation vs coupling)
# ---------------------------------------------------------------------------


def test_citation_similarity_invalid_method_envelope() -> None:
    """citation_similarity with an unknown method carries error_code='invalid_method'."""
    conn = _make_conn()
    out = _dispatch_tool(
        conn,
        "citation_similarity",
        {"bibcode": "2024ApJ...001A", "method": "telepathy"},
    )
    data = json.loads(out)
    _assert_envelope(data, "invalid_method")


# ---------------------------------------------------------------------------
# claim_blame / find_replications — invalid scope
# ---------------------------------------------------------------------------


def test_claim_blame_invalid_scope_envelope() -> None:
    """claim_blame rejects malformed scope payload with error_code='invalid_scope'."""
    conn = _make_conn()
    # scope=42 fails scope_from_dict's TypeError path before any DB call.
    out = _dispatch_tool(
        conn, "claim_blame", {"claim_text": "x", "scope": 42}
    )
    data = json.loads(out)
    _assert_envelope(data, "invalid_scope")


def test_find_replications_invalid_scope_envelope() -> None:
    """find_replications rejects malformed scope payload with error_code='invalid_scope'."""
    conn = _make_conn()
    out = _dispatch_tool(
        conn,
        "find_replications",
        {"target_bibcode": "2024ApJ...001A", "scope": 42},
    )
    data = json.loads(out)
    _assert_envelope(data, "invalid_scope")


# ---------------------------------------------------------------------------
# chunk_search — invalid filters
# ---------------------------------------------------------------------------


def test_chunk_search_invalid_filters_envelope() -> None:
    """chunk_search rejects un-coercible filters with error_code='invalid_filters'.

    chunk_search is env-hidden in prod (Qdrant collection not built); the
    handler short-circuits with ``qdrant_disabled`` before reaching the
    filters validation when ``_qdrant_enabled()`` is False. We patch the
    gate to True so the test exercises the documented ``invalid_filters``
    branch — which is what an agent on a real Qdrant-enabled deployment
    would see.
    """
    conn = _make_conn()
    with patch("scix.mcp_server._qdrant_enabled", return_value=True):
        out = _dispatch_tool(
            conn,
            "chunk_search",
            {"query": "x", "filters": {"year_min": "not_a_year"}},
        )
    data = json.loads(out)
    _assert_envelope(data, "invalid_filters")


# ---------------------------------------------------------------------------
# Documented enum (registry of stable error_codes)
# ---------------------------------------------------------------------------


def test_documented_error_codes_in_audit_doc() -> None:
    """The audit doc must list every error_code asserted above.

    Matches the backtick-quoted form (`code_name`) used in the registry
    table so a stray prose mention cannot satisfy the check.
    """
    import re
    from pathlib import Path

    audit_path = Path(__file__).resolve().parents[1] / "docs" / "mcp_tool_audit_2026-04.md"
    text = audit_path.read_text()

    expected_codes = {
        "missing_required_params",  # already documented (zjt9)
        "unscoped_broad_query",
        "entity_legacy_extraction_type",
        "invalid_action",
        "invalid_limit",
        "invalid_mode",
        "invalid_direction",
        "invalid_method",
        "invalid_filters",
        "invalid_scope",
    }
    missing = {
        code for code in expected_codes
        if not re.search(r"`" + re.escape(code) + r"`", text)
    }
    assert not missing, (
        f"audit doc is missing stable error_codes (backtick-quoted form): "
        f"{sorted(missing)}. Each new error_code asserted in this file MUST "
        "also appear in the registry table in "
        "docs/mcp_tool_audit_2026-04.md so agents and operators can find it."
    )
