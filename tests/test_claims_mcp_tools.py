"""Tests for the read_paper_claims and find_claims MCP tools.

Three layers:

1. Tool registration — the tools appear in EXPECTED_TOOLS, in
   TOOL_TIMEOUTS, and in the live list_tools() handler with the
   documented JSON Schemas. No DB required.

2. Dispatch error envelopes — invalid args produce a structured JSON
   ``{"error": ...}`` response (no exception escapes the dispatcher).
   Uses a no-op connection stub; never reaches the database.

3. Integration — applies migration 062, seeds rows, calls the
   dispatcher with a real connection, and asserts the response shape.
   Skipped if scix_test isn't reachable.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from scix.mcp_server import (
    EXPECTED_TOOLS,
    TOOL_TIMEOUTS,
    _dispatch_tool,
    _handle_find_claims,
    _handle_read_paper_claims,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_PATH = REPO_ROOT / "migrations" / "062_paper_claims.sql"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Both tools must be wired into the tool listing + timeouts."""

    def test_read_paper_claims_in_expected_tools(self) -> None:
        assert "read_paper_claims" in EXPECTED_TOOLS

    def test_find_claims_in_expected_tools(self) -> None:
        assert "find_claims" in EXPECTED_TOOLS

    def test_read_paper_claims_in_tool_timeouts(self) -> None:
        assert "read_paper_claims" in TOOL_TIMEOUTS
        assert TOOL_TIMEOUTS["read_paper_claims"] > 0

    def test_find_claims_in_tool_timeouts(self) -> None:
        assert "find_claims" in TOOL_TIMEOUTS
        assert TOOL_TIMEOUTS["find_claims"] > 0

    def test_tools_appear_in_list_tools(self) -> None:
        try:
            import asyncio

            from mcp.types import ListToolsRequest
        except ImportError:
            pytest.skip("mcp SDK not installed")

        with patch("scix.mcp_server._init_model_impl"):
            from scix.mcp_server import create_server

            server = create_server(_run_self_test=False)

        handler = server.request_handlers[ListToolsRequest]
        result = asyncio.run(handler(ListToolsRequest(method="tools/list")))
        tools = result.root.tools if hasattr(result, "root") else result.tools
        names = {t.name for t in tools}
        assert "read_paper_claims" in names
        assert "find_claims" in names

    def test_read_paper_claims_input_schema(self) -> None:
        try:
            import asyncio

            from mcp.types import ListToolsRequest
        except ImportError:
            pytest.skip("mcp SDK not installed")

        with patch("scix.mcp_server._init_model_impl"):
            from scix.mcp_server import create_server

            server = create_server(_run_self_test=False)

        handler = server.request_handlers[ListToolsRequest]
        result = asyncio.run(handler(ListToolsRequest(method="tools/list")))
        tools = result.root.tools if hasattr(result, "root") else result.tools
        by_name = {t.name: t for t in tools}
        schema = by_name["read_paper_claims"].inputSchema

        assert schema["type"] == "object"
        assert "bibcode" in schema["properties"]
        assert schema["properties"]["bibcode"]["type"] == "string"
        assert "claim_type" in schema["properties"]
        assert "limit" in schema["properties"]
        assert schema.get("required") == ["bibcode"]

    def test_find_claims_input_schema(self) -> None:
        try:
            import asyncio

            from mcp.types import ListToolsRequest
        except ImportError:
            pytest.skip("mcp SDK not installed")

        with patch("scix.mcp_server._init_model_impl"):
            from scix.mcp_server import create_server

            server = create_server(_run_self_test=False)

        handler = server.request_handlers[ListToolsRequest]
        result = asyncio.run(handler(ListToolsRequest(method="tools/list")))
        tools = result.root.tools if hasattr(result, "root") else result.tools
        by_name = {t.name: t for t in tools}
        schema = by_name["find_claims"].inputSchema

        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert "claim_type" in schema["properties"]
        assert "entity_id" in schema["properties"]
        assert "limit" in schema["properties"]
        assert schema.get("required") == ["query"]


# ---------------------------------------------------------------------------
# Static dispatch error envelopes (no DB)
# ---------------------------------------------------------------------------


class _NoOpCursor:
    def __enter__(self) -> "_NoOpCursor":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:  # noqa: D401
        raise AssertionError("dispatcher reached the DB despite invalid args")


class _NoOpConn:
    def cursor(self) -> _NoOpCursor:
        return _NoOpCursor()


class TestDispatchErrorEnvelopes:
    """Invalid args yield ``{"error": ...}`` — never raise to the caller."""

    def test_read_paper_claims_missing_bibcode(self) -> None:
        out = _handle_read_paper_claims(_NoOpConn(), {})
        data = json.loads(out)
        assert "error" in data
        assert "bibcode" in data["error"]

    def test_read_paper_claims_empty_bibcode(self) -> None:
        out = _handle_read_paper_claims(_NoOpConn(), {"bibcode": ""})
        data = json.loads(out)
        assert "error" in data

    def test_read_paper_claims_unknown_claim_type(self) -> None:
        out = _handle_read_paper_claims(
            _NoOpConn(), {"bibcode": "x", "claim_type": "bogus"}
        )
        data = json.loads(out)
        assert "error" in data

    def test_find_claims_missing_query(self) -> None:
        out = _handle_find_claims(_NoOpConn(), {})
        data = json.loads(out)
        assert "error" in data
        assert "query" in data["error"]

    def test_find_claims_empty_query(self) -> None:
        out = _handle_find_claims(_NoOpConn(), {"query": ""})
        data = json.loads(out)
        assert "error" in data

    def test_find_claims_non_int_entity_id(self) -> None:
        out = _handle_find_claims(
            _NoOpConn(), {"query": "x", "entity_id": "abc"}
        )
        data = json.loads(out)
        assert "error" in data
        assert "entity_id" in data["error"]


# ---------------------------------------------------------------------------
# Integration — apply migration 062 + seed rows + dispatch
# ---------------------------------------------------------------------------


def _resolve_test_dsn() -> str | None:
    dsn = os.environ.get("SCIX_TEST_DSN")
    if dsn:
        if "scix_test" not in dsn:
            pytest.fail(
                "SCIX_TEST_DSN must reference scix_test — got: " + dsn,
                pytrace=False,
            )
        return dsn
    fallback = "dbname=scix_test"
    try:
        import psycopg
    except ImportError:
        return None
    try:
        with psycopg.connect(fallback, connect_timeout=2):
            pass
    except Exception:
        return None
    return fallback


TEST_DSN = _resolve_test_dsn()
INTEGRATION_REASON = "scix_test database not reachable (set SCIX_TEST_DSN)"


@pytest.fixture(scope="module")
def dsn() -> str:
    if TEST_DSN is None:
        pytest.skip(INTEGRATION_REASON, allow_module_level=False)
    return TEST_DSN


@pytest.fixture(scope="module")
def applied_migration(dsn: str) -> str:
    result = subprocess.run(
        ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"migration 062 failed to apply: stderr=\n{result.stderr}"
    )
    return dsn


def _ensure_test_bibcode(dsn: str) -> str:
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT bibcode FROM papers LIMIT 1")
        row = cur.fetchone()
        if row is not None:
            return row[0]
        synthetic = "9999paper_claims_mcp_X"
        cur.execute(
            "INSERT INTO papers (bibcode) VALUES (%s) "
            "ON CONFLICT (bibcode) DO NOTHING",
            (synthetic,),
        )
        conn.commit()
        return synthetic


_TEST_MARKER = "test-claims-mcp"


@pytest.fixture
def seeded_claims(applied_migration: str) -> Iterator[dict[str, Any]]:
    """Seed a small fixed set of paper_claims rows."""
    import psycopg

    bibcode = _ensure_test_bibcode(applied_migration)
    seed_rows = [
        (0, 0, 0, 50, "The Hubble constant H0 is 73 km/s/Mpc.", "factual"),
        (
            0,
            1,
            60,
            120,
            "We measured galaxy distances using Cepheid variables.",
            "methodological",
        ),
        (
            1,
            0,
            0,
            70,
            "Future JWST observations may resolve the Hubble tension.",
            "speculative",
        ),
    ]
    inserted: list[uuid.UUID] = []
    with psycopg.connect(applied_migration) as conn:
        with conn.cursor() as cur:
            for sec, para, s, e, text, ctype in seed_rows:
                cid = uuid.uuid4()
                inserted.append(cid)
                cur.execute(
                    """
                    INSERT INTO paper_claims (
                        claim_id, bibcode, section_index, paragraph_index,
                        char_span_start, char_span_end,
                        claim_text, claim_type,
                        extraction_model, extraction_prompt_version
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        str(cid),
                        bibcode,
                        sec,
                        para,
                        s,
                        e,
                        text,
                        ctype,
                        _TEST_MARKER,
                        "v1",
                    ),
                )
        conn.commit()

    try:
        yield {"dsn": applied_migration, "bibcode": bibcode}
    finally:
        with psycopg.connect(applied_migration) as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM paper_claims WHERE extraction_model = %s",
                (_TEST_MARKER,),
            )
            conn.commit()


class TestDispatchReadPaperClaims:
    """End-to-end ``_dispatch_tool('read_paper_claims', ...)``."""

    @patch("scix.mcp_server._log_query")
    def test_returns_seeded_rows(
        self, _mock_log: Any, seeded_claims: dict[str, Any]
    ) -> None:
        import psycopg

        dsn = seeded_claims["dsn"]
        bibcode = seeded_claims["bibcode"]
        with psycopg.connect(dsn) as conn:
            out = _dispatch_tool(
                conn,
                "read_paper_claims",
                {"bibcode": bibcode},
            )
        data = json.loads(out)
        assert "error" not in data, data
        assert data["bibcode"] == bibcode
        assert data["total"] == 3
        assert len(data["claims"]) == 3
        # Schema acceptance: every claim dict carries the documented keys.
        required_keys = {
            "bibcode",
            "section_index",
            "paragraph_index",
            "char_span_start",
            "char_span_end",
            "claim_text",
            "claim_type",
            "subject",
            "predicate",
            "object",
            "confidence",
        }
        for claim in data["claims"]:
            assert required_keys <= set(claim.keys()), claim

    @patch("scix.mcp_server._log_query")
    def test_claim_type_filter(
        self, _mock_log: Any, seeded_claims: dict[str, Any]
    ) -> None:
        import psycopg

        dsn = seeded_claims["dsn"]
        bibcode = seeded_claims["bibcode"]
        with psycopg.connect(dsn) as conn:
            out = _dispatch_tool(
                conn,
                "read_paper_claims",
                {"bibcode": bibcode, "claim_type": "factual"},
            )
        data = json.loads(out)
        assert "error" not in data
        assert all(c["claim_type"] == "factual" for c in data["claims"])


class TestDispatchFindClaims:
    """End-to-end ``_dispatch_tool('find_claims', ...)``."""

    @patch("scix.mcp_server._log_query")
    def test_returns_results_for_known_phrase(
        self, _mock_log: Any, seeded_claims: dict[str, Any]
    ) -> None:
        import psycopg

        dsn = seeded_claims["dsn"]
        bibcode = seeded_claims["bibcode"]
        with psycopg.connect(dsn) as conn:
            out = _dispatch_tool(
                conn,
                "find_claims",
                {"query": "Hubble tension"},
            )
        data = json.loads(out)
        assert "error" not in data, data
        assert data["query"] == "Hubble tension"
        seeded = [c for c in data["claims"] if c["bibcode"] == bibcode]
        # The seed contains exactly one row mentioning "Hubble tension".
        assert len(seeded) >= 1
        assert any(
            "Hubble tension" in c["claim_text"] for c in seeded
        )

    @patch("scix.mcp_server._log_query")
    def test_limit_is_honored(
        self, _mock_log: Any, seeded_claims: dict[str, Any]
    ) -> None:
        import psycopg

        dsn = seeded_claims["dsn"]
        with psycopg.connect(dsn) as conn:
            out = _dispatch_tool(
                conn,
                "find_claims",
                {"query": "Hubble", "limit": 1},
            )
        data = json.loads(out)
        assert "error" not in data
        assert len(data["claims"]) <= 1
