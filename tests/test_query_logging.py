"""Tests for MCP query logging: migration DDL, call_tool logging, and analysis script."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from scix.mcp_server import _log_query

# ---------------------------------------------------------------------------
# Migration DDL verification
# ---------------------------------------------------------------------------


class TestMigrationDDL:
    """Verify the migration file contains the expected schema."""

    def test_migration_file_creates_query_log_table(self) -> None:
        with open("migrations/013_query_log.sql") as f:
            sql = f.read()
        assert "CREATE TABLE" in sql
        assert "query_log" in sql
        for col in [
            "id",
            "tool_name",
            "params_json",
            "latency_ms",
            "success",
            "error_msg",
            "created_at",
        ]:
            assert col in sql, f"Missing column: {col}"

    def test_migration_has_indexes(self) -> None:
        with open("migrations/013_query_log.sql") as f:
            sql = f.read()
        assert "idx_query_log_tool_name" in sql
        assert "idx_query_log_created_at" in sql


# ---------------------------------------------------------------------------
# _log_query unit tests
# ---------------------------------------------------------------------------


class TestLogQuery:
    def test_log_query_inserts_row(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        _log_query(mock_conn, "get_paper", {"bibcode": "X"}, 5.2, True, None)

        mock_cur.execute.assert_called_once()
        sql_arg = mock_cur.execute.call_args[0][0]
        assert "INSERT INTO query_log" in sql_arg
        params = mock_cur.execute.call_args[0][1]
        assert params[0] == "get_paper"
        assert json.loads(params[1]) == {"bibcode": "X"}
        assert params[2] == 5.2
        assert params[3] is True
        assert params[4] is None
        mock_conn.commit.assert_called_once()

    def test_log_query_records_failure(self) -> None:
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        _log_query(mock_conn, "semantic_search", {"query": "q"}, 100.0, False, "timeout")

        params = mock_cur.execute.call_args[0][1]
        assert params[3] is False
        assert params[4] == "timeout"

    def test_log_query_swallows_exceptions(self) -> None:
        """Logging failures must not propagate."""
        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = RuntimeError("db down")

        # Should not raise
        _log_query(mock_conn, "test", {}, 0.0, True, None)


# ---------------------------------------------------------------------------
# call_tool integration: verify logging wraps dispatch
# ---------------------------------------------------------------------------


class TestCallToolLogging:
    """Verify that call_tool logs every tool dispatch."""

    @staticmethod
    def _make_request(tool_name: str, arguments: dict[str, Any]) -> Any:
        """Build a CallToolRequest for the given tool."""
        from mcp.types import CallToolRequest, CallToolRequestParams

        return CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=tool_name, arguments=arguments),
        )

    @patch("scix.mcp_server._log_query")
    @patch("scix.mcp_server._dispatch_tool")
    @patch("scix.mcp_server._get_conn")
    def test_three_tools_produce_three_log_rows(
        self,
        mock_get_conn: MagicMock,
        mock_dispatch: MagicMock,
        mock_log: MagicMock,
    ) -> None:
        """Call 3 different tools via the server and check _log_query called 3 times."""
        import asyncio

        try:
            from mcp.types import CallToolRequest
        except ImportError:
            pytest.skip("mcp SDK not installed")

        # Setup mock connection context manager
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        mock_dispatch.return_value = '{"ok": true}'

        from scix.mcp_server import create_server

        with patch("scix.mcp_server._init_model_impl"):
            server = create_server()

        handler = server.request_handlers[CallToolRequest]

        tools = [
            ("get_paper", {"bibcode": "2024ApJ...001A"}),
            ("keyword_search", {"terms": "dark matter"}),
            ("health_check", {}),
        ]

        loop = asyncio.new_event_loop()
        for tool_name, args in tools:
            loop.run_until_complete(handler(self._make_request(tool_name, args)))
        loop.close()

        assert mock_log.call_count == 3
        logged_tools = [c.args[1] for c in mock_log.call_args_list]
        assert logged_tools == ["get_paper", "keyword_search", "health_check"]
        # All should be success=True
        for c in mock_log.call_args_list:
            assert c.args[4] is True  # success

    @patch("scix.mcp_server._log_query")
    @patch("scix.mcp_server._dispatch_tool", side_effect=RuntimeError("boom"))
    @patch("scix.mcp_server._get_conn")
    def test_failed_dispatch_still_logs(
        self,
        mock_get_conn: MagicMock,
        mock_dispatch: MagicMock,
        mock_log: MagicMock,
    ) -> None:
        """Even when dispatch raises, a log row is written with success=False."""
        import asyncio

        try:
            from mcp.types import CallToolRequest
        except ImportError:
            pytest.skip("mcp SDK not installed")

        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        from scix.mcp_server import create_server

        with patch("scix.mcp_server._init_model_impl"):
            server = create_server()

        handler = server.request_handlers[CallToolRequest]

        loop = asyncio.new_event_loop()
        # The MCP framework may catch the exception internally,
        # so we just verify logging happened with success=False.
        try:
            loop.run_until_complete(handler(self._make_request("get_paper", {"bibcode": "X"})))
        except RuntimeError:
            pass  # Expected if MCP doesn't catch it
        finally:
            loop.close()

        assert mock_log.call_count == 1
        assert mock_log.call_args.args[4] is False  # success
        assert "boom" in mock_log.call_args.args[5]  # error_msg


# ---------------------------------------------------------------------------
# Analysis script unit tests
# ---------------------------------------------------------------------------


class TestAnalyzeQueryLog:
    """Test the analysis functions with mocked DB cursors."""

    def test_top_queries(self) -> None:
        from scripts.analyze_query_log import top_queries

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = [
            ("get_paper", {"bibcode": "X"}, 10),
            ("keyword_search", {"terms": "q"}, 5),
        ]

        result = top_queries(mock_conn, limit=50)
        assert len(result) == 2
        assert result[0]["tool_name"] == "get_paper"
        assert result[0]["call_count"] == 10

    def test_failure_rate_by_tool(self) -> None:
        from scripts.analyze_query_log import failure_rate_by_tool

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = [
            ("semantic_search", 100, 5, 5.00),
            ("get_paper", 50, 0, 0.00),
        ]

        result = failure_rate_by_tool(mock_conn)
        assert len(result) == 2
        assert result[0]["failures"] == 5
        assert result[0]["failure_pct"] == 5.0

    def test_entity_type_requests(self) -> None:
        from scripts.analyze_query_log import entity_type_requests

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = [
            ("methods", 20),
            ("datasets", 10),
            ("instruments", 5),
        ]

        result = entity_type_requests(mock_conn)
        assert len(result) == 3
        assert result[0]["entity_type"] == "methods"

    def test_generate_report_has_required_keys(self) -> None:
        from scripts.analyze_query_log import generate_report

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = []

        report = generate_report(mock_conn)
        assert "top_queries" in report
        assert "failure_rate_by_tool" in report
        assert "entity_type_requests" in report
