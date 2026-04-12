"""Query log write API for MCP tool instrumentation (M3.5.0).

The ``query_log`` table has been extended by migration 031 with the
instrumentation columns ``ts, tool, query, result_count, session_id,
is_test``. This module provides a thin wrapper that writes to those
columns while also filling the legacy NOT NULL columns
(``tool_name``, ``success``) so existing constraints are satisfied.

This is a write-only API. Analytics live in ``scripts/analyze_query_log.py``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import psycopg

from scix.db import DEFAULT_DSN, get_connection

logger = logging.getLogger(__name__)


def _resolve_dsn(dsn: str | None) -> str:
    """Pick an explicit DSN, otherwise SCIX_TEST_DSN, otherwise the default."""
    if dsn is not None:
        return dsn
    test_dsn = os.environ.get("SCIX_TEST_DSN")
    if test_dsn:
        return test_dsn
    return DEFAULT_DSN


def log_query(
    tool: str,
    query: str,
    result_count: int,
    session_id: Optional[str] = None,
    is_test: bool = False,
    *,
    conn: psycopg.Connection | None = None,
    dsn: str | None = None,
) -> None:
    """Record an MCP tool call in ``query_log``.

    Writes both the new instrumentation columns
    (``tool, query, result_count, session_id, is_test``) and the
    legacy NOT NULL columns (``tool_name``, ``success``) in a single row.
    ``ts`` is populated by the column default (``now()``).

    Args:
        tool: MCP tool name (e.g. ``"search_dual"``).
        query: User/agent query string.
        result_count: Number of results returned (0 = zero-result).
        session_id: Opaque session identifier (optional).
        is_test: True for synthetic/test traffic so curation can filter it.
        conn: Optional open connection to reuse (will NOT be committed on
            failure; caller owns its lifecycle). If omitted, a short-lived
            connection is opened using ``dsn`` or environment defaults.
        dsn: DSN override for the short-lived connection path.
    """
    owned_conn = False
    if conn is None:
        conn = get_connection(_resolve_dsn(dsn), autocommit=False)
        owned_conn = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_log (
                    tool_name,
                    success,
                    tool,
                    query,
                    result_count,
                    session_id,
                    is_test
                ) VALUES (%s, TRUE, %s, %s, %s, %s, %s)
                """,
                (tool, tool, query, result_count, session_id, is_test),
            )
        conn.commit()
    except Exception:
        logger.warning("Failed to log query for tool=%s", tool, exc_info=True)
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if owned_conn:
            conn.close()
