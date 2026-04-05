#!/usr/bin/env python3
"""Analyze MCP query log and produce a JSON report.

Output keys:
  - top_queries: Top 50 (tool_name, params) combinations by frequency
  - failure_rate_by_tool: Failure count / total count per tool
  - entity_type_requests: Distribution of entity_type values from entity_search calls

Usage:
    python scripts/analyze_query_log.py [--dsn DSN] [--pretty]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import psycopg

from scix.db import DEFAULT_DSN


def _connect(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn)


def top_queries(conn: psycopg.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """Top N most frequent (tool_name, params_json) combinations."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tool_name, params_json, COUNT(*) AS call_count
            FROM query_log
            GROUP BY tool_name, params_json
            ORDER BY call_count DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [{"tool_name": row[0], "params": row[1], "call_count": row[2]} for row in rows]


def failure_rate_by_tool(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Failure rate per tool: total calls, failures, and rate."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tool_name,
                   COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE NOT success) AS failures,
                   ROUND(
                       COUNT(*) FILTER (WHERE NOT success)::numeric
                       / GREATEST(COUNT(*), 1) * 100, 2
                   ) AS failure_pct
            FROM query_log
            GROUP BY tool_name
            ORDER BY failures DESC, total DESC
            """)
        rows = cur.fetchall()
    return [
        {
            "tool_name": row[0],
            "total": row[1],
            "failures": row[2],
            "failure_pct": float(row[3]),
        }
        for row in rows
    ]


def entity_type_requests(conn: psycopg.Connection) -> list[dict[str, Any]]:
    """Distribution of entity_type values from entity_search tool calls."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT params_json->>'entity_type' AS entity_type,
                   COUNT(*) AS call_count
            FROM query_log
            WHERE tool_name = 'entity_search'
              AND params_json->>'entity_type' IS NOT NULL
            GROUP BY entity_type
            ORDER BY call_count DESC
            """)
        rows = cur.fetchall()
    return [{"entity_type": row[0], "call_count": row[1]} for row in rows]


def generate_report(conn: psycopg.Connection) -> dict[str, Any]:
    """Generate the full analysis report."""
    return {
        "top_queries": top_queries(conn),
        "failure_rate_by_tool": failure_rate_by_tool(conn),
        "entity_type_requests": entity_type_requests(conn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MCP query log")
    parser.add_argument(
        "--dsn",
        default=os.environ.get("SCIX_DSN", DEFAULT_DSN),
        help="PostgreSQL DSN (default: $SCIX_DSN or scix.db.DEFAULT_DSN)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    args = parser.parse_args()

    conn = _connect(args.dsn)
    try:
        report = generate_report(conn)
        indent = 2 if args.pretty else None
        json.dump(report, sys.stdout, indent=indent, default=str)
        sys.stdout.write("\n")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
