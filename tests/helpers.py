"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import os

import psycopg
import psycopg.errors

# Default DSN for read-only tests; destructive tests MUST use SCIX_TEST_DSN.
DSN = os.environ.get("SCIX_DSN", "dbname=scix")

_PRODUCTION_DB_NAMES = {"scix"}


def is_production_dsn(dsn: str) -> bool:
    """Return True if DSN appears to point at a production database."""
    for token in dsn.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key.strip() == "dbname" and value.strip() in _PRODUCTION_DB_NAMES:
                return True
    return False


def get_test_dsn() -> str | None:
    """Return DSN for destructive tests, or None if not configured.

    Destructive tests MUST call this instead of using DSN directly.
    Returns SCIX_TEST_DSN if set and not pointing at production.
    """
    test_dsn = os.environ.get("SCIX_TEST_DSN")
    if test_dsn is None:
        return None
    if is_production_dsn(test_dsn):
        return None
    return test_dsn

# Per-query timeout in seconds (configurable for slow environments)
STMT_TIMEOUT_S = int(os.environ.get("SCIX_TEST_TIMEOUT", "60"))


# ---------------------------------------------------------------------------
# Database availability checks (used by multiple test modules)
# ---------------------------------------------------------------------------


def has_papers(conn: psycopg.Connection) -> bool:
    """Check if the papers table has any rows."""
    with conn.cursor() as cur:
        cur.execute("SELECT EXISTS(SELECT 1 FROM papers LIMIT 1)")
        return cur.fetchone()[0]


def has_tsv_column(conn: psycopg.Connection) -> bool:
    """Check if the tsv column exists on papers (migration 003 applied)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'papers' AND column_name = 'tsv'
            )
        """)
        return cur.fetchone()[0]


def has_citation_edges(conn: psycopg.Connection) -> bool:
    """Check if the citation_edges table has any rows."""
    with conn.cursor() as cur:
        cur.execute("SELECT EXISTS(SELECT 1 FROM citation_edges LIMIT 1)")
        return cur.fetchone()[0]


def get_cited_bibcode(conn: psycopg.Connection) -> str | None:
    """Get a bibcode that has incoming citations (is a target in citation_edges)."""
    with conn.cursor() as cur:
        cur.execute("SELECT target_bibcode FROM citation_edges LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else None


def get_citing_bibcode(conn: psycopg.Connection) -> str | None:
    """Get a bibcode that cites other papers (is a source in citation_edges)."""
    with conn.cursor() as cur:
        cur.execute("SELECT source_bibcode FROM citation_edges LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else None


def rollback_and_reset(conn: psycopg.Connection) -> None:
    """Rollback a failed transaction and restore statement_timeout."""
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {STMT_TIMEOUT_S * 1000}")
