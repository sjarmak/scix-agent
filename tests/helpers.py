"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
import psycopg.errors

# Make src/ importable so we can share the canonical production-DSN guard
# with the library code (src/scix/db.py). Keeping a separate implementation
# in this file caused a URI-form bypass (tests/helpers saw only key=value
# DSNs; src/scix/ads_body used a superset parser) until 2026-04-13.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scix.db import is_production_dsn as _is_production_dsn  # noqa: E402

# SCIX_TEST_DSN wins, per CLAUDE.md: exporting it is documented as the way to
# point the suite away from production. Reading SCIX_DSN first broke that
# contract in both directions: modules that connect here reached production
# even with the guard exported, and the destructive modules that gate on
# is_production_dsn(DSN) then skipped silently instead of running.
DSN = os.environ.get("SCIX_TEST_DSN") or os.environ.get("SCIX_DSN", "dbname=scix")


def is_production_dsn(dsn: str | None) -> bool:
    """Return True if DSN appears to point at a production database.

    Thin re-export of ``scix.db.is_production_dsn`` — kept as a function so
    the many ``from helpers import is_production_dsn`` call sites across the
    test suite continue to resolve without edits.
    """
    return _is_production_dsn(dsn)


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
# Throwaway databases for migration-replay tests
# ---------------------------------------------------------------------------


def _maintenance_dsn(test_dsn: str) -> str:
    """Return ``test_dsn`` with its dbname swapped for ``postgres``.

    CREATE/DROP DATABASE cannot run while connected to the target database.
    """
    kept = [tok for tok in test_dsn.split() if not tok.startswith("dbname=")]
    return " ".join([*kept, "dbname=postgres"])


def _replace_dbname(test_dsn: str, dbname: str) -> str:
    kept = [tok for tok in test_dsn.split() if not tok.startswith("dbname=")]
    return " ".join([*kept, f"dbname={dbname}"])


@contextmanager
def throwaway_db(migrations: list[str], repo_root: Path) -> Iterator[str]:
    """Create a fresh database, apply ``migrations`` in order, drop it after.

    Migration-replay tests must NOT run against the shared ``scix_test``.
    Replaying ``001_initial_schema.sql`` there re-creates ``paper_embeddings``,
    which ADR-015 dropped and migration 074 records as retired — so whether
    that table existed depended on test order, and unrelated modules failed or
    passed according to which ran first.

    Yields the DSN of the throwaway database. Raises if any migration fails;
    the database is dropped either way.
    """
    test_dsn = get_test_dsn()
    if test_dsn is None:
        raise RuntimeError("throwaway_db requires a non-production SCIX_TEST_DSN")

    dbname = f"scix_tmp_{uuid.uuid4().hex[:12]}"
    admin = _maintenance_dsn(test_dsn)
    target = _replace_dbname(test_dsn, dbname)

    with psycopg.connect(admin, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{dbname}"')
    try:
        for fname in migrations:
            path = repo_root / "migrations" / fname
            if not path.exists():
                raise AssertionError(f"missing migration file: {fname}")
            result = subprocess.run(
                ["psql", target, "-v", "ON_ERROR_STOP=1", "-f", str(path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise AssertionError(
                    f"failed to apply {fname} to {dbname}:\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
        yield target
    finally:
        with psycopg.connect(admin, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (dbname,),
            )
            cur.execute(f'DROP DATABASE IF EXISTS "{dbname}"')


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
        cur.execute(
            """
            SELECT EXISTS(
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'papers' AND column_name = 'tsv'
            )
        """
        )
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
