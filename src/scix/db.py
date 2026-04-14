"""Database helpers: connection, index management, and ingestion log."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

logger = logging.getLogger(__name__)

DEFAULT_DSN = os.environ.get("SCIX_DSN", "dbname=scix")

_PRODUCTION_DB_NAMES: frozenset[str] = frozenset({"scix"})


def is_production_dsn(dsn: str | None) -> bool:
    """Return True if DSN appears to point at a production database.

    Delegates parsing to libpq via ``psycopg.conninfo.conninfo_to_dict`` so
    key=value, URI (``postgresql://``, ``postgres://``), and spaces-around-=
    variants are all handled uniformly. Returns False for empty/None inputs —
    callers must resolve the effective DSN (via ``DEFAULT_DSN`` fallback)
    BEFORE calling this, or an unset dsn will silently slip past the guard.
    """
    if not dsn:
        return False
    try:
        params = conninfo_to_dict(dsn)
    except psycopg.ProgrammingError:
        return False
    dbname = params.get("dbname")
    return isinstance(dbname, str) and dbname in _PRODUCTION_DB_NAMES


def get_connection(dsn: str | None = None, autocommit: bool = False) -> psycopg.Connection:
    """Open a connection to the scix database."""
    conn = psycopg.connect(dsn or DEFAULT_DSN)
    conn.autocommit = autocommit
    return conn


# ---------------------------------------------------------------------------
# pgvector version detection and iterative scan support
# ---------------------------------------------------------------------------


def get_pgvector_version(conn: psycopg.Connection) -> tuple[int, ...] | None:
    """Return the installed pgvector version as a tuple, e.g. (0, 8, 0).

    Returns None if the vector extension is not installed.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
        if row is None:
            return None
        return tuple(int(x) for x in row[0].split("."))


def supports_iterative_scan(conn: psycopg.Connection) -> bool:
    """Check if the installed pgvector supports iterative index scans (>=0.8.0)."""
    version = get_pgvector_version(conn)
    if version is None:
        return False
    return version >= (0, 8, 0)


IterativeScanMode = Literal["off", "relaxed_order", "strict_order"]


def configure_iterative_scan(
    conn: psycopg.Connection,
    mode: IterativeScanMode = "relaxed_order",
) -> bool:
    """Enable iterative index scans for the current transaction (pgvector 0.8.0+).

    Uses SET LOCAL so the setting applies only to the current transaction and
    is automatically reverted on COMMIT/ROLLBACK.

    Args:
        conn: Database connection (must NOT be in autocommit mode).
        mode: Scan mode — "relaxed_order" (best recall), "strict_order"
              (exact ordering), or "off" (disable).

    Returns True if the setting was applied, False if pgvector < 0.8.0.
    """
    if not supports_iterative_scan(conn):
        logger.debug("pgvector < 0.8.0 — iterative scan not available, skipping")
        return False

    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL hnsw.iterative_scan = {mode}")

    logger.debug("Enabled iterative scan mode: %s", mode)
    return True


@dataclass(frozen=True)
class IndexDef:
    """A stored index definition for drop/recreate cycles."""

    name: str
    table: str
    definition: str  # full CREATE INDEX statement


class IndexManager:
    """Drop and recreate non-PK indexes on a table for bulk load performance."""

    def __init__(self, conn: psycopg.Connection, table: str = "papers") -> None:
        self._conn = conn
        self._table = table

    def get_non_pk_indexes(self) -> list[IndexDef]:
        """Read all non-primary-key index definitions from pg_indexes."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.indexname, i.indexdef
                FROM pg_indexes i
                LEFT JOIN pg_constraint c
                    ON c.conname = i.indexname AND c.conrelid = i.tablename::regclass
                WHERE i.tablename = %s
                  AND i.schemaname = 'public'
                  AND (c.contype IS NULL OR c.contype != 'p')
                """,
                (self._table,),
            )
            return [
                IndexDef(name=row[0], table=self._table, definition=row[1])
                for row in cur.fetchall()
            ]

    def drop_indexes(self) -> list[IndexDef]:
        """Drop all non-PK indexes on the table. Returns definitions for recreate."""
        indexes = self.get_non_pk_indexes()
        with self._conn.cursor() as cur:
            for idx in indexes:
                logger.info("Dropping index %s", idx.name)
                cur.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(idx.name)))
        self._conn.commit()
        return indexes

    def recreate_indexes(self, indexes: list[IndexDef]) -> None:
        """Recreate indexes from stored definitions."""
        with self._conn.cursor() as cur:
            for idx in indexes:
                logger.info("Creating index %s", idx.name)
                cur.execute(idx.definition)
        self._conn.commit()


class IngestLog:
    """Track ingestion progress per file for resumability."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def is_complete(self, filename: str) -> bool:
        """Check if a file has already been fully ingested."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM ingest_log WHERE filename = %s",
                (filename,),
            )
            row = cur.fetchone()
            return row is not None and row[0] == "complete"

    def start(self, filename: str) -> None:
        """Mark a file as in-progress. Resets counters if re-ingesting."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingest_log (filename, status, started_at)
                VALUES (%s, 'in_progress', NOW())
                ON CONFLICT (filename) DO UPDATE SET
                    status = 'in_progress',
                    records_loaded = 0,
                    errors_skipped = 0,
                    edges_loaded = 0,
                    started_at = NOW(),
                    finished_at = NULL
                """,
                (filename,),
            )
        self._conn.commit()

    def update_counts(self, filename: str, records: int, errors: int, edges: int) -> None:
        """Update cumulative counts for a file."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingest_log SET
                    records_loaded = %s,
                    errors_skipped = %s,
                    edges_loaded = %s
                WHERE filename = %s
                """,
                (records, errors, edges, filename),
            )
        self._conn.commit()

    def finish(self, filename: str) -> None:
        """Mark a file as complete."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingest_log SET
                    status = 'complete',
                    finished_at = NOW()
                WHERE filename = %s
                """,
                (filename,),
            )
        self._conn.commit()

    def mark_failed(self, filename: str) -> None:
        """Mark a file as failed."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingest_log SET
                    status = 'failed',
                    finished_at = NOW()
                WHERE filename = %s
                """,
                (filename,),
            )
        self._conn.commit()
