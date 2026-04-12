"""Write-path helpers for the ``document_entities_canonical`` materialized view.

This module is the *write* side of the fused-confidence MV (M8):

* :func:`mark_dirty` flips the ``fusion_mv_state.dirty`` bit so a downstream
  refresher knows there is work to do. Called by linker writes that touch
  ``document_entities``.
* :func:`refresh_if_due` checks the dirty bit and the last refresh timestamp
  and, if at least ``min_interval_seconds`` have passed, runs
  ``REFRESH MATERIALIZED VIEW CONCURRENTLY document_entities_canonical``.
  This function is intended to be driven by a cron / loop and rate-limits
  refreshes to at most one per ``min_interval_seconds`` (default: 1 hour).

The module deliberately does NOT SELECT against the fused MV itself — the
M13 resolver lint (``scripts/ast_lint_resolver.py``) forbids such reads
outside ``src/scix/resolve_entities.py``. Callers that need to read the
MV go through M13.
"""

from __future__ import annotations

import logging

import psycopg

from .db import DEFAULT_DSN

logger = logging.getLogger(__name__)

# REFRESH cannot run inside a transaction block, so this SQL is always
# executed on an autocommit connection.
_REFRESH_SQL = "REFRESH MATERIALIZED VIEW CONCURRENTLY document_entities_canonical"


def _resolve_dsn(conn: psycopg.Connection | None) -> str:
    """Return the DSN to open a fresh autocommit connection against.

    If the caller supplied a connection, reuse its connection parameters so
    the fresh connection hits the same database. Otherwise fall back to the
    process-wide default.
    """
    if conn is None:
        return DEFAULT_DSN
    params = conn.info.get_parameters()
    # psycopg's dsn_parameters excludes the password; we only need
    # host/port/dbname/user for an autocommit sibling connection.
    parts = []
    for key in ("host", "hostaddr", "port", "dbname", "user"):
        value = params.get(key)
        if value:
            parts.append(f"{key}={value}")
    return " ".join(parts) if parts else DEFAULT_DSN


def mark_dirty(conn: psycopg.Connection | None = None) -> None:
    """Mark the fusion MV as dirty so the next refresh will fire.

    Idempotent: calling this repeatedly is cheap. Safe to call inside an
    existing transaction — uses the caller's connection if supplied and
    leaves commit/rollback to the caller. If no connection is supplied,
    opens a short-lived one and commits before returning.
    """
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fusion_mv_state (id, dirty, last_refresh_at)
                VALUES (1, true, NULL)
                ON CONFLICT (id) DO UPDATE SET dirty = true
                """)
        return

    with psycopg.connect(DEFAULT_DSN) as owned:
        with owned.cursor() as cur:
            cur.execute("""
                INSERT INTO fusion_mv_state (id, dirty, last_refresh_at)
                VALUES (1, true, NULL)
                ON CONFLICT (id) DO UPDATE SET dirty = true
                """)
        owned.commit()


def refresh_if_due(
    conn: psycopg.Connection | None = None,
    min_interval_seconds: int = 3600,
) -> bool:
    """Refresh the fusion MV if dirty and ``min_interval_seconds`` have elapsed.

    Returns
    -------
    bool
        ``True`` if a refresh was executed, ``False`` otherwise (not dirty,
        or rate-limited).

    Notes
    -----
    * Runs ``REFRESH MATERIALIZED VIEW CONCURRENTLY`` which requires a
      UNIQUE index on the MV — see migration 033. CONCURRENTLY cannot run
      inside a transaction, so this function always opens its own
      autocommit connection (even when the caller supplies ``conn``),
      using the same DSN as the caller's connection when possible.
    * If the caller supplies ``conn``, any pending transaction on it is
      committed first so the state update the refresh writes is visible
      to subsequent reads on ``conn``.
    """
    if min_interval_seconds < 0:
        raise ValueError(f"min_interval_seconds must be >= 0, got {min_interval_seconds}")

    dsn = _resolve_dsn(conn)
    # Flush the caller's pending work so our autocommit sibling sees the
    # latest dirty bit / linker writes.
    if conn is not None:
        conn.commit()

    with psycopg.connect(dsn, autocommit=True) as owned:
        with owned.cursor() as cur:
            # Ensure a state row exists so the first-ever call can proceed.
            cur.execute("""
                INSERT INTO fusion_mv_state (id, dirty, last_refresh_at)
                VALUES (1, true, NULL)
                ON CONFLICT (id) DO NOTHING
                """)
            cur.execute(
                """
                SELECT
                    dirty,
                    last_refresh_at,
                    (last_refresh_at IS NULL)
                        OR (now() - last_refresh_at >= make_interval(secs => %s))
                        AS interval_elapsed
                FROM fusion_mv_state
                WHERE id = 1
                """,
                (min_interval_seconds,),
            )
            row = cur.fetchone()

            if row is None:
                logger.warning("fusion_mv_state row missing after upsert; skipping refresh")
                return False

            dirty, last_refresh_at, interval_elapsed = row
            if not dirty:
                logger.debug("fusion MV not dirty; skipping refresh")
                return False
            if not interval_elapsed:
                logger.debug(
                    "fusion MV refresh rate-limited (last_refresh_at=%s, min_interval=%ss)",
                    last_refresh_at,
                    min_interval_seconds,
                )
                return False

            logger.info("Refreshing document_entities_canonical (CONCURRENTLY)")
            cur.execute(_REFRESH_SQL)
            cur.execute("""
                UPDATE fusion_mv_state
                   SET dirty = false,
                       last_refresh_at = now()
                 WHERE id = 1
                """)
            return True
