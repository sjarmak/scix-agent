"""Materialized view refresh helpers for agent context views."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

import psycopg

logger = logging.getLogger(__name__)

# Ordered list of agent context views.  Refresh order matters when views
# depend on each other — document first (base data), entity second
# (references documents), dataset last (references both).
AGENT_VIEWS: Sequence[str] = (
    "agent_document_context",
    "agent_entity_context",
    "agent_dataset_context",
)


@dataclass(frozen=True)
class RefreshResult:
    """Outcome of a single materialized view refresh."""

    view_name: str
    duration_s: float
    success: bool
    error: str | None = None


def refresh_view(conn: psycopg.Connection, name: str) -> RefreshResult:
    """Refresh a single materialized view concurrently.

    Requires the view to have a unique index (all agent views do).
    The connection must be in autocommit mode because
    ``REFRESH MATERIALIZED VIEW CONCURRENTLY`` cannot run inside a
    transaction block.

    Args:
        conn: Database connection (**must** have ``autocommit=True``).
        name: Materialized view name (unqualified, ``public`` schema assumed).

    Returns:
        A :class:`RefreshResult` with timing and success/failure info.
    """
    start = time.monotonic()
    try:
        with conn.cursor() as cur:
            # SET per-session parallelism for index maintenance during refresh.
            cur.execute("SET max_parallel_maintenance_workers = 7")
            cur.execute("SET maintenance_work_mem = '4GB'")
            cur.execute(
                psycopg.sql.SQL("REFRESH MATERIALIZED VIEW CONCURRENTLY {}").format(
                    psycopg.sql.Identifier(name)
                )
            )
        elapsed = time.monotonic() - start
        logger.info("Refreshed %s in %.2fs", name, elapsed)
        return RefreshResult(view_name=name, duration_s=elapsed, success=True)
    except psycopg.errors.ObjectNotInPrerequisiteState as exc:
        # View exists but has no unique index — CONCURRENTLY not possible.
        elapsed = time.monotonic() - start
        msg = f"Cannot refresh concurrently (missing unique index): {exc}"
        logger.error("Failed to refresh %s: %s", name, msg)
        return RefreshResult(view_name=name, duration_s=elapsed, success=False, error=msg)
    except psycopg.errors.UndefinedTable as exc:
        elapsed = time.monotonic() - start
        msg = f"View does not exist: {exc}"
        logger.error("Failed to refresh %s: %s", name, msg)
        return RefreshResult(view_name=name, duration_s=elapsed, success=False, error=msg)
    except psycopg.Error as exc:
        elapsed = time.monotonic() - start
        msg = str(exc)
        logger.error("Failed to refresh %s: %s", name, msg)
        return RefreshResult(view_name=name, duration_s=elapsed, success=False, error=msg)


def refresh_all_views(conn: psycopg.Connection) -> list[RefreshResult]:
    """Refresh all agent context views in dependency order.

    Args:
        conn: Database connection (**must** have ``autocommit=True``).

    Returns:
        List of :class:`RefreshResult` — one per view, in refresh order.
    """
    results: list[RefreshResult] = []
    total_start = time.monotonic()

    for view_name in AGENT_VIEWS:
        result = refresh_view(conn, view_name)
        results.append(result)

    total_elapsed = time.monotonic() - total_start
    ok = sum(1 for r in results if r.success)
    logger.info(
        "Refreshed %d/%d views in %.2fs",
        ok,
        len(results),
        total_elapsed,
    )
    return results
