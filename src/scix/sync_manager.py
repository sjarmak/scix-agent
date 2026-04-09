"""Incremental sync manager: track harvest freshness and trigger refreshes.

Queries the ``harvest_runs`` table to determine when each source was last
synced, compares against a configurable cadence, and reports staleness.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

# Default cadence per source (hours).  Override via environment variable
# ``SCIX_SYNC_CADENCE_<SOURCE>=<hours>`` (e.g. ``SCIX_SYNC_CADENCE_SPDF=48``).
DEFAULT_CADENCE_HOURS = 24


@dataclass(frozen=True)
class SourceStatus:
    """Freshness status for a single harvest source."""

    source: str
    last_sync: datetime | None
    cadence: timedelta
    is_stale: bool
    next_sync: datetime | None


def _cadence_for(source: str) -> timedelta:
    """Return the configured cadence for a source.

    Checks ``SCIX_SYNC_CADENCE_<SOURCE>`` env var (value in hours),
    falls back to ``DEFAULT_CADENCE_HOURS``.
    """
    env_key = f"SCIX_SYNC_CADENCE_{source.upper()}"
    hours = int(os.environ.get(env_key, DEFAULT_CADENCE_HOURS))
    return timedelta(hours=hours)


def get_last_sync(
    conn: psycopg.Connection,
    source: str,
) -> datetime | None:
    """Return the ``finished_at`` of the most recent completed harvest for a source.

    Returns ``None`` if no completed run exists.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT finished_at
            FROM harvest_runs
            WHERE source = %s AND status = 'completed'
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            (source,),
        )
        row = cur.fetchone()
        return row[0] if row else None


def needs_refresh(
    conn: psycopg.Connection,
    source: str,
) -> bool:
    """Return True if the source is stale (last sync older than cadence)."""
    last = get_last_sync(conn, source)
    if last is None:
        return True
    cadence = _cadence_for(source)
    return datetime.now(timezone.utc) - last > cadence


def get_all_sources(conn: psycopg.Connection) -> list[str]:
    """Return all distinct source names from harvest_runs."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT source FROM harvest_runs ORDER BY source")
        return [row[0] for row in cur.fetchall()]


def sync_status(
    conn: psycopg.Connection,
    sources: list[str] | None = None,
) -> list[SourceStatus]:
    """Return freshness status for all (or specified) sources.

    Args:
        conn: Database connection.
        sources: Optional list of sources to check. If None, checks all
            sources that have at least one harvest_run.

    Returns:
        List of SourceStatus sorted by source name.
    """
    if sources is None:
        sources = get_all_sources(conn)

    results: list[SourceStatus] = []
    now = datetime.now(timezone.utc)

    for source in sorted(sources):
        last = get_last_sync(conn, source)
        cadence = _cadence_for(source)
        is_stale = last is None or (now - last > cadence)
        next_sync = (last + cadence) if last is not None else None

        results.append(
            SourceStatus(
                source=source,
                last_sync=last,
                cadence=cadence,
                is_stale=is_stale,
                next_sync=next_sync,
            )
        )

    return results


def format_sync_status(statuses: list[SourceStatus]) -> str:
    """Format sync status as a human-readable table.

    Returns a multi-line string suitable for CLI output.
    """
    if not statuses:
        return "No harvest sources found."

    lines = [f"{'Source':<20} {'Last Sync':<22} {'Cadence':<10} {'Status':<8} {'Next Sync':<22}"]
    lines.append("-" * 82)

    for s in statuses:
        last_str = s.last_sync.strftime("%Y-%m-%d %H:%M UTC") if s.last_sync else "never"
        cadence_str = f"{int(s.cadence.total_seconds() // 3600)}h"
        status_str = "STALE" if s.is_stale else "OK"
        next_str = s.next_sync.strftime("%Y-%m-%d %H:%M UTC") if s.next_sync else "now"
        lines.append(f"{s.source:<20} {last_str:<22} {cadence_str:<10} {status_str:<8} {next_str:<22}")

    return "\n".join(lines)
