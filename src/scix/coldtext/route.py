"""Read-side routing to NAS cold-text shards (ADR-016 Phase 1a).

When a sealed-year ``papers_fulltext`` row is no longer in Postgres, fetch its
structured ``sections`` from the per-year NAS shard instead. The route is
triggered by a Postgres miss, so it is a no-op until the sealed rows are dropped
(Phase 1b) — which lets us land and test it ahead of the destructive step.

Shard readers are cached per year. Shards are read-only/immutable, so a reader
is safe to keep open and share across the MCP server's request threads.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path

import psycopg

from scix.coldtext.shard import ColdTextReader, shard_path

logger = logging.getLogger(__name__)

DEFAULT_COLDTEXT_ROOT = "/mnt/scix_coldtext/v1"

# ADR-016 hot window: years >= this stay full-fat in Postgres; older years are
# sealed to NAS. Shared so the seal driver, the read route, and the fulltext
# populator all agree on the boundary.
HOT_WINDOW_START_YEAR = 2025

# NULL-year rows seal into the year-0 sentinel shard (see seal_fulltext_to_nas).
_NULL_YEAR = 0

# Cache keyed by year: a reader (shard present), or None (shard legitimately
# absent for that year — e.g. a hot year with no sealed shard). Transient NFS
# failures are NOT cached, so they self-heal on the next call. Guarded by a lock
# because the MCP server populates it from concurrent request threads.
_readers: dict[int, ColdTextReader | None] = {}
_readers_lock = threading.Lock()


def coldtext_root() -> Path | None:
    """Configured cold-text root, or ``None`` if it isn't present.

    ``None`` means "no cold tier" — callers degrade gracefully (Postgres-only),
    which is the correct behavior before Phase 1b and on hosts without the NAS
    mount.
    """
    root = Path(os.environ.get("SCIX_COLDTEXT_ROOT", DEFAULT_COLDTEXT_ROOT))
    return root if root.is_dir() else None


def _reader_for_year(year: int) -> ColdTextReader | None:
    """Cached read-only reader for a year's shard, or ``None`` if unavailable.

    A confirmed-absent shard (root present, file missing) is cached as ``None``.
    A missing mount or a transient open error returns ``None`` *without* caching,
    so the cold tier self-heals once the NAS is reachable.
    """
    if year in _readers:  # fast path, no lock
        return _readers[year]
    with _readers_lock:
        if year in _readers:  # re-check under lock
            return _readers[year]
        root = coldtext_root()
        if root is None:
            return None  # no mount — don't cache, may appear later
        path = shard_path(root, year)
        try:
            if not path.exists():
                _readers[year] = None  # legit absence for this year
                return None
            reader = ColdTextReader(path)
        except (OSError, sqlite3.Error) as exc:
            logger.warning("cold-text shard unavailable for year %d: %s", year, exc)
            return None  # transient — don't cache, retry next call
        _readers[year] = reader
        return reader


def _resolve_year(conn: psycopg.Connection, bibcode: str) -> int | None:
    """Publication year for ``bibcode`` (NULL → year-0 sentinel), or ``None``
    if the paper is unknown to Postgres."""
    with conn.cursor() as cur:
        cur.execute("SELECT year FROM papers WHERE bibcode = %s", (bibcode,))
        row = cur.fetchone()
    if row is None:
        return None
    return _NULL_YEAR if row[0] is None else int(row[0])


def fetch_sections_cold(conn: psycopg.Connection, bibcode: str) -> object | None:
    """``sections`` JSON for ``bibcode`` from its NAS shard, or ``None``.

    ``None`` whenever the cold tier can't serve it (no mount, no shard for the
    year, or the bibcode isn't in the shard) — the caller then falls back to its
    existing behavior.
    """
    year = _resolve_year(conn, bibcode)
    if year is None:
        return None
    reader = _reader_for_year(year)
    if reader is None:
        return None
    return reader.fetch_sections(bibcode)


def clear_cache() -> None:
    """Close and drop all cached readers (test isolation / config reload)."""
    with _readers_lock:
        for reader in _readers.values():
            if reader is not None:
                reader.close()
        _readers.clear()
