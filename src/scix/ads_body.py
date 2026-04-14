"""Loader for the ADS-harvested `body` field into papers_ads_body.

Reads a JSONL file of ADS records (as harvested by scripts/harvest_*.py),
streams the `(bibcode, body, entry_date)` triples, pre-filters unknown
bibcodes via a JOIN to papers, and COPY-loads into papers_ads_body. After
each batch commits, papers_external_ids.has_ads_body is flipped to true for
the loaded bibcodes.

Resumability and progress tracking reuse the project's existing IngestLog
(src/scix/db.py). The CLI lives in scripts/ingest_ads_body.py.

SAFETY:
    * Refuses to target a production DSN unless LoaderConfig.yes_production
      is explicitly set. The guard resolves the effective DSN (falling back to
      scix.db.DEFAULT_DSN when cfg.dsn is None) so a missing --dsn flag cannot
      silently connect to production via SCIX_DSN defaulting.
    * Uses binary COPY into a staging TEMP table + INSERT ... ON CONFLICT so a
      partial load can be resumed. Bibcodes absent from papers are skipped
      (counted in records_skipped) so the FK never fires a row-level error.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from scix.db import (
    DEFAULT_DSN,
    IndexDef,
    IndexManager,
    IngestLog,
    get_connection,
    is_production_dsn,
)
from scix.ingest import open_jsonl

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility with tests/test_ads_body_unit.py
__all__ = [
    "AdsBodyLoader",
    "LoaderConfig",
    "LoaderStats",
    "ProductionGuardError",
    "is_production_dsn",
]


def _redact_dsn(dsn: str) -> str:
    """Return a DSN-safe string for logging: drops passwords and secrets.

    For key=value DSNs, keeps only ``dbname``/``host``/``port``/``user``.
    For URI DSNs, replaces ``user:password@`` with ``user:***@`` and drops
    the query string.
    """
    if "://" in dsn:
        scheme, _, rest = dsn.partition("://")
        # strip query string
        rest = rest.split("?", 1)[0]
        if "@" in rest:
            userinfo, _, host_and_path = rest.partition("@")
            user = userinfo.split(":", 1)[0]
            rest = f"{user}:***@{host_and_path}"
        return f"{scheme}://{rest}"

    safe_keys = {"dbname", "host", "port", "user"}
    parts: list[str] = []
    for token in dsn.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if key.strip() in safe_keys:
                parts.append(f"{key.strip()}={value.strip()}")
    return " ".join(parts) if parts else "<redacted>"


class ProductionGuardError(RuntimeError):
    """Raised when the loader would target a production DSN without explicit opt-in."""


# ---------------------------------------------------------------------------
# Config + stats (both frozen — the loader treats its config as immutable and
# returns an immutable stats record from run()).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoaderConfig:
    """Immutable configuration for the ADS body loader."""

    dsn: str | None
    jsonl_path: Path
    batch_size: int = 10_000
    dry_run: bool = False
    yes_production: bool = False
    drop_indexes: bool = False


@dataclass(frozen=True)
class LoaderStats:
    """Immutable result record returned from AdsBodyLoader.run()."""

    filename: str
    records_loaded: int
    records_skipped: int
    elapsed_seconds: float
    dry_run: bool
    already_complete: bool
    batches: int = 0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


# SQL used by the loader. Kept as module-level string constants so they can be
# audited without reading through the class. No user input is ever
# concatenated into these strings — only parameterized values via %s and
# psycopg.sql identifier quoting.
_STAGING_DDL = (
    "CREATE TEMP TABLE IF NOT EXISTS _ads_body_staging ("
    "bibcode TEXT, body_text TEXT, body_length INT, harvested_at TIMESTAMPTZ"
    ") ON COMMIT DELETE ROWS"
)

_STAGING_COPY = "COPY _ads_body_staging (bibcode, body_text, body_length, harvested_at) FROM STDIN"

# INSERT into papers_ads_body from staging, but only for bibcodes that already
# exist in papers. The JOIN pre-filters unknown bibcodes so the FK never
# rejects a row mid-batch.
_MERGE_SQL = (
    "INSERT INTO papers_ads_body (bibcode, body_text, body_length, harvested_at) "
    "SELECT s.bibcode, s.body_text, s.body_length, s.harvested_at "
    "FROM _ads_body_staging s "
    "JOIN papers p ON p.bibcode = s.bibcode "
    "ON CONFLICT (bibcode) DO UPDATE SET "
    "    body_text = EXCLUDED.body_text, "
    "    body_length = EXCLUDED.body_length, "
    "    harvested_at = EXCLUDED.harvested_at"
)

# Flip has_ads_body=true for every bibcode that made it into papers_ads_body
# in this batch. Upserts into papers_external_ids in case the row does not
# exist yet (W1 bulk populate happens in a separate ops task).
_FLIP_FLAG_SQL = (
    "INSERT INTO papers_external_ids (bibcode, has_ads_body) "
    "SELECT bibcode, true FROM _ads_body_staging s "
    "WHERE EXISTS (SELECT 1 FROM papers p WHERE p.bibcode = s.bibcode) "
    "ON CONFLICT (bibcode) DO UPDATE SET has_ads_body = true"
)

# Count of rows in the staging batch whose bibcode is known to papers. Used
# together with rowcount to compute records_skipped accurately.
_COUNT_KNOWN_SQL = (
    "SELECT COUNT(*) FROM _ads_body_staging s "
    "WHERE EXISTS (SELECT 1 FROM papers p WHERE p.bibcode = s.bibcode)"
)


_LoaderRow = tuple[str, str, int, datetime]


class AdsBodyLoader:
    """Stream a JSONL file of ADS records into papers_ads_body.

    See module docstring for safety guarantees and SQL layout.
    """

    def __init__(self, config: LoaderConfig) -> None:
        self._cfg = config

    # -- Public entry point --------------------------------------------------

    def run(self) -> LoaderStats:
        self._check_production_guard()

        if not self._cfg.jsonl_path.exists():
            raise FileNotFoundError(f"ADS body JSONL not found: {self._cfg.jsonl_path}")

        filename = self._cfg.jsonl_path.name
        t_start = time.monotonic()

        conn = get_connection(self._cfg.dsn)
        try:
            ingest_log = IngestLog(conn)

            if not self._cfg.dry_run and ingest_log.is_complete(filename):
                logger.info("Skipping %s (already complete per ingest_log)", filename)
                return LoaderStats(
                    filename=filename,
                    records_loaded=0,
                    records_skipped=0,
                    elapsed_seconds=time.monotonic() - t_start,
                    dry_run=self._cfg.dry_run,
                    already_complete=True,
                )

            if not self._cfg.dry_run:
                ingest_log.start(filename)

            saved_indexes: list[IndexDef] = []
            if self._cfg.drop_indexes and not self._cfg.dry_run:
                idx_mgr = IndexManager(conn, "papers_ads_body")
                saved_indexes = idx_mgr.drop_indexes()
                logger.info(
                    "Dropped %d indexes on papers_ads_body for bulk load",
                    len(saved_indexes),
                )

            try:
                records_loaded, records_skipped, batches = self._stream_file(
                    conn=conn, filename=filename, ingest_log=ingest_log
                )
            except Exception:
                logger.exception("ADS body load failed: %s", filename)
                conn.rollback()
                if not self._cfg.dry_run:
                    ingest_log.mark_failed(filename)
                raise
            finally:
                if saved_indexes:
                    logger.info("Recreating %d indexes on papers_ads_body", len(saved_indexes))
                    IndexManager(conn, "papers_ads_body").recreate_indexes(saved_indexes)

            if not self._cfg.dry_run:
                ingest_log.update_counts(filename, records_loaded, records_skipped, edges=0)
                ingest_log.finish(filename)

            elapsed = time.monotonic() - t_start
            logger.info(
                "%s: loaded=%d skipped=%d batches=%d elapsed=%.2fs dry_run=%s",
                filename,
                records_loaded,
                records_skipped,
                batches,
                elapsed,
                self._cfg.dry_run,
            )

            return LoaderStats(
                filename=filename,
                records_loaded=records_loaded,
                records_skipped=records_skipped,
                elapsed_seconds=elapsed,
                dry_run=self._cfg.dry_run,
                already_complete=False,
                batches=batches,
            )
        finally:
            conn.close()

    # -- Internals -----------------------------------------------------------

    def _check_production_guard(self) -> None:
        """Refuse to run against production unless explicitly authorized.

        Resolves the same effective DSN that ``get_connection`` will use
        (cfg.dsn → DEFAULT_DSN → libpq defaults). Without this fallback, a
        caller that omits ``--dsn`` would bypass the guard: the cfg value
        would be None, the guard would see an empty string, and the actual
        connection would fall through to ``DEFAULT_DSN = "dbname=scix"``.

        Raises ProductionGuardError with a DSN-redacted message so passwords
        in the DSN never leak into logs or error messages.
        """
        effective_dsn = self._cfg.dsn or DEFAULT_DSN
        if is_production_dsn(effective_dsn) and not self._cfg.yes_production:
            raise ProductionGuardError(
                "Refusing to run ADS body loader against production DSN "
                f"({_redact_dsn(effective_dsn)}). Pass yes_production=True "
                "(or --yes-production on the CLI) to override. See "
                "CLAUDE.md §Testing — Database Safety."
            )

    def _stream_file(
        self,
        conn: psycopg.Connection,
        filename: str,
        ingest_log: IngestLog,
    ) -> tuple[int, int, int]:
        """Stream the JSONL file in batches. Returns (loaded, skipped, batches)."""
        total_loaded = 0
        total_skipped = 0
        batches = 0
        batch: list[_LoaderRow] = []

        def flush() -> None:
            nonlocal total_loaded, total_skipped, batches
            if not batch:
                return
            loaded, skipped = self._flush_batch(conn, batch)
            total_loaded += loaded
            total_skipped += skipped
            batches += 1
            batch.clear()

        with open_jsonl(self._cfg.jsonl_path) as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parsed = _parse_record(line, filename=filename, line_no=line_no)
                if parsed is None:
                    total_skipped += 1
                    continue
                batch.append(parsed)
                if len(batch) >= self._cfg.batch_size:
                    flush()
                    if not self._cfg.dry_run:
                        ingest_log.update_counts(filename, total_loaded, total_skipped, edges=0)

        flush()
        return total_loaded, total_skipped, batches

    def _flush_batch(self, conn: psycopg.Connection, rows: list[_LoaderRow]) -> tuple[int, int]:
        """COPY+merge one batch. Returns (loaded, skipped_unknown_bibcodes)."""
        if not rows:
            return 0, 0

        with conn.cursor() as cur:
            cur.execute(_STAGING_DDL)
            with cur.copy(_STAGING_COPY) as copy:
                for row in rows:
                    copy.write_row(row)

            # Count staging rows whose bibcode is known to papers so we can
            # report skipped-unknown accurately, independent of any
            # pre-existing papers_ads_body rows (which affect rowcount under
            # ON CONFLICT DO UPDATE).
            cur.execute(_COUNT_KNOWN_SQL)
            known_row = cur.fetchone()
            known = int(known_row[0]) if known_row is not None else 0

            if self._cfg.dry_run:
                conn.rollback()
                return known, len(rows) - known

            cur.execute(_MERGE_SQL)
            cur.execute(_FLIP_FLAG_SQL)

        conn.commit()
        return known, len(rows) - known


# ---------------------------------------------------------------------------
# Record parsing helpers (module-level, no side effects, easy to unit test)
# ---------------------------------------------------------------------------


def _parse_record(
    line: str, *, filename: str, line_no: int
) -> tuple[str, str, int, datetime] | None:
    """Parse a single JSONL line into a loader row, or None if it must be skipped.

    Skips records with:
        - invalid JSON (logged as a warning);
        - missing bibcode;
        - missing, null, empty, or whitespace-only body.

    harvested_at is derived from `entry_date` if present, else the Unix epoch
    (never fails the load because a single record has no timestamp).
    """
    try:
        rec = json.loads(line)
    except json.JSONDecodeError as exc:
        logger.warning("%s line %d: invalid JSON: %s", filename, line_no, exc)
        return None

    if not isinstance(rec, dict):
        logger.warning("%s line %d: record is not an object", filename, line_no)
        return None

    bibcode = rec.get("bibcode")
    if not isinstance(bibcode, str) or not bibcode:
        return None

    body = rec.get("body")
    if not isinstance(body, str) or not body.strip():
        return None

    harvested_at = _parse_entry_date(rec.get("entry_date"))
    return (bibcode, body, len(body), harvested_at)


def _parse_entry_date(raw: object) -> datetime:
    """Parse ADS entry_date into a tz-aware datetime, falling back to epoch."""
    if isinstance(raw, str) and raw:
        # ADS uses ISO-8601-ish dates, sometimes without timezone. Accept both.
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return datetime(1970, 1, 1, tzinfo=timezone.utc)
