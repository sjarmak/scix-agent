"""Ingestion pipeline: stream JSONL files into PostgreSQL papers + citation_edges."""

from __future__ import annotations

import gzip
import json
import logging
import lzma
import time
from pathlib import Path
from typing import IO, Any

import psycopg

from scix.db import IndexManager, IngestLog, get_connection
from scix.field_mapping import COLUMN_ORDER, transform_record

logger = logging.getLogger(__name__)

# SQL column list for COPY
_COPY_COLS = ", ".join(COLUMN_ORDER)

# Papers staging: COPY into temp table, then merge with ON CONFLICT DO NOTHING.
# This makes retry safe — duplicate bibcodes from a partially-ingested file are skipped.
_PAPER_STAGING_DDL = (
    "CREATE TEMP TABLE IF NOT EXISTS _paper_staging "
    f"(LIKE papers INCLUDING DEFAULTS) ON COMMIT DELETE ROWS"
)
_PAPER_STAGING_COPY = f"COPY _paper_staging ({_COPY_COLS}) FROM STDIN"
_PAPER_MERGE_SQL = (
    f"INSERT INTO papers ({_COPY_COLS}) "
    f"SELECT {_COPY_COLS} FROM _paper_staging "
    "ON CONFLICT (bibcode) DO NOTHING"
)

_EDGE_STAGING_DDL = (
    "CREATE TEMP TABLE IF NOT EXISTS _edge_staging "
    "(source_bibcode TEXT, target_bibcode TEXT) ON COMMIT DELETE ROWS"
)
_EDGE_MERGE_SQL = (
    "INSERT INTO citation_edges (source_bibcode, target_bibcode) "
    "SELECT DISTINCT source_bibcode, target_bibcode FROM _edge_staging "
    "ON CONFLICT DO NOTHING"
)


def open_jsonl(filepath: Path) -> IO[str]:
    """Open a JSONL file, auto-detecting compression from extension."""
    name = filepath.name
    if name.endswith(".jsonl.xz"):
        return lzma.open(filepath, "rt", encoding="utf-8")
    elif name.endswith(".jsonl.gz"):
        return gzip.open(filepath, "rt", encoding="utf-8")
    elif name.endswith(".jsonl"):
        return open(filepath, "r", encoding="utf-8")
    else:
        raise ValueError(f"Unknown file format: {name}")


def discover_files(data_dir: Path) -> list[Path]:
    """Find all JSONL files in the data directory, sorted by name."""
    patterns = ["*.jsonl", "*.jsonl.gz", "*.jsonl.xz"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(data_dir.glob(pattern))
    return sorted(set(files))


class IngestPipeline:
    """Stream JSONL files into PostgreSQL papers and citation_edges tables."""

    def __init__(
        self,
        data_dir: Path,
        dsn: str | None = None,
        batch_size: int = 10_000,
    ) -> None:
        self._data_dir = data_dir
        self._dsn = dsn
        self._batch_size = batch_size

    def run(self, drop_indexes: bool = True, single_file: Path | None = None) -> None:
        """Full ingestion: discover files, optionally drop indexes, ingest, recreate."""
        conn = get_connection(self._dsn)
        try:
            ingest_log = IngestLog(conn)

            if single_file:
                files = [single_file]
            else:
                files = discover_files(self._data_dir)

            if not files:
                logger.warning("No JSONL files found in %s", self._data_dir)
                return

            logger.info("Found %d file(s) to process", len(files))

            # Drop indexes for bulk load performance
            saved_indexes: list[Any] = []
            if drop_indexes and not single_file:
                idx_mgr = IndexManager(conn, "papers")
                saved_indexes = idx_mgr.drop_indexes()
                logger.info("Dropped %d indexes for bulk load", len(saved_indexes))

            for filepath in files:
                filename = filepath.name
                if ingest_log.is_complete(filename):
                    logger.info("Skipping %s (already complete)", filename)
                    continue
                self._ingest_file(conn, filepath, ingest_log)

            # Recreate indexes
            if saved_indexes:
                logger.info("Recreating %d indexes...", len(saved_indexes))
                idx_mgr.recreate_indexes(saved_indexes)
                logger.info("Index recreation complete")
        finally:
            conn.close()

    def _ingest_file(
        self, conn: psycopg.Connection, filepath: Path, ingest_log: IngestLog
    ) -> None:
        """Stream-ingest a single JSONL file."""
        filename = filepath.name
        logger.info("Starting ingestion: %s", filename)
        ingest_log.start(filename)

        total_records = 0
        total_errors = 0
        total_edges = 0
        paper_batch: list[tuple[Any, ...]] = []
        edge_batch: list[tuple[str, str]] = []
        t_start = time.monotonic()

        try:
            with open_jsonl(filepath) as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        rec = json.loads(line)
                        row, edges = transform_record(rec)
                        paper_batch.append(row)
                        edge_batch.extend(edges)
                    except (json.JSONDecodeError, ValueError) as e:
                        total_errors += 1
                        logger.warning(
                            "%s line %d: %s", filename, line_no, e
                        )
                        continue

                    if len(paper_batch) >= self._batch_size:
                        inserted, edge_count = self._flush_batch(
                            conn, paper_batch, edge_batch
                        )
                        total_records += inserted
                        total_edges += edge_count
                        paper_batch.clear()
                        edge_batch.clear()

                        elapsed = time.monotonic() - t_start
                        rate = total_records / elapsed if elapsed > 0 else 0
                        logger.info(
                            "%s: %d records (%d edges, %d errors) %.0f rec/s",
                            filename,
                            total_records,
                            total_edges,
                            total_errors,
                            rate,
                        )
                        ingest_log.update_counts(
                            filename, total_records, total_errors, total_edges
                        )

            # Flush remaining
            if paper_batch:
                inserted, edge_count = self._flush_batch(
                    conn, paper_batch, edge_batch
                )
                total_records += inserted
                total_edges += edge_count

            elapsed = time.monotonic() - t_start
            rate = total_records / elapsed if elapsed > 0 else 0
            logger.info(
                "%s complete: %d records, %d edges, %d errors, %.0f rec/s (%.1fs)",
                filename,
                total_records,
                total_edges,
                total_errors,
                rate,
                elapsed,
            )
            ingest_log.update_counts(
                filename, total_records, total_errors, total_edges
            )
            ingest_log.finish(filename)

        except Exception:
            logger.exception("Failed ingesting %s", filename)
            conn.rollback()  # Clear error state so mark_failed can execute
            ingest_log.mark_failed(filename)
            raise

    def _flush_batch(
        self,
        conn: psycopg.Connection,
        paper_rows: list[tuple[Any, ...]],
        edge_rows: list[tuple[str, str]],
    ) -> tuple[int, int]:
        """COPY paper rows and merge edges. Returns (papers_inserted, edges_inserted)."""
        papers_inserted = 0
        edges_inserted = 0

        with conn.cursor() as cur:
            # Papers via staging table (idempotent: ON CONFLICT DO NOTHING)
            cur.execute(_PAPER_STAGING_DDL)
            with cur.copy(_PAPER_STAGING_COPY) as copy:
                for row in paper_rows:
                    copy.write_row(row)
            cur.execute(_PAPER_MERGE_SQL)
            papers_inserted = cur.rowcount

            # Edges via staging table
            if edge_rows:
                cur.execute(_EDGE_STAGING_DDL)
                with cur.copy("COPY _edge_staging FROM STDIN") as copy:
                    for edge in edge_rows:
                        copy.write_row(edge)
                cur.execute(_EDGE_MERGE_SQL)
                edges_inserted = cur.rowcount

        conn.commit()
        return papers_inserted, edges_inserted
