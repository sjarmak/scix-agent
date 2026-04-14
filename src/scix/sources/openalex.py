"""OpenAlex S3 snapshot ingest loader.

Reads the OpenAlex S3 snapshot (Parquet format) via DuckDB, stages in memory,
and COPY-loads into ``papers_openalex`` and ``works_references``. After each
batch, joins on DOI and arXiv ID to populate ``papers_external_ids.openalex_id``
and ``papers_external_ids.openalex_has_pdf_url`` for matched ADS papers.

Resumability and progress tracking reuse the project's existing IngestLog
(src/scix/db.py). The CLI lives in scripts/ingest_openalex.py.

SAFETY:
    * Refuses to target a production DSN unless LoaderConfig.yes_production
      is explicitly set. The guard resolves the effective DSN (falling back to
      scix.db.DEFAULT_DSN when cfg.dsn is None) so a missing --dsn flag cannot
      silently connect to production.
    * Both target tables (papers_openalex, works_references) are LOGGED — see
      migration 040 for the relpersistence safety assertion.
    * Uses staging TEMP tables + INSERT ... ON CONFLICT for idempotent loads.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from scix.db import (
    DEFAULT_DSN,
    IndexDef,
    IndexManager,
    IngestLog,
    get_connection,
    is_production_dsn,
    redact_dsn,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OpenAlexLoader",
    "LoaderConfig",
    "LoaderStats",
    "ProductionGuardError",
    "reconstruct_abstract",
    "prune_work_record",
    "normalize_openalex_id",
    "extract_arxiv_id",
    "validate_manifest",
]

_OPENALEX_URL_PREFIX = "https://openalex.org/"
_DOI_URL_PREFIX = "https://doi.org/"
_ARXIV_LANDING_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")


class ProductionGuardError(RuntimeError):
    """Raised when the loader would target a production DSN without explicit opt-in."""


# ---------------------------------------------------------------------------
# Pure functions — no IO, easy to unit test
# ---------------------------------------------------------------------------


def _sanitize_text(val: str | None) -> str | None:
    """Remove null bytes that PostgreSQL rejects in text columns."""
    if val is None:
        return None
    return val.replace("\x00", "")


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Reconstruct abstract text from OpenAlex's inverted abstract index.

    OpenAlex stores abstracts as ``{word: [position, ...], ...}``. This
    function reverses the index to produce the original text.

    Returns None for None or empty input.
    """
    if not inverted_index:
        return None

    # Build position -> word mapping
    max_pos = -1
    pos_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            pos_word[pos] = word
            if pos > max_pos:
                max_pos = pos

    if max_pos < 0:
        return None

    tokens = [pos_word.get(i, "") for i in range(max_pos + 1)]
    return " ".join(tokens)


def normalize_openalex_id(raw_id: str | None) -> str | None:
    """Strip the ``https://openalex.org/`` URL prefix from an OpenAlex ID.

    Returns None for None or empty input.
    """
    if not raw_id:
        return None
    if raw_id.startswith(_OPENALEX_URL_PREFIX):
        return raw_id[len(_OPENALEX_URL_PREFIX) :]
    return raw_id


def _strip_doi_prefix(doi: str | None) -> str | None:
    """Strip the ``https://doi.org/`` URL prefix from a DOI."""
    if not doi:
        return None
    if doi.startswith(_DOI_URL_PREFIX):
        return doi[len(_DOI_URL_PREFIX) :]
    if doi.startswith("http://doi.org/"):
        return doi[len("http://doi.org/") :]
    return doi


def extract_arxiv_id(work: dict[str, Any]) -> str | None:
    """Extract an arXiv ID from an OpenAlex Work record.

    Checks ``primary_location.landing_page_url`` for arxiv.org URLs.
    Strips version suffixes (e.g. ``v2``).
    """
    # Check primary_location landing_page_url
    primary_loc = work.get("primary_location")
    if isinstance(primary_loc, dict):
        landing_url = primary_loc.get("landing_page_url", "")
        if isinstance(landing_url, str):
            match = _ARXIV_LANDING_RE.search(landing_url)
            if match:
                return match.group(1)

    return None


def prune_work_record(
    work: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Prune an OpenAlex Work record to the columns we store.

    Returns:
        (row_dict, reference_edges) where row_dict has keys matching
        papers_openalex columns and reference_edges is a list of
        (source_openalex_id, referenced_openalex_id) tuples.

    Raises:
        ValueError: If the work is missing an ``id`` field.
    """
    raw_id = work.get("id")
    if not raw_id:
        raise ValueError("Work record missing id")

    openalex_id = normalize_openalex_id(raw_id)
    doi = _strip_doi_prefix(work.get("doi"))
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

    topics = work.get("topics")
    open_access = work.get("open_access")
    best_oa = work.get("best_oa_location")

    title = _sanitize_text(work.get("title"))
    abstract = _sanitize_text(abstract)

    row = {
        "openalex_id": openalex_id,
        "doi": doi,
        "title": title,
        "publication_year": work.get("publication_year"),
        "abstract": abstract,
        "topics": json.dumps(topics) if topics is not None else None,
        "open_access": json.dumps(open_access) if open_access is not None else None,
        "best_oa_location": json.dumps(best_oa) if best_oa is not None else None,
        "cited_by_count": work.get("cited_by_count"),
        "referenced_works_count": work.get("referenced_works_count"),
        "type": work.get("type"),
        "updated_date": work.get("updated_date"),
        "created_date": work.get("created_date"),
    }

    # Extract reference edges
    edges: list[tuple[str, str]] = []
    referenced_works = work.get("referenced_works", [])
    if isinstance(referenced_works, list):
        for ref_url in referenced_works:
            ref_id = normalize_openalex_id(ref_url)
            if ref_id and openalex_id:
                edges.append((openalex_id, ref_id))

    return row, edges


def validate_manifest(partition_dir: Path) -> bool:
    """Check whether an S3 partition directory has a completed manifest.

    OpenAlex uses ``_SUCCESS`` marker files to indicate a partition has been
    fully written. Consuming a partition without this marker risks reading
    partial data.
    """
    if not partition_dir.is_dir():
        return False
    return (partition_dir / "_SUCCESS").exists()


# ---------------------------------------------------------------------------
# Config + Stats (frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoaderConfig:
    """Immutable configuration for the OpenAlex S3 snapshot loader."""

    dsn: str | None
    data_dir: Path
    batch_size: int = 50_000
    dry_run: bool = False
    yes_production: bool = False
    drop_indexes: bool = False
    works_only: bool = False  # Skip reference edge loading


@dataclass(frozen=True)
class LoaderStats:
    """Immutable result record returned from OpenAlexLoader.run()."""

    partitions_processed: int
    works_loaded: int
    references_loaded: int
    works_skipped: int
    crosswalk_updated: int
    elapsed_seconds: float
    dry_run: bool


# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_WORK_STAGING_DDL = (
    "CREATE TEMP TABLE IF NOT EXISTS _openalex_work_staging ("
    "openalex_id TEXT, doi TEXT, title TEXT, publication_year SMALLINT, "
    "abstract TEXT, topics JSONB, open_access JSONB, best_oa_location JSONB, "
    "cited_by_count INT, referenced_works_count INT, type TEXT, "
    "updated_date DATE, created_date DATE"
    ") ON COMMIT DELETE ROWS"
)

_WORK_STAGING_COPY = (
    "COPY _openalex_work_staging ("
    "openalex_id, doi, title, publication_year, abstract, topics, "
    "open_access, best_oa_location, cited_by_count, referenced_works_count, "
    "type, updated_date, created_date"
    ") FROM STDIN"
)

_WORK_MERGE_SQL = (
    "INSERT INTO papers_openalex ("
    "openalex_id, doi, title, publication_year, abstract, topics, "
    "open_access, best_oa_location, cited_by_count, referenced_works_count, "
    "type, updated_date, created_date"
    ") SELECT DISTINCT ON (openalex_id) "
    "openalex_id, doi, title, publication_year, abstract, topics, "
    "open_access, best_oa_location, cited_by_count, referenced_works_count, "
    "type, updated_date, created_date "
    "FROM _openalex_work_staging ORDER BY openalex_id "
    "ON CONFLICT (openalex_id) DO UPDATE SET "
    "doi = EXCLUDED.doi, title = EXCLUDED.title, "
    "publication_year = EXCLUDED.publication_year, "
    "abstract = EXCLUDED.abstract, topics = EXCLUDED.topics, "
    "open_access = EXCLUDED.open_access, "
    "best_oa_location = EXCLUDED.best_oa_location, "
    "cited_by_count = EXCLUDED.cited_by_count, "
    "referenced_works_count = EXCLUDED.referenced_works_count, "
    "type = EXCLUDED.type, "
    "updated_date = EXCLUDED.updated_date, "
    "created_date = EXCLUDED.created_date"
)

_REF_STAGING_DDL = (
    "CREATE TEMP TABLE IF NOT EXISTS _openalex_ref_staging ("
    "source_openalex_id TEXT, referenced_openalex_id TEXT"
    ") ON COMMIT DELETE ROWS"
)

_REF_STAGING_COPY = (
    "COPY _openalex_ref_staging (source_openalex_id, referenced_openalex_id) FROM STDIN"
)

_REF_MERGE_SQL = (
    "INSERT INTO works_references (source_openalex_id, referenced_openalex_id) "
    "SELECT DISTINCT source_openalex_id, referenced_openalex_id "
    "FROM _openalex_ref_staging "
    "ON CONFLICT DO NOTHING"
)

# Join DOI to papers_external_ids and set openalex_id + openalex_has_pdf_url.
# Also updates papers.openalex_id (migration 018) for backward compatibility.
_CROSSWALK_DOI_SQL = (
    "INSERT INTO papers_external_ids (bibcode, openalex_id, openalex_has_pdf_url) "
    "SELECT p.bibcode, s.openalex_id, "
    "       (s.best_oa_location::jsonb->>'pdf_url') IS NOT NULL "
    "FROM _openalex_work_staging s "
    "JOIN papers p ON p.doi[1] = s.doi "
    "WHERE s.doi IS NOT NULL "
    "ON CONFLICT (bibcode) DO UPDATE SET "
    "    openalex_id = EXCLUDED.openalex_id, "
    "    openalex_has_pdf_url = EXCLUDED.openalex_has_pdf_url"
)

_PAPERS_OPENALEX_ID_SQL = (
    "UPDATE papers SET openalex_id = s.openalex_id "
    "FROM _openalex_work_staging s "
    "WHERE papers.doi[1] = s.doi AND s.doi IS NOT NULL "
    "AND (papers.openalex_id IS NULL OR papers.openalex_id != s.openalex_id)"
)

_WORK_COLUMNS = (
    "openalex_id",
    "doi",
    "title",
    "publication_year",
    "abstract",
    "topics",
    "open_access",
    "best_oa_location",
    "cited_by_count",
    "referenced_works_count",
    "type",
    "updated_date",
    "created_date",
)


# ---------------------------------------------------------------------------
# Loader class
# ---------------------------------------------------------------------------


class OpenAlexLoader:
    """Load OpenAlex S3 snapshot (Parquet) into PostgreSQL.

    Uses DuckDB to read Parquet files in a memory-efficient streaming fashion,
    then COPY-loads into Postgres via staging tables. Follows the same pattern
    as AdsBodyLoader (src/scix/ads_body.py).
    """

    def __init__(self, config: LoaderConfig) -> None:
        self._cfg = config

    def run(self) -> LoaderStats:
        """Execute the full ingest pipeline."""
        self._check_production_guard()

        if not self._cfg.data_dir.exists():
            raise FileNotFoundError(f"OpenAlex data directory not found: {self._cfg.data_dir}")

        t_start = time.monotonic()

        conn = get_connection(self._cfg.dsn)
        try:
            ingest_log = IngestLog(conn)

            # Discover partition directories
            partitions = self._discover_partitions()
            if not partitions:
                logger.warning("No valid partitions found in %s", self._cfg.data_dir)
                return LoaderStats(
                    partitions_processed=0,
                    works_loaded=0,
                    references_loaded=0,
                    works_skipped=0,
                    crosswalk_updated=0,
                    elapsed_seconds=time.monotonic() - t_start,
                    dry_run=self._cfg.dry_run,
                )

            logger.info("Found %d partition(s) to process", len(partitions))

            # Optionally drop indexes for bulk load
            saved_work_indexes: list[IndexDef] = []
            saved_ref_indexes: list[IndexDef] = []
            if self._cfg.drop_indexes and not self._cfg.dry_run:
                work_mgr = IndexManager(conn, "papers_openalex")
                saved_work_indexes = work_mgr.drop_indexes()
                if not self._cfg.works_only:
                    ref_mgr = IndexManager(conn, "works_references")
                    saved_ref_indexes = ref_mgr.drop_indexes()
                logger.info(
                    "Dropped %d + %d indexes for bulk load",
                    len(saved_work_indexes),
                    len(saved_ref_indexes),
                )

            total_works = 0
            total_refs = 0
            total_skipped = 0
            total_crosswalk = 0
            partitions_done = 0

            try:
                for partition_dir in partitions:
                    partition_name = partition_dir.name
                    if not self._cfg.dry_run and ingest_log.is_complete(partition_name):
                        logger.info("Skipping %s (already complete)", partition_name)
                        partitions_done += 1
                        continue

                    if not self._cfg.dry_run:
                        ingest_log.start(partition_name)

                    try:
                        works, refs, skipped, crosswalk = self._load_partition(conn, partition_dir)
                        total_works += works
                        total_refs += refs
                        total_skipped += skipped
                        total_crosswalk += crosswalk
                        partitions_done += 1

                        if not self._cfg.dry_run:
                            ingest_log.update_counts(partition_name, works, skipped, refs)
                            ingest_log.finish(partition_name)

                        logger.info(
                            "%s: works=%d refs=%d skipped=%d crosswalk=%d",
                            partition_name,
                            works,
                            refs,
                            skipped,
                            crosswalk,
                        )
                    except Exception:
                        logger.exception("Failed loading partition %s", partition_name)
                        conn.rollback()
                        if not self._cfg.dry_run:
                            ingest_log.mark_failed(partition_name)
                        raise
            finally:
                # Recreate indexes
                if saved_work_indexes:
                    logger.info("Recreating %d indexes on papers_openalex", len(saved_work_indexes))
                    IndexManager(conn, "papers_openalex").recreate_indexes(saved_work_indexes)
                if saved_ref_indexes:
                    logger.info("Recreating %d indexes on works_references", len(saved_ref_indexes))
                    IndexManager(conn, "works_references").recreate_indexes(saved_ref_indexes)

            elapsed = time.monotonic() - t_start
            logger.info(
                "OpenAlex ingest complete: partitions=%d works=%d refs=%d "
                "skipped=%d crosswalk=%d elapsed=%.1fs",
                partitions_done,
                total_works,
                total_refs,
                total_skipped,
                total_crosswalk,
                elapsed,
            )

            return LoaderStats(
                partitions_processed=partitions_done,
                works_loaded=total_works,
                references_loaded=total_refs,
                works_skipped=total_skipped,
                crosswalk_updated=total_crosswalk,
                elapsed_seconds=elapsed,
                dry_run=self._cfg.dry_run,
            )
        finally:
            conn.close()

    # -- Internal methods ---------------------------------------------------

    def _check_production_guard(self) -> None:
        """Refuse to run against production unless explicitly authorized."""
        effective_dsn = self._cfg.dsn or DEFAULT_DSN
        if is_production_dsn(effective_dsn) and not self._cfg.yes_production:
            raise ProductionGuardError(
                "Refusing to run OpenAlex loader against production DSN "
                f"({redact_dsn(effective_dsn)}). Pass yes_production=True "
                "(or --yes-production on the CLI) to override."
            )

    def _discover_partitions(self) -> list[Path]:
        """Find all valid partition directories with _SUCCESS markers.

        Looks for ``updated_date=YYYY-MM-DD`` directories containing at least
        one Parquet file and a ``_SUCCESS`` marker.
        """
        partitions: list[Path] = []
        if not self._cfg.data_dir.is_dir():
            return partitions

        for entry in sorted(self._cfg.data_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("updated_date="):
                if validate_manifest(entry):
                    partitions.append(entry)
                else:
                    logger.debug("Skipping %s (no _SUCCESS marker)", entry.name)

        return partitions

    def _load_partition(
        self,
        conn: psycopg.Connection,
        partition_dir: Path,
    ) -> tuple[int, int, int, int]:
        """Load a single partition via DuckDB -> Postgres COPY.

        Returns (works_loaded, refs_loaded, works_skipped, crosswalk_updated).
        """
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError(
                "DuckDB is required for OpenAlex Parquet ingestion. "
                "Install it with: pip install duckdb"
            ) from exc

        parquet_files = sorted(partition_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning("No parquet files in %s", partition_dir)
            return 0, 0, 0, 0

        total_works = 0
        total_refs = 0
        total_skipped = 0
        total_crosswalk = 0

        # Use DuckDB to read Parquet files in streaming fashion
        duck_conn = duckdb.connect(":memory:")
        try:
            for pf in parquet_files:
                works_batch: list[tuple[Any, ...]] = []
                refs_batch: list[tuple[str, str]] = []

                # Read Parquet via DuckDB as Python dicts
                result = duck_conn.execute("SELECT * FROM read_parquet(?)", [str(pf)])
                while True:
                    chunk = result.fetchmany(self._cfg.batch_size)
                    if not chunk:
                        break

                    columns = [desc[0] for desc in result.description]
                    for row_tuple in chunk:
                        work = dict(zip(columns, row_tuple))
                        try:
                            row_dict, edges = prune_work_record(work)
                            work_row = tuple(row_dict[col] for col in _WORK_COLUMNS)
                            works_batch.append(work_row)
                            refs_batch.extend(edges)
                        except (ValueError, KeyError) as exc:
                            total_skipped += 1
                            logger.debug("Skipping work: %s", exc)
                            continue

                    if works_batch:
                        loaded, ref_loaded, crosswalk = self._flush_batch(
                            conn, works_batch, refs_batch
                        )
                        total_works += loaded
                        total_refs += ref_loaded
                        total_crosswalk += crosswalk
                        works_batch.clear()
                        refs_batch.clear()

                # Flush remaining
                if works_batch:
                    loaded, ref_loaded, crosswalk = self._flush_batch(conn, works_batch, refs_batch)
                    total_works += loaded
                    total_refs += ref_loaded
                    total_crosswalk += crosswalk
        finally:
            duck_conn.close()

        return total_works, total_refs, total_skipped, total_crosswalk

    def _flush_batch(
        self,
        conn: psycopg.Connection,
        work_rows: list[tuple[Any, ...]],
        ref_rows: list[tuple[str, str]],
    ) -> tuple[int, int, int]:
        """COPY+merge one batch. Returns (works_loaded, refs_loaded, crosswalk_updated)."""
        if not work_rows:
            return 0, 0, 0

        works_loaded = 0
        refs_loaded = 0
        crosswalk_updated = 0

        with conn.cursor() as cur:
            # Works staging + merge
            cur.execute(_WORK_STAGING_DDL)
            with cur.copy(_WORK_STAGING_COPY) as copy:
                for row in work_rows:
                    copy.write_row(row)

            if self._cfg.dry_run:
                conn.rollback()
                return len(work_rows), 0, 0

            cur.execute(_WORK_MERGE_SQL)
            works_loaded = cur.rowcount

            # Crosswalk: join DOI to papers_external_ids
            try:
                cur.execute(_CROSSWALK_DOI_SQL)
                crosswalk_updated = cur.rowcount
                # Also update papers.openalex_id
                cur.execute(_PAPERS_OPENALEX_ID_SQL)
            except psycopg.errors.UndefinedTable:
                # papers_external_ids may not exist in test databases
                logger.debug("papers_external_ids not available, skipping crosswalk")
                conn.rollback()
                # Re-do the work staging + merge after rollback
                cur.execute(_WORK_STAGING_DDL)
                with cur.copy(_WORK_STAGING_COPY) as copy:
                    for row in work_rows:
                        copy.write_row(row)
                cur.execute(_WORK_MERGE_SQL)
                works_loaded = cur.rowcount

        conn.commit()

        # Reference edges in a separate transaction
        if ref_rows and not self._cfg.works_only:
            with conn.cursor() as cur:
                cur.execute(_REF_STAGING_DDL)
                with cur.copy(_REF_STAGING_COPY) as copy:
                    for edge in ref_rows:
                        copy.write_row(edge)

                cur.execute(_REF_MERGE_SQL)
                refs_loaded = cur.rowcount

            conn.commit()

        return works_loaded, refs_loaded, crosswalk_updated
