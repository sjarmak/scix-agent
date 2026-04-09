"""Document-dataset linking pipeline.

Reads extracted dataset mentions from the extractions table, resolves them
against the datasets table, and writes results to document_datasets.
Processes in configurable chunks with per-batch commits for resumability.

Three match methods:
  1. extraction — resolve dataset names from entity_extraction payloads
  2. ads_data   — parse the ADS ``data`` field (repository:count pairs)
  3. doi_cite   — match dataset DOIs in paper reference lists

Currently (1) is the primary method; (2) and (3) activate when the
corresponding data is populated.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import psycopg

from scix.harvest_utils import LINK_TYPE_ANALYZES_DATASET

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ADS data field parsing
# ---------------------------------------------------------------------------


def _parse_ads_data_field(data: list[str] | None) -> dict[str, int]:
    """Parse ADS ``data`` field entries like ``["CDS:1", "IRSA:2"]``.

    Returns:
        Dict mapping repository code to record count.
    """
    if not data:
        return {}
    result: dict[str, int] = {}
    for entry in data:
        if not isinstance(entry, str) or ":" not in entry:
            continue
        parts = entry.rsplit(":", 1)
        if len(parts) != 2:
            continue
        repo, count_str = parts
        try:
            result[repo] = int(count_str)
        except ValueError:
            continue
    return result


# ---------------------------------------------------------------------------
# Mention extraction
# ---------------------------------------------------------------------------


def _extract_dataset_mentions(
    payload: dict[str, Any],
    extraction_type: str,
) -> list[tuple[str, str]]:
    """Extract (mention, match_method) pairs from an extraction payload.

    Only extracts dataset-related mentions:
    - Combined payload: reads ``datasets`` key only
    - Per-type payload: reads ``entities`` key when extraction_type is ``datasets``
    """
    mentions: list[tuple[str, str]] = []

    # Combined payload — only the "datasets" key
    items = payload.get("datasets")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, str) and item.strip():
                mentions.append((item.strip(), "extraction"))

    # Per-type payload (legacy style)
    entities = payload.get("entities")
    if isinstance(entities, list) and not mentions:
        for item in entities:
            if isinstance(item, str) and item.strip():
                mentions.append((item.strip(), "extraction"))

    return mentions


# ---------------------------------------------------------------------------
# Dataset resolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetResolverMatch:
    """A single dataset resolution result."""

    dataset_id: int
    confidence: float
    match_method: str


class DatasetResolver:
    """Resolve mention strings to dataset IDs via bulk-loaded cache.

    Loads the full dataset name mapping on first use so subsequent lookups
    are pure dict hits.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn
        self._cache: dict[str, DatasetResolverMatch] | None = None

    def _build_cache(self) -> dict[str, DatasetResolverMatch]:
        cache: dict[str, DatasetResolverMatch] = {}
        with self._conn.cursor() as cur:
            cur.execute("SELECT id, name FROM datasets")
            for row in cur.fetchall():
                dataset_id, name = row
                key = name.strip().lower()
                if key and key not in cache:
                    cache[key] = DatasetResolverMatch(
                        dataset_id=dataset_id,
                        confidence=1.0,
                        match_method="name_exact",
                    )
        return cache

    def resolve(self, mention: str) -> DatasetResolverMatch | None:
        if self._cache is None:
            self._cache = self._build_cache()
        key = mention.strip().lower()
        return self._cache.get(key)


# ---------------------------------------------------------------------------
# Batch linking
# ---------------------------------------------------------------------------


def link_datasets_batch(
    conn: psycopg.Connection,
    *,
    batch_size: int = 1000,
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Link extracted dataset mentions to datasets in batches.

    Args:
        conn: Database connection (must NOT be in autocommit mode).
        batch_size: Number of bibcodes per commit chunk.
        resume: If True, skip bibcodes already in document_datasets.
        dry_run: If True, count mentions without writing.

    Returns:
        Summary dict with bibcodes_processed, links_created, skipped_no_match,
        and by_method breakdown.
    """
    resolver = DatasetResolver(conn)

    with conn.cursor() as cur:
        if resume:
            cur.execute(
                """
                SELECT DISTINCT e.bibcode
                FROM extractions e
                WHERE e.extraction_type = 'datasets'
                  AND NOT EXISTS (
                      SELECT 1 FROM document_datasets dd WHERE dd.bibcode = e.bibcode
                  )
                """,
            )
        else:
            cur.execute(
                "SELECT DISTINCT bibcode FROM extractions WHERE extraction_type = 'datasets'",
            )
        bibcodes = [row[0] for row in cur.fetchall()]

    if not bibcodes:
        logger.info("No dataset extractions to process")
        return {
            "bibcodes_processed": 0,
            "links_created": 0,
            "skipped_no_match": 0,
            "by_method": {},
        }

    if resume:
        logger.info("Resume: %d bibcodes remaining to link", len(bibcodes))

    total_links = 0
    total_skipped = 0
    total_processed = 0
    by_method: dict[str, int] = {}

    for batch_start in range(0, len(bibcodes), batch_size):
        batch = bibcodes[batch_start : batch_start + batch_size]

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bibcode, extraction_type, payload
                FROM extractions
                WHERE extraction_type = 'datasets' AND bibcode = ANY(%s)
                """,
                (batch,),
            )
            rows = cur.fetchall()

        insert_params: list[tuple[str, int, str, float, str]] = []
        batch_skipped = 0

        for bibcode, ext_type, payload_raw in rows:
            payload = (
                payload_raw if isinstance(payload_raw, dict) else json.loads(payload_raw)
            )
            mentions = _extract_dataset_mentions(payload, ext_type)

            for mention_text, match_method_source in mentions:
                match = resolver.resolve(mention_text)
                if match is None:
                    batch_skipped += 1
                    continue

                insert_params.append(
                    (
                        bibcode,
                        match.dataset_id,
                        LINK_TYPE_ANALYZES_DATASET,
                        match.confidence,
                        match.match_method,
                    )
                )
                by_method[match_method_source] = (
                    by_method.get(match_method_source, 0) + 1
                )

        batch_links = len(insert_params)

        if not dry_run and insert_params:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO document_datasets
                        (bibcode, dataset_id, link_type, confidence, match_method)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (bibcode, dataset_id, link_type) DO NOTHING
                    """,
                    insert_params,
                )
            conn.commit()

        total_links += batch_links
        total_skipped += batch_skipped
        total_processed += len(batch)

        logger.info(
            "Batch %d-%d: %d links, %d unresolved",
            batch_start,
            batch_start + len(batch),
            batch_links,
            batch_skipped,
        )

    summary = {
        "bibcodes_processed": total_processed,
        "links_created": total_links,
        "skipped_no_match": total_skipped,
        "by_method": by_method,
    }
    logger.info("Dataset linking complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


def get_dataset_linking_progress(conn: psycopg.Connection) -> dict[str, int]:
    """Return dataset linking progress statistics."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(DISTINCT bibcode) FROM extractions WHERE extraction_type = 'datasets'"
        )
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT bibcode) FROM document_datasets")
        linked = cur.fetchone()[0]

    return {
        "total_bibcodes": total,
        "linked_bibcodes": linked,
        "pending_bibcodes": total - linked,
    }
