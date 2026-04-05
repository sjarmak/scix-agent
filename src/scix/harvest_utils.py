"""Shared utilities for harvest pipelines.

Provides harvest_run lifecycle management and entity graph upsert helpers
used by all harvesters (GCMD, PDS4, SPDF, etc.).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Harvest run lifecycle
# ---------------------------------------------------------------------------


class HarvestRunLog:
    """Track a harvest run in the harvest_runs table."""

    def __init__(self, conn: psycopg.Connection, source: str) -> None:
        self._conn = conn
        self._source = source
        self._run_id: int | None = None

    @property
    def run_id(self) -> int:
        if self._run_id is None:
            raise RuntimeError("Harvest run not started — call start() first")
        return self._run_id

    def start(self, config: dict[str, Any] | None = None) -> int:
        """Create a harvest_runs row with status='running'. Returns run id."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO harvest_runs (source, status, config)
                VALUES (%s, 'running', %s)
                RETURNING id
                """,
                (self._source, json.dumps(config or {})),
            )
            self._run_id = cur.fetchone()[0]
        self._conn.commit()
        return self._run_id

    def complete(
        self,
        *,
        records_fetched: int,
        records_upserted: int,
        counts: dict[str, int] | None = None,
    ) -> None:
        """Mark the harvest run as completed with final counts."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE harvest_runs
                SET finished_at = now(),
                    status = 'completed',
                    records_fetched = %s,
                    records_upserted = %s,
                    counts = %s
                WHERE id = %s
                """,
                (
                    records_fetched,
                    records_upserted,
                    json.dumps(counts or {}),
                    self.run_id,
                ),
            )
        self._conn.commit()

    def fail(self, error_message: str) -> None:
        """Mark the harvest run as failed."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE harvest_runs
                SET finished_at = now(),
                    status = 'failed',
                    error_message = %s
                WHERE id = %s
                """,
                (error_message, self.run_id),
            )
        self._conn.commit()


# ---------------------------------------------------------------------------
# Entity graph upsert helpers
# ---------------------------------------------------------------------------


def upsert_entity(
    conn: psycopg.Connection,
    *,
    canonical_name: str,
    entity_type: str,
    source: str,
    discipline: str,
    harvest_run_id: int,
    properties: dict[str, Any] | None = None,
) -> int:
    """Upsert an entity and return its id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entities
                (canonical_name, entity_type, discipline, source,
                 harvest_run_id, properties)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (canonical_name, entity_type, source) DO UPDATE SET
                discipline = EXCLUDED.discipline,
                harvest_run_id = EXCLUDED.harvest_run_id,
                properties = EXCLUDED.properties,
                updated_at = NOW()
            RETURNING id
            """,
            (
                canonical_name,
                entity_type,
                discipline,
                source,
                harvest_run_id,
                json.dumps(properties or {}),
            ),
        )
        return cur.fetchone()[0]


def upsert_entity_identifier(
    conn: psycopg.Connection,
    *,
    entity_id: int,
    id_scheme: str,
    external_id: str,
    is_primary: bool = False,
) -> None:
    """Upsert an entity identifier."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entity_identifiers (entity_id, id_scheme, external_id, is_primary)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id_scheme, external_id) DO UPDATE SET
                entity_id = EXCLUDED.entity_id,
                is_primary = EXCLUDED.is_primary
            """,
            (entity_id, id_scheme, external_id, is_primary),
        )


def upsert_entity_alias(
    conn: psycopg.Connection,
    *,
    entity_id: int,
    alias: str,
    alias_source: str,
) -> None:
    """Upsert an entity alias."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entity_aliases (entity_id, alias, alias_source)
            VALUES (%s, %s, %s)
            ON CONFLICT (entity_id, alias) DO NOTHING
            """,
            (entity_id, alias, alias_source),
        )


def upsert_entity_relationship(
    conn: psycopg.Connection,
    *,
    subject_entity_id: int,
    predicate: str,
    object_entity_id: int,
    source: str,
    harvest_run_id: int,
    confidence: float = 1.0,
) -> None:
    """Upsert an entity relationship."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO entity_relationships
                (subject_entity_id, predicate, object_entity_id,
                 source, harvest_run_id, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (subject_entity_id, predicate, object_entity_id) DO UPDATE SET
                source = EXCLUDED.source,
                harvest_run_id = EXCLUDED.harvest_run_id,
                confidence = EXCLUDED.confidence
            """,
            (
                subject_entity_id,
                predicate,
                object_entity_id,
                source,
                harvest_run_id,
                confidence,
            ),
        )
