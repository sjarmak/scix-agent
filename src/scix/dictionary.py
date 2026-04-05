"""Entity dictionary: canonical names, aliases, and metadata for scientific entities.

Provides upsert, lookup, bulk load, and stats operations against the
entity_dictionary table (migration 013).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


def upsert_entry(
    conn: psycopg.Connection,
    *,
    canonical_name: str,
    entity_type: str,
    source: str,
    external_id: str | None = None,
    aliases: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert or update a single entity dictionary entry.

    On conflict (canonical_name, entity_type, source), updates external_id,
    aliases, and metadata.

    Args:
        conn: Database connection.
        canonical_name: The canonical name of the entity.
        entity_type: Entity type (e.g. 'software', 'instrument', 'mission').
        source: Source of the entry (e.g. 'manual', 'wikidata', 'ads').
        external_id: Optional external identifier (e.g. Wikidata QID).
        aliases: Optional list of alternate names.
        metadata: Optional JSONB metadata dict.

    Returns:
        Dict with the upserted row's fields.
    """
    resolved_aliases = aliases if aliases is not None else []
    resolved_metadata = metadata if metadata is not None else {}

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO entity_dictionary
                (canonical_name, entity_type, source, external_id, aliases, metadata)
            VALUES (%(canonical_name)s, %(entity_type)s, %(source)s,
                    %(external_id)s, %(aliases)s, %(metadata)s)
            ON CONFLICT (canonical_name, entity_type, source) DO UPDATE SET
                external_id = EXCLUDED.external_id,
                aliases = EXCLUDED.aliases,
                metadata = EXCLUDED.metadata
            RETURNING id, canonical_name, entity_type, source,
                      external_id, aliases, metadata
            """,
            {
                "canonical_name": canonical_name,
                "entity_type": entity_type,
                "source": source,
                "external_id": external_id,
                "aliases": resolved_aliases,
                "metadata": json.dumps(resolved_metadata),
            },
        )
        row = cur.fetchone()

    conn.commit()
    return dict(row)


def lookup(
    conn: psycopg.Connection,
    name: str,
    *,
    entity_type: str | None = None,
) -> dict[str, Any] | None:
    """Look up an entity by canonical name or alias.

    Searches canonical_name (exact, case-insensitive) first, then falls back
    to searching aliases.  If entity_type is provided, filters by type.

    Args:
        conn: Database connection.
        name: Name to search for.
        entity_type: Optional entity type filter.

    Returns:
        Dict with keys canonical_name, entity_type, source, external_id,
        aliases, metadata — or None if not found.
    """
    # Try canonical_name first
    type_clause = "AND entity_type = %(entity_type)s" if entity_type else ""

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT id, canonical_name, entity_type, source,
                   external_id, aliases, metadata
            FROM entity_dictionary
            WHERE lower(canonical_name) = lower(%(name)s)
            {type_clause}
            LIMIT 1
            """,
            {"name": name, "entity_type": entity_type},
        )
        row = cur.fetchone()

    if row is not None:
        return dict(row)

    # Fall back to alias search
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT id, canonical_name, entity_type, source,
                   external_id, aliases, metadata
            FROM entity_dictionary
            WHERE %(name)s = ANY(aliases)
            {type_clause}
            LIMIT 1
            """,
            {"name": name, "entity_type": entity_type},
        )
        row = cur.fetchone()

    return dict(row) if row is not None else None


def bulk_load(
    conn: psycopg.Connection,
    entries: list[dict[str, Any]],
) -> int:
    """Batch-insert entity dictionary entries with ON CONFLICT DO UPDATE.

    Each entry dict should have keys: canonical_name, entity_type, source.
    Optional keys: external_id, aliases, metadata.

    Args:
        conn: Database connection.
        entries: List of entry dicts.

    Returns:
        Number of rows upserted.
    """
    if not entries:
        return 0

    total = 0
    with conn.cursor() as cur:
        for entry in entries:
            canonical_name = entry["canonical_name"]
            entity_type = entry["entity_type"]
            source = entry["source"]
            external_id = entry.get("external_id")
            aliases = entry.get("aliases", [])
            metadata = entry.get("metadata", {})

            cur.execute(
                """
                INSERT INTO entity_dictionary
                    (canonical_name, entity_type, source, external_id, aliases, metadata)
                VALUES (%(canonical_name)s, %(entity_type)s, %(source)s,
                        %(external_id)s, %(aliases)s, %(metadata)s)
                ON CONFLICT (canonical_name, entity_type, source) DO UPDATE SET
                    external_id = EXCLUDED.external_id,
                    aliases = EXCLUDED.aliases,
                    metadata = EXCLUDED.metadata
                """,
                {
                    "canonical_name": canonical_name,
                    "entity_type": entity_type,
                    "source": source,
                    "external_id": external_id,
                    "aliases": aliases,
                    "metadata": json.dumps(metadata),
                },
            )
            total += cur.rowcount

    conn.commit()
    logger.info("Bulk loaded %d entity dictionary entries", total)
    return total


def get_stats(conn: psycopg.Connection) -> dict[str, Any]:
    """Return summary statistics for the entity dictionary.

    Returns:
        Dict with keys: total (int), by_type (dict mapping entity_type to count).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM entity_dictionary")
        total = cur.fetchone()[0]

        cur.execute("""
            SELECT entity_type, COUNT(*) AS cnt
            FROM entity_dictionary
            GROUP BY entity_type
            ORDER BY cnt DESC
            """)
        by_type = {row[0]: row[1] for row in cur.fetchall()}

    return {"total": total, "by_type": by_type}
