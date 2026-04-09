"""Entity merge/split audit log.

Records and queries entity resolution changes — merges (two entities
become one) and splits (one entity becomes several).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeEntry:
    """A single entity merge audit record."""

    id: int
    old_entity_id: int
    new_entity_id: int
    reason: str | None
    merged_by: str | None
    merged_at: datetime


@dataclass(frozen=True)
class SplitEntry:
    """A single entity split audit record."""

    id: int
    parent_entity_id: int
    child_entity_ids: tuple[int, ...]
    reason: str | None
    split_by: str | None
    split_at: datetime


def record_merge(
    conn: psycopg.Connection,
    old_entity_id: int,
    new_entity_id: int,
    *,
    reason: str | None = None,
    merged_by: str | None = None,
) -> MergeEntry:
    """Record an entity merge in the audit log.

    Args:
        conn: Database connection.
        old_entity_id: The entity being retired (merged away).
        new_entity_id: The surviving entity.
        reason: Free-text explanation of why the merge happened.
        merged_by: Actor who performed the merge (user, pipeline, etc.).

    Returns:
        The created MergeEntry.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO entity_merge_log (old_entity_id, new_entity_id, reason, merged_by)
            VALUES (%(old)s, %(new)s, %(reason)s, %(merged_by)s)
            RETURNING id, old_entity_id, new_entity_id, reason, merged_by, merged_at
            """,
            {
                "old": old_entity_id,
                "new": new_entity_id,
                "reason": reason,
                "merged_by": merged_by,
            },
        )
        row = cur.fetchone()
    conn.commit()
    logger.info(
        "Recorded merge: entity %d -> %d (reason: %s)",
        old_entity_id,
        new_entity_id,
        reason,
    )
    return MergeEntry(
        id=row["id"],
        old_entity_id=row["old_entity_id"],
        new_entity_id=row["new_entity_id"],
        reason=row["reason"],
        merged_by=row["merged_by"],
        merged_at=row["merged_at"],
    )


def record_split(
    conn: psycopg.Connection,
    parent_entity_id: int,
    child_entity_ids: list[int],
    *,
    reason: str | None = None,
    split_by: str | None = None,
) -> SplitEntry:
    """Record an entity split in the audit log.

    Args:
        conn: Database connection.
        parent_entity_id: The entity being split (may be retired).
        child_entity_ids: The new child entities created from the split.
        reason: Free-text explanation of why the split happened.
        split_by: Actor who performed the split.

    Returns:
        The created SplitEntry.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO entity_split_log (parent_entity_id, child_entity_ids, reason, split_by)
            VALUES (%(parent)s, %(children)s, %(reason)s, %(split_by)s)
            RETURNING id, parent_entity_id, child_entity_ids, reason, split_by, split_at
            """,
            {
                "parent": parent_entity_id,
                "children": child_entity_ids,
                "reason": reason,
                "split_by": split_by,
            },
        )
        row = cur.fetchone()
    conn.commit()
    logger.info(
        "Recorded split: entity %d -> %s (reason: %s)",
        parent_entity_id,
        child_entity_ids,
        reason,
    )
    return SplitEntry(
        id=row["id"],
        parent_entity_id=row["parent_entity_id"],
        child_entity_ids=tuple(row["child_entity_ids"]),
        reason=row["reason"],
        split_by=row["split_by"],
        split_at=row["split_at"],
    )


def get_merge_history(
    conn: psycopg.Connection,
    entity_id: int,
) -> list[MergeEntry]:
    """Get all merge records involving an entity (as old or new).

    Args:
        conn: Database connection.
        entity_id: Entity ID to look up.

    Returns:
        List of MergeEntry sorted by merged_at DESC.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, old_entity_id, new_entity_id, reason, merged_by, merged_at
            FROM entity_merge_log
            WHERE old_entity_id = %(eid)s OR new_entity_id = %(eid)s
            ORDER BY merged_at DESC
            """,
            {"eid": entity_id},
        )
        return [
            MergeEntry(
                id=row["id"],
                old_entity_id=row["old_entity_id"],
                new_entity_id=row["new_entity_id"],
                reason=row["reason"],
                merged_by=row["merged_by"],
                merged_at=row["merged_at"],
            )
            for row in cur.fetchall()
        ]


def get_split_history(
    conn: psycopg.Connection,
    entity_id: int,
) -> list[SplitEntry]:
    """Get all split records involving an entity (as parent or child).

    Args:
        conn: Database connection.
        entity_id: Entity ID to look up.

    Returns:
        List of SplitEntry sorted by split_at DESC.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, parent_entity_id, child_entity_ids, reason, split_by, split_at
            FROM entity_split_log
            WHERE parent_entity_id = %(eid)s OR %(eid)s = ANY(child_entity_ids)
            ORDER BY split_at DESC
            """,
            {"eid": entity_id},
        )
        return [
            SplitEntry(
                id=row["id"],
                parent_entity_id=row["parent_entity_id"],
                child_entity_ids=tuple(row["child_entity_ids"]),
                reason=row["reason"],
                split_by=row["split_by"],
                split_at=row["split_at"],
            )
            for row in cur.fetchall()
        ]


def get_audit_history(
    conn: psycopg.Connection,
    entity_id: int,
) -> dict[str, list[MergeEntry] | list[SplitEntry]]:
    """Get complete audit history (merges + splits) for an entity.

    Args:
        conn: Database connection.
        entity_id: Entity ID to look up.

    Returns:
        Dict with keys "merges" and "splits".
    """
    return {
        "merges": get_merge_history(conn, entity_id),
        "splits": get_split_history(conn, entity_id),
    }
