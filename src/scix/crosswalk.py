"""SPDF-SPASE ID crosswalk: bidirectional lookup between CDAWeb dataset IDs
and SPASE ResourceIDs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrosswalkEntry:
    """A single SPDF <-> SPASE mapping."""

    id: int
    spdf_id: str
    spase_id: str
    source: str


def upsert_crosswalk(
    conn: psycopg.Connection,
    spdf_id: str,
    spase_id: str,
    *,
    source: str = "spdf_harvest",
) -> CrosswalkEntry:
    """Insert or ignore a SPDF-SPASE mapping.

    Args:
        conn: Database connection.
        spdf_id: CDAWeb dataset identifier (e.g. ``AC_H2_MFI``).
        spase_id: SPASE ResourceID (e.g. ``spase://NASA/NumericalData/ACE/MAG/L2/PT16S``).
        source: Provenance tag.

    Returns:
        The upserted CrosswalkEntry.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO spdf_spase_crosswalk (spdf_id, spase_id, source)
            VALUES (%(spdf)s, %(spase)s, %(source)s)
            ON CONFLICT (spdf_id, spase_id) DO NOTHING
            RETURNING id, spdf_id, spase_id, source
            """,
            {"spdf": spdf_id, "spase": spase_id, "source": source},
        )
        row = cur.fetchone()
        if row is None:
            # Already existed — fetch it
            cur.execute(
                """
                SELECT id, spdf_id, spase_id, source
                FROM spdf_spase_crosswalk
                WHERE spdf_id = %(spdf)s AND spase_id = %(spase)s
                """,
                {"spdf": spdf_id, "spase": spase_id},
            )
            row = cur.fetchone()
    conn.commit()
    return CrosswalkEntry(
        id=row["id"],
        spdf_id=row["spdf_id"],
        spase_id=row["spase_id"],
        source=row["source"],
    )


def lookup_by_spdf_id(
    conn: psycopg.Connection,
    spdf_id: str,
) -> list[CrosswalkEntry]:
    """Find all SPASE ResourceIDs mapped to a CDAWeb dataset ID.

    Args:
        conn: Database connection.
        spdf_id: CDAWeb dataset identifier.

    Returns:
        List of CrosswalkEntry (may be empty).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, spdf_id, spase_id, source
            FROM spdf_spase_crosswalk
            WHERE spdf_id = %(spdf)s
            ORDER BY spase_id
            """,
            {"spdf": spdf_id},
        )
        return [
            CrosswalkEntry(
                id=row["id"],
                spdf_id=row["spdf_id"],
                spase_id=row["spase_id"],
                source=row["source"],
            )
            for row in cur.fetchall()
        ]


def lookup_by_spase_id(
    conn: psycopg.Connection,
    spase_id: str,
) -> list[CrosswalkEntry]:
    """Find all CDAWeb dataset IDs mapped to a SPASE ResourceID.

    Args:
        conn: Database connection.
        spase_id: SPASE ResourceID.

    Returns:
        List of CrosswalkEntry (may be empty).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, spdf_id, spase_id, source
            FROM spdf_spase_crosswalk
            WHERE spase_id = %(spase)s
            ORDER BY spdf_id
            """,
            {"spase": spase_id},
        )
        return [
            CrosswalkEntry(
                id=row["id"],
                spdf_id=row["spdf_id"],
                spase_id=row["spase_id"],
                source=row["source"],
            )
            for row in cur.fetchall()
        ]
