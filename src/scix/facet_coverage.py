"""Completeness metadata for paper facets."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row


def _estimated_null_pct(
    conn: psycopg.Connection,
    facet_field: str,
) -> float | None:
    """Read the planner's corpus-level null estimate without another scan."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT null_frac
            FROM pg_stats
            WHERE schemaname = ANY(current_schemas(false))
              AND tablename = 'papers'
              AND attname = %s
            ORDER BY array_position(current_schemas(false), schemaname)
            LIMIT 1
            """,
            [facet_field],
        )
        row = cur.fetchone()

    raw_null_frac = row.get("null_frac") if isinstance(row, dict) else None
    if not isinstance(raw_null_frac, (int, float)) or isinstance(raw_null_frac, bool):
        return None
    return round(float(raw_null_frac) * 100.0, 2)


def facet_coverage(
    conn: psycopg.Connection,
    facet_field: str,
    *,
    is_array: bool,
) -> dict[str, Any]:
    """Return explicit, approximate corpus-level completeness metadata."""
    null_pct = _estimated_null_pct(conn, facet_field)
    exclusions = ["null", "empty_array"] if is_array else ["null"]
    exclusion_text = "null and empty-array" if is_array else "null"

    if null_pct is None:
        note = (
            f"Corpus null-rate statistics for {facet_field} are unavailable. "
            f"Facet counts exclude {exclusion_text} values, so an empty "
            "result cannot distinguish no matching papers from missing metadata."
        )
    else:
        missing_kind = "classifications" if facet_field == "arxiv_class" else "metadata"
        note = (
            f"Estimated {null_pct:.2f}% of corpus papers have null {facet_field} metadata. "
            f"Facet counts exclude {exclusion_text} values; sparse facets may "
            f"reflect missing {missing_kind} rather than no matching papers."
        )

    return {
        "field": facet_field,
        "scope": "corpus",
        "basis": "postgresql_statistics",
        "estimated": True,
        "estimated_null_pct": null_pct,
        "counts_exclude": exclusions,
        "note": note,
    }
