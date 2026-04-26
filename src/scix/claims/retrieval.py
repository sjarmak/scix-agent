"""Retrieval helpers over the ``paper_claims`` table (migration 062).

Two public functions back the MCP tools ``read_paper_claims`` and
``find_claims``:

* :func:`read_paper_claims` — return claims for a single paper, ordered
  by their position in the source text.
* :func:`find_claims` — full-text search over claim text using the GIN
  ``to_tsvector('english', claim_text)`` index, optionally filtered by
  a linked entity id (subject OR object) and / or claim_type.

The shape of every returned row is fixed and matches the JSON schema
the MCP server emits. Keeping the projection here (rather than in the
server) lets us unit-test the SQL without spinning up the MCP layer.
"""

from __future__ import annotations

from typing import Any

import psycopg

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: CHECK-constrained claim_type values per migration 062. Kept here so callers
#: can validate input symmetrically with the SQL CHECK constraint without
#: making a round-trip to the database. Validation must reject anything
#: outside this set so we never push a stray label through to the planner.
VALID_CLAIM_TYPES: frozenset[str] = frozenset(
    {
        "factual",
        "methodological",
        "comparative",
        "speculative",
        "cited_from_other",
    }
)


# Columns projected by both helpers. Single source of truth so the MCP
# response schema and the SQL stay in lockstep.
_CLAIM_COLUMNS: tuple[str, ...] = (
    "bibcode",
    "section_index",
    "paragraph_index",
    "char_span_start",
    "char_span_end",
    "claim_text",
    "claim_type",
    "subject",
    "predicate",
    "object",
    "confidence",
)


def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    """Project a tuple of the documented columns to a dict.

    Pure helper — the column ordering must match :data:`_CLAIM_COLUMNS`
    and every SELECT in this module.
    """
    return {col: row[i] for i, col in enumerate(_CLAIM_COLUMNS)}


def _validate_claim_type(claim_type: str | None) -> None:
    """Reject claim_type values outside the migration-062 CHECK set.

    A None value means "no filter" and is always valid.
    """
    if claim_type is None:
        return
    if claim_type not in VALID_CLAIM_TYPES:
        raise ValueError(
            f"invalid claim_type: {claim_type!r}; "
            f"must be one of {sorted(VALID_CLAIM_TYPES)}"
        )


def _validate_limit(limit: int) -> int:
    """Coerce + bounds-check ``limit``. Returns the validated int.

    Negative or zero limits are rejected. The upper bound mirrors the
    convention used elsewhere in the MCP layer (cap fanout to keep
    blast radius bounded).
    """
    try:
        n = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"limit must be an integer, got {limit!r}") from exc
    if n <= 0:
        raise ValueError(f"limit must be positive, got {n}")
    return min(n, 500)


def read_paper_claims(
    conn: psycopg.Connection,
    bibcode: str,
    claim_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return claims for a single paper, in source-text order.

    Ordering is ``(section_index, paragraph_index, char_span_start)`` so
    a downstream caller iterating the result list walks the paper from
    front to back deterministically.

    Args:
        conn: Open psycopg connection.
        bibcode: Source paper bibcode (FK to ``papers``).
        claim_type: Optional filter on claim_type. Must be one of
            :data:`VALID_CLAIM_TYPES` if provided.
        limit: Max rows to return. Coerced to a positive int, capped at 500.

    Returns:
        A list of dicts, one per claim, with the keys documented in
        :data:`_CLAIM_COLUMNS`. Empty list if no claims match.
    """
    if not isinstance(bibcode, str) or not bibcode.strip():
        raise ValueError("bibcode must be a non-empty string")
    _validate_claim_type(claim_type)
    n = _validate_limit(limit)

    columns_sql = ", ".join(_CLAIM_COLUMNS)
    params: list[Any] = [bibcode]
    where = ["bibcode = %s"]
    if claim_type is not None:
        where.append("claim_type = %s")
        params.append(claim_type)
    params.append(n)

    sql = (
        f"SELECT {columns_sql} "
        "FROM paper_claims "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY section_index, paragraph_index, char_span_start "
        "LIMIT %s"
    )

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [_row_to_dict(row) for row in rows]


def find_claims(
    conn: psycopg.Connection,
    query: str,
    claim_type: str | None = None,
    entity_id: int | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Full-text search over ``paper_claims.claim_text``.

    Uses ``plainto_tsquery('english', %s)`` so the user's query is treated
    as a natural-language phrase (no operators required). Ranking is by
    ``ts_rank(to_tsvector('english', claim_text), plainto_tsquery(...))``
    descending; ties broken by ``extracted_at`` descending so newer
    extractions surface first.

    The match predicate references ``to_tsvector('english', claim_text)``
    verbatim so the planner picks the GIN index
    ``ix_paper_claims_claim_text_tsv`` (created in migration 062).

    Args:
        conn: Open psycopg connection.
        query: User query string. Must be non-empty after stripping.
        claim_type: Optional filter on claim_type. Must be one of
            :data:`VALID_CLAIM_TYPES` if provided.
        entity_id: Optional filter — restricts to rows where
            ``linked_entity_subject_id = %s OR linked_entity_object_id = %s``.
        limit: Max rows to return. Coerced to a positive int, capped at 500.

    Returns:
        A list of dicts, one per claim, with the keys documented in
        :data:`_CLAIM_COLUMNS`. Empty list if no claims match.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    _validate_claim_type(claim_type)
    n = _validate_limit(limit)

    if entity_id is not None:
        try:
            entity_id_int: int | None = int(entity_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"entity_id must be an integer, got {entity_id!r}"
            ) from exc
    else:
        entity_id_int = None

    columns_sql = ", ".join(_CLAIM_COLUMNS)

    # Build WHERE clauses + bound params in lock-step. The ORDER BY clause
    # appends one more %s for the ts_rank tsquery, and LIMIT appends the
    # final one. Param order must match placeholder order in the final SQL.
    params: list[Any] = [query]  # WHERE plainto_tsquery
    where = ["to_tsvector('english', claim_text) @@ plainto_tsquery('english', %s)"]
    if claim_type is not None:
        where.append("claim_type = %s")
        params.append(claim_type)
    if entity_id_int is not None:
        where.append(
            "(linked_entity_subject_id = %s OR linked_entity_object_id = %s)"
        )
        params.append(entity_id_int)
        params.append(entity_id_int)
    # ORDER BY ts_rank tsquery
    params.append(query)
    # LIMIT
    params.append(n)

    sql = (
        f"SELECT {columns_sql} "
        "FROM paper_claims "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY ts_rank(to_tsvector('english', claim_text), "
        "plainto_tsquery('english', %s)) DESC, "
        "extracted_at DESC "
        "LIMIT %s"
    )

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [_row_to_dict(row) for row in rows]
