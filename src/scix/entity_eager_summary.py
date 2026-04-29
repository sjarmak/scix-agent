"""Eager mini-summary helpers for the MCP ``entity`` tool.

Folds two cheap context fields into the response of
``entity(action='resolve')`` so callers can triage candidates without a
follow-up ``entity_context`` round trip:

  co_occurring[k]   - top-K canonical_names that share >=N papers with the
                      focal entity, sampled from the focal entity's
                      highest-confidence papers in
                      ``document_entities_canonical``.
  recent_examples[k] - bibcode + year + title for the K most-recent papers
                      tagged with the focal entity, ordered (year DESC,
                      fused_confidence DESC).

A single batched query covers every focal entity_id passed in, so the
cost grows with the number of resolver candidates (typically <= 5), not
with the corpus size. See bead ``scix_experiments-3lw4.1`` for the
agent-side rationale (50% of ``entity`` calls were followed by
``entity_context`` in 30-day query_log).
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

#: Batched mini-summary SQL.
#:
#: Strategy:
#:   1. ``focal`` = the resolver candidate entity_ids passed by the caller.
#:   2. ``top_papers`` = top ``%(sample)s`` papers per focal entity by
#:      ``fused_confidence DESC`` (uses ``idx_dec_entity_fused``, ~1-2 ms
#:      per focal even at hub-entity cardinalities).
#:   3. ``co_occur`` = re-join ``document_entities_canonical`` on the
#:      sampled bibcodes to count co-mentioned entities, ranked per focal.
#:   4. ``recent_papers`` = same sampled paper set, joined to ``papers``
#:      for year/title, ranked (year DESC, fused_confidence DESC).
#:
#: Both branches share ``top_papers`` and surface as a UNION ALL with a
#: ``kind`` discriminator so we get both halves in one round trip.
_MINI_SUMMARY_SQL = """
WITH focal(eid) AS (
    SELECT * FROM unnest(%(focal_ids)s::int[])
),
top_papers AS (
    SELECT f.eid AS focal_id, dec.bibcode, dec.fused_confidence
    FROM focal f
    CROSS JOIN LATERAL (
        SELECT bibcode, fused_confidence
        FROM document_entities_canonical
        WHERE entity_id = f.eid
        ORDER BY fused_confidence DESC
        LIMIT %(sample)s
    ) dec
),
co_occur AS (
    SELECT tp.focal_id, dec2.entity_id AS co_id, COUNT(*) AS shared_papers,
           ROW_NUMBER() OVER (
               PARTITION BY tp.focal_id
               ORDER BY COUNT(*) DESC, dec2.entity_id
           ) AS rn
    FROM top_papers tp
    JOIN document_entities_canonical dec2
      ON dec2.bibcode = tp.bibcode
     AND dec2.entity_id <> tp.focal_id
    GROUP BY tp.focal_id, dec2.entity_id
),
co_top AS (
    SELECT c.focal_id, e.canonical_name, e.entity_type, c.shared_papers, c.rn
    FROM co_occur c
    JOIN entities e ON e.id = c.co_id
    WHERE c.rn <= %(co_top_k)s
      AND c.shared_papers >= %(co_min)s
),
recent_papers AS (
    SELECT tp.focal_id, p.bibcode, p.year, p.title, tp.fused_confidence,
           ROW_NUMBER() OVER (
               PARTITION BY tp.focal_id
               ORDER BY p.year DESC NULLS LAST, tp.fused_confidence DESC,
                        p.bibcode
           ) AS rn
    FROM top_papers tp
    LEFT JOIN papers p ON p.bibcode = tp.bibcode
)
SELECT 'co'::text AS kind,
       focal_id,
       rn,
       canonical_name,
       entity_type,
       shared_papers,
       NULL::text AS bibcode,
       NULL::int  AS year,
       NULL::text AS title,
       NULL::double precision AS fused_confidence
FROM co_top
UNION ALL
SELECT 'recent'::text AS kind,
       focal_id,
       rn,
       NULL::text     AS canonical_name,
       NULL::text     AS entity_type,
       NULL::bigint   AS shared_papers,
       bibcode,
       year,
       title,
       fused_confidence
FROM recent_papers
WHERE rn <= %(recent_top_k)s
"""


def fetch_entity_mini_summaries(
    conn: psycopg.Connection,
    entity_ids: list[int],
    *,
    co_occur_min: int = 2,
    co_occur_top_k: int = 3,
    recent_top_k: int = 2,
    sample_papers: int = 100,
) -> dict[int, dict[str, Any]]:
    """Batched co-occurrence + recent-example lookup for resolver candidates.

    Args:
        conn: Live psycopg connection (caller manages tx scope).
        entity_ids: Focal entity_ids to enrich. Empty list -> empty dict.
        co_occur_min: Minimum number of shared sampled papers for a co-
            occurring entity to surface. Defaults to 2 to filter out
            singleton co-mentions in the sampled paper set.
        co_occur_top_k: Cap on co_occurring list per focal entity.
        recent_top_k: Cap on recent_examples list per focal entity.
        sample_papers: Top-N highest-confidence papers per focal entity to
            scan when computing co-occurrence and recent examples. Bounds
            cost; the index ``idx_dec_entity_fused`` makes this cheap.

    Returns:
        ``{entity_id: {"co_occurring": list | None,
                        "recent_examples": list | None}}``

        Entries with no qualifying rows surface as ``None`` (not ``[]``)
        per bead D2 ("emit null instead of `[]` so absence is unambiguous").
        Entity_ids with no rows in ``document_entities_canonical`` at all
        are still keyed in the result dict, both fields ``None``.
    """
    if not entity_ids:
        return {}

    # Deduplicate while preserving caller-visible IDs.
    focal_ids = list(dict.fromkeys(int(eid) for eid in entity_ids))

    params = {
        "focal_ids": focal_ids,
        "sample": int(sample_papers),
        "co_top_k": int(co_occur_top_k),
        "co_min": int(co_occur_min),
        "recent_top_k": int(recent_top_k),
    }

    # Pre-seed the result with nulls so callers always get an entry per
    # requested entity_id, regardless of data availability.
    result: dict[int, dict[str, Any]] = {
        eid: {"co_occurring": None, "recent_examples": None} for eid in focal_ids
    }

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_MINI_SUMMARY_SQL, params)
        rows = cur.fetchall()

    co_buckets: dict[int, list[dict[str, Any]]] = {eid: [] for eid in focal_ids}
    recent_buckets: dict[int, list[dict[str, Any]]] = {eid: [] for eid in focal_ids}

    for row in rows:
        focal_id = row["focal_id"]
        kind = row["kind"]
        if kind == "co":
            co_buckets.setdefault(focal_id, []).append(
                {
                    "canonical_name": row["canonical_name"],
                    "entity_type": row["entity_type"],
                    "shared_papers": int(row["shared_papers"]),
                }
            )
        elif kind == "recent":
            # Recent rows are emitted regardless of whether papers join
            # produced a year/title (LEFT JOIN); skip rows that have no
            # bibcode -- those would be empty placeholders.
            if row["bibcode"] is None:
                continue
            recent_buckets.setdefault(focal_id, []).append(
                {
                    "bibcode": row["bibcode"],
                    "year": int(row["year"]) if row["year"] is not None else None,
                    "title": row["title"],
                }
            )

    for eid in focal_ids:
        co = co_buckets.get(eid) or []
        recent = recent_buckets.get(eid) or []
        result[eid]["co_occurring"] = co if co else None
        result[eid]["recent_examples"] = recent if recent else None

    return result
