"""Citation-consistency precision proxy (PRD §S1).

For a paper P and entity E, citation_consistency(P, E) is the fraction of
papers that P cites that *also* link to E at the same link_type.

Intuition: if P cites 10 papers and 7 of those cited papers also mention the
Hubble Space Telescope, then P->HST has high citation consistency; if P's
link to HST is a one-off match among unrelated citations, consistency is low.

This is a precision *proxy*, not ground truth. Computed lazily and persisted
into ``document_entities.citation_consistency`` by offline jobs. This module
provides the pure computation; the writer lives in the resolver service
(M13) and is out of scope here.

SQL access pattern:
    SELECT COUNT(*) AS total_cites,
           COUNT(DISTINCT de.bibcode) AS matching_cites
    FROM citation_edges ce
    LEFT JOIN document_entities de
           ON de.bibcode = ce.target_bibcode
          AND de.entity_id = %(entity_id)s
          AND de.link_type = %(link_type)s
          AND (%(tier)s IS NULL OR de.tier = %(tier)s)
    WHERE ce.source_bibcode = %(bibcode)s;

READS from ``document_entities`` are allowed by the M13 AST lint — only
writes go through ``resolve_entities.py``. See ``scripts/ast_lint_resolver.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import psycopg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsistencyResult:
    """Outcome of a citation-consistency computation."""

    bibcode: str
    entity_id: int
    link_type: str
    tier: Optional[int]
    total_cites: int
    matching_cites: int
    consistency: Optional[float]


def compute_consistency(
    bibcode: str,
    entity_id: int,
    *,
    conn: psycopg.Connection,
    link_type: str = "mention",
    tier: Optional[int] = None,
) -> ConsistencyResult:
    """Compute citation consistency for (bibcode, entity_id).

    Returns a :class:`ConsistencyResult`. ``consistency`` is ``None`` when
    the paper has zero outbound citations — callers must handle this case
    (e.g. leave the column NULL rather than writing 0.0).

    Parameters
    ----------
    bibcode
        Citing paper (source_bibcode in citation_edges).
    entity_id
        Entity whose consistency we're measuring.
    conn
        Open psycopg connection. Caller owns commit/rollback.
    link_type
        ``document_entities.link_type`` to filter on. Defaults to ``"mention"``.
    tier
        Optional tier filter. ``None`` means "any tier".
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) AS total_cites,
                COUNT(DISTINCT de.bibcode) AS matching_cites
            FROM citation_edges ce
            LEFT JOIN document_entities de
                   ON de.bibcode = ce.target_bibcode
                  AND de.entity_id = %(entity_id)s
                  AND de.link_type = %(link_type)s
                  AND (%(tier)s::smallint IS NULL OR de.tier = %(tier)s::smallint)
            WHERE ce.source_bibcode = %(bibcode)s
            """,
            {
                "bibcode": bibcode,
                "entity_id": entity_id,
                "link_type": link_type,
                "tier": tier,
            },
        )
        row = cur.fetchone()

    total_cites = int(row[0]) if row and row[0] is not None else 0
    matching_cites = int(row[1]) if row and row[1] is not None else 0

    if total_cites == 0:
        consistency: Optional[float] = None
    else:
        consistency = matching_cites / total_cites

    return ConsistencyResult(
        bibcode=bibcode,
        entity_id=entity_id,
        link_type=link_type,
        tier=tier,
        total_cites=total_cites,
        matching_cites=matching_cites,
        consistency=consistency,
    )
