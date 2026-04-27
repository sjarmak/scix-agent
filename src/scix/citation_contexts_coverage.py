"""Citation-contexts coverage probe for claim_blame / find_replications.

Surfaces *whether* a set of seed bibcodes is covered by the
``citation_contexts`` table at all (via the ``v_claim_edges`` view), so
agents can distinguish two responses that look identical:

* **No events** — the seed papers ARE covered by ``citation_contexts``,
  but no replication / blame events were found. The signal is real;
  trust the empty result.
* **No coverage** — the seed papers are NOT in ``citation_contexts``.
  An empty result is silent about the underlying topic — the agent
  should use a different tool (``citation_traverse``,
  ``concept_search``) or broaden its query.

``citation_contexts`` covers ~0.27% of citation edges
(see ``docs/citation_contexts_coverage.md``); the gap is the dominant
reason these tools return empty results on niche topics. Without this
signal the agent treats both cases as 'topic genuinely has no events',
which is wrong about half the time on out-of-corpus subdomains
(bead scix_experiments-7avw).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import psycopg

logger = logging.getLogger(__name__)

# Path used in the response ``note`` so agents can surface it inline.
COVERAGE_DOC_PATH: str = "docs/citation_contexts_coverage.md"

DEFAULT_COVERAGE_NOTE: str = (
    "citation_contexts has ~0.27% edge coverage; results may be undercounting. "
    f"See {COVERAGE_DOC_PATH} for the no-events vs no-coverage distinction."
)


def compute_coverage(
    conn: psycopg.Connection,
    seeds: Iterable[str],
) -> dict[str, Any]:
    """Return a coverage block for ``seeds`` against ``v_claim_edges``.

    A seed is *covered* when it appears in ``v_claim_edges`` as either
    ``source_bibcode`` (i.e. the paper has outgoing citation contexts —
    relevant to ``claim_blame``'s reverse walk) or ``target_bibcode``
    (i.e. the paper has incoming citation contexts — relevant to
    ``find_replications``'s forward walk). The two indexes
    ``idx_v_claim_edges_source_intent`` and
    ``idx_v_claim_edges_target_intent`` (migration 057) keep this probe
    cheap.

    Args:
        conn: Live psycopg connection. Caller owns lifecycle.
        seeds: Iterable of ADS bibcodes. Duplicates are deduped.

    Returns:
        Dict with keys:
            * ``covered_seeds`` (int): unique seed bibcodes present in
              ``v_claim_edges`` on either side.
            * ``total_seeds`` (int): unique seed bibcodes provided.
            * ``coverage_pct`` (float): ``covered_seeds / total_seeds``,
              or ``0.0`` when ``total_seeds == 0`` (avoids div-by-zero
              while still emitting a coverage block).
            * ``note`` (str): canonical advisory string referencing
              :data:`COVERAGE_DOC_PATH`.
    """
    unique_seeds: list[str] = sorted({s for s in seeds if isinstance(s, str) and s})
    total = len(unique_seeds)

    if total == 0:
        return _coverage_block(covered=0, total=0)

    sql = (
        "SELECT COUNT(DISTINCT bib) FROM ("
        "  SELECT source_bibcode AS bib FROM v_claim_edges "
        "    WHERE source_bibcode = ANY(%s) "
        "  UNION "
        "  SELECT target_bibcode AS bib FROM v_claim_edges "
        "    WHERE target_bibcode = ANY(%s) "
        ") AS covered"
    )

    try:
        with conn.cursor() as cur:
            cur.execute(sql, (unique_seeds, unique_seeds))
            row = cur.fetchone()
    except psycopg.Error as exc:  # pragma: no cover — defensive
        logger.warning("citation_contexts_coverage: probe failed: %s", exc)
        return _coverage_block(covered=0, total=total)

    covered = int(row[0]) if row and row[0] is not None else 0
    return _coverage_block(covered=covered, total=total)


def empty_coverage() -> dict[str, Any]:
    """Return a zero-coverage block for response paths that bypass the DB.

    Useful for early-return paths (e.g. empty input validation) so the
    response shape stays uniform for downstream agents.
    """
    return _coverage_block(covered=0, total=0)


def _coverage_block(*, covered: int, total: int) -> dict[str, Any]:
    pct = (covered / total) if total > 0 else 0.0
    return {
        "covered_seeds": covered,
        "total_seeds": total,
        "coverage_pct": pct,
        "note": DEFAULT_COVERAGE_NOTE,
    }


__all__ = [
    "COVERAGE_DOC_PATH",
    "DEFAULT_COVERAGE_NOTE",
    "compute_coverage",
    "empty_coverage",
]
