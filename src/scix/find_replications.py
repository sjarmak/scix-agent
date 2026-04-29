"""find_replications — forward citations to a target paper, ranked by intent.

Implements the second half of MH-4 of the SciX Deep Search v1 PRD
(``docs/prd/scix_deep_search_v1.md``). Returns papers that cite a given
target bibcode, annotated with citation intent, an inferred replication
relation, and a hedge-presence flag drawn from the citation context.

Hedge / relation inference (heuristic substitute for NegBERT)
-------------------------------------------------------------

NegBERT (Khandelwal & Sawant 2020, arXiv:1911.04211) is a BERT model
fine-tuned for negation cue detection. It is **not** wired into this repo
yet; per PRD MH-4, we ship a documented hand-rolled heuristic substitute
that detects:

* **Hedge cues** — case-insensitive lookup of: ``may``, ``might``,
  ``could``, ``suggests``, ``suggesting``, ``appears``, ``appear``,
  ``preliminary``, ``tentative``, ``tentatively``, ``possibly``,
  ``perhaps``, ``likely``, ``unclear``, ``seem``, ``seems``.
* **Comparison verbs** — case-insensitive substring match of:

    - replicates: ``agrees with``, ``consistent with``, ``confirm``
      (matches confirms/confirmed/we confirm), ``reproduce``
      (matches reproduces/reproduced), ``replicate`` (matches
      replicates/replicated), ``in agreement with``
    - refutes: ``contradicts``, ``disagrees with``, ``inconsistent with``,
      ``refutes``, ``refuted``, ``rules out``, ``excludes``,
      ``in tension with``
    - qualifies: ``extends``, ``qualifies``, ``narrows``, ``clarifies``,
      ``revises``, ``supersedes``
    - partial: ``partially``, ``mixed``, ``in part``,
      ``qualified support``

Inference order: ``partial`` → ``refutes`` → ``qualifies`` → ``replicates``.
A hedge cue paired with a replicates match is downgraded to ``qualifies``
(hedged agreement is rarely a clean replication). When no pattern matches,
``relation_inferred`` is ``"unknown"``.

NegBERT remains the future drop-in: when wired up it will replace
:func:`_detect_hedge` (cue extraction) and contribute calibrated
probabilities to :func:`_infer_relation`.

Ranking
-------

Rows are ranked by ``intent_weight`` descending (per the MH-4 spec) using
the same weight table as :mod:`scix.claim_blame`::

    {result_comparison: 1.0, method: 0.6, background: 0.3}

Year ascending breaks ties so the chronologically earliest replication is
surfaced first.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

import psycopg

from scix.citation_contexts_coverage import compute_coverage, empty_coverage
from scix.citation_intent import DEFAULT_INTENT_WEIGHT, INTENT_WEIGHTS
from scix.research_scope import ResearchScope, scope_to_sql_clauses

logger = logging.getLogger(__name__)

Relation = Literal["replicates", "refutes", "qualifies", "partial"]

VALID_RELATIONS: frozenset[str] = frozenset({"replicates", "refutes", "qualifies", "partial"})

HEDGE_CUES: frozenset[str] = frozenset(
    {
        "may",
        "might",
        "could",
        "suggests",
        "suggesting",
        "appears",
        "appear",
        "preliminary",
        "tentative",
        "tentatively",
        "possibly",
        "perhaps",
        "likely",
        "unclear",
        "seem",
        "seems",
    }
)

REPLICATION_PATTERNS: tuple[str, ...] = (
    "agrees with",
    "consistent with",
    "confirm",  # also matches "confirms" / "confirmed" / "we confirm"
    "reproduce",  # also matches "reproduces" / "reproduced"
    "replicate",  # also matches "replicates" / "replicated"
    "in agreement with",
)

REFUTATION_PATTERNS: tuple[str, ...] = (
    "contradicts",
    "disagrees with",
    "inconsistent with",
    "refutes",
    "refuted",
    "rules out",
    "excludes",
    "in tension with",
)

QUALIFIES_PATTERNS: tuple[str, ...] = (
    "extends",
    "qualifies",
    "narrows",
    "clarifies",
    "revises",
    "supersedes",
)

PARTIAL_PATTERNS: tuple[str, ...] = (
    "partially",
    "mixed",
    "in part",
    "qualified support",
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """One forward citation returned by :func:`find_replications`.

    Attributes:
        citing_bibcode: ADS bibcode of the paper that cites the target.
        year: Publication year of the citing paper, or ``None``.
        intent: Citation-intent class, or ``None`` when SciCite has not
            backfilled this row.
        intent_weight: Numeric weight per the MH-4 spec.
        context_snippet: ≤1000 chars of citation context.
        relation_inferred: One of ``replicates``, ``refutes``, ``qualifies``,
            ``partial``, or ``unknown``.
        hedge_present: True if any HEDGE_CUE token appears in the context.
    """

    citing_bibcode: str
    year: int | None
    intent: str | None
    intent_weight: float
    context_snippet: str
    relation_inferred: str
    hedge_present: bool


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def find_replications(
    target_bibcode: str,
    relation: Relation | None = None,
    scope: ResearchScope | None = None,
    db_pool: Any = None,
    *,
    conn: psycopg.Connection | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Return forward citations to ``target_bibcode`` ranked by intent_weight.

    Args:
        target_bibcode: ADS bibcode whose forward citations to enumerate.
        relation: Optional filter — keep only citations whose
            ``relation_inferred`` matches.
        scope: Optional :class:`ResearchScope` filters (year_window applies
            to the citing paper's year).
        db_pool: Optional psycopg connection pool. When ``None`` and
            ``conn`` is also ``None``, falls back to
            ``mcp_server._get_pool()``.
        conn: Optional pre-acquired connection (used by MCP dispatch).
        limit: Maximum citations to return.

    Returns:
        Dict with keys:

        * ``citations``: list of :class:`Citation` dicts sorted by
          intent_weight DESC, year ASC. Empty list when
          ``target_bibcode`` is unknown or has no forward citations in
          ``v_claim_edges``.
        * ``coverage``: coverage block (see
          :mod:`scix.citation_contexts_coverage`) so callers can
          distinguish 'no events' from 'no coverage'.
    """
    if not isinstance(target_bibcode, str) or not target_bibcode.strip():
        return {"citations": [], "coverage": empty_coverage()}
    if relation is not None and relation not in VALID_RELATIONS:
        raise ValueError(
            f"relation must be one of {sorted(VALID_RELATIONS)} or None; " f"got {relation!r}"
        )

    effective_scope = scope or ResearchScope()

    if conn is not None:
        rows = _query_citations(conn, target_bibcode, effective_scope, limit)
        coverage = compute_coverage(conn, [target_bibcode])
    else:
        pool = db_pool if db_pool is not None else _default_pool()
        with pool.connection() as acquired:
            rows = _query_citations(acquired, target_bibcode, effective_scope, limit)
            coverage = compute_coverage(acquired, [target_bibcode])

    citations: list[Citation] = []
    for row in rows:
        snippet = row.get("context_snippet") or ""
        intent = row.get("intent")
        weight = INTENT_WEIGHTS.get(intent, DEFAULT_INTENT_WEIGHT)
        hedge = _detect_hedge(snippet)
        rel = _infer_relation(snippet, hedge)
        citations.append(
            Citation(
                citing_bibcode=row["citing_bibcode"],
                year=row.get("year"),
                intent=intent,
                intent_weight=weight,
                context_snippet=snippet,
                relation_inferred=rel,
                hedge_present=hedge,
            )
        )

    if relation is not None:
        citations = [c for c in citations if c.relation_inferred == relation]

    citations.sort(
        key=lambda c: (
            -c.intent_weight,
            c.year if c.year is not None else 9999,
            c.citing_bibcode,
        )
    )

    return {
        "citations": [asdict(c) for c in citations],
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# SQL helper
# ---------------------------------------------------------------------------


def _query_citations(
    conn: psycopg.Connection,
    target_bibcode: str,
    scope: ResearchScope,
    limit: int,
) -> list[dict[str, Any]]:
    base_sql = (
        "SELECT vce.source_bibcode AS citing_bibcode, "
        "vce.source_year AS year, "
        "vce.intent, "
        "vce.context_snippet, "
        "vce.section_name "
        "FROM v_claim_edges vce "
    )

    where_parts = ["vce.target_bibcode = %s"]
    params: list[Any] = [target_bibcode]

    # year_window we honour directly against vce.source_year (avoids needing
    # the papers join).
    if scope.year_window is not None:
        lo, hi = scope.year_window
        where_parts.append(
            "(vce.source_year IS NULL OR (vce.source_year >= %s AND vce.source_year <= %s))"
        )
        params.extend([lo, hi])

    # Other scope fields require the papers/paper_metrics join — wire it up
    # when any of them is set. We delegate the clause derivation to
    # research_scope.scope_to_sql_clauses.
    other_scope_active = any(
        getattr(scope, name) is not None
        for name in (
            "community_ids",
            "methodology_class",
            "instruments",
            "exclude_authors",
            "exclude_funders",
            "min_venue_tier",
        )
    )
    if other_scope_active:
        base_sql += "JOIN papers p ON p.bibcode = vce.source_bibcode "
        if scope.community_ids is not None:
            base_sql += "JOIN paper_metrics pm ON pm.bibcode = vce.source_bibcode "
        # Build the scope clause without year_window (already applied above)
        # by constructing a copy with year_window cleared.
        scope_no_year = ResearchScope(
            community_ids=scope.community_ids,
            year_window=None,
            methodology_class=scope.methodology_class,
            instruments=scope.instruments,
            exclude_authors=scope.exclude_authors,
            exclude_funders=scope.exclude_funders,
            min_venue_tier=scope.min_venue_tier,
            leiden_resolution=scope.leiden_resolution,
        )
        clause, scope_params = scope_to_sql_clauses(
            scope_no_year, {"papers": "p", "paper_metrics": "pm"}
        )
        if clause:
            where_parts.append(clause)
            params.extend(scope_params)

    base_sql += "WHERE " + " AND ".join(where_parts)
    base_sql += " ORDER BY vce.source_year ASC NULLS LAST, vce.char_offset ASC NULLS LAST"
    base_sql += " LIMIT %s"
    params.append(limit)

    try:
        with conn.cursor() as cur:
            cur.execute(base_sql, params)
            rows = cur.fetchall()
    except psycopg.Error as exc:
        logger.warning("find_replications: query failed for %s: %s", target_bibcode, exc)
        return []

    return [
        {
            "citing_bibcode": r[0],
            "year": r[1],
            "intent": r[2],
            "context_snippet": r[3] or "",
            "section_name": r[4],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Heuristic substitute for NegBERT
# ---------------------------------------------------------------------------


def _detect_hedge(text: str) -> bool:
    """Hand-rolled heuristic for hedge-cue detection — NegBERT substitute.

    Tokenises ``text`` on whitespace + punctuation and checks whether any
    token (case-folded) is in :data:`HEDGE_CUES`. NegBERT will replace this
    with a calibrated cue-extraction model (PRD MH-4).
    """
    if not text:
        return False
    lowered = text.lower()
    # Quick reject before tokenising.
    if not any(cue in lowered for cue in HEDGE_CUES):
        return False
    # Tokenise on non-alphanumeric for cheap word-boundary semantics.
    tokens = set(re.findall(r"[a-z]+", lowered))
    return bool(tokens & HEDGE_CUES)


def _infer_relation(text: str, hedge_present: bool) -> str:
    """Heuristic relation inference. See module docstring for the order."""
    if not text:
        return "unknown"
    lowered = text.lower()

    if any(p in lowered for p in PARTIAL_PATTERNS):
        return "partial"
    if any(p in lowered for p in REFUTATION_PATTERNS):
        return "refutes"
    if any(p in lowered for p in QUALIFIES_PATTERNS):
        return "qualifies"
    if any(p in lowered for p in REPLICATION_PATTERNS):
        # Hedged agreement isn't a clean replication.
        return "qualifies" if hedge_present else "replicates"
    return "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_pool() -> Any:
    """Lazy import to avoid cycles between mcp_server and find_replications."""
    from scix.mcp_server import _get_pool

    return _get_pool()


__all__ = [
    "Citation",
    "DEFAULT_INTENT_WEIGHT",
    "HEDGE_CUES",
    "INTENT_WEIGHTS",
    "PARTIAL_PATTERNS",
    "QUALIFIES_PATTERNS",
    "REFUTATION_PATTERNS",
    "REPLICATION_PATTERNS",
    "Relation",
    "VALID_RELATIONS",
    "find_replications",
]
