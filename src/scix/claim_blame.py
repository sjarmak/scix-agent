"""claim_blame — trace a claim back to its chronologically earliest origin.

Implements MH-4 of the SciX Deep Search v1 PRD
(``docs/prd/scix_deep_search_v1.md``). Walks reverse references over **all**
citation contexts (no hard intent filter) and returns the chronologically
earliest non-retracted paper as the origin, with a Hop chain that surfaces
``intent`` and ``intent_weight`` on every step.

Ranking
-------

Per the PRD Q2 resolution, the ordering is exactly:

    (chronological_priority, intent_weight, semantic_match)

where intent_weight is::

    {result_comparison: 1.0, method: 0.6, background: 0.3}

Chronological-earliest is the laundering guard: citation laundering is
fundamentally a temporal problem. ``intent_weight`` is the secondary
precision signal — it shapes which citations earn the most credit when two
papers share a year. ``semantic_match`` (cosine similarity between the
claim's INDUS embedding and each candidate's paper embedding) is the
tiebreaker.

Retraction handling
-------------------

A candidate is excluded from origin selection when ``papers.correction_events``
contains an event of ``type='retraction'``. Any such bibcode that appears in
the candidate or hop chain is surfaced in the response's
``retraction_warnings`` list — never silently dropped — so callers can
mark the lineage as contested. Errata, expressions of concern, and other
non-retraction events do **not** disqualify a paper from origin selection in
v1; they will land in v1.1 once the broader correction-event UX matures.

Scope handling
--------------

The optional :class:`scix.research_scope.ResearchScope` is honoured on:

* candidate seeding (``year_window`` filters which papers can be candidates),
* hop traversal (``year_window`` filters lineage hops),
* community / methodology / instrument / venue filters (delegated to
  :func:`scope_to_sql_clauses`).

Confidence
----------

The returned ``confidence`` is a deterministic combination of three signals
in ``[0, 1]``::

    confidence = clamp(
        0.5 * intent_weight_of_origin
      + 0.3 * chronology_score
      + 0.2 * semantic_match,
        0.0, 1.0,
    )

``chronology_score`` is ``1.0`` when origin year strictly precedes every
non-retracted candidate's year, ``0.5`` otherwise. ``semantic_match`` is the
INDUS cosine similarity in ``[-1, 1]`` mapped to ``[0, 1]`` via
``(s + 1) / 2``; if no embedding is available, ``0.5`` is substituted.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable

import psycopg

from scix.citation_contexts_coverage import compute_coverage, empty_coverage
from scix.citation_intent import DEFAULT_INTENT_WEIGHT, INTENT_WEIGHTS
from scix.research_scope import ResearchScope, scope_to_sql_clauses

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hop:
    """One step in the lineage chain returned by :func:`claim_blame`.

    Attributes:
        bibcode: ADS bibcode of the cited / cited-by paper at this hop.
        year: Publication year of the paper at this hop, or ``None`` if
            unknown.
        intent: Citation-intent class for the hop, or ``None`` when the
            SciBERT-SciCite backfill has not run on the underlying row.
        intent_weight: Numeric weight for the intent (per the MH-4 spec).
        context_snippet: ≤1000 chars of the in-text citation context.
        section_name: Section the citation appeared in, or ``None``.
    """

    bibcode: str
    year: int | None
    intent: str | None
    intent_weight: float
    context_snippet: str
    section_name: str | None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


SeedFn = Callable[
    [psycopg.Connection, str, ResearchScope, int],
    list[tuple[str, "int | None", "list[float] | None"]],
]
"""Signature of the candidate seeder. Returns ``(bibcode, year, embedding)``
tuples; the embedding may be ``None`` (semantic_match defaults to ``0.5``).
"""


EmbedFn = Callable[[str], "list[float] | None"]
"""Signature of the query embedder. Receives the claim text, returns a 768d
INDUS embedding (or ``None`` when embedding is unavailable)."""


def claim_blame(
    claim_text: str,
    scope: ResearchScope | None = None,
    db_pool: Any = None,
    *,
    conn: psycopg.Connection | None = None,
    seed_candidates_fn: SeedFn | None = None,
    embed_query_fn: EmbedFn | None = None,
    candidate_limit: int = 20,
    lineage_limit: int = 10,
) -> dict[str, Any]:
    """Trace ``claim_text`` back to its chronologically earliest origin.

    Args:
        claim_text: The natural-language assertion to trace (e.g.
            ``"local H0 measurement is 73 km/s/Mpc"``).
        scope: Optional :class:`ResearchScope` filters. ``year_window`` is
            applied both to candidate seeding and to lineage hops.
        db_pool: Optional psycopg connection pool. When ``None`` and ``conn``
            is also ``None``, falls back to ``mcp_server._get_pool()``. Pool
            acquisition uses the standard ``with pool.connection() as c:``
            pattern.
        conn: Optional pre-acquired connection (used by the MCP dispatch
            path so we don't double-acquire). Mutually exclusive with
            ``db_pool`` in practice; if both are passed, ``conn`` wins.
        seed_candidates_fn: Optional override for candidate seeding (used by
            tests). Production default scans ``papers.title``/``abstract``
            with a lexical match.
        embed_query_fn: Optional override for INDUS embedding of
            ``claim_text``. Production default loads INDUS via
            :func:`scix.embed.load_model` and runs one ``embed_batch`` call.
        candidate_limit: Maximum candidates to seed.
        lineage_limit: Maximum hops to walk per candidate.

    Returns:
        Dict with keys:

        * ``origin``: bibcode of the chronologically earliest non-retracted
          paper supporting the claim (``""`` when no candidates found).
        * ``lineage``: list of :class:`Hop` dicts (origin first, then
          chronological ascending).
        * ``confidence``: float in ``[0, 1]`` (see module docstring).
        * ``retraction_warnings``: list of bibcodes touched by the lineage
          that have a retraction event in ``papers.correction_events``.
    """
    if not isinstance(claim_text, str) or not claim_text.strip():
        # No DB call possible without claim text — emit a zero coverage
        # block so the response shape stays uniform for callers.
        return {
            "origin": "",
            "lineage": [],
            "confidence": 0.0,
            "retraction_warnings": [],
            "coverage": empty_coverage(),
        }

    effective_scope = scope or ResearchScope()

    if conn is not None:
        return _run_claim_blame(
            conn,
            claim_text,
            effective_scope,
            seed_candidates_fn=seed_candidates_fn,
            embed_query_fn=embed_query_fn,
            candidate_limit=candidate_limit,
            lineage_limit=lineage_limit,
        )

    pool = db_pool if db_pool is not None else _default_pool()
    with pool.connection() as acquired:
        return _run_claim_blame(
            acquired,
            claim_text,
            effective_scope,
            seed_candidates_fn=seed_candidates_fn,
            embed_query_fn=embed_query_fn,
            candidate_limit=candidate_limit,
            lineage_limit=lineage_limit,
        )


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _run_claim_blame(
    conn: psycopg.Connection,
    claim_text: str,
    scope: ResearchScope,
    *,
    seed_candidates_fn: SeedFn | None,
    embed_query_fn: EmbedFn | None,
    candidate_limit: int,
    lineage_limit: int,
) -> dict[str, Any]:
    seed_fn = seed_candidates_fn or _seed_candidates_default
    embed_fn = embed_query_fn or _embed_query_default

    query_embedding = embed_fn(claim_text)
    candidates = seed_fn(conn, claim_text, scope, candidate_limit)
    # Coverage probe: do candidate seeds appear in citation_contexts?
    # The result lets agents distinguish 'no events' from 'no coverage'.
    coverage = compute_coverage(conn, [c[0] for c in candidates])
    if not candidates:
        return {
            "origin": "",
            "lineage": [],
            "confidence": 0.0,
            "retraction_warnings": [],
            "coverage": coverage,
        }

    # Walk reverse references for each candidate. ``hops_by_target`` maps
    # ``(target_bibcode)`` -> the highest-weight hop record we've seen for
    # that referenced paper, so we can rank targets directly.
    all_hops: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int | None]] = set()
    candidate_years: dict[str, int | None] = {c[0]: c[1] for c in candidates}

    for cand_bibcode, _cand_year, _cand_embedding in candidates:
        rows = _walk_reverse_references(conn, cand_bibcode, scope, lineage_limit)
        for row in rows:
            target_bib = row["target_bibcode"]
            offset = row.get("char_offset")
            key = (cand_bibcode, target_bib, offset)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_hops.append(row)

    # Build the candidate set used for origin selection — both the seed
    # candidates AND the targets of their reverse references are
    # claim-supporting papers. Year is taken from v_claim_edges' target_year
    # for hop-derived bibcodes and from the seed result for candidates.
    universe: dict[str, int | None] = dict(candidate_years)
    for hop in all_hops:
        bib = hop["target_bibcode"]
        if bib not in universe:
            universe[bib] = hop.get("target_year")

    # Apply year_window to the universe — scope honours acceptance criteria
    # 8(c). Candidate seeding already applies year_window via SQL, but a hop
    # target may fall outside the window (the source paper is in-scope, the
    # cited paper isn't). Drop those.
    if scope.year_window is not None:
        lo, hi = scope.year_window
        universe = {bib: yr for bib, yr in universe.items() if yr is None or lo <= yr <= hi}

    if not universe:
        return {
            "origin": "",
            "lineage": [],
            "confidence": 0.0,
            "retraction_warnings": [],
            "coverage": coverage,
        }

    retracted = _lookup_retractions(conn, list(universe.keys()))

    # Origin selection: chronologically earliest non-retracted bibcode.
    # When years tie, fall back to intent_weight (max across hops to that
    # bibcode) and then semantic_match (max likewise). All three signals
    # combine into a sort key per the PRD Q2 ranking.
    intent_weight_by_bib: dict[str, float] = {}
    semantic_by_bib: dict[str, float] = {}
    for hop in all_hops:
        bib = hop["target_bibcode"]
        weight = INTENT_WEIGHTS.get(hop.get("intent"), DEFAULT_INTENT_WEIGHT)
        if weight > intent_weight_by_bib.get(bib, -1.0):
            intent_weight_by_bib[bib] = weight
        sm = hop.get("semantic_match", 0.0) or 0.0
        if sm > semantic_by_bib.get(bib, -2.0):
            semantic_by_bib[bib] = sm
    # Seed candidates also count as claim-supporting; assign them the
    # highest intent_weight (they directly assert the claim by virtue of
    # matching the seed query) so a candidate that isn't a hop target still
    # ranks fairly.
    for cand_bib, _cand_year, cand_embedding in candidates:
        intent_weight_by_bib.setdefault(cand_bib, INTENT_WEIGHTS["result_comparison"])
        sm = _cosine(query_embedding, cand_embedding)
        if sm is not None and sm > semantic_by_bib.get(cand_bib, -2.0):
            semantic_by_bib[cand_bib] = sm

    candidates_for_origin: list[tuple[str, int | None, float, float]] = []
    for bib, year in universe.items():
        if bib in retracted:
            continue
        candidates_for_origin.append(
            (
                bib,
                year,
                intent_weight_by_bib.get(bib, DEFAULT_INTENT_WEIGHT),
                semantic_by_bib.get(bib, 0.0),
            )
        )

    if not candidates_for_origin:
        # All universe papers are retracted — return empty origin but still
        # surface the warning list so the caller can see the lineage was
        # tainted.
        return {
            "origin": "",
            "lineage": [],
            "confidence": 0.0,
            "retraction_warnings": sorted(retracted),
            "coverage": coverage,
        }

    # Sort: chronological ASC (None last), then intent_weight DESC,
    # then semantic_match DESC, then bibcode ASC for deterministic ties.
    candidates_for_origin.sort(
        key=lambda t: (
            t[1] if t[1] is not None else 9999,
            -t[2],
            -t[3],
            t[0],
        )
    )

    origin_bib, origin_year, origin_weight, origin_semantic = candidates_for_origin[0]

    # Build the lineage. Surface intent and intent_weight on every Hop
    # (acceptance criterion 3). The lineage starts at the origin and walks
    # forward through each candidate's hop record that pointed at it. If a
    # candidate didn't have a hop to the origin, we add a synthetic Hop
    # carrying the candidate's metadata and ``intent=None`` (no citation
    # context exists in v_claim_edges).
    lineage = _build_lineage(
        origin_bib,
        origin_year,
        all_hops,
        candidates,
    )

    # Confidence (per module docstring).
    chronology_score = (
        1.0
        if all(
            (year is None or origin_year is None or origin_year < year)
            for bib, year, _, _ in candidates_for_origin[1:]
        )
        else 0.5
    )
    semantic_score_01 = (origin_semantic + 1.0) / 2.0
    confidence = 0.5 * origin_weight + 0.3 * chronology_score + 0.2 * semantic_score_01
    confidence = max(0.0, min(1.0, confidence))

    return {
        "origin": origin_bib,
        "lineage": [asdict(h) for h in lineage],
        "confidence": confidence,
        "retraction_warnings": sorted(retracted),
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def _walk_reverse_references(
    conn: psycopg.Connection,
    source_bibcode: str,
    scope: ResearchScope,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch v_claim_edges rows where ``source_bibcode = %s``.

    Each row represents one in-text citation from ``source_bibcode`` to a
    referenced paper. We don't apply intent filtering — the PRD walks
    "all citation contexts" — but we do apply the scope's year_window to
    ``target_year`` so out-of-scope ancestors are dropped at the SQL layer.
    """
    base_sql = (
        "SELECT source_bibcode, target_bibcode, context_snippet, intent, "
        "section_name, source_year, target_year, char_offset "
        "FROM v_claim_edges "
        "WHERE source_bibcode = %s"
    )
    params: list[Any] = [source_bibcode]

    if scope.year_window is not None:
        lo, hi = scope.year_window
        base_sql += " AND (target_year IS NULL OR (target_year >= %s AND target_year <= %s))"
        params.extend([lo, hi])

    base_sql += " ORDER BY target_year ASC NULLS LAST, char_offset ASC NULLS LAST LIMIT %s"
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute(base_sql, params)
        rows = cur.fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "source_bibcode": row[0],
                "target_bibcode": row[1],
                "context_snippet": row[2] or "",
                "intent": row[3],
                "section_name": row[4],
                "source_year": row[5],
                "target_year": row[6],
                "char_offset": row[7],
            }
        )
    return out


def _lookup_retractions(
    conn: psycopg.Connection,
    bibcodes: list[str],
) -> set[str]:
    """Return the subset of ``bibcodes`` that have a retraction event.

    Reads ``papers.correction_events`` (migration 058). Only events of
    ``type='retraction'`` are considered for origin disqualification; other
    correction-event types are surfaced in v1.1 once the broader
    correction-event UX lands.
    """
    if not bibcodes:
        return set()

    sql = (
        "SELECT bibcode FROM papers "
        "WHERE bibcode = ANY(%s) "
        "AND EXISTS (SELECT 1 FROM jsonb_array_elements(correction_events) AS ev "
        "WHERE ev->>'type' = 'retraction')"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (list(bibcodes),))
        rows = cur.fetchall()
    return {r[0] for r in rows}


def _seed_candidates_default(
    conn: psycopg.Connection,
    claim_text: str,
    scope: ResearchScope,
    limit: int,
) -> list[tuple[str, int | None, list[float] | None]]:
    """Seed candidate papers via lexical title/abstract match.

    Production default: tokenise the claim into the longest few words,
    build an ILIKE pattern, filter by scope, and return the top N by year
    ascending. Tests override via ``seed_candidates_fn``.
    """
    # Pick the longest ~3 tokens as a coarse keyword filter — good enough
    # to seed candidates without introducing a heavyweight FTS dependency.
    tokens = sorted(
        (t for t in claim_text.split() if len(t) >= 4),
        key=len,
        reverse=True,
    )[:3]
    if not tokens:
        return []

    pattern = "%" + "%".join(tokens) + "%"

    base_sql = (
        "SELECT p.bibcode, p.year FROM papers p " "WHERE (p.title ILIKE %s OR p.abstract ILIKE %s)"
    )
    params: list[Any] = [pattern, pattern]

    scope_clause, scope_params = scope_to_sql_clauses(scope, {"papers": "p"})
    if scope_clause:
        base_sql += " AND " + scope_clause
        params.extend(scope_params)

    base_sql += " ORDER BY p.year ASC NULLS LAST, p.bibcode ASC LIMIT %s"
    params.append(limit)

    try:
        with conn.cursor() as cur:
            cur.execute(base_sql, params)
            rows = cur.fetchall()
    except psycopg.Error as exc:
        logger.warning("claim_blame: seed query failed: %s", exc)
        return []

    return [(r[0], r[1], None) for r in rows]


def _embed_query_default(claim_text: str) -> list[float] | None:
    """INDUS embedding of ``claim_text`` for the semantic_match tiebreaker.

    Lazily imports :mod:`scix.embed` so test paths that override
    ``embed_query_fn`` never have to load torch.
    """
    try:
        from scix.embed import embed_batch, load_model
    except ImportError:
        return None

    try:
        import os

        device = os.environ.get("SCIX_EMBED_DEVICE", "cpu")
        model, tokenizer = load_model("indus", device=device)
        vectors = embed_batch(model, tokenizer, [claim_text], batch_size=1, pooling="mean")
        return vectors[0] if vectors else None
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("claim_blame: query embedding failed: %s", exc)
        return None


def _default_pool() -> Any:
    """Lazy import to avoid cycles between mcp_server and claim_blame."""
    from scix.mcp_server import _get_pool

    return _get_pool()


# ---------------------------------------------------------------------------
# Lineage construction
# ---------------------------------------------------------------------------


def _build_lineage(
    origin_bib: str,
    origin_year: int | None,
    all_hops: list[dict[str, Any]],
    candidates: list[tuple[str, int | None, list[float] | None]],
) -> list[Hop]:
    """Construct the Hop chain from ``origin`` forward through candidates.

    Every Hop carries ``intent`` and ``intent_weight`` (acceptance criterion
    3). The origin hop carries the candidate's citation-context snippet
    when one exists; otherwise an empty snippet with ``intent=None`` and
    ``intent_weight=DEFAULT_INTENT_WEIGHT``.
    """
    lineage: list[Hop] = []

    # Find the hop that points at the origin (most authoritative one — pick
    # by intent_weight DESC then char_offset ASC for determinism).
    origin_hops = [h for h in all_hops if h["target_bibcode"] == origin_bib]
    origin_hops.sort(
        key=lambda h: (
            -INTENT_WEIGHTS.get(h.get("intent"), DEFAULT_INTENT_WEIGHT),
            h.get("char_offset") or 0,
        )
    )

    if origin_hops:
        oh = origin_hops[0]
        lineage.append(
            Hop(
                bibcode=origin_bib,
                year=oh.get("target_year") or origin_year,
                intent=oh.get("intent"),
                intent_weight=INTENT_WEIGHTS.get(oh.get("intent"), DEFAULT_INTENT_WEIGHT),
                context_snippet=oh.get("context_snippet", "") or "",
                section_name=oh.get("section_name"),
            )
        )
    else:
        lineage.append(
            Hop(
                bibcode=origin_bib,
                year=origin_year,
                intent=None,
                intent_weight=DEFAULT_INTENT_WEIGHT,
                context_snippet="",
                section_name=None,
            )
        )

    # Then walk forward: each candidate that cited (directly or transitively)
    # the origin contributes a Hop. We use the candidate-level hop record
    # that pointed AT the origin when available; otherwise a synthetic Hop.
    appended: set[str] = {origin_bib}

    # Sort candidates chronologically ascending so the lineage reads
    # forward in time.
    sorted_cands = sorted(
        candidates,
        key=lambda c: (c[1] if c[1] is not None else 9999, c[0]),
    )

    for cand_bib, cand_year, _embedding in sorted_cands:
        if cand_bib in appended:
            continue
        # Find the candidate's hop record pointing at the origin (if any).
        cand_hop = next(
            (
                h
                for h in all_hops
                if h["source_bibcode"] == cand_bib and h["target_bibcode"] == origin_bib
            ),
            None,
        )
        if cand_hop is not None:
            lineage.append(
                Hop(
                    bibcode=cand_bib,
                    year=cand_hop.get("source_year") or cand_year,
                    intent=cand_hop.get("intent"),
                    intent_weight=INTENT_WEIGHTS.get(cand_hop.get("intent"), DEFAULT_INTENT_WEIGHT),
                    context_snippet=cand_hop.get("context_snippet", "") or "",
                    section_name=cand_hop.get("section_name"),
                )
            )
        else:
            lineage.append(
                Hop(
                    bibcode=cand_bib,
                    year=cand_year,
                    intent=None,
                    intent_weight=DEFAULT_INTENT_WEIGHT,
                    context_snippet="",
                    section_name=None,
                )
            )
        appended.add(cand_bib)

    return lineage


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _cosine(
    a: list[float] | None,
    b: list[float] | None,
) -> float | None:
    """Cosine similarity in ``[-1, 1]``. Returns ``None`` if either is absent."""
    if a is None or b is None or len(a) != len(b) or not a:
        return None
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    import math

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


__all__ = [
    "DEFAULT_INTENT_WEIGHT",
    "Hop",
    "INTENT_WEIGHTS",
    "claim_blame",
]
