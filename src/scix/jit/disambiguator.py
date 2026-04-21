"""Query-time entity disambiguator (PRD §D1).

Given a query string, extract entity mentions via alias / canonical-name matching
against the entity graph, then identify mentions whose candidate entities are
*ambiguous* — i.e. mentions that could refer to entities of materially different
types (e.g. "Hubble" → telescope vs. person).

This module is a pure DB-lookup utility. No AI calls, no model inference. It is
consumed by the MCP `disambiguate_query` tool (wired in a separate work unit).

Public API
----------

* :class:`EntityCandidate` — frozen dataclass: one candidate entity for a mention.
* :class:`MentionDisambiguation` — frozen dataclass: one mention with its
  scored candidates and ambiguity verdict.
* :func:`disambiguate_query` — main entry point. Takes a psycopg3 connection and
  a query string, returns a list of :class:`MentionDisambiguation`.

Mention extraction strategy
---------------------------

Query is lowercased, whitespace-tokenized, stripped of punctuation at token
boundaries, and expanded into contiguous 1..4-gram sequences. The ngram set is
handed to a single SQL lookup that joins against both ``entity_aliases.alias``
(lowercased) and ``entities.canonical_name`` (lowercased), and pulls
``citing_paper_count`` from the pre-aggregated ``agent_entity_context`` MV.

Ambiguity rule
--------------

A mention is ``ambiguous=True`` iff:

1. It has ≥2 distinct candidate entities,
2. At least two of those candidates have ``paper_count > min_paper_count``, AND
3. Those ≥2 candidates have *different* ``entity_type`` values.

Same-type collisions (e.g. two different stars both called "Vega") are NOT
ambiguous for this module — downstream logic handles intra-type disambiguation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Longest ngram window tried against the alias / canonical-name index.
# "James Webb Space Telescope" = 4 tokens, which is a reasonable upper bound
# for named-entity surface forms in a query.
MAX_NGRAM_LEN: int = 4

# Token-splitting regex: contiguous alphanumeric (plus Unicode word chars)
# runs, treating punctuation as separator. Apostrophes and hyphens split the
# token — agents can work around that by sending underscored queries.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityCandidate:
    """A single candidate entity for one mention.

    Attributes
    ----------
    entity_id
        The ``entities.id`` primary key.
    entity_type
        The entity's declared type (e.g. ``mission``, ``person``, ``instrument``).
    display
        The entity's canonical name — what a caller should show to the user.
    score
        Normalized ``paper_count``: ``paper_count / max(paper_counts_in_mention)``.
        Top candidate is always 1.0. If every candidate has zero papers, all
        scores are 0.0.
    paper_count
        Raw citing-paper count from ``agent_entity_context.citing_paper_count``.
    """

    entity_id: int
    entity_type: str
    display: str
    score: float
    paper_count: int


@dataclass(frozen=True)
class MentionDisambiguation:
    """Disambiguation result for a single mention extracted from the query.

    Attributes
    ----------
    mention
        The exact surface form (case-preserved) matched in the query.
    ambiguous
        True iff the candidate set meets the ambiguity rule described in the
        module docstring.
    candidates
        Tuple of :class:`EntityCandidate`, sorted by ``paper_count`` DESC,
        then ``entity_id`` ASC for deterministic ties.
    default_type
        Entity type of the top candidate (highest paper count). ``None`` only
        when there are no candidates (shouldn't happen in normal flow — if a
        mention has no candidates the mention isn't emitted at all).
    """

    mention: str
    ambiguous: bool
    candidates: tuple[EntityCandidate, ...]
    default_type: str | None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def disambiguate_query(
    conn: psycopg.Connection,
    query: str,
    *,
    min_paper_count: int = 10,
) -> list[MentionDisambiguation]:
    """Return one :class:`MentionDisambiguation` per entity mention found in ``query``.

    Only mentions with at least one candidate are returned. Mentions are
    deduplicated by their lowercased surface form.

    Args:
        conn: psycopg3 connection with read access to the entity graph tables
            and the ``agent_entity_context`` MV.
        query: Free-form query string from the agent.
        min_paper_count: Minimum ``paper_count`` for a candidate to count
            toward the ambiguity verdict. Candidates below the threshold still
            appear in ``candidates`` but are ignored for the ambiguity check.

    Returns:
        List of :class:`MentionDisambiguation` in mention-appearance order (first
        occurrence of each distinct surface form).
    """
    if not query or not query.strip():
        return []

    ngrams = _extract_ngrams(query, MAX_NGRAM_LEN)
    if not ngrams:
        return []

    # DB lookup: for every ngram, find every entity whose canonical_name OR
    # any alias case-insensitively equals the ngram. One round trip.
    rows = _lookup_candidates(conn, [ng.lower() for ng, _ in ngrams])
    if not rows:
        return []

    # Group rows by matched ngram (lowercase) → list of candidate rows.
    by_ngram: dict[str, list[dict]] = {}
    for row in rows:
        by_ngram.setdefault(row["matched_ngram"], []).append(row)

    # Emit mentions in query appearance order, deduped by lowercased surface.
    seen: set[str] = set()
    results: list[MentionDisambiguation] = []
    for surface, lowered in ngrams:
        if lowered in seen:
            continue
        hits = by_ngram.get(lowered)
        if not hits:
            continue
        seen.add(lowered)

        candidates = _build_candidates(hits)
        if not candidates:
            continue

        results.append(
            MentionDisambiguation(
                mention=surface,
                ambiguous=_is_ambiguous(candidates, min_paper_count),
                candidates=tuple(candidates),
                default_type=candidates[0].entity_type,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extract_ngrams(query: str, max_len: int) -> list[tuple[str, str]]:
    """Extract contiguous 1..max_len word ngrams from ``query``.

    Returns a list of ``(surface, lowered)`` tuples preserving original case in
    ``surface`` and query-appearance order. Overlapping and nested ngrams are
    all emitted — the caller decides which ones match the DB.
    """
    # Find all word-token spans so we can rebuild surfaces from the raw query
    # (preserves original casing, punctuation handling is uniform).
    spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(query)]
    if not spans:
        return []

    ngrams: list[tuple[str, str]] = []
    n_tokens = len(spans)
    for i in range(n_tokens):
        for length in range(1, max_len + 1):
            j = i + length
            if j > n_tokens:
                break
            start = spans[i][0]
            end = spans[j - 1][1]
            surface = query[start:end]
            ngrams.append((surface, surface.lower()))
    return ngrams


def _lookup_candidates(
    conn: psycopg.Connection,
    lowered_ngrams: list[str],
) -> list[dict]:
    """Single-round-trip DB lookup for all ngrams.

    Returns rows with columns:
      matched_ngram TEXT  -- the lowercased ngram that matched
      entity_id     INT
      canonical_name TEXT
      entity_type   TEXT
      paper_count   INT
    """
    if not lowered_ngrams:
        return []

    # Two CTEs — one matches canonical_name, one matches alias. UNION then
    # left-joins agent_entity_context for the pre-aggregated paper count.
    sql = """
        WITH ngrams(g) AS (
            SELECT unnest(%(ngrams)s::text[])
        ),
        hits AS (
            SELECT DISTINCT e.id AS entity_id,
                            e.canonical_name,
                            e.entity_type,
                            n.g AS matched_ngram
            FROM ngrams n
            JOIN entities e ON lower(e.canonical_name) = n.g
            UNION
            SELECT DISTINCT e.id AS entity_id,
                            e.canonical_name,
                            e.entity_type,
                            n.g AS matched_ngram
            FROM ngrams n
            JOIN entity_aliases ea ON lower(ea.alias) = n.g
            JOIN entities e ON e.id = ea.entity_id
        )
        SELECT h.matched_ngram,
               h.entity_id,
               h.canonical_name,
               h.entity_type,
               COALESCE(aec.citing_paper_count, 0)::int AS paper_count
        FROM hits h
        LEFT JOIN agent_entity_context aec ON aec.entity_id = h.entity_id
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, {"ngrams": lowered_ngrams})
        return list(cur.fetchall())


def _build_candidates(rows: Iterable[dict]) -> list[EntityCandidate]:
    """Turn raw DB rows into a sorted, scored, deduped candidate list."""
    # Dedup by entity_id (a single entity can match via both its canonical and
    # an alias in the same ngram).
    by_id: dict[int, dict] = {}
    for row in rows:
        eid = row["entity_id"]
        if eid not in by_id:
            by_id[eid] = row

    if not by_id:
        return []

    max_count = max(int(r["paper_count"]) for r in by_id.values())
    max_count = max(max_count, 0)

    candidates = [
        EntityCandidate(
            entity_id=int(r["entity_id"]),
            entity_type=r["entity_type"],
            display=r["canonical_name"],
            score=(int(r["paper_count"]) / max_count) if max_count > 0 else 0.0,
            paper_count=int(r["paper_count"]),
        )
        for r in by_id.values()
    ]

    # Sort by paper_count DESC, entity_id ASC for deterministic ties.
    candidates.sort(key=lambda c: (-c.paper_count, c.entity_id))
    return candidates


def _is_ambiguous(candidates: list[EntityCandidate], min_paper_count: int) -> bool:
    """Apply the 3-clause ambiguity rule."""
    if len(candidates) < 2:
        return False
    above_threshold = [c for c in candidates if c.paper_count > min_paper_count]
    if len(above_threshold) < 2:
        return False
    distinct_types = {c.entity_type for c in above_threshold}
    return len(distinct_types) >= 2


__all__ = [
    "EntityCandidate",
    "MentionDisambiguation",
    "disambiguate_query",
]
