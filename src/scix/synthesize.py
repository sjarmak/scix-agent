"""Mechanical aggregation: working set + section shape -> grounded outline.

This module backs the ``synthesize_findings`` MCP tool. It is **pure
mechanism** — no LLM call lives inside this orchestration code (per the
ZFC pattern in ``rules/common/patterns.md``). The model writes the
synthesis prose using the structured output as scaffolding; this module
just bins working-set bibcodes into named sections using three existing
signals plus a citation-count safety net:

  1. ``citation_contexts.intent`` — modal intent across rows where the
     working-set bibcode is the *target* (i.e., how others cite it). Maps
     ``method`` -> "methods", ``background`` -> "background",
     ``result_comparison`` -> "results". Coverage is small (~0.27% of
     edges per bead 79n) so this signal lights up only a minority of
     papers.

  2. Weighted community fall-through (bead 37wj) — when no intent
     coverage exists for a paper, classify its community by the
     community's share of the working set:

       * ``share >= _CORE_SHARE_THRESHOLD`` (0.15) -> 'core' tier ->
         ``background``. Anchor topics for the working set.
       * ``share >= _SUPPORTING_SHARE_THRESHOLD`` (0.05) -> 'supporting'
         tier -> ``methods`` (or ``background`` overflow when ``methods``
         is not in the requested section list). Adjacent or
         methodological communities that aren't the dominant theme but
         carry enough mass to be more than noise.
       * ``share < _SUPPORTING_SHARE_THRESHOLD`` -> 'peripheral' tier ->
         leave unattributed (so the paper is eligible for the Tier 3
         citation-count fallback below). Peripheral noise should not
         dilute the main outline.

     Replaces the pre-37wj binary modal/non-modal rule, which dumped any
     non-modal-community paper into ``open_questions`` regardless of how
     much mass the second-largest community carried — the failure mode
     surfaced by the cross-disciplinary granular-mech demo.

  3. Empty-section citation-count fallback (bead spj0) — when a section
     remains empty after tiers 1 and 2, fill it from the unattributed
     remainder sorted by ``papers.citation_count`` desc, capped at
     ``max_papers_per_section // 2`` so the fallback is clearly
     secondary. Each pulled paper carries
     ``signal_used='citation_count_fallback'``.

Papers with no signal in any layer land in ``unattributed_bibcodes``.

Each section emits two theme fields (bead 4la8):

  * ``theme_summary`` — legacy formatted string of the section's
    most-common community labels (deprecated; preserved for backwards
    compat with pre-4la8 MCP clients).
  * ``theme`` — a structured payload of raw signals
    (``communities[].{community_id, label, top_keywords[],
    top_arxiv_classes[], paper_count_in_section}`` and
    ``top_papers_by_citation[]``) for the agent to compose its own
    thematic framing. Do **not** treat ``communities[].label`` as
    prose — it is a metadata tag, not a description.

References
----------
  * Bead scix_experiments-cfh9 (initial three-tier implementation)
  * Bead scix_experiments-spj0 (Tier 3 citation-count fallback)
  * Bead scix_experiments-37wj (weighted Tier 2 share-tier classifier)
  * Bead scix_experiments-4la8 (structured theme payload)
  * Bead 79n (citation_contexts.intent partial coverage)
  * docs/CLAUDE.md "ZFC" guidance
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Mapping, Sequence

import psycopg

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEFAULT_SECTIONS: tuple[str, ...] = (
    "background",
    "methods",
    "results",
    "open_questions",
)

# Mapping from citation_contexts.intent values to outline section names.
# The three known intent labels (method, background, result_comparison)
# come from the citation-intent classifier in src/scix/extract; see
# bead 79n for coverage notes.
INTENT_TO_SECTION: Mapping[str, str] = {
    "background": "background",
    "method": "methods",
    "result_comparison": "results",
}

# Maximum number of bibcodes we will accept in a single call. Mirrors
# find_gaps' implicit cap (200) so behaviour is consistent across tools.
_MAX_WORKING_SET_BIBCODES = 200

# Weighted-share thresholds for the Tier 2 community classifier (bead 37wj).
# A community's share of the working set is ``papers_in_community /
# working_set_size``. These constants are tunable in one place; the rest of
# the module derives the share-tier label from them.
_CORE_SHARE_THRESHOLD: Final[float] = 0.15
_SUPPORTING_SHARE_THRESHOLD: Final[float] = 0.05

ShareTier = Literal["core", "supporting", "peripheral"]


def _classify_share_tier(share: float) -> ShareTier:
    """Classify a working-set community share into a share-tier label.

    Pure arithmetic — no DB, no model. The thresholds live in the
    module-level constants :data:`_CORE_SHARE_THRESHOLD` and
    :data:`_SUPPORTING_SHARE_THRESHOLD` so they're trivial to retune.
    """
    if share >= _CORE_SHARE_THRESHOLD:
        return "core"
    if share >= _SUPPORTING_SHARE_THRESHOLD:
        return "supporting"
    return "peripheral"


# ---------------------------------------------------------------------------
# Result dataclasses (frozen — immutable per coding-style guidance)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionBucket:
    """A single named section in the synthesised outline.

    The ``theme`` field (bead 4la8) is a structured payload of raw
    signals — community membership counts, top arxiv classes, top
    keywords, and the section's top-cited papers — for the agent to
    compose its own thematic framing. ``theme_summary`` is the legacy
    string field (concatenated community labels) preserved for
    backwards-compat with pre-4la8 MCP clients; new callers should
    prefer ``theme``.
    """

    name: str
    cited_papers: list[dict[str, Any]] = field(default_factory=list)
    theme_summary: str = ""
    theme: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SynthesisResult:
    """Top-level result returned by :func:`synthesize_findings`."""

    sections: list[SectionBucket]
    unattributed_bibcodes: list[str]
    # int counters plus dict[str, int] for fallback_pulled_per_section.
    coverage: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation for the MCP wire format.

        Note: the section-assignment accounting is keyed as
        ``assignment_coverage`` (NOT ``coverage``) to avoid colliding with
        the ``coverage`` envelope emitted by ``claim_blame`` /
        ``find_replications``, which describes citation_contexts edge
        coverage — a structurally different concept.
        """
        return {
            "sections": [
                {
                    "name": s.name,
                    "cited_papers": list(s.cited_papers),
                    "theme_summary": s.theme_summary,
                    # Explicit reconstruction (not dict(s.theme)) so the
                    # nested communities / top_papers_by_citation lists
                    # are independent copies — protects the frozen
                    # SectionBucket from accidental downstream mutation.
                    "theme": {
                        "communities": list(s.theme.get("communities", [])),
                        "top_papers_by_citation": list(
                            s.theme.get("top_papers_by_citation", [])
                        ),
                    },
                }
                for s in self.sections
            ],
            "unattributed_bibcodes": list(self.unattributed_bibcodes),
            "assignment_coverage": dict(self.coverage),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def synthesize_findings(
    conn: psycopg.Connection,
    working_set_bibcodes: Sequence[str] | None,
    *,
    sections: Sequence[str] | None = None,
    max_papers_per_section: int = 8,
    section_overrides: Mapping[str, str] | None = None,
    include_full_abstracts: bool = False,
    include_citation_contexts: bool = False,
) -> SynthesisResult:
    """Bin a working set of papers into named sections.

    Each entry in a section's ``cited_papers`` list carries a
    ``signal_used`` tag (``intent_modal`` | ``community_fallthrough`` |
    ``override`` | ``citation_count_fallback``), a ``signals`` payload
    exposing the raw inputs that drove the assignment (intent histogram,
    community membership, modal community share), and an
    ``alternative_sections`` list naming the other sections the paper
    *could* have landed in under the available signals — so an agent
    can confidently re-bucket a paper it disagrees with.

    Parameters
    ----------
    conn:
        Open psycopg connection. The function issues three short SELECTs:
        one against ``papers``, one against ``citation_contexts``, one
        against ``paper_metrics`` joined to ``communities``.
    working_set_bibcodes:
        Bibcodes to synthesise. ``None`` or an empty sequence yields an
        empty result (the MCP wrapper falls through to the session
        focused-papers list before calling this function).
    sections:
        Section names to emit, in order. Defaults to
        :data:`DEFAULT_SECTIONS`. Section names that don't appear in the
        intent map (e.g., ``open_questions``) only receive papers via
        explicit ``section_overrides`` or the Tier-3 citation-count
        fallback — under the weighted community classifier (bead 37wj),
        Tier-2 routes only to ``background`` (core) or ``methods``
        (supporting), never to ``open_questions``.
    max_papers_per_section:
        Cap on per-section ``cited_papers`` length (deterministic order:
        first by year desc, then bibcode asc). The Tier-3 citation-count
        fallback is capped at ``max_papers_per_section // 2``, so values
        of 1 disable the fallback entirely (floor division yields 0).
    section_overrides:
        Optional ``{bibcode: section_name}`` mapping that pins specific
        papers to specific sections, overriding the intent-modal and
        community-fallthrough rules. The override target must appear in
        ``sections``; out-of-set targets are silently ignored and the
        paper falls back to the normal rules. Non-string keys/values
        are skipped defensively. Overridden papers carry
        ``signal_used='override'`` while still exposing the underlying
        intent/community signals so an agent can audit the override.
    include_full_abstracts:
        When True, every ``cited_papers`` row also carries an
        ``abstract_full`` field with the untruncated ``papers.abstract``
        text alongside the existing ``abstract_snippet`` (bead tq0t).
        Default False — preserves the default wire format. Enables the
        agent to write grounded synthesis without round-tripping
        through ``read_paper`` for each pivotal seed.
    include_citation_contexts:
        When True, papers attributed via ``signal_used='intent_modal'``
        carry a ``citation_excerpts`` field with up to 3 deterministic
        ``{context_text, intent, citing_bibcode}`` rows from
        ``citation_contexts`` (bead tq0t). Papers attributed via
        community / override / citation-count fallback do NOT receive
        excerpts (their bucket assignment did not come from a
        ``citation_contexts`` row, so a citing sentence is not
        evidence for it). Default False — preserves the default wire
        format. Triggers ONE extra SELECT against ``citation_contexts``
        when enabled.

    Returns
    -------
    :class:`SynthesisResult`
        Each ``cited_papers`` entry has the schema::

            {
              "bibcode": str,
              "title": str | None,
              "year": int | None,
              "first_author": str | None,    # bead tq0t / AC1 (always present)
              "abstract_snippet": str,
              # Bead tq0t / AC2: present only when
              # ``include_full_abstracts=True``.
              "abstract_full": str,           # optional
              "role": str,                    # alias of section_assigned
              "section_assigned": str,
              "signal_used": "intent_modal" | "community_fallthrough" |
                             "override" | "citation_count_fallback",
              "signals": {
                "intent_counts": {intent: n_rows, ...},
                "intent_total_rows": int,
                "community_id": int | None,
                "community_share": float,    # fraction of working set in same community
                # Bead 37wj: weighted Tier 2 share-tier classification.
                # 'core' (share >= 0.15) | 'supporting' (>= 0.05) |
                # 'peripheral' (< 0.05) | None when no community membership.
                "share_tier": "core" | "supporting" | "peripheral" | None,
                "is_modal_community": bool,  # observational; kept for backwards compat
                "modal_community_id": int | None,
              },
              "alternative_sections": [section_name, ...],
              # Bead tq0t / AC3: present only when
              # ``include_citation_contexts=True`` AND
              # ``signal_used == 'intent_modal'``. Up to 3 rows from
              # ``citation_contexts`` whose ``target_bibcode`` matches.
              "citation_excerpts": [
                {context_text: str, intent: str, citing_bibcode: str},
                ...  # capped at 3
              ],
            }

        Each :class:`SectionBucket` also carries a ``theme`` dict
        (bead 4la8) with structured signals for the agent to compose
        its own thematic framing — do not treat
        ``theme.communities[].label`` as prose; it is a metadata tag::

            theme: {
              communities: [
                {
                  community_id: int,
                  label: str | None,
                  paper_count_in_section: int,
                  top_arxiv_classes: [str, ...],
                  top_keywords: [str, ...],
                },
                ...  # sorted by paper_count desc, capped at top 3
              ],
              top_papers_by_citation: [
                {
                  bibcode: str,
                  title: str | None,
                  first_author: str | None,  # bead tq0t / AC1
                  citation_count: int,
                },
                ...  # sorted by citation_count desc, capped at top 3
              ],
            }
    """
    sections_list = list(sections) if sections else list(DEFAULT_SECTIONS)
    bibcodes = _prepare_bibcodes(working_set_bibcodes)

    if not bibcodes:
        return SynthesisResult(
            sections=[],
            unattributed_bibcodes=[],
            coverage={
                "total_bibcodes": 0,
                "assigned_bibcodes": 0,
                "unattributed_bibcodes": 0,
                "intent_assigned_bibcodes": 0,
                "community_assigned_bibcodes": 0,
                "override_assigned_bibcodes": 0,
                "fallback_pulled_bibcodes": 0,
                "fallback_pulled_per_section": {name: 0 for name in sections_list},
            },
            metadata={
                "message": (
                    "Empty working set. Pass working_set_bibcodes or "
                    "populate the session via lit_review/get_paper first."
                ),
                "sections_requested": sections_list,
            },
        )

    paper_meta = _fetch_paper_metadata(conn, bibcodes)
    intent_hist = _fetch_intent_histogram(conn, bibcodes)
    community_map = _fetch_community_assignments(conn, bibcodes)
    overrides = _normalize_overrides(section_overrides, set(sections_list))

    # Bead tq0t: optional citation-excerpt enrichment for intent_modal
    # papers. Fetched up-front (one query) and threaded through assembly
    # so ``_paper_row`` can attach excerpts only to papers whose
    # ``signal_used`` ends up as ``intent_modal``. ``None`` when the
    # kwarg is False — the assembly step then OMITS the
    # ``citation_excerpts`` field entirely (preserves default wire
    # format). When the kwarg is True, the dict may still be empty if
    # no rows match — in that case intent-modal papers get an explicit
    # empty list (so the schema is uniform when the flag is on).
    citation_excerpts: dict[str, list[dict[str, Any]]] | None = None
    if include_citation_contexts:
        citation_excerpts = _fetch_citation_excerpts(conn, bibcodes)

    return _assemble_sections(
        bibcodes=bibcodes,
        sections=sections_list,
        paper_meta=paper_meta,
        intent_hist=intent_hist,
        community_map=community_map,
        max_papers_per_section=max_papers_per_section,
        overrides=overrides,
        include_full_abstracts=include_full_abstracts,
        citation_excerpts=citation_excerpts,
    )


def _normalize_overrides(raw: Mapping[str, str] | None, valid_sections: set[str]) -> dict[str, str]:
    """Return a clean ``{bibcode: section_name}`` dict.

    Drops non-string keys/values and any entry whose section name isn't
    in the requested ``sections`` set (so callers can't sneak in unknown
    section names). Pure validation; no DB or model calls.
    """
    if not raw:
        return {}
    out: dict[str, str] = {}
    for bibcode, section in raw.items():
        if not isinstance(bibcode, str) or not isinstance(section, str):
            continue
        if section not in valid_sections:
            continue
        out[bibcode.strip()] = section
    return out


# ---------------------------------------------------------------------------
# Bibcode preparation
# ---------------------------------------------------------------------------


def _prepare_bibcodes(raw: Sequence[str] | None) -> list[str]:
    """Deduplicate, strip, and cap the input bibcode list."""
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for b in raw:
        if not isinstance(b, str):
            continue
        bb = b.strip()
        if not bb or bb in seen:
            continue
        seen.add(bb)
        out.append(bb)
        if len(out) >= _MAX_WORKING_SET_BIBCODES:
            break
    return out


# ---------------------------------------------------------------------------
# DB queries (each kept in its own small function for readability)
# ---------------------------------------------------------------------------


def _fetch_paper_metadata(
    conn: psycopg.Connection, bibcodes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Return ``{bibcode: {title, year, abstract, abstract_snippet,
    citation_count, arxiv_class, keywords, first_author}}``.

    ``papers.citation_count`` is INTEGER NULL — ``COALESCE`` to 0 so the
    Tier-3 (citation-count) fallback always has a comparable scalar. The
    ``citation_count`` is the *corpus-wide* tally; the fallback ranks
    within the working set, so this is just a per-paper attribute used
    for sorting.

    ``arxiv_class`` and ``keywords`` are ``text[]`` arrays — partial
    coverage on prod (~8% / ~49% per spot-check 2026-04-27) — used by
    the bead-4la8 ``theme`` aggregation. ``first_author`` (added in
    bead tq0t) is sourced verbatim from ``papers.first_author``;
    production coverage is high but a small number of legacy rows have
    it as NULL.

    Production rows always return 8 columns from the SELECT below; the
    ``len(row) > 5/6/7`` guards exist *only* to keep pre-4la8/pre-tq0t
    short-tuple test fixtures working (the test suite uses 5-tuples in
    older suites, 7-tuples in TestSectionTheme, and 8-tuples in
    TestAdditiveGroundingFields). Migrating those fixtures to 8-tuples
    is tracked as bead k27h; once done, drop the guards. Do NOT take
    inspiration from these guards for new code — the SELECT contract
    guarantees the column count.

    The full ``abstract`` string is stashed under the ``abstract`` key
    (in addition to the truncated ``abstract_snippet``) so that
    callers passing ``include_full_abstracts=True`` can surface it in
    the per-paper payload without a second query.
    """
    sql = """
        SELECT bibcode, title, year, abstract, COALESCE(citation_count, 0),
               arxiv_class, keywords, first_author
        FROM papers
        WHERE bibcode = ANY(%s)
    """
    out: dict[str, dict[str, Any]] = {}
    with conn.cursor() as cur:
        cur.execute(sql, (list(bibcodes),))
        for row in cur.fetchall():
            bibcode = row[0]
            abstract = row[3] or ""
            citation_count = int(row[4])  # COALESCE in SQL guarantees non-NULL
            # Fixture-compat guards: see docstring above. Production rows
            # always return 8 columns; remove these once the legacy
            # short-tuple test fixtures are migrated (bead k27h).
            arxiv_class: list[str] = list(row[5]) if len(row) > 5 and row[5] else []
            keywords: list[str] = list(row[6]) if len(row) > 6 and row[6] else []
            first_author: str | None = row[7] if len(row) > 7 else None
            out[bibcode] = {
                "bibcode": bibcode,
                "title": row[1],
                "year": row[2],
                "abstract": abstract,
                "abstract_snippet": _snippet(abstract),
                "citation_count": citation_count,
                "arxiv_class": arxiv_class,
                "keywords": keywords,
                "first_author": first_author,
            }
    return out


def _fetch_intent_histogram(
    conn: psycopg.Connection, bibcodes: Sequence[str]
) -> dict[str, Counter[str]]:
    """Return ``{bibcode: Counter({intent: n_rows})}`` from citation_contexts.

    Rows are aggregated server-side via ``GROUP BY``. Rows with NULL
    ``intent`` are skipped (they carry no signal for section assignment).
    """
    sql = """
        SELECT target_bibcode, intent, COUNT(*) AS n_rows
        FROM citation_contexts
        WHERE target_bibcode = ANY(%s)
          AND intent IS NOT NULL
        GROUP BY target_bibcode, intent
    """
    out: dict[str, Counter[str]] = {}
    with conn.cursor() as cur:
        cur.execute(sql, (list(bibcodes),))
        for row in cur.fetchall():
            target, intent, n_rows = row[0], row[1], int(row[2])
            out.setdefault(target, Counter())[intent] = n_rows
    return out


def _fetch_community_assignments(
    conn: psycopg.Connection, bibcodes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Return ``{bibcode: {community_id, community_label}}``.

    Uses the semantic-medium partition (``community_semantic_medium``)
    by default — this matches the ``find_gaps`` default and is the only
    partition with full 32M-paper coverage. The Phase-A sentinel ``-1``
    on the citation partition is filtered server-side (the semantic
    partition has no sentinel but the filter is harmless).
    """
    sql = """
        SELECT pm.bibcode,
               pm.community_semantic_medium AS community_id,
               c.label AS community_label
        FROM paper_metrics pm
        LEFT JOIN communities c
               ON c.signal = 'semantic'
              AND c.resolution = 'medium'
              AND c.community_id = pm.community_semantic_medium
        WHERE pm.bibcode = ANY(%s)
          AND pm.community_semantic_medium IS NOT NULL
          AND pm.community_semantic_medium <> -1
    """
    out: dict[str, dict[str, Any]] = {}
    with conn.cursor() as cur:
        cur.execute(sql, (list(bibcodes),))
        for row in cur.fetchall():
            bibcode = row[0]
            out[bibcode] = {
                "community_id": int(row[1]),
                "community_label": row[2],
            }
    return out


# Bead-tq0t cap: at most 3 citation-context excerpts per paper in the
# synthesise output. Module-level constant so it's easy to retune
# alongside the other ``_THEME_MAX_*`` caps.
_CITATION_EXCERPTS_MAX_PER_PAPER: Final[int] = 3


def _fetch_citation_excerpts(
    conn: psycopg.Connection, bibcodes: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{target_bibcode: [{context_text, intent, citing_bibcode}, ...]}``.

    One row in ``citation_contexts`` per (source -> target) citation
    incident. Bead tq0t surfaces up to
    :data:`_CITATION_EXCERPTS_MAX_PER_PAPER` rows per paper so the agent
    can ground bucket assignments on actual citing-sentence evidence.

    Determinism: ORDER BY (intent ASC, source_bibcode ASC, char_offset
    ASC NULLS LAST) so repeated calls return the same excerpts in the
    same order. Filters out rows with NULL intent (they carry no signal
    for the synthesise output and are excluded from intent-modal
    bucketing upstream).

    Implementation notes:
      * One query, one IN/ANY filter, bounded by the working-set cap
        (200 bibcodes).
      * Top-K sliced in Python rather than via SQL ROW_NUMBER —
        simpler, and the row count for a 200-bibcode working set is
        tiny in practice.
    """
    sql = """
        SELECT target_bibcode, context_text, intent, source_bibcode
        FROM citation_contexts
        WHERE target_bibcode = ANY(%s)
          AND intent IS NOT NULL
        ORDER BY target_bibcode ASC,
                 intent ASC,
                 source_bibcode ASC,
                 char_offset ASC NULLS LAST
    """
    out: dict[str, list[dict[str, Any]]] = {}
    with conn.cursor() as cur:
        cur.execute(sql, (list(bibcodes),))
        for row in cur.fetchall():
            target = row[0]
            bucket = out.setdefault(target, [])
            if len(bucket) >= _CITATION_EXCERPTS_MAX_PER_PAPER:
                continue  # cap reached; skip the rest for this target
            bucket.append(
                {
                    "context_text": row[1],
                    "intent": row[2],
                    "citing_bibcode": row[3],
                }
            )
    return out


# ---------------------------------------------------------------------------
# Pure-arithmetic assembly
# ---------------------------------------------------------------------------


def _modal_intent(hist: Counter[str]) -> str | None:
    """Return the modal intent label, or ``None`` if the counter is empty.

    Ties are broken by the canonical order
    (``background`` < ``method`` < ``result_comparison``) so the
    function is deterministic.
    """
    if not hist:
        return None
    canonical = ("background", "method", "result_comparison")
    rank = {label: i for i, label in enumerate(canonical)}
    return max(
        hist.items(),
        key=lambda kv: (kv[1], -rank.get(kv[0], 999)),
    )[0]


def _section_for_share_tier(tier: ShareTier, section_set: set[str]) -> str | None:
    """Map a share-tier label to a section name (or None for peripheral).

    Explicit ladder per bead 37wj AC1:
      * 'core'       -> 'background' (anchor topics).
      * 'supporting' -> 'methods' if requested, else 'background' overflow,
                        else None (no valid section absorbs it; falls into
                        unattributed).
      * 'peripheral' -> None (leave for Tier 3 fallback).

    Pure arithmetic; no DB, no model.
    """
    if tier == "core":
        return "background" if "background" in section_set else None
    if tier == "supporting":
        if "methods" in section_set:
            return "methods"
        if "background" in section_set:
            return "background"
        return None
    # tier == "peripheral"
    return None


def _modal_community(
    bibcodes: Sequence[str], community_map: Mapping[str, dict[str, Any]]
) -> int | None:
    """Return the most-common community_id across the working set, or None."""
    counts: Counter[int] = Counter()
    for b in bibcodes:
        info = community_map.get(b)
        if info is not None:
            counts[info["community_id"]] += 1
    if not counts:
        return None
    # Tiebreak by community_id ascending for determinism.
    return min(
        counts.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )[0]


def _theme_summary_for(
    bibcodes: Sequence[str],
    community_map: Mapping[str, dict[str, Any]],
    top_k: int = 3,
) -> str:
    """Build a one-line theme summary by joining top community labels.

    Pure aggregation: the top-K most-common community labels among the
    section's papers, joined with ``; ``. No LLM, no semantic
    generation. Returns an empty string if no labels are available.

    .. deprecated:: 4la8
        Prefer :func:`_theme_for` which exposes the underlying signals
        (community sizes, arxiv classes, keywords, top-cited papers) so
        the agent can compose its own thematic framing. Kept here for
        backwards compatibility — see ``SectionBucket.theme_summary``.
    """
    label_counts: Counter[str] = Counter()
    for b in bibcodes:
        info = community_map.get(b)
        if info and info.get("community_label"):
            label_counts[info["community_label"]] += 1
    if not label_counts:
        return ""
    top = [label for label, _ in label_counts.most_common(top_k)]
    return "; ".join(top)


# Top-K caps for the structured theme payload. Module-level constants so
# they are trivial to retune without hunting through the assembly code.
_THEME_MAX_COMMUNITIES: Final[int] = 3
_THEME_MAX_TOP_PAPERS: Final[int] = 3
_THEME_MAX_KEYWORDS_PER_COMMUNITY: Final[int] = 5
_THEME_MAX_ARXIV_CLASSES_PER_COMMUNITY: Final[int] = 5
# Rough English/scientific stopword list for the title-token keyword
# fallback (per CLAUDE.md memory ``community_labels_pipeline.md`` —
# ``papers.keywords`` is NULL on ~52% of rows so we need a fallback).
_TITLE_TOKEN_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "via",
        "vs",
        "is",
        "are",
        "be",
        "this",
        "that",
        "these",
        "those",
        "its",
        "their",
        "our",
        "we",
        "i",
        "ii",
        "iii",
        "iv",
        "v",
        "study",
        "studies",
        "paper",
        "papers",
        "review",
        "based",
        "using",
        "use",
        "used",
        "new",
    }
)


def _title_tokens(title: str | None) -> list[str]:
    """Tokenize a paper title into lowercase content words.

    Splits on non-alphanumeric, drops short tokens (<3 chars) and
    English/scientific stopwords. Returns the deduplicated list in
    first-seen order so a paper contributes each token at most once to
    the section's keyword counter.
    """
    if not title:
        return []
    raw = re.split(r"[^A-Za-z0-9]+", title.lower())
    seen: set[str] = set()
    out: list[str] = []
    for tok in raw:
        if len(tok) < 3:
            continue
        if tok in _TITLE_TOKEN_STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _theme_for(
    bibcodes: Sequence[str],
    paper_meta: Mapping[str, dict[str, Any]],
    community_map: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the structured ``theme`` payload for a section (bead 4la8).

    Returns ``{"communities": [...], "top_papers_by_citation": [...]}``.

    ``communities`` is sorted by ``paper_count_in_section`` desc
    (tiebreak: ``community_id`` asc) and capped at
    :data:`_THEME_MAX_COMMUNITIES`. Each entry carries
    ``community_id``, ``label``, ``paper_count_in_section``,
    ``top_arxiv_classes`` (Counter over the section's papers' arxiv
    arrays, top N), and ``top_keywords`` (Counter over keywords, with
    title-token fallback when the keyword arrays are all empty).

    ``top_papers_by_citation`` is the section's papers sorted by
    ``citation_count`` desc (tiebreak: ``bibcode`` asc) and capped at
    :data:`_THEME_MAX_TOP_PAPERS`.

    Pure aggregation — no DB, no model. Empty section returns
    ``{"communities": [], "top_papers_by_citation": []}`` (no crash).
    """
    # Group bibcodes by community_id (skip papers with no community).
    by_community: dict[int, list[str]] = {}
    community_label: dict[int, str | None] = {}
    for b in bibcodes:
        info = community_map.get(b)
        if info is None:
            continue
        cid = int(info["community_id"])
        by_community.setdefault(cid, []).append(b)
        if cid not in community_label:
            community_label[cid] = info.get("community_label")

    # Sort communities by (count desc, community_id asc) for determinism.
    ordered_cids = sorted(
        by_community.keys(),
        key=lambda cid: (-len(by_community[cid]), cid),
    )[:_THEME_MAX_COMMUNITIES]

    communities_payload: list[dict[str, Any]] = []
    for cid in ordered_cids:
        members = by_community[cid]
        arxiv_counter: Counter[str] = Counter()
        keyword_counter: Counter[str] = Counter()
        for member in members:
            meta = paper_meta.get(member, {})
            for ax in meta.get("arxiv_class") or []:
                if isinstance(ax, str) and ax:
                    arxiv_counter[ax] += 1
            for kw in meta.get("keywords") or []:
                if isinstance(kw, str) and kw:
                    keyword_counter[kw.lower()] += 1
        # Title-token fallback when no keyword data is available across
        # the community's section papers (papers.keywords is NULL on
        # ~52% of prod rows per the labels-pipeline note in CLAUDE.md).
        #
        # Known limitation: this fires only when *all* members have empty
        # `keywords`. Mixed-coverage communities (some papers have
        # keywords, others don't) skip the fallback entirely — the
        # `top_keywords` then reflect only the keyword-bearing minority.
        # That asymmetry is acceptable for now (the keyword signal is
        # advisory metadata, not load-bearing for routing) but worth
        # revisiting if `papers.keywords` coverage changes materially.
        if not keyword_counter:
            for member in members:
                meta = paper_meta.get(member, {})
                for tok in _title_tokens(meta.get("title")):
                    keyword_counter[tok] += 1
        communities_payload.append(
            {
                "community_id": cid,
                "label": community_label.get(cid),
                "paper_count_in_section": len(members),
                "top_arxiv_classes": [
                    label
                    for label, _ in arxiv_counter.most_common(
                        _THEME_MAX_ARXIV_CLASSES_PER_COMMUNITY
                    )
                ],
                "top_keywords": [
                    label
                    for label, _ in keyword_counter.most_common(_THEME_MAX_KEYWORDS_PER_COMMUNITY)
                ],
            }
        )

    # Top papers by citation_count (desc, tiebreak bibcode asc).
    ranked = sorted(
        bibcodes,
        key=lambda b: (
            -int(paper_meta.get(b, {}).get("citation_count", 0) or 0),
            b,
        ),
    )[:_THEME_MAX_TOP_PAPERS]
    top_papers: list[dict[str, Any]] = []
    for b in ranked:
        meta = paper_meta.get(b, {})
        top_papers.append(
            {
                "bibcode": b,
                "title": meta.get("title"),
                # Bead tq0t / AC1: first_author surfaced alongside the
                # other thumbnail metadata so the agent can attribute
                # without round-tripping through ``get_paper``.
                "first_author": meta.get("first_author"),
                "citation_count": int(meta.get("citation_count", 0) or 0),
            }
        )

    return {
        "communities": communities_payload,
        "top_papers_by_citation": top_papers,
    }


def _excerpts_for(
    *,
    bibcode: str,
    signal_used: str,
    excerpts_map: Mapping[str, list[dict[str, Any]]] | None,
) -> list[dict[str, Any]] | None:
    """Pick the citation-excerpts payload for a single paper (bead tq0t).

    Three-state result:
      * ``None``  -> the kwarg was off OR the paper wasn't bucketed via
                     ``intent_modal``. ``_paper_row`` will OMIT the
                     ``citation_excerpts`` field entirely.
      * ``[]``    -> kwarg was on, paper is intent_modal, no rows in
                     ``citation_contexts`` for this target.
      * ``list[dict]`` -> kwarg on, paper is intent_modal, and rows
                     exist (capped upstream by
                     :data:`_CITATION_EXCERPTS_MAX_PER_PAPER`).
    """
    if excerpts_map is None:
        return None
    if signal_used != "intent_modal":
        return None
    return list(excerpts_map.get(bibcode, []))


def _assemble_sections(
    *,
    bibcodes: Sequence[str],
    sections: Sequence[str],
    paper_meta: Mapping[str, dict[str, Any]],
    intent_hist: Mapping[str, Counter[str]],
    community_map: Mapping[str, dict[str, Any]],
    max_papers_per_section: int,
    overrides: Mapping[str, str],
    include_full_abstracts: bool = False,
    citation_excerpts: Mapping[str, list[dict[str, Any]]] | None = None,
) -> SynthesisResult:
    """Bin bibcodes into requested sections via override -> intent -> community
    -> citation-count fallback (for empty sections only)."""

    # Per-section bucket of bibcodes (preserves working-set order).
    buckets: dict[str, list[str]] = {name: [] for name in sections}
    # Per-bibcode signal_used label, for the cited_papers payload.
    signal_used: dict[str, str] = {}
    unattributed: list[str] = []
    intent_assigned = 0
    community_assigned = 0
    override_assigned = 0

    modal_comm = _modal_community(bibcodes, community_map)
    section_set = set(sections)

    # Working-set community share: how many bibcodes share each community_id?
    community_size: Counter[int] = Counter()
    for b in bibcodes:
        info = community_map.get(b)
        if info is not None:
            community_size[info["community_id"]] += 1
    total_bibcodes = len(bibcodes)

    for bibcode in bibcodes:
        target_section: str | None = None

        # Tier 0: explicit override (highest precedence).
        override = overrides.get(bibcode)
        if override is not None:
            target_section = override
            signal_used[bibcode] = "override"
            override_assigned += 1

        # Tier 1: intent histogram.
        if target_section is None:
            modal = _modal_intent(intent_hist.get(bibcode, Counter()))
            if modal is not None:
                mapped = INTENT_TO_SECTION.get(modal)
                if mapped is not None and mapped in section_set:
                    target_section = mapped
                    signal_used[bibcode] = "intent_modal"
                    intent_assigned += 1

        # Tier 2: weighted community fall-through (bead 37wj).
        # Classify the paper's community by its share of the working set:
        #   core      -> background
        #   supporting -> methods (or background overflow)
        #   peripheral -> leave unattributed (eligible for Tier 3)
        if target_section is None:
            comm = community_map.get(bibcode)
            if comm is not None and total_bibcodes > 0:
                share = community_size.get(comm["community_id"], 0) / total_bibcodes
                tier = _classify_share_tier(share)
                fallback = _section_for_share_tier(tier, section_set)
                if fallback is not None:
                    target_section = fallback
                    signal_used[bibcode] = "community_fallthrough"
                    community_assigned += 1

        if target_section is None:
            unattributed.append(bibcode)
        else:
            buckets[target_section].append(bibcode)

    # Tier 3: empty-section citation-count fallback (bead spj0). Pulls from
    # the ``unattributed`` pool only — already-attributed (override / intent /
    # community) papers are NEVER moved or duplicated into other sections.
    fallback_pulled_per_section = _apply_citation_count_fallback(
        sections=sections,
        buckets=buckets,
        unattributed=unattributed,
        paper_meta=paper_meta,
        signal_used=signal_used,
        max_papers_per_section=max_papers_per_section,
    )

    # Build SectionBucket list in the requested order; cap each at
    # max_papers_per_section using a deterministic sort (year desc then
    # bibcode asc) so callers get a stable outline.
    out_sections: list[SectionBucket] = []
    for name in sections:
        chosen = _select_top_papers(buckets[name], paper_meta, max_papers_per_section)
        cited = [
            _paper_row(
                b,
                paper_meta,
                name,
                signals=_build_signals(
                    bibcode=b,
                    intent_hist=intent_hist,
                    community_map=community_map,
                    modal_comm=modal_comm,
                    community_size=community_size,
                    total_bibcodes=total_bibcodes,
                ),
                signal_used=signal_used.get(b, ""),
                alternative_sections=_alternative_sections(
                    bibcode=b,
                    chosen_section=name,
                    intent_hist=intent_hist,
                    community_map=community_map,
                    community_size=community_size,
                    total_bibcodes=total_bibcodes,
                    section_set=section_set,
                ),
                include_full_abstract=include_full_abstracts,
                # Bead tq0t / AC3: forward an excerpt list only when the
                # caller enabled the kwarg (citation_excerpts is not
                # None) AND the paper was bucketed via intent_modal.
                # Other tiers / disabled kwarg -> None -> field omitted.
                citation_excerpts=_excerpts_for(
                    bibcode=b,
                    signal_used=signal_used.get(b, ""),
                    excerpts_map=citation_excerpts,
                ),
            )
            for b in chosen
        ]
        theme_summary = _theme_summary_for(chosen, community_map)
        theme = _theme_for(chosen, paper_meta, community_map)
        out_sections.append(
            SectionBucket(
                name=name,
                cited_papers=cited,
                theme_summary=theme_summary,
                theme=theme,
            )
        )

    fallback_pulled_total = sum(fallback_pulled_per_section.values())
    coverage = {
        "total_bibcodes": len(bibcodes),
        # ``unattributed`` was mutated in-place by the Tier-3 helper to remove
        # any bibcode that was successfully fallback-pulled, so the
        # ``assigned_bibcodes`` total reflects all four tiers.
        "assigned_bibcodes": len(bibcodes) - len(unattributed),
        "unattributed_bibcodes": len(unattributed),
        "intent_assigned_bibcodes": intent_assigned,
        "community_assigned_bibcodes": community_assigned,
        "override_assigned_bibcodes": override_assigned,
        "fallback_pulled_bibcodes": fallback_pulled_total,
        "fallback_pulled_per_section": dict(fallback_pulled_per_section),
    }

    return SynthesisResult(
        sections=out_sections,
        unattributed_bibcodes=unattributed,
        coverage=coverage,
        metadata={
            "sections_requested": list(sections),
            "modal_community_id": modal_comm,
            "override_count": len(overrides),
        },
    )


def _apply_citation_count_fallback(
    *,
    sections: Sequence[str],
    buckets: dict[str, list[str]],
    unattributed: list[str],
    paper_meta: Mapping[str, dict[str, Any]],
    signal_used: dict[str, str],
    max_papers_per_section: int,
) -> dict[str, int]:
    """Fill empty sections from the unattributed pool by citation_count desc.

    Mutates ``buckets``, ``unattributed``, and ``signal_used`` in place.
    Returns a ``{section_name: int}`` map of how many papers were pulled
    per section (always populated for every section in ``sections``,
    even if zero).

    Why ``unattributed`` only? AC3 — fallback must not poach papers
    that were already attributed by a primary signal (override / intent
    / community). They keep their primary assignment; only orphaned
    papers are eligible to fill empty sections.

    Sections process in input order; each pulls up to
    ``max_papers_per_section // 2`` papers from the *remaining* pool, so
    a paper goes to at most one section.
    """
    pulled_per_section: dict[str, int] = {name: 0 for name in sections}
    cap = max_papers_per_section // 2
    if cap <= 0 or not unattributed:
        return pulled_per_section

    # Snapshot the pool sorted by (citation_count desc, bibcode asc) so
    # cross-section pulls follow a single deterministic ranking.
    remaining = sorted(
        unattributed,
        key=lambda b: (-int(paper_meta.get(b, {}).get("citation_count", 0) or 0), b),
    )

    pulled_set: set[str] = set()
    for section_name in sections:
        if buckets.get(section_name):
            continue  # tier 0/1/2 already populated this section
        if not remaining:
            break
        take = remaining[:cap]
        for bibcode in take:
            buckets[section_name].append(bibcode)
            signal_used[bibcode] = "citation_count_fallback"
            pulled_set.add(bibcode)
        remaining = remaining[cap:]
        pulled_per_section[section_name] = len(take)

    # Single O(n) rebuild of `unattributed` instead of N x O(n) `remove()`
    # calls inside the loop above.
    if pulled_set:
        unattributed[:] = [b for b in unattributed if b not in pulled_set]

    return pulled_per_section


def _build_signals(
    *,
    bibcode: str,
    intent_hist: Mapping[str, Counter[str]],
    community_map: Mapping[str, dict[str, Any]],
    modal_comm: int | None,
    community_size: Mapping[int, int],
    total_bibcodes: int,
) -> dict[str, Any]:
    """Build the per-paper ``signals`` payload (pure aggregation)."""
    hist = intent_hist.get(bibcode, Counter())
    intent_counts = dict(hist)
    intent_total = sum(intent_counts.values())

    comm_info = community_map.get(bibcode)
    community_id = comm_info["community_id"] if comm_info is not None else None

    if community_id is not None and total_bibcodes > 0:
        share = community_size.get(community_id, 0) / total_bibcodes
        share_tier: str | None = _classify_share_tier(share)
    else:
        share = 0.0
        share_tier = None

    is_modal = community_id is not None and community_id == modal_comm

    return {
        "intent_counts": intent_counts,
        "intent_total_rows": intent_total,
        "community_id": community_id,
        "community_share": share,
        # Bead 37wj: weighted Tier 2 share-tier classification.
        "share_tier": share_tier,
        # Observational, retained for backwards-compat with pre-37wj clients.
        "is_modal_community": bool(is_modal),
        "modal_community_id": modal_comm,
    }


def _alternative_sections(
    *,
    bibcode: str,
    chosen_section: str,
    intent_hist: Mapping[str, Counter[str]],
    community_map: Mapping[str, dict[str, Any]],
    community_size: Mapping[int, int],
    total_bibcodes: int,
    section_set: set[str],
) -> list[str]:
    """Return sections (other than ``chosen_section``) the paper could have
    landed in given the available signals — sorted for determinism.

    Sources of alternatives (post bead 37wj):
      * Every intent label with at least one row maps to its section
        (per :data:`INTENT_TO_SECTION`).
      * Community share-tier alternatives:
        - ``core`` -> ``background`` (the canonical core routing).
        - ``supporting`` -> both ``methods`` and ``background``
          (the supporting -> methods/background overflow ladder).
        - ``peripheral`` -> none (peripheral papers do not have a
          community-driven section).
    """
    alts: set[str] = set()
    for intent_label in intent_hist.get(bibcode, Counter()):
        mapped = INTENT_TO_SECTION.get(intent_label)
        if mapped is not None and mapped in section_set:
            alts.add(mapped)

    comm_info = community_map.get(bibcode)
    if comm_info is not None and total_bibcodes > 0:
        share = community_size.get(comm_info["community_id"], 0) / total_bibcodes
        tier = _classify_share_tier(share)
        if tier == "core":
            if "background" in section_set:
                alts.add("background")
        elif tier == "supporting":
            if "methods" in section_set:
                alts.add("methods")
            if "background" in section_set:
                alts.add("background")
        # peripheral: no community-driven alternative

    alts.discard(chosen_section)
    return sorted(alts)


def _select_top_papers(
    bibcodes: Sequence[str],
    paper_meta: Mapping[str, dict[str, Any]],
    cap: int,
) -> list[str]:
    """Sort a section's bibcodes by (year desc, bibcode asc) and cap."""
    if cap <= 0:
        return []

    def sort_key(b: str) -> tuple[int, str]:
        meta = paper_meta.get(b)
        year = (meta or {}).get("year") or 0
        # Negative year so descending; bibcode ascending for stable tiebreak.
        return (-int(year), b)

    return sorted(bibcodes, key=sort_key)[:cap]


def _paper_row(
    bibcode: str,
    paper_meta: Mapping[str, dict[str, Any]],
    role: str,
    *,
    signals: dict[str, Any] | None = None,
    signal_used: str = "",
    alternative_sections: list[str] | None = None,
    include_full_abstract: bool = False,
    citation_excerpts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the per-paper dict for a section's ``cited_papers`` list.

    ``role`` and ``section_assigned`` are aliases — ``role`` is kept for
    back-compat with earlier callers; ``section_assigned`` is the name
    used by the bead-gtsx signals schema.

    ``first_author`` is always present (bead tq0t / AC1) — populated
    from ``papers.first_author`` via :func:`_fetch_paper_metadata`. May
    be ``None`` for legacy fixtures or rows where the column is NULL.

    ``abstract_full`` (bead tq0t / AC2) is added only when
    ``include_full_abstract`` is True. Coexists with ``abstract_snippet``
    (additive, not replacement) so wave-1+ wire format remains stable.

    ``citation_excerpts`` (bead tq0t / AC3) is added only when the
    caller passes a non-None list — by convention, the assembly step
    forwards a list (possibly empty) for ``signal_used='intent_modal'``
    papers and forwards ``None`` for everyone else, so this field is
    omitted entirely on community / override / fallback rows.
    """
    meta = paper_meta.get(bibcode, {})
    row: dict[str, Any] = {
        "bibcode": bibcode,
        "title": meta.get("title"),
        "year": meta.get("year"),
        "first_author": meta.get("first_author"),
        "abstract_snippet": meta.get("abstract_snippet", ""),
        "role": role,
        "section_assigned": role,
        "signal_used": signal_used,
        "signals": signals if signals is not None else {},
        "alternative_sections": list(alternative_sections or []),
    }
    if include_full_abstract:
        row["abstract_full"] = meta.get("abstract", "")
    if citation_excerpts is not None:
        row["citation_excerpts"] = list(citation_excerpts)
    return row


def _snippet(text: str, max_chars: int = 280) -> str:
    """Truncate an abstract for the outline payload."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
