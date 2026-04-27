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

  2. Community-membership fall-through — when no intent coverage exists
     for a paper, the modal community across the working set defines
     "background" and out-of-modal-community papers go to
     "open_questions" (the cross-community signal).

  3. Empty-section citation-count fallback (bead spj0) — when a section
     remains empty after tiers 1 and 2, fill it from the unattributed
     remainder sorted by ``papers.citation_count`` desc, capped at
     ``max_papers_per_section // 2`` so the fallback is clearly
     secondary. Each pulled paper carries
     ``signal_used='citation_count_fallback'``.

Papers with no signal in any layer land in ``unattributed_bibcodes``.

The ``theme_summary`` per section is a deterministic concatenation of
the section's most-common community labels — no semantic generation.

References
----------
  * Bead scix_experiments-cfh9 (this implementation)
  * Bead 79n (citation_contexts.intent partial coverage)
  * docs/CLAUDE.md "ZFC" guidance
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

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


# ---------------------------------------------------------------------------
# Result dataclasses (frozen — immutable per coding-style guidance)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionBucket:
    """A single named section in the synthesised outline."""

    name: str
    cited_papers: list[dict[str, Any]] = field(default_factory=list)
    theme_summary: str = ""


@dataclass(frozen=True)
class SynthesisResult:
    """Top-level result returned by :func:`synthesize_findings`."""

    sections: list[SectionBucket]
    unattributed_bibcodes: list[str]
    coverage: dict[str, int]
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
        the community fall-through.
    max_papers_per_section:
        Cap on per-section ``cited_papers`` length (deterministic order:
        first by year desc, then bibcode asc).
    section_overrides:
        Optional ``{bibcode: section_name}`` mapping that pins specific
        papers to specific sections, overriding the intent-modal and
        community-fallthrough rules. The override target must appear in
        ``sections``; out-of-set targets are silently ignored and the
        paper falls back to the normal rules. Non-string keys/values
        are skipped defensively. Overridden papers carry
        ``signal_used='override'`` while still exposing the underlying
        intent/community signals so an agent can audit the override.

    Returns
    -------
    :class:`SynthesisResult`
        Each ``cited_papers`` entry has the schema::

            {
              "bibcode": str,
              "title": str | None,
              "year": int | None,
              "abstract_snippet": str,
              "role": str,                    # alias of section_assigned
              "section_assigned": str,
              "signal_used": "intent_modal" | "community_fallthrough" |
                             "override" | "citation_count_fallback",
              "signals": {
                "intent_counts": {intent: n_rows, ...},
                "intent_total_rows": int,
                "community_id": int | None,
                "community_share": float,    # fraction of working set in same community
                "is_modal_community": bool,
                "modal_community_id": int | None,
              },
              "alternative_sections": [section_name, ...],
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

    return _assemble_sections(
        bibcodes=bibcodes,
        sections=sections_list,
        paper_meta=paper_meta,
        intent_hist=intent_hist,
        community_map=community_map,
        max_papers_per_section=max_papers_per_section,
        overrides=overrides,
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
    """Return ``{bibcode: {title, year, abstract_snippet, citation_count}}``.

    ``papers.citation_count`` is INTEGER NULL — ``COALESCE`` to 0 so the
    Tier-3 (citation-count) fallback always has a comparable scalar. The
    ``citation_count`` is the *corpus-wide* tally; the fallback ranks
    within the working set, so this is just a per-paper attribute used
    for sorting.
    """
    sql = """
        SELECT bibcode, title, year, abstract, COALESCE(citation_count, 0)
        FROM papers
        WHERE bibcode = ANY(%s)
    """
    out: dict[str, dict[str, Any]] = {}
    with conn.cursor() as cur:
        cur.execute(sql, (list(bibcodes),))
        for row in cur.fetchall():
            bibcode = row[0]
            abstract = row[3] or ""
            citation_count = int(row[4]) if len(row) > 4 and row[4] is not None else 0
            out[bibcode] = {
                "bibcode": bibcode,
                "title": row[1],
                "year": row[2],
                "abstract_snippet": _snippet(abstract),
                "citation_count": citation_count,
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


def _assemble_sections(
    *,
    bibcodes: Sequence[str],
    sections: Sequence[str],
    paper_meta: Mapping[str, dict[str, Any]],
    intent_hist: Mapping[str, Counter[str]],
    community_map: Mapping[str, dict[str, Any]],
    max_papers_per_section: int,
    overrides: Mapping[str, str],
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

        # Tier 2: community fall-through.
        if target_section is None:
            comm = community_map.get(bibcode)
            if comm is not None:
                if modal_comm is not None and comm["community_id"] == modal_comm:
                    fallback = "background"
                else:
                    fallback = "open_questions"
                if fallback in section_set:
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
                    modal_comm=modal_comm,
                    section_set=section_set,
                ),
            )
            for b in chosen
        ]
        theme = _theme_summary_for(chosen, community_map)
        out_sections.append(
            SectionBucket(
                name=name,
                cited_papers=cited,
                theme_summary=theme,
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

    for section_name in sections:
        if buckets.get(section_name):
            continue  # tier 0/1/2 already populated this section
        if not remaining:
            break
        take = remaining[:cap]
        for bibcode in take:
            buckets[section_name].append(bibcode)
            signal_used[bibcode] = "citation_count_fallback"
            unattributed.remove(bibcode)
        remaining = remaining[cap:]
        pulled_per_section[section_name] = len(take)

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
    else:
        share = 0.0

    is_modal = community_id is not None and community_id == modal_comm

    return {
        "intent_counts": intent_counts,
        "intent_total_rows": intent_total,
        "community_id": community_id,
        "community_share": share,
        "is_modal_community": bool(is_modal),
        "modal_community_id": modal_comm,
    }


def _alternative_sections(
    *,
    bibcode: str,
    chosen_section: str,
    intent_hist: Mapping[str, Counter[str]],
    community_map: Mapping[str, dict[str, Any]],
    modal_comm: int | None,
    section_set: set[str],
) -> list[str]:
    """Return sections (other than ``chosen_section``) the paper could have
    landed in given the available signals — sorted for determinism.

    Sources of alternatives:
      * Every intent label with at least one row maps to its section
        (per :data:`INTENT_TO_SECTION`).
      * If the paper is in the modal community, ``background`` is an
        alternative. If it's in any other community, ``open_questions``
        is an alternative.
    """
    alts: set[str] = set()
    for intent_label in intent_hist.get(bibcode, Counter()):
        mapped = INTENT_TO_SECTION.get(intent_label)
        if mapped is not None and mapped in section_set:
            alts.add(mapped)

    comm_info = community_map.get(bibcode)
    if comm_info is not None:
        if modal_comm is not None and comm_info["community_id"] == modal_comm:
            if "background" in section_set:
                alts.add("background")
        else:
            if "open_questions" in section_set:
                alts.add("open_questions")

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
) -> dict[str, Any]:
    """Build the per-paper dict for a section's ``cited_papers`` list.

    ``role`` and ``section_assigned`` are aliases — ``role`` is kept for
    back-compat with earlier callers; ``section_assigned`` is the name
    used by the bead-gtsx signals schema.
    """
    meta = paper_meta.get(bibcode, {})
    return {
        "bibcode": bibcode,
        "title": meta.get("title"),
        "year": meta.get("year"),
        "abstract_snippet": meta.get("abstract_snippet", ""),
        "role": role,
        "section_assigned": role,
        "signal_used": signal_used,
        "signals": signals if signals is not None else {},
        "alternative_sections": list(alternative_sections or []),
    }


def _snippet(text: str, max_chars: int = 280) -> str:
    """Truncate an abstract for the outline payload."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
