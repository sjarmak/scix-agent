"""Section-grain entity linker — bead scix_experiments-67e.

Pure library that scans the section text of every paper in
``papers_fulltext`` and produces (bibcode, section_index, entity_id) link
candidates using the Tier 2 Aho-Corasick automaton from
:mod:`scix.aho_corasick`. Section roles are canonicalized via
:mod:`scix.section_role`.

Why a separate module
---------------------

:mod:`scix.aho_corasick` already exposes :func:`link_abstract` which is
text-agnostic — it scans any string. Section linking is just "run
``link_abstract`` per section text and remember which section the hit came
from." Wrapping that in a thin module lets the CLI driver in
``scripts/link_section_entities.py`` stay focused on DB plumbing while the
matching logic stays here, deterministic and unit-testable.

Output shape
------------

:class:`SectionLinkCandidate` carries everything the writer needs to
populate the ``section_entities`` row (migration 063): the entity id, the
section index + heading + canonical role, the matched span, and the
underlying tier-2 :class:`scix.aho_corasick.LinkCandidate` for evidence.

Dedupe is per ``(section_index, entity_id)``: the same entity can match
multiple surface forms within one section and we only emit the earliest
mention. Cross-section duplicates are kept — the same entity legitimately
appears in introduction AND methods, and that is the signal section-grain
linking exists to capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

from scix.aho_corasick import (
    AhocorasickAutomaton,
    LinkCandidate,
    link_abstract,
)
from scix.section_role import classify_section_role


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """One entry of ``papers_fulltext.sections``.

    ``section_index`` is the 0-based position in the source JSONB array,
    matching :mod:`section_embeddings.section_index` and the
    ``section_entities.section_index`` column on migration 063.
    """

    section_index: int
    heading: Optional[str]
    text: str


@dataclass(frozen=True)
class SectionLinkCandidate:
    """One section-grain entity match emitted by :func:`link_paper_sections`.

    Attributes
    ----------
    section_index, section_heading
        Provenance back to ``papers_fulltext.sections``.
    section_role
        Canonical 5-role label from :mod:`scix.section_role`.
    candidate
        The underlying :class:`scix.aho_corasick.LinkCandidate`. Carries
        ``entity_id``, ``matched_surface``, ``start``/``end`` offsets within
        the section text, ``confidence``, ``ambiguity_class``, ``is_alias``.
    """

    section_index: int
    section_heading: Optional[str]
    section_role: str
    candidate: LinkCandidate

    @property
    def entity_id(self) -> int:
        return self.candidate.entity_id


# ---------------------------------------------------------------------------
# Sections JSONB normalization
# ---------------------------------------------------------------------------


def parse_sections(sections_jsonb: Optional[Sequence[dict]]) -> list[Section]:
    """Normalize a ``papers_fulltext.sections`` JSONB array into Sections.

    Tolerates missing ``heading``/``text`` keys (returns ``None``/``""``)
    and silently drops entries that have neither — those are parser
    artifacts that should never make it to entity matching.

    The 0-based index is computed from the source array order, NOT from any
    field on the entry, because the JSONB schema does not currently carry
    an explicit index and we rely on stable ordering inside the array.
    """
    if not sections_jsonb:
        return []
    out: list[Section] = []
    for idx, entry in enumerate(sections_jsonb):
        if not isinstance(entry, dict):
            continue
        heading = entry.get("heading")
        text = entry.get("text") or ""
        if not text and not heading:
            continue
        out.append(
            Section(
                section_index=idx,
                heading=heading if heading else None,
                text=text,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------


def _dedupe_per_section(
    section_index: int,
    candidates: Iterable[LinkCandidate],
) -> list[LinkCandidate]:
    """Keep the earliest match per ``entity_id`` within a single section.

    Mirrors :func:`scripts.link_tier2._dedupe_candidates` so section_entities
    rows have the same one-row-per-(unit, entity) shape as document_entities.
    """
    seen: dict[int, LinkCandidate] = {}
    for cand in candidates:
        prev = seen.get(cand.entity_id)
        if prev is None or cand.start < prev.start:
            seen[cand.entity_id] = cand
    return list(seen.values())


def link_paper_sections(
    sections: Sequence[Section],
    automaton: AhocorasickAutomaton,
) -> list[SectionLinkCandidate]:
    """Run the tier-2 automaton against every section text in ``sections``.

    Returns a flat list of :class:`SectionLinkCandidate`, one per
    ``(section_index, entity_id)`` pair. Cross-section duplicates are
    intentionally preserved — the whole point of section linking is to know
    that the same entity appears in different sections.
    """
    out: list[SectionLinkCandidate] = []
    for section in sections:
        if not section.text:
            continue
        raw = link_abstract(section.text, automaton)
        if not raw:
            continue
        deduped = _dedupe_per_section(section.section_index, raw)
        role = classify_section_role(section.heading or "")
        for cand in deduped:
            out.append(
                SectionLinkCandidate(
                    section_index=section.section_index,
                    section_heading=section.heading,
                    section_role=role,
                    candidate=cand,
                )
            )
    return out


__all__ = [
    "Section",
    "SectionLinkCandidate",
    "parse_sections",
    "link_paper_sections",
]
