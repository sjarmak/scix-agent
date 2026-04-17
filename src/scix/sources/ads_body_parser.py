"""Tier 1 ADS body section splitter (regex-based).

Splits the ``body`` text of a paper (as stored in ADS JSONL / ``papers.body``)
into structured sections by matching family-specific heading patterns. This
is the cheap, deterministic first tier of the body-parsing ladder — higher
tiers (LLM classification, GROBID, etc.) are reserved for bodies that this
parser flags as low confidence.

The parser is:

* **Pure**: no IO, no DB, no subprocess.
* **Family-aware**: the ``bibstem`` argument selects a regex family
  (``mnras``, ``apj``, ``physrev``) with a ``fallback`` that tries a bank
  of bare-heading shapes. Unknown / missing bibstems route to the fallback.
* **Versioned**: ``PARSER_VERSION`` is bumped any time the regex bank or
  semantics change. Downstream consumers persist this string with every
  parse so we can re-parse stale rows after a bump.

Returned metadata fields are fixed by the public contract:

``n_sections``
    Number of ``Section`` records parsed.
``coverage_frac``
    Fraction of the input body that is attributed to a section body
    (sum of ``len(section.text)`` / ``len(body)``), clamped to ``[0, 1]``.
``first_heading_offset``
    Character offset of the first detected heading, or ``-1`` when no
    heading matched.
``bibstem_family``
    Which regex family was selected (one of ``mnras``, ``apj``, ``physrev``,
    ``fallback``).
``patterns_tried``
    Number of patterns actually evaluated. ``1`` for a named family,
    ``len(fallback bank)`` for the fallback family.

``compute_confidence`` returns a scalar in ``[0, 1]`` that downstream
routing uses to decide whether to accept the Tier 1 parse or escalate.
"""

from __future__ import annotations

import logging
import re

from scix.sources.ar5iv import Section

logger = logging.getLogger(__name__)

PARSER_VERSION: str = "ads_body_regex@v1"

__all__ = ["PARSER_VERSION", "Section", "compute_confidence", "parse_ads_body"]


# ---------------------------------------------------------------------------
# Regex bank
# ---------------------------------------------------------------------------

# Named family patterns. Each pattern matches a single heading line and
# captures the heading text (without the numbering prefix).
_MNRAS_RE: re.Pattern[str] = re.compile(
    r"^(?P<num>\d+)\s+(?P<heading>[A-Z][^\n]*?)\s*$",
    re.MULTILINE,
)
_APJ_RE: re.Pattern[str] = re.compile(
    r"^(?P<num>\d+)\.\s+(?P<heading>[A-Z][a-z][^\n]*?)\s*$",
    re.MULTILINE,
)
_PHYSREV_RE: re.Pattern[str] = re.compile(
    r"^(?P<num>[IVX]+)\.\s+(?P<heading>[A-Z][^\n]*?)\s*$",
    re.MULTILINE,
)

# Fallback bare-heading patterns. Tried in order; the one yielding the most
# matches wins.
_BARE_ALLCAPS_RE: re.Pattern[str] = re.compile(
    r"^(?P<heading>[A-Z][A-Z0-9 \-/&]{2,})\s*$",
    re.MULTILINE,
)
_BARE_TITLECASE_RE: re.Pattern[str] = re.compile(
    r"^(?P<heading>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})\s*$",
    re.MULTILINE,
)

_NAMED_FAMILIES: dict[str, re.Pattern[str]] = {
    "mnras": _MNRAS_RE,
    "apj": _APJ_RE,
    "physrev": _PHYSREV_RE,
}

_FALLBACK_BANK: tuple[re.Pattern[str], ...] = (
    _BARE_ALLCAPS_RE,
    _BARE_TITLECASE_RE,
)


# Bibstem → family. Bibstems not in this map (or ``None``) route to
# ``"fallback"``. Bibstem matching is case-sensitive against the canonical
# ADS form.
_BIBSTEM_TO_FAMILY: dict[str, str] = {
    # MNRAS family
    "MNRAS": "mnras",
    "MNRASL": "mnras",
    # ApJ / A&A / AJ / PASP family (ApJ-style numbered "1. Introduction")
    "ApJ": "apj",
    "ApJS": "apj",
    "ApJL": "apj",
    "AJ": "apj",
    "A&A": "apj",
    "A&ARv": "apj",
    "AAS": "apj",
    "PASP": "apj",
    "PASA": "apj",
    # PhysRev family (roman-numeral headings)
    "PhRvD": "physrev",
    "PhRvL": "physrev",
    "PhRvA": "physrev",
    "PhRvB": "physrev",
    "PhRvC": "physrev",
    "PhRvE": "physrev",
    "PhRvX": "physrev",
    "RvMP": "physrev",
}


def _family_for(bibstem: str | None) -> str:
    """Return the regex family for a given bibstem (or ``"fallback"``)."""
    if bibstem is None:
        return "fallback"
    return _BIBSTEM_TO_FAMILY.get(bibstem, "fallback")


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def compute_confidence(
    n_sections: int,
    coverage_frac: float,
    first_heading_offset: int,
) -> float:
    """Return a Tier 1 parse-quality score in ``[0, 1]``.

    The score is monotonically non-decreasing in ``n_sections`` (saturating
    at 6 headings) and in ``coverage_frac``. A late first heading reduces
    the score slightly. Weights: 0.45 on headings, 0.45 on coverage, 0.10
    on the offset term.
    """
    headings_score = min(max(n_sections, 0) / 6.0, 1.0)
    coverage_score = max(0.0, min(coverage_frac, 1.0))

    if first_heading_offset < 0:
        offset_score = 0.0
    else:
        offset_score = max(0.0, 1.0 - first_heading_offset / 5000.0)

    score = 0.45 * headings_score + 0.45 * coverage_score + 0.10 * offset_score
    # Numerical safety — keep inside [0, 1].
    return max(0.0, min(score, 1.0))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _collect_matches(
    body: str, pattern: re.Pattern[str]
) -> list[tuple[int, int, str]]:
    """Return ``(start, end, heading)`` tuples for every match of ``pattern``."""
    hits: list[tuple[int, int, str]] = []
    for m in pattern.finditer(body):
        # Prefer the named "heading" group when present; else first group.
        heading = m.group("heading") if "heading" in m.groupdict() else m.group(0)
        hits.append((m.start(), m.end(), heading.strip()))
    return hits


def _build_sections(
    body: str, matches: list[tuple[int, int, str]]
) -> list[Section]:
    """Build ``Section`` list from ordered ``(start, end, heading)`` matches."""
    sections: list[Section] = []
    for i, (start, end, heading) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(body)
        text = body[end:next_start]
        sections.append(
            Section(
                heading=heading,
                level=1,
                text=text,
                offset=start,
            )
        )
    return sections


def parse_ads_body(
    body: str,
    bibstem: str | None = None,
) -> tuple[list[Section], dict]:
    """Split an ADS body text into ``Section`` records by regex family.

    Parameters
    ----------
    body:
        Plain-text body of a paper (as stored in ``papers.body``). May be
        empty.
    bibstem:
        Canonical ADS bibstem (``"MNRAS"``, ``"ApJ"``, ``"PhRvD"``, …) used
        to select the regex family. ``None`` or an unknown value routes to
        the bare-heading fallback.

    Returns
    -------
    tuple[list[Section], dict]
        ``(sections, metadata)`` where ``metadata`` has keys
        ``n_sections``, ``coverage_frac``, ``first_heading_offset``,
        ``bibstem_family``, and ``patterns_tried``.
    """
    family = _family_for(bibstem)

    # Empty-body short-circuit: preserves the public contract with deterministic
    # metadata rather than dividing by zero on coverage_frac.
    if not body:
        return [], {
            "n_sections": 0,
            "coverage_frac": 0.0,
            "first_heading_offset": -1,
            "bibstem_family": family,
            "patterns_tried": 0,
        }

    if family in _NAMED_FAMILIES:
        matches = _collect_matches(body, _NAMED_FAMILIES[family])
        patterns_tried = 1
    else:
        # Fallback: try every pattern, pick the one with the most matches.
        best_matches: list[tuple[int, int, str]] = []
        for pattern in _FALLBACK_BANK:
            found = _collect_matches(body, pattern)
            if len(found) > len(best_matches):
                best_matches = found
        matches = best_matches
        patterns_tried = len(_FALLBACK_BANK)

    # Matches from ``finditer`` are already in document order, but be defensive:
    # a future composite fallback could concatenate hits from several patterns.
    matches.sort(key=lambda t: t[0])

    sections = _build_sections(body, matches)

    # Metadata
    n_sections = len(sections)
    total_text = sum(len(s.text) for s in sections)
    coverage_frac = max(0.0, min(total_text / len(body), 1.0)) if body else 0.0
    first_heading_offset = sections[0].offset if sections else -1

    metadata: dict = {
        "n_sections": n_sections,
        "coverage_frac": coverage_frac,
        "first_heading_offset": first_heading_offset,
        "bibstem_family": family,
        "patterns_tried": patterns_tried,
    }
    return sections, metadata


__all__ = [
    "PARSER_VERSION",
    "Section",
    "compute_confidence",
    "parse_ads_body",
]
