"""Tier 1 ADS body section splitter (inline-keyword-anchor).

Splits the ``body`` text of a paper (as stored in ADS JSONL / ``papers.body``)
into structured sections by scanning for a small, canonical vocabulary of
section markers (INTRODUCTION, METHODS, RESULTS, REFERENCES, …) with
word-boundary anchors. Built for flat, single-line bodies produced by
PDF text extraction, where line-anchored regexes (``^heading$``) fail
because headings are embedded mid-line in the extracted stream.

The parser is:

* **Pure**: no IO, no DB, no subprocess.
* **Bibstem-agnostic**: the ``bibstem`` argument is accepted for backward
  compatibility but does not affect behaviour. One compiled vocabulary
  regex runs against every body regardless of journal.
* **Versioned**: ``PARSER_VERSION`` is bumped any time the vocabulary or
  matching semantics change. Downstream consumers persist this string
  with every parse so we can re-parse stale rows after a bump.

Returned metadata fields are fixed by the public contract:

``n_sections``
    Number of ``Section`` records parsed.
``coverage_frac``
    Fraction of the input body attributed to a section body
    (sum of ``len(section.text)`` / ``len(body)``), clamped to ``[0, 1]``.
``first_heading_offset``
    Character offset of the first detected heading (including any numeric
    prefix that was stripped from the heading string), or ``-1`` when no
    heading matched.
``bibstem_family``
    Retained for backward compatibility. Always ``"inline_v2"`` — the
    bibstem is no longer dispatched to a family.
``patterns_tried``
    Number of compiled patterns evaluated per call. Always ``1``.

``compute_confidence`` returns a scalar in ``[0, 1]`` that downstream
routing uses to decide whether to accept the Tier 1 parse or escalate.
"""

from __future__ import annotations

import logging
import re

from scix.sources.ar5iv import Section

logger = logging.getLogger(__name__)

PARSER_VERSION: str = "ads_body_inline_v2"

__all__ = ["PARSER_VERSION", "Section", "compute_confidence", "parse_ads_body"]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Canonical section-marker vocabulary. ORDER MATTERS within the alternation:
# ``re`` picks the leftmost match, and at the same start position it picks
# the first-listed alternative. So longer/more-specific alternatives must
# come first where they overlap at the same starting character:
#
# * ``APPENDIX\s+[A-Z]\b`` before ``APPENDIX\b`` so ``APPENDIX A`` matches
#   as the two-token form.
# * ``CONCLUSIONS\b`` before ``CONCLUSION\b`` so the plural wins.
# * ``ACKNOWLEDGEMENTS\b`` before ``ACKNOWLEDGMENTS\b`` (one is a spelling
#   superset of the other — list the longer first).
# * ``METHODOLOGY\b`` before ``METHODS\b`` — they don't actually overlap
#   under ``\b`` anchors since one ends at ``Y`` and the other at ``S``,
#   but keep the longer word first as a defence-in-depth choice.
_CANONICAL_MARKERS: tuple[str, ...] = (
    # APPENDIX with optional trailing letter — two-token form first.
    r"APPENDIX\s+[A-Z]\b",
    r"APPENDIX\b",
    # Plural-before-singular pairs.
    r"CONCLUSIONS\b",
    r"CONCLUSION\b",
    r"ACKNOWLEDGEMENTS\b",
    r"ACKNOWLEDGMENTS\b",
    # Methodology before Methods (same start-letter, same prefix).
    r"METHODOLOGY\b",
    r"METHODS\b",
    # Remaining markers — no same-start overlap, order is not load-bearing.
    r"ABSTRACT\b",
    r"INTRODUCTION\b",
    r"BACKGROUND\b",
    r"RELATED\s+WORK\b",
    r"OBSERVATIONS\b",
    r"DATA\b",
    r"MODEL\b",
    r"THEORY\b",
    r"ANALYSIS\b",
    r"RESULTS\b",
    r"DISCUSSION\b",
    r"SUMMARY\b",
    r"REFERENCES\b",
    r"BIBLIOGRAPHY\b",
)

# Single compiled vocabulary regex — one pass per body.
_MARKER_RE: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(_CANONICAL_MARKERS) + r")",
    re.IGNORECASE,
)

# Numeric-prefix regex for ``1 INTRODUCTION`` / ``1. Introduction`` /
# ``1.1 Background`` shapes. Matched against the small lookback window
# immediately preceding a keyword hit. The prefix must abut the keyword
# (end of the prefix string == start of the keyword match).
_NUMERIC_PREFIX_RE: re.Pattern[str] = re.compile(r"\d+(?:\.\d+)*[.\s]+\Z")

# How many characters to look back for a numeric prefix. Comfortably
# covers up to ``"99.99 "`` (6 chars) with slack for multi-dot variants.
_PREFIX_LOOKBACK: int = 12

# Family tag — retained for backward compatibility with downstream code
# that reads ``metadata["bibstem_family"]``.
_FAMILY_TAG: str = "inline_v2"


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


def _canonical_heading(raw: str) -> str:
    """Normalize a matched heading to the canonical UPPERCASE form.

    Collapses runs of internal whitespace to a single ASCII space so that
    ``"Appendix   a"`` and ``"appendix\tA"`` both emit ``"APPENDIX A"``.
    """
    return re.sub(r"\s+", " ", raw).strip().upper()


def _extend_start_for_numeric_prefix(body: str, start: int) -> int:
    """If a numeric prefix immediately precedes ``start``, return the new start.

    Returns ``start`` unchanged when no abutting prefix is present. The
    prefix is consumed from the heading STRING (``_canonical_heading`` runs
    on the keyword only) but included in the heading SPAN so that
    ``section.offset`` points at the true start of the visual heading
    ("1. INTRODUCTION" → offset at the '1').
    """
    if start == 0:
        return start
    window_lo = max(0, start - _PREFIX_LOOKBACK)
    window = body[window_lo:start]
    m = _NUMERIC_PREFIX_RE.search(window)
    if m is None:
        return start
    return window_lo + m.start()


def _collect_marker_hits(body: str) -> list[tuple[int, int, str]]:
    """Return ordered ``(start, end, canonical_heading)`` tuples for marker hits.

    ``start`` is extended backward to include any abutting numeric prefix so
    that downstream text-slicing does not attribute the prefix to the
    previous section's trailing text.
    """
    hits: list[tuple[int, int, str]] = []
    for m in _MARKER_RE.finditer(body):
        raw_start, raw_end = m.start(), m.end()
        heading = _canonical_heading(m.group(0))
        span_start = _extend_start_for_numeric_prefix(body, raw_start)
        hits.append((span_start, raw_end, heading))
    return hits


def _build_sections(
    body: str, hits: list[tuple[int, int, str]]
) -> list[Section]:
    """Build ``Section`` list from ordered ``(start, end, heading)`` tuples."""
    sections: list[Section] = []
    for i, (start, end, heading) in enumerate(hits):
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(body)
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


def _empty_metadata() -> dict:
    return {
        "n_sections": 0,
        "coverage_frac": 0.0,
        "first_heading_offset": -1,
        "bibstem_family": _FAMILY_TAG,
        "patterns_tried": 1,
    }


def parse_ads_body(
    body: str,
    bibstem: str | None = None,
) -> tuple[list[Section], dict]:
    """Split an ADS body text into ``Section`` records by keyword-anchor scan.

    Parameters
    ----------
    body:
        Plain-text body of a paper (as stored in ``papers.body``). May be
        empty. Expected to be a flat, possibly single-line, stream from a
        PDF text extractor.
    bibstem:
        Accepted for backward compatibility. Ignored — the inline-anchor
        parser applies the same canonical vocabulary regardless of journal.

    Returns
    -------
    tuple[list[Section], dict]
        ``(sections, metadata)`` where ``metadata`` has keys
        ``n_sections``, ``coverage_frac``, ``first_heading_offset``,
        ``bibstem_family`` (always ``"inline_v2"``), and ``patterns_tried``
        (always ``1``).

    Notes
    -----
    * Headings are emitted in canonical UPPERCASE regardless of input case.
    * Numeric prefixes (``"1 "``, ``"1. "``, ``"1.1 "``) are stripped from
      the emitted heading string but included in the ``offset`` and
      inter-heading text-slice spans.
    * If fewer than 2 distinct canonical headings are found, the body is
      considered unparseable and an empty section list is returned.
    """
    del bibstem  # Retained in signature only.

    # Empty-body short-circuit.
    if not body:
        return [], _empty_metadata()

    hits = _collect_marker_hits(body)

    # Minimum-threshold gate: need at least 2 DISTINCT canonical headings.
    # ``APPENDIX A`` and ``APPENDIX B`` count as two distinct headings.
    distinct = {h for _, _, h in hits}
    if len(distinct) < 2:
        return [], _empty_metadata()

    # finditer yields matches in document order, but after extending starts
    # backward for numeric prefixes we sort defensively in case a prefix
    # pushes a later match's span ahead of an earlier one (this cannot
    # happen with non-overlapping keyword matches, but the cost is trivial).
    hits.sort(key=lambda t: t[0])

    sections = _build_sections(body, hits)

    n_sections = len(sections)
    total_text = sum(len(s.text) for s in sections)
    coverage_frac = max(0.0, min(total_text / len(body), 1.0))
    first_heading_offset = sections[0].offset

    metadata: dict = {
        "n_sections": n_sections,
        "coverage_frac": coverage_frac,
        "first_heading_offset": first_heading_offset,
        "bibstem_family": _FAMILY_TAG,
        "patterns_tried": 1,
    }
    return sections, metadata
