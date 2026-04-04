"""Regex-based section splitter for astronomy papers.

Splits body text into (section_name, start_char, end_char, text) tuples
for standard IMRaD sections. Pure Python utility with no DB dependencies.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Canonical section names we normalize to
SECTION_NAMES: frozenset[str] = frozenset(
    {
        "abstract",
        "introduction",
        "methods",
        "observations",
        "data",
        "results",
        "discussion",
        "conclusions",
        "summary",
        "acknowledgments",
        "references",
    }
)

# Mapping of common variants to canonical names
_ALIASES: dict[str, str] = {
    "introduction": "introduction",
    "motivation": "introduction",
    "methods": "methods",
    "methodology": "methods",
    "method": "methods",
    "observations": "observations",
    "observation": "observations",
    "data": "data",
    "data reduction": "data",
    "data analysis": "data",
    "results": "results",
    "result": "results",
    "analysis": "results",
    "discussion": "discussion",
    "conclusions": "conclusions",
    "conclusion": "conclusions",
    "summary": "summary",
    "summary and conclusions": "conclusions",
    "conclusions and summary": "conclusions",
    "discussion and conclusions": "conclusions",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "acknowledgment": "acknowledgments",
    "references": "references",
    "bibliography": "references",
    "abstract": "abstract",
}

# Pattern to match section headers.
# Matches optional numbering (e.g. "1.", "2.1", "II.") followed by a header name.
# Headers may appear on their own line, possibly preceded by whitespace.
_NUMBERING = r"(?:[0-9]+\.?[0-9]*\.?\s+|[IVXLC]+\.?\s+)?"
_HEADER_NAMES = "|".join(re.escape(name) for name in sorted(_ALIASES.keys(), key=len, reverse=True))
_SECTION_RE = re.compile(
    rf"^[ \t]*{_NUMBERING}({_HEADER_NAMES})[ \t]*\.?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


class Section(NamedTuple):
    """A parsed section of a paper body."""

    name: str
    start: int
    end: int
    text: str


def _normalize_name(raw: str) -> str:
    """Normalize a raw section header to a canonical name."""
    key = raw.strip().lower()
    return _ALIASES.get(key, key)


def parse_sections(body: str) -> list[tuple[str, int, int, str]]:
    """Split paper body text into sections.

    Parameters
    ----------
    body : str
        Plain-text body of a paper.

    Returns
    -------
    list[tuple[str, int, int, str]]
        Each tuple is (section_name, start_char, end_char, text).
        If no recognizable section headers are found, returns a single
        ``('full', 0, len(body), body)`` tuple.
    """
    if not body:
        return [("full", 0, 0, "")]

    matches: list[tuple[str, int, int]] = []
    for m in _SECTION_RE.finditer(body):
        name = _normalize_name(m.group(1))
        header_end = m.end()
        matches.append((name, m.start(), header_end))

    if not matches:
        return [("full", 0, len(body), body)]

    sections: list[tuple[str, int, int, str]] = []

    # If there is text before the first section header, capture it as preamble
    if matches[0][1] > 0:
        preamble_text = body[: matches[0][1]].strip()
        if preamble_text:
            sections.append(("preamble", 0, matches[0][1], body[: matches[0][1]]))

    for i, (name, start, header_end) in enumerate(matches):
        if i + 1 < len(matches):
            end = matches[i + 1][1]
        else:
            end = len(body)
        section_text = body[header_end:end]
        sections.append((name, start, end, section_text))

    return sections
