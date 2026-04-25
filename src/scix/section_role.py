"""Rule-based section role classifier for scientific paper headers.

Maps a raw section header string into one of five canonical roles:
{background, method, result, conclusion, other}.

The classifier strips leading numbering (Arabic, Roman, "Section N:") and
matches against canonical keywords drawn from
:mod:`scix.section_parser`. The mapping is:

- introduction, motivation, background → ``background``
- methods, methodology, observations, data, data reduction,
  data analysis, experimental → ``method``
- results, result, analysis, findings → ``result``
- discussion, conclusions, conclusion, summary, summary and conclusions,
  conclusions and summary, discussion and conclusions → ``conclusion``
- abstract, acknowledgments, references, bibliography, full,
  preamble, appendix → ``other``

Anything that does not match any keyword falls back to ``other``.
"""

from __future__ import annotations

import re
from typing import Final

# Canonical role labels — keep the set small and stable.
ROLE_BACKGROUND: Final[str] = "background"
ROLE_METHOD: Final[str] = "method"
ROLE_RESULT: Final[str] = "result"
ROLE_CONCLUSION: Final[str] = "conclusion"
ROLE_OTHER: Final[str] = "other"

ROLES: Final[frozenset[str]] = frozenset(
    {ROLE_BACKGROUND, ROLE_METHOD, ROLE_RESULT, ROLE_CONCLUSION, ROLE_OTHER}
)

# Keyword → role mapping. Order does not matter; we test each keyword
# against the cleaned header and return on the first match within a
# priority-ordered scan (conclusion before result before method before
# background) so that compound headers like "Results and Discussion"
# resolve to ``conclusion`` (the more specific signal).
_BACKGROUND_KEYWORDS: Final[tuple[str, ...]] = (
    "introduction",
    "motivation",
    "background",
    "related work",
    "prior work",
    "literature review",
)

_METHOD_KEYWORDS: Final[tuple[str, ...]] = (
    "methodology",
    "methods",
    "method",
    "observations",
    "observation",
    "data reduction",
    "data analysis",
    "data acquisition",
    "data",
    "experimental",
    "experiment",
    "approach",
    "implementation",
    "model",
    "modeling",
    "modelling",
    "simulation",
    "simulations",
    "techniques",
    "procedure",
    "materials and methods",
)

_RESULT_KEYWORDS: Final[tuple[str, ...]] = (
    "results",
    "result",
    "analysis",
    "findings",
    "measurements",
)

_CONCLUSION_KEYWORDS: Final[tuple[str, ...]] = (
    "discussion and conclusions",
    "conclusions and summary",
    "summary and conclusions",
    "discussion",
    "conclusions",
    "conclusion",
    "summary",
    "concluding remarks",
    "outlook",
)

_OTHER_KEYWORDS: Final[tuple[str, ...]] = (
    "abstract",
    "acknowledgments",
    "acknowledgements",
    "acknowledgment",
    "references",
    "bibliography",
    "appendix",
    "appendices",
    "preamble",
    "full",
    "table of contents",
    "nomenclature",
    "notation",
    "glossary",
)

# Numbering prefix patterns. We strip these BEFORE matching keywords so
# that "2.1 Data Reduction" and "III. Results" reduce to bare keywords.
# - Arabic numbering: "1", "1.", "2.1", "2.1.", "2.1.3"
# - Roman numerals (I-XXXIX-ish): "I", "IV.", "VIII"
# - Trailing separator (`.`, `:`, `)`, `-`) and/or whitespace REQUIRED
#   after the numeral so that "Introduction" (which starts with "I")
#   is not misread as the Roman numeral I followed by "ntroduction".
_NUMBERING_PREFIX = re.compile(
    r"""
    ^\s*
    (?:
        \d+(?:\.\d+)*\.?           # 1, 2.1, 3.4.5
      | [ivxlcdm]+                  # roman numerals (case-insensitive)
    )
    (?:[\.:\)\-]\s*|\s+)            # MANDATORY separator: dot/colon/etc OR whitespace
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Strip a leading "Section N", "Sec. N", "Chapter N", "Part N" word
# (with optional numbering and separator). Run BEFORE _NUMBERING_PREFIX
# so that the bare numeric/roman component left behind can be stripped
# in a second pass.
_SECTION_WORD_PREFIX = re.compile(
    r"""
    ^\s*
    (?:section|sec\.?|chapter|chap\.?|part)
    \s*
    (?:
        \d+(?:\.\d+)*\.?           # optional Arabic numbering
      | [ivxlcdm]+                  # or Roman numerals
    )?
    \s*
    [:\.\-]?
    \s*
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize(header: str) -> str:
    """Return ``header`` lowercased with leading numbering stripped.

    The output is intended only for keyword matching; it is not a
    canonical name in the :mod:`scix.section_parser` sense.
    """
    if not header:
        return ""
    cleaned = header.strip()
    # Strip "Section 4: " (or "Section IV.") prefix first, then any
    # remaining bare numbering. Two passes of _NUMBERING_PREFIX handle
    # cases where the section-word strip leaves behind nothing visible.
    cleaned = _SECTION_WORD_PREFIX.sub("", cleaned)
    cleaned = _NUMBERING_PREFIX.sub("", cleaned)
    # Collapse internal whitespace and trailing punctuation.
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .:-)\t")
    return cleaned.lower()


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Whole-word containment test for any keyword in ``keywords``."""
    for kw in keywords:
        # Word-boundary match — guards against "ablation" matching "lat".
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return True
    return False


def classify_section_role(name: str) -> str:
    """Classify a section header into one of the five canonical roles.

    Parameters
    ----------
    name : str
        Raw section header text. May include numbering ("2.1 Data
        Reduction"), Roman numerals ("III. Results and Discussion"), or
        "Section N:" prefixes ("Section 4: Methodology").

    Returns
    -------
    str
        One of ``background``, ``method``, ``result``, ``conclusion``,
        ``other``. Headers that do not match any keyword fall back to
        ``other``.
    """
    if not name or not isinstance(name, str):
        return ROLE_OTHER

    cleaned = _normalize(name)
    if not cleaned:
        return ROLE_OTHER

    # Priority order matters for compound headers:
    #   "Results and Discussion" → conclusion (discussion wins)
    #   "Summary of Methods"     → method (method wins over summary)
    # We pick conclusion-class keywords first because compound headers
    # like "Discussion and Results" usually represent the synthesis
    # section, not the bare results section. Then method, since
    # "Materials and Methods" should beat "Materials". Then result, then
    # background. Other-class keywords are checked last only if nothing
    # else matched.
    if _contains_keyword(cleaned, _CONCLUSION_KEYWORDS):
        return ROLE_CONCLUSION
    if _contains_keyword(cleaned, _METHOD_KEYWORDS):
        return ROLE_METHOD
    if _contains_keyword(cleaned, _RESULT_KEYWORDS):
        return ROLE_RESULT
    if _contains_keyword(cleaned, _BACKGROUND_KEYWORDS):
        return ROLE_BACKGROUND
    if _contains_keyword(cleaned, _OTHER_KEYWORDS):
        return ROLE_OTHER

    return ROLE_OTHER


__all__ = [
    "ROLES",
    "ROLE_BACKGROUND",
    "ROLE_METHOD",
    "ROLE_RESULT",
    "ROLE_CONCLUSION",
    "ROLE_OTHER",
    "classify_section_role",
]
