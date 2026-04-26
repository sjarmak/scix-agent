"""Pure-function canonicalizer for paper section headings.

Section headings in the wild are noisy: ``"Materials and Methods"``,
``"METHODS"``, ``"Methodology"``, ``"  methods  "`` all denote the same
logical section. This module collapses those surface variants to a small,
fixed canonical vocabulary so the chunk-ingest pipeline can write a single
``section_heading_norm`` payload field that downstream code (Qdrant
filter, MV joins, eval reports) can group on without re-implementing the
same string-munging in three places.

Public API:

- :data:`HEADING_MAP` — ``dict[str, str]`` mapping pre-normalized variants
  to canonical labels. Keys are lowercase + whitespace-collapsed; callers
  may introspect or extend this mapping in-place if they need to handle a
  paper-specific synonym not yet covered.
- :func:`normalize_heading` — turns a raw heading string (or ``None``)
  into one of the canonical labels listed in :data:`CANONICAL_HEADINGS`,
  or the sentinel ``'unknown'`` for missing input and ``'other'`` for
  recognized-shape-but-not-mapped input.

No I/O, no DB, no model — importing this module must remain free of
side effects so chunkers, ingest scripts, and tests can pull it in
without dragging in heavyweight dependencies.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical vocabulary
# ---------------------------------------------------------------------------

#: Sentinel returned when the input is ``None`` / empty / whitespace-only.
UNKNOWN: str = "unknown"

#: Sentinel returned when the input is a non-empty string that does not
#: match any known synonym in :data:`HEADING_MAP`. Distinct from
#: :data:`UNKNOWN` so downstream code can tell "no heading on this chunk"
#: apart from "heading present but unrecognized".
OTHER: str = "other"

#: Frozen v1 canonical-label set. Adding a label is a v2 change — bump
#: the chunker's heading-norm version stamp so downstream filters can
#: distinguish v1 from v2 normalizations on stored chunks.
CANONICAL_HEADINGS: frozenset[str] = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "related_work",
        "methods",
        "experiments",
        "data",
        "results",
        "discussion",
        "conclusion",
        "acknowledgments",
        "references",
        "appendix",
        "other",
    }
)


# ---------------------------------------------------------------------------
# Internal normalization
# ---------------------------------------------------------------------------


def _key(s: str) -> str:
    """Collapse whitespace + lowercase a heading for ``HEADING_MAP`` lookup.

    All keys in :data:`HEADING_MAP` are produced through this helper so
    callers extending the map should pass their keys through it too —
    otherwise lookups will silently miss on the trailing-whitespace or
    mixed-case variants the function is meant to absorb.
    """
    return " ".join(s.split()).lower()


# ---------------------------------------------------------------------------
# Heading map — synonym -> canonical label
# ---------------------------------------------------------------------------


def _build_heading_map() -> dict[str, str]:
    """Build :data:`HEADING_MAP` from the per-canonical synonym tables.

    Synonyms are intentionally inlined rather than loaded from disk: this
    is a small, hand-curated table; growing it past ~200 entries is a
    signal to switch to a learned classifier, not to externalize the file.
    """
    synonyms: dict[str, tuple[str, ...]] = {
        "abstract": (
            "abstract",
            "summary abstract",
        ),
        "introduction": (
            "introduction",
            "intro",
        ),
        "background": (
            "background",
            "motivation",
            "preliminaries",
        ),
        "related_work": (
            "related work",
            "related works",
            "prior work",
            "previous work",
            "literature review",
        ),
        "methods": (
            "methods",
            "method",
            "methodology",
            "materials and methods",
            "experimental procedures",
            "materials methods",
            "methods and materials",
            "approach",
        ),
        "experiments": (
            "experiments",
            "experiment",
            "experimental setup",
            "experimental results",
            "evaluation",
        ),
        "data": (
            "data",
            "datasets",
            "dataset",
            "observations",
            "sample",
            "samples",
            "data collection",
        ),
        "results": (
            "results",
            "result",
            "findings",
        ),
        "discussion": (
            "discussion",
            "discussions",
        ),
        "conclusion": (
            "conclusion",
            "conclusions",
            "concluding remarks",
            "summary",
            "summary and conclusions",
        ),
        "acknowledgments": (
            "acknowledgments",
            "acknowledgements",
            "acknowledgment",
            "acknowledgement",
            "funding",
        ),
        "references": (
            "references",
            "bibliography",
            "works cited",
        ),
        "appendix": (
            "appendix",
            "appendices",
            "supplementary material",
            "supplementary materials",
            "supplementary information",
            "supporting information",
        ),
    }

    out: dict[str, str] = {}
    for canonical, variants in synonyms.items():
        for raw in variants:
            out[_key(raw)] = canonical
    return out


#: Lookup table from pre-normalized heading variant to canonical label.
#: Keys are lower-cased + whitespace-collapsed (see :func:`_key`).
HEADING_MAP: dict[str, str] = _build_heading_map()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_heading(raw: str | None) -> str:
    """Canonicalize a raw section heading.

    Returns one of the labels in :data:`CANONICAL_HEADINGS`:

    - :data:`UNKNOWN` (``'unknown'``) when ``raw`` is ``None``, empty, or
      whitespace-only.
    - The mapped canonical label when ``raw`` (after lower + whitespace
      collapse) appears in :data:`HEADING_MAP`.
    - :data:`OTHER` (``'other'``) for any other non-empty heading.

    The function is pure — same input always returns the same output and
    no module-level state is mutated.
    """
    if raw is None:
        return UNKNOWN
    key = _key(raw)
    if not key:
        return UNKNOWN
    return HEADING_MAP.get(key, OTHER)


__all__ = [
    "CANONICAL_HEADINGS",
    "HEADING_MAP",
    "OTHER",
    "UNKNOWN",
    "normalize_heading",
]
