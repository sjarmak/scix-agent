"""Section-aware negative-results / null-finding detector (PRD M3).

Detects spans in paper body text that look like null results, non-detections,
upper-limit reports, refutations, or retractions. Emits structured spans with
provenance suitable for ``staging.extractions`` rows.

Design contract
---------------
A span is classified as a negative-result iff BOTH:

  (a) The containing section is in ``{results, discussion, conclusions, summary}``
      (the "section guard"); OR — when the body has no recognizable headers —
      the span matches a tier=3 (high-confidence) pattern.

  (b) The span matches one of a curated set of hedging / null-result patterns.

Pattern catalog tiers
---------------------
- tier=3 / 'high'   : unambiguous null statements ("no significant detection",
                      "failed to detect", "rejected at 3 sigma", "retracted").
- tier=2 / 'medium' : hedged null statements ("cannot rule out",
                      "consistent with no signal", "no statistically significant").
- tier=1 / 'low'    : ambiguous hedging ("if real", "marginal detection",
                      "tentative"). Only fires inside in-scope sections.

The detector is pure-Python (no model dependency) and ZFC-allowed because:
- patterns are mechanical regexes, not semantic judgements;
- the section guard is structural;
- confidence_tier is derived from explicit pattern strength, not LLM scoring.

Public surface
--------------
- ``NegativeResultSpan``           — frozen dataclass (one detected span)
- ``detect_negative_results``      — pure detector over (body, sections)
- ``insert_extractions``           — write spans to ``staging.extractions``
- ``EXTRACTION_TYPE``              — ``'negative_result'``
- ``EXTRACTION_VERSION`` / ``SOURCE`` — provenance stamps
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from psycopg.types.json import Jsonb

from scix.section_parser import parse_sections

if TYPE_CHECKING:  # pragma: no cover — typing-only import
    import psycopg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provenance constants
# ---------------------------------------------------------------------------

EXTRACTION_TYPE: str = "negative_result"
EXTRACTION_VERSION: str = "neg_results_v1"
SOURCE: str = "neg_results_v1"

# Sections in which medium- and low-tier patterns are eligible to fire.
# High-tier patterns also fire here; section_guard() handles the relaxation
# for full-body / preamble fall-through cases.
SECTIONS_OF_INTEREST: frozenset[str] = frozenset(
    {"results", "discussion", "conclusions", "summary"}
)

# Sections from which we NEVER emit (any tier).
SECTIONS_BLOCKED: frozenset[str] = frozenset(
    {
        "abstract",
        "introduction",
        "methods",
        "observations",
        "data",
        "acknowledgments",
        "references",
    }
)

# Evidence span window (centered on the match midpoint).
EVIDENCE_SPAN_CHARS: int = 250

# ---------------------------------------------------------------------------
# Pattern catalog
# ---------------------------------------------------------------------------
# Each entry: (pattern_id, regex, tier_int, tier_label).
# Patterns are case-insensitive and avoid \b on phrases that may contain
# punctuation; explicit lookarounds keep "no significantly different" from
# stealing the high-tier "no significant" prefix when the latter applies.
#
# To extend: keep tier=3 patterns precise (low false-positive rate); admit new
# tier=2 patterns only after they survive the gold fixture eval at P>=0.70.

_PATTERNS_RAW: list[tuple[str, str, int, str]] = [
    # --- tier=3 / high ---
    ("no_significant",
     r"\bno\s+(?:statistically\s+)?significant\b",
     3, "high"),
    ("null_result",
     r"\bnull\s+results?\b",
     3, "high"),
    ("failed_to_detect",
     r"\bfailed?\s+to\s+detect\b",
     3, "high"),
    ("do_not_detect",
     r"\b(?:do(?:es)?|did)\s+not\s+detect\b",
     3, "high"),
    ("we_do_not_find",
     r"\bwe\s+(?:do\s+not|did\s+not|cannot)\s+(?:find|see|observe|measure)\b",
     3, "high"),
    ("no_evidence",
     r"\bno\s+evidence\s+(?:for|of)\b",
     3, "high"),
    ("non_detection",
     r"\bnon[- ]?detections?\b",
     3, "high"),
    ("not_detected",
     r"\b(?:is|are|was|were)\s+not\s+detected\b",
     3, "high"),
    ("rejected_sigma",
     r"\b(?:rejected?|excluded?|ruled\s+out)\s+(?:at|with|to)\s+(?:more\s+than\s+)?"
     r"(?:>\s*)?\d+(?:\.\d+)?\s*(?:[σ]|sigma|standard\s+deviations?)\b",
     3, "high"),
    ("ruled_out",
     r"\b(?:rule|ruled|ruling)\s+out\b",
     3, "high"),
    ("retracted",
     r"\bretracted\b",
     3, "high"),
    ("refuted",
     r"\b(?:was|has\s+been|is)\s+(?:refuted|disproved|disproven)\b",
     3, "high"),
    ("no_detection",
     r"\bno\s+detections?\s+(?:of|at|in|was|were|is|are)\b",
     3, "high"),

    # --- tier=2 / medium ---
    ("cannot_rule_out",
     r"\b(?:cannot|can\s+not|are\s+unable\s+to|unable\s+to)\s+rule\s+out\b",
     2, "medium"),
    ("consistent_with_no",
     r"\bconsistent\s+with\s+(?:no|zero|the\s+absence\s+of)\b",
     2, "medium"),
    ("not_consistent_with",
     r"\b(?:not\s+consistent|inconsistent)\s+with\b",
     2, "medium"),
    ("upper_limit",
     r"\b(?:place|placing|set|setting|derive|deriving|report|reporting|provide|providing)"
     r"\s+(?:a|an|the)?\s*(?:\d+(?:\.\d+)?\s*[σ]\s+)?upper\s+limits?\s+(?:of|on)\b",
     2, "medium"),
    ("lower_limit",
     r"\b(?:place|placing|set|setting|derive|deriving|report|reporting|provide|providing)"
     r"\s+(?:a|an|the)?\s*(?:\d+(?:\.\d+)?\s*[σ]\s+)?lower\s+limits?\s+(?:of|on)\b",
     2, "medium"),
    ("no_correlation",
     r"\bno\s+(?:significant\s+)?correlation\b",
     2, "medium"),
    ("no_clear_signal",
     r"\bno\s+clear\s+(?:evidence|signal|trend|detection|excess|feature)\b",
     2, "medium"),
    ("no_difference",
     r"\bno\s+(?:significant\s+)?(?:difference|differences)\s+(?:between|in|was|were)\b",
     2, "medium"),

    # --- tier=1 / low ---
    ("if_real",
     r"\b(?:the\s+(?:signal|feature|excess|detection|effect)\s*,?\s+if\s+real)\b",
     1, "low"),
    ("marginal_detection",
     r"\bmarginal(?:ly)?\s+(?:significant|detected|consistent)\b",
     1, "low"),
    ("tentative_detection",
     r"\btentative(?:ly)?\s+(?:detection|detected|identified)\b",
     1, "low"),
]


_PATTERNS: list[tuple[str, re.Pattern[str], int, str]] = [
    (pid, re.compile(rx, re.IGNORECASE), tier, label)
    for (pid, rx, tier, label) in _PATTERNS_RAW
]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NegativeResultSpan:
    """One detected negative-result span.

    ``start_char`` and ``end_char`` are absolute offsets into the original
    body string (NOT into the section text). ``evidence_span`` is exactly
    ``EVIDENCE_SPAN_CHARS`` characters of context centered on the match,
    space-padded if the body is too short to provide a full window.
    """

    section: str
    pattern_id: str
    confidence_tier: int  # 1 / 2 / 3
    confidence_label: str  # 'low' / 'medium' / 'high'
    match_text: str
    start_char: int
    end_char: int
    evidence_span: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_evidence(body: str, start: int, end: int) -> str:
    """Return exactly ``EVIDENCE_SPAN_CHARS`` characters centered on the match.

    If the body is shorter than the window, pads on the right with spaces.
    Always returns a string of length ``EVIDENCE_SPAN_CHARS``.
    """
    n = len(body)
    if n == 0:
        return " " * EVIDENCE_SPAN_CHARS

    mid = (start + end) // 2
    half = EVIDENCE_SPAN_CHARS // 2
    win_start = max(0, mid - half)
    win_end = win_start + EVIDENCE_SPAN_CHARS
    if win_end > n:
        # Shift the window left so we keep EVIDENCE_SPAN_CHARS chars when possible.
        win_end = n
        win_start = max(0, n - EVIDENCE_SPAN_CHARS)

    snippet = body[win_start:win_end]
    if len(snippet) < EVIDENCE_SPAN_CHARS:
        snippet = snippet + " " * (EVIDENCE_SPAN_CHARS - len(snippet))
    return snippet


def _section_allows(section_name: str, tier: int, has_real_sections: bool) -> bool:
    """Decide whether to emit a tier-N match found inside ``section_name``.

    - ``SECTIONS_BLOCKED`` always blocks.
    - Inside ``SECTIONS_OF_INTEREST`` all tiers are allowed.
    - ``'preamble'`` / ``'full'`` (no real headers detected): only tier=3.
    """
    if section_name in SECTIONS_BLOCKED:
        return False
    if section_name in SECTIONS_OF_INTEREST:
        return True
    # Section name is unrecognized OR body had no headers ('full'/'preamble').
    if not has_real_sections and tier >= 3:
        return True
    return False


def _has_real_sections(sections: list[tuple[str, int, int, str]]) -> bool:
    """Return True iff ``parse_sections`` found at least one canonical header."""
    for name, _, _, _ in sections:
        if name not in {"full", "preamble"}:
            return True
    return False


def _dedup_overlapping(spans: list[NegativeResultSpan]) -> list[NegativeResultSpan]:
    """Drop overlapping matches, keeping the highest tier (then earliest start).

    Two spans overlap when their ``[start_char, end_char)`` intervals intersect.
    Sort by (-tier, start_char) and greedily accept non-overlapping spans.
    """
    if not spans:
        return spans
    ordered = sorted(spans, key=lambda s: (-s.confidence_tier, s.start_char))
    accepted: list[NegativeResultSpan] = []
    for cand in ordered:
        clash = False
        for kept in accepted:
            if cand.start_char < kept.end_char and kept.start_char < cand.end_char:
                clash = True
                break
        if not clash:
            accepted.append(cand)
    accepted.sort(key=lambda s: s.start_char)
    return accepted


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def detect_negative_results(
    body: str,
    sections: list[tuple[str, int, int, str]] | None = None,
) -> list[NegativeResultSpan]:
    """Detect negative-result spans in ``body``.

    Parameters
    ----------
    body
        Plain-text body of a paper.
    sections
        Optional pre-computed output of
        :func:`scix.section_parser.parse_sections`. If ``None``, the parser
        is invoked on ``body``.

    Returns
    -------
    list[NegativeResultSpan]
        One entry per accepted span, in body-order, with overlapping
        matches deduplicated (highest tier wins).
    """
    if not body:
        return []

    if sections is None:
        sections = parse_sections(body)

    has_real = _has_real_sections(sections)

    out: list[NegativeResultSpan] = []
    for name, start, _end, text in sections:
        if not text:
            continue
        # The text returned by parse_sections starts AFTER the header line,
        # but ``start`` is the offset of the header itself in the body. We
        # need an absolute-offset base for the section text.
        # Recover it by scanning forward from start until we land on the
        # first character of `text` in body. The contract from parse_sections
        # is that body[header_end:end] == text (for non-preamble sections),
        # so body[start:].find(text[:64]) is robust.
        if name == "preamble":
            base = 0
        else:
            sample = text[:64] if len(text) >= 64 else text
            offset = body.find(sample, start) if sample else -1
            base = offset if offset != -1 else start

        for pattern_id, regex, tier, label in _PATTERNS:
            for m in regex.finditer(text):
                if not _section_allows(name, tier, has_real):
                    continue
                abs_start = base + m.start()
                abs_end = base + m.end()
                evidence = _extract_evidence(body, abs_start, abs_end)
                out.append(
                    NegativeResultSpan(
                        section=name,
                        pattern_id=pattern_id,
                        confidence_tier=tier,
                        confidence_label=label,
                        match_text=m.group(0),
                        start_char=abs_start,
                        end_char=abs_end,
                        evidence_span=evidence,
                    )
                )

    return _dedup_overlapping(out)


# ---------------------------------------------------------------------------
# DB writer
# ---------------------------------------------------------------------------


_INSERT_SQL = (
    "INSERT INTO staging.extractions "
    "(bibcode, extraction_type, extraction_version, payload, source, confidence_tier) "
    "VALUES (%s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (bibcode, extraction_type, extraction_version) "
    "DO UPDATE SET payload = EXCLUDED.payload, "
    "              source = EXCLUDED.source, "
    "              confidence_tier = EXCLUDED.confidence_tier"
)


def _build_payload(spans: list[NegativeResultSpan]) -> dict[str, object]:
    """Serialize spans into the JSONB payload written to staging.extractions."""
    tier_counts = {"high": 0, "medium": 0, "low": 0}
    for s in spans:
        tier_counts[s.confidence_label] = tier_counts.get(s.confidence_label, 0) + 1
    return {
        "spans": [asdict(s) for s in spans],
        "n_spans": len(spans),
        "tier_counts": tier_counts,
        "extractor": SOURCE,
    }


def insert_extractions(
    conn: "psycopg.Connection",
    bibcode: str,
    spans: list[NegativeResultSpan],
) -> int:
    """Upsert detected spans for ``bibcode`` into ``staging.extractions``.

    Always writes exactly one row per (bibcode, extraction_type,
    extraction_version) — even when ``spans`` is empty (records the
    "scanned, found nothing" outcome and prevents reprocessing).

    Returns the number of rows written (always 1).
    """
    payload = _build_payload(spans)
    max_tier = max((s.confidence_tier for s in spans), default=0) or None
    with conn.cursor() as cur:
        cur.execute(
            _INSERT_SQL,
            (
                bibcode,
                EXTRACTION_TYPE,
                EXTRACTION_VERSION,
                Jsonb(payload),
                SOURCE,
                max_tier,
            ),
        )
    return 1
