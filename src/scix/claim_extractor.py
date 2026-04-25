"""Quantitative-claim extractor for paper bodies (M4).

Regex-first pipeline that captures claims of the form
``quantity = value ± uncertainty unit`` where ``±`` may be the unicode
character (``±``), ASCII ``+/-``, the LaTeX macro ``\\pm``, or the
asymmetric ``^{+a}_{-b}`` form.

Quantity recognition is driven by a curated dictionary of canonical
cosmology names (``H0``, ``Omega_m``, ``sigma_8``, ...). Surface
variants seen in the literature (``H_0``, ``Hubble constant``,
``\\Omega_m``, ``Ω_m``, ``\\sigma_8``, ...) all normalise back to the
canonical key.

A second-tier ``llm_disambiguate`` hook is defined for completeness but
intentionally NOT implemented — the project rule
``feedback_no_paid_apis`` (see CLAUDE.md) bars the use of paid APIs.

Public API:

    extract_claims(body: str) -> list[ClaimSpan]
    to_payload(claims: list[ClaimSpan]) -> dict
    llm_disambiguate(span: ClaimSpan) -> ClaimSpan  # NotImplementedError

Design notes:

* The extractor is deterministic and side-effect free; database writes
  live in ``scripts/run_claim_extractor.py``.
* Output is sorted by ``span.start`` for stable downstream iteration.
* Overlapping matches (e.g. ``H0`` matching inside ``H_0`` or longer
  surface forms) are deduplicated by preferring the longest surface that
  starts at the same position.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants — extraction_type / version stamps written to staging.extractions
# ---------------------------------------------------------------------------

EXTRACTION_TYPE: str = "quant_claim"
EXTRACTION_VERSION: str = "quant_claim_regex_v1"
EXTRACTION_SOURCE: str = "claim_extractor_regex"

# Confidence tiers (mirrors the M4 NER convention: 1=high, 2=medium, 3=low).
CONFIDENCE_HIGH: int = 1
CONFIDENCE_MED: int = 2
CONFIDENCE_LOW: int = 3


# ---------------------------------------------------------------------------
# Quantity dictionary — canonical name -> ordered list of surface variants.
# Variants are sorted longest-first per quantity so the alternation regex
# prefers the most specific surface (e.g. ``Hubble constant`` over ``H0``).
# ---------------------------------------------------------------------------

QUANTITY_DICT: dict[str, list[str]] = {
    "H0": [
        "Hubble parameter",
        "Hubble constant",
        "H_{0}",
        "H_0",
        "H0",
    ],
    "Omega_m": [
        "\\Omega_{m}",
        "\\Omega_m",
        "Omega_matter",
        "Omega_M",
        "Omega_m",
        "Ω_m",  # Ω_m
    ],
    "Omega_b": [
        "\\Omega_{b}",
        "\\Omega_b",
        "Omega_baryon",
        "Omega_b",
        "Ω_b",  # Ω_b
    ],
    "Omega_Lambda": [
        "\\Omega_{\\Lambda}",
        "\\Omega_\\Lambda",
        "Omega_Lambda",
        "Ω_Λ",  # Ω_Λ
    ],
    "sigma_8": [
        "\\sigma_{8}",
        "\\sigma_8",
        "sigma_8",
        "sigma8",
        "σ_8",  # σ_8
    ],
    "n_s": [
        "n_{s}",
        "\\n_s",
        "n_s",
    ],
    "w0": [
        "w_{0}",
        "w_0",
        "w0",
    ],
}


# ---------------------------------------------------------------------------
# ClaimSpan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimSpan:
    """A single quantitative claim extracted from text.

    Attributes:
        quantity:        Canonical name (e.g. ``H0``).
        value:           Best estimate parsed from the assignment.
        uncertainty:     Symmetric uncertainty if present, else None. For
                         asymmetric forms this is the average of pos/neg.
        unit:            Trailing unit string if matched, else None.
        span:            ``(start, end)`` character offsets in the source body.
        uncertainty_pos: For asymmetric ``^{+a}_{-b}`` forms.
        uncertainty_neg: For asymmetric ``^{+a}_{-b}`` forms.
        surface:         Raw substring matched (debug / provenance).
        confidence_tier: 1=high (regex hit with uncertainty + unit),
                         2=medium (regex hit, missing unit OR uncertainty),
                         3=low (regex hit but value-only, no uncertainty AND no unit).
    """

    quantity: str
    value: float
    uncertainty: float | None
    unit: str | None
    span: tuple[int, int]
    uncertainty_pos: float | None = None
    uncertainty_neg: float | None = None
    surface: str = ""
    confidence_tier: int = CONFIDENCE_MED


# ---------------------------------------------------------------------------
# Regex building blocks
# ---------------------------------------------------------------------------

# Numeric value: optional sign, digits, optional fraction, optional exponent.
_NUMBER_RE: str = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

# Unit: a permissive token allowing letters, digits, /, ^, -, \, {, }, _.
# Bounded to 1-30 chars to avoid runaway matches over body text.
_UNIT_RE: str = r"[A-Za-z][A-Za-z0-9/^\-\\{}_\.]{0,29}"

# The "plus or minus" symbol family.
_PM_ALT: str = r"(?:\\pm|±|\+/-|\+\\?-)"

# Compiled once for use in extraction.
_VALUE_PAT = re.compile(_NUMBER_RE)


def _surface_pattern_for(quantity: str) -> re.Pattern[str]:
    """Compile a regex matching any surface variant for a single quantity.

    Variants are escaped (the dictionary already contains literal LaTeX
    backslashes) and joined longest-first so the alternation prefers the
    most specific surface (``Hubble constant`` before ``H_0``).
    """
    variants = QUANTITY_DICT[quantity]
    sorted_variants = sorted(variants, key=len, reverse=True)
    escaped = [re.escape(v) for v in sorted_variants]
    return re.compile("(?:" + "|".join(escaped) + ")")


def _full_assignment_pattern_for(quantity: str) -> re.Pattern[str]:
    """Compile the per-quantity full assignment regex.

    The returned pattern matches three alternatives in priority order:

      A. asymmetric:   ``surf = value^{+a}_{-b} [unit]``
      B. symmetric:    ``surf = value <pm> uncertainty [unit]``
      C. value-only:   ``surf = value [unit]``

    Named groups:
        value       — the central value
        u_pos / u_neg — asymmetric uncertainties
        u_sym       — symmetric uncertainty
        unit_a / unit_b / unit_c — unit per branch
    """
    surf = _surface_pattern_for(quantity).pattern
    # Whitespace is permissive (any unicode space) — the body text often
    # contains LaTeX-induced double spaces, NBSPs, or newline wraps.
    ws = r"[\s ]*"

    asymmetric = (
        rf"{surf}{ws}={ws}(?P<value>{_NUMBER_RE})"
        rf"{ws}\^{{?{ws}\+{ws}(?P<u_pos>{_NUMBER_RE}){ws}}}?"
        rf"{ws}_{{?{ws}-{ws}(?P<u_neg>{_NUMBER_RE}){ws}}}?"
        rf"(?:{ws}(?P<unit_a>{_UNIT_RE}))?"
    )
    symmetric = (
        rf"{surf}{ws}={ws}(?P<value2>{_NUMBER_RE})"
        rf"{ws}{_PM_ALT}{ws}(?P<u_sym>{_NUMBER_RE})"
        rf"(?:{ws}(?P<unit_b>{_UNIT_RE}))?"
    )
    value_only = (
        rf"{surf}{ws}={ws}(?P<value3>{_NUMBER_RE})"
        rf"(?:{ws}(?P<unit_c>{_UNIT_RE}))?"
    )

    full = rf"(?:{asymmetric})|(?:{symmetric})|(?:{value_only})"
    return re.compile(full)


# Cache compiled patterns at import time — the dict is small and static.
_PATTERNS: dict[str, re.Pattern[str]] = {
    q: _full_assignment_pattern_for(q) for q in QUANTITY_DICT
}


# ---------------------------------------------------------------------------
# Unit cleanup
# ---------------------------------------------------------------------------

# A short blocklist of "unit" tails that are obviously sentence continuations,
# not units. Prevents false-unit attachment like
# ``H0 = 73 and Omega_m = 0.3`` -> unit="and".
_UNIT_STOPWORDS: frozenset[str] = frozenset(
    {
        "and",
        "or",
        "with",
        "is",
        "are",
        "while",
        "but",
        "in",
        "for",
        "to",
        "the",
        "a",
        "an",
        "we",
        "this",
        "that",
        "from",
        "at",
        "on",
        "by",
        "as",
        "was",
        "were",
    }
)


def _clean_unit(raw: str | None) -> str | None:
    """Trim and reject obviously-not-a-unit tails."""
    if raw is None:
        return None
    cleaned = raw.strip().rstrip(".,;:")
    if not cleaned:
        return None
    if cleaned.lower() in _UNIT_STOPWORDS:
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_claims(body: str) -> list[ClaimSpan]:
    """Extract all quantitative claims from ``body``.

    Returns a list of :class:`ClaimSpan` sorted by ``span.start``. When
    multiple surface variants overlap at the same start position (e.g.
    ``H_0`` and ``H0``), the longest surface match wins.

    The function is deterministic and pure — it performs no IO. Empty or
    falsy input returns an empty list.
    """
    if not body:
        return []

    candidates: list[ClaimSpan] = []
    for quantity, pattern in _PATTERNS.items():
        for m in pattern.finditer(body):
            value_str = m.group("value") or m.group("value2") or m.group("value3")
            if value_str is None:
                continue
            value = float(value_str)

            u_pos_str = m.group("u_pos")
            u_neg_str = m.group("u_neg")
            u_sym_str = m.group("u_sym")
            unit_raw = m.group("unit_a") or m.group("unit_b") or m.group("unit_c")
            unit = _clean_unit(unit_raw)

            uncertainty: float | None
            uncertainty_pos: float | None = None
            uncertainty_neg: float | None = None

            if u_pos_str is not None and u_neg_str is not None:
                uncertainty_pos = float(u_pos_str)
                uncertainty_neg = float(u_neg_str)
                # Use the average magnitude as a single-number summary.
                uncertainty = (uncertainty_pos + uncertainty_neg) / 2.0
            elif u_sym_str is not None:
                uncertainty = float(u_sym_str)
            else:
                uncertainty = None

            if uncertainty is not None and unit is not None:
                tier = CONFIDENCE_HIGH
            elif uncertainty is not None or unit is not None:
                tier = CONFIDENCE_MED
            else:
                tier = CONFIDENCE_LOW

            candidates.append(
                ClaimSpan(
                    quantity=quantity,
                    value=value,
                    uncertainty=uncertainty,
                    unit=unit,
                    span=(m.start(), m.end()),
                    uncertainty_pos=uncertainty_pos,
                    uncertainty_neg=uncertainty_neg,
                    surface=body[m.start() : m.end()],
                    confidence_tier=tier,
                )
            )

    return _dedupe_overlaps(candidates)


def _dedupe_overlaps(spans: Iterable[ClaimSpan]) -> list[ClaimSpan]:
    """Deduplicate overlapping matches.

    Strategy:
      1. Sort by (start asc, length desc) so the longest match at each
         start position is encountered first.
      2. Walk the list, dropping any candidate whose span is fully
         contained in the previous accepted span.
    """
    items = sorted(spans, key=lambda s: (s.span[0], -(s.span[1] - s.span[0])))
    accepted: list[ClaimSpan] = []
    for s in items:
        if accepted:
            prev = accepted[-1]
            # Fully contained inside the previous accepted span -> drop.
            if s.span[0] >= prev.span[0] and s.span[1] <= prev.span[1]:
                continue
        accepted.append(s)
    # Stable sort by start for the public-facing return.
    return sorted(accepted, key=lambda s: s.span[0])


# ---------------------------------------------------------------------------
# Payload builder — feeds staging.extractions.payload (JSONB)
# ---------------------------------------------------------------------------


def to_payload(claims: list[ClaimSpan]) -> dict:
    """Build the JSONB payload written to ``staging.extractions``.

    The unique constraint on ``staging.extractions(bibcode,
    extraction_type, extraction_version)`` allows only one row per
    (paper, quant_claim, version), so we aggregate all per-paper claims
    under a top-level ``claims`` array. Each claim contains the
    PRD-required fields ``{quantity, value, uncertainty, unit, span}``
    plus asymmetric components when present.
    """
    return {
        "extraction_type": EXTRACTION_TYPE,
        "extraction_version": EXTRACTION_VERSION,
        "source": EXTRACTION_SOURCE,
        "claims": [
            {
                "quantity": c.quantity,
                "value": c.value,
                "uncertainty": c.uncertainty,
                "unit": c.unit,
                "span": list(c.span),
                "uncertainty_pos": c.uncertainty_pos,
                "uncertainty_neg": c.uncertainty_neg,
                "surface": c.surface,
                "confidence_tier": c.confidence_tier,
            }
            for c in claims
        ],
    }


# ---------------------------------------------------------------------------
# LLM hook — intentionally NOT implemented (see CLAUDE.md)
# ---------------------------------------------------------------------------


def llm_disambiguate(span: ClaimSpan) -> ClaimSpan:
    """LLM-tier disambiguation pass for low-confidence spans.

    Defined as a stable hook so the regex tier can call into a future
    cheaper-than-Claude model without a downstream API change. The
    implementation is intentionally absent because the project rule
    ``feedback_no_paid_apis`` (see CLAUDE.md) bans paid APIs.

    Raises:
        NotImplementedError: always.
    """
    del span  # explicit "we read this, did nothing with it" marker
    raise NotImplementedError(
        "Requires paid API; see CLAUDE.md feedback_no_paid_apis"
    )


# ---------------------------------------------------------------------------
# Convenience: dataclass -> dict (used by callers serialising to JSON)
# ---------------------------------------------------------------------------


def claim_to_dict(claim: ClaimSpan) -> dict:
    """Return a plain-dict representation of a :class:`ClaimSpan`."""
    d = asdict(claim)
    # asdict() turns the tuple into a tuple; coerce to list for JSON.
    d["span"] = list(d["span"])
    return d


# Silence unused-import warnings if downstream tools strip optional ones.
_ = field
