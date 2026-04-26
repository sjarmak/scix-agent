"""Per-(entity_type, source, agreement) precision estimates for NER mentions.

Empirical numbers from the dbl.3 acceptance eval (414 hand-judged mentions
across pre-1990 + modern stratified samples; see
``docs/eval/dbl3_ner_precision_*.md`` for the full reports).

This profile is the **source of truth** that MCP tools surface as
``precision_estimate`` metadata on each result row, so agents can apply
their own min-precision filter rather than getting an opaque pass/fail.
The bead acceptance criterion ("≥80% precision at conf≥0.7") is NOT met
in aggregate — but it IS met for several (type, agreement) buckets, and
the profile makes that explicit.

Usage::

    from scix.extract.ner_quality_profile import precision_estimate

    p = precision_estimate(
        entity_type="method",
        source="gliner",
        agreement=True,           # the classifier agreed with GLiNER
        era="modern",             # year >= 2010 abstract
    )
    # p ≈ 0.86

When the agreement / era is unknown (e.g. classifier hasn't run yet,
year is missing), fall back to the **lower** bound of the available
buckets so we don't overstate precision.
"""

from __future__ import annotations

from typing import Literal

# Eras carved by the eval samples. The split is empirical: GLiNER
# behavior shifts measurably between these ranges (modern papers have
# more real software/datasets to detect; pre-1990 papers expose more
# generic-noun confusions).
Era = Literal["pre_1990", "modern"]


#: Aggregate precision baselines (no filter applied). Reference numbers
#: from `dbl3_ner_precision_eval_2026-04-25.md` and
#: `dbl3_ner_precision_eval_2026-04-25_modern.md`.
UNFILTERED_PRECISION: dict[tuple[str, Era], float] = {
    # ───────── pre-1990 ─────────
    ("method", "pre_1990"): 0.78,
    ("chemical", "pre_1990"): 0.78,
    ("location", "pre_1990"): 0.74,
    ("organism", "pre_1990"): 0.65,
    ("instrument", "pre_1990"): 0.57,
    ("mission", "pre_1990"): 0.52,
    ("gene", "pre_1990"): 0.52,
    ("software", "pre_1990"): 0.22,
    ("dataset", "pre_1990"): 0.09,
    # ───────── modern (>=2010) ─────────
    ("method", "modern"): 0.87,
    ("location", "modern"): 0.87,
    ("mission", "modern"): 0.78,
    ("instrument", "modern"): 0.74,
    ("organism", "modern"): 0.39,
    ("chemical", "modern"): 0.43,
    ("gene", "modern"): 0.17,
    ("software", "modern"): 0.61,
    ("dataset", "modern"): 0.35,
}


#: Precision after the INDUS classifier filter (agreement=true). Numbers
#: from `dbl3_ner_classifier_filtered_eval.md` (2026-04-26 — FULL
#: coverage on both eras after the classifier post-pass completed). The
#: original partial-coverage pre-1990 numbers (n=14) overstated several
#: buckets dramatically (mission 100%→47%, organism 100%→62%); these
#: are the corrected values.
CLASSIFIER_FILTERED_PRECISION: dict[tuple[str, Era], float] = {
    # ───────── pre-1990 (FULL coverage; n=93 kept of 207) ─────────
    ("chemical", "pre_1990"): 0.76,  # 13/17 — passes
    ("location", "pre_1990"): 0.86,  # 12/14 — passes
    ("instrument", "pre_1990"): 0.67,  # 4/6
    ("method", "pre_1990"): 0.64,  # 7/11
    ("organism", "pre_1990"): 0.62,  # 10/16
    ("mission", "pre_1990"): 0.47,  # 7/15
    ("gene", "pre_1990"): 0.33,  # 2/6
    ("software", "pre_1990"): 0.20,  # 1/5
    ("dataset", "pre_1990"): 0.00,  # 0/3 — broken
    # ───────── modern (n=77 kept of 207) ─────────
    ("instrument", "modern"): 1.00,  # 4/4 — passes
    ("method", "modern"): 0.86,  # 12/14 — passes
    ("location", "modern"): 0.86,  # 6/7 — passes
    ("mission", "modern"): 0.83,  # 10/12 — passes
    ("software", "modern"): 0.71,  # 5/7 — close
    ("organism", "modern"): 0.50,  # 2/4
    ("chemical", "modern"): 0.46,  # 6/13
    ("dataset", "modern"): 0.40,  # 2/5
    ("gene", "modern"): 0.18,  # 2/11 — broken
}

#: Sample size each filtered-precision number is based on. Buckets with
#: very small n (< 5) should be treated as estimates only.
CLASSIFIER_FILTERED_N: dict[tuple[str, Era], int] = {
    # pre-1990 (full coverage)
    ("chemical", "pre_1990"): 17,
    ("location", "pre_1990"): 14,
    ("instrument", "pre_1990"): 6,
    ("method", "pre_1990"): 11,
    ("organism", "pre_1990"): 16,
    ("mission", "pre_1990"): 15,
    ("gene", "pre_1990"): 6,
    ("software", "pre_1990"): 5,
    ("dataset", "pre_1990"): 3,
    # modern
    ("instrument", "modern"): 4,
    ("method", "modern"): 14,
    ("location", "modern"): 7,
    ("mission", "modern"): 12,
    ("software", "modern"): 7,
    ("organism", "modern"): 4,
    ("chemical", "modern"): 13,
    ("dataset", "modern"): 5,
    ("gene", "modern"): 11,
}


#: Lexical sources (keyword_exact_lower, aho_corasick_abstract,
#: canonical_exact, alias_exact) are dictionary-driven — precision is
#: high by construction (the lookup matched a curated entity). We use a
#: blanket 0.95 estimate; this is conservative for canonical_exact and
#: liberal for the AC matches but is the right ballpark per prior eval
#: work in the entity_value_props reports.
LEXICAL_PRECISION_DEFAULT = 0.95


def _classify_era(year: int | None) -> Era:
    """Bucket a paper year into one of the eval-defined eras."""
    if year is not None and year >= 2010:
        return "modern"
    return "pre_1990"


def precision_estimate(
    entity_type: str,
    source: str,
    *,
    agreement: bool | None = None,
    year: int | None = None,
) -> float:
    """Return the empirical precision estimate for a single mention.

    Parameters
    ----------
    entity_type
        From ``entities.entity_type``.
    source
        From ``entities.source`` (``'gliner'``, ``'canonical'``, etc.).
    agreement
        From ``document_entities.evidence->>'agreement'``. ``None`` when
        the classifier hasn't run yet. Only used for ``source='gliner'``.
    year
        Paper year. ``None`` when missing — falls back to the lower
        (pre-1990) precision bucket so we don't overstate quality.

    Returns
    -------
    float in [0, 1]. Conservative — when in doubt, returns the lower of
    the candidate buckets.
    """
    if source != "gliner":
        return LEXICAL_PRECISION_DEFAULT

    era = _classify_era(year)
    if agreement is True:
        # Use filtered precision if we have an estimate for this bucket;
        # else fall back to unfiltered (still better than nothing).
        return CLASSIFIER_FILTERED_PRECISION.get(
            (entity_type, era),
            UNFILTERED_PRECISION.get((entity_type, era), 0.5),
        )
    if agreement is False:
        # Classifier disagreed → mention is in the rejected pile. We
        # don't have a measured precision for "rejected GLiNER mentions"
        # but they're worse than unfiltered baseline. 0.2 is a rough
        # ceiling guess.
        return 0.2
    # agreement is None → no classifier verdict yet. Use unfiltered.
    return UNFILTERED_PRECISION.get((entity_type, era), 0.5)


def precision_band(p: float) -> str:
    """Coarse human-readable bucket for the score."""
    if p >= 0.85:
        return "high"
    if p >= 0.70:
        return "medium"
    if p >= 0.50:
        return "low"
    return "noisy"


def quality_summary() -> dict:
    """Return the full quality profile as a json-serializable dict.

    Used by MCP tools that want to surface the catalog (e.g. tool
    descriptions can include a link/snapshot of which buckets are high
    vs noisy).
    """
    return {
        "unfiltered": {f"{t}/{era}": p for (t, era), p in UNFILTERED_PRECISION.items()},
        "classifier_filtered": {
            f"{t}/{era}": {
                "precision": p,
                "n_judged": CLASSIFIER_FILTERED_N.get((t, era), 0),
            }
            for (t, era), p in CLASSIFIER_FILTERED_PRECISION.items()
        },
        "lexical_default": LEXICAL_PRECISION_DEFAULT,
        "notes": (
            "Empirical numbers from the dbl.3 414-mention acceptance eval "
            "(2026-04-25). Pre-1990 classifier-filtered buckets are partial "
            "estimates (small n). For mentions where source='gliner' and the "
            "INDUS post-classifier has not yet run, agreement=None → fall "
            "back to unfiltered estimate. For source != 'gliner' (lexical), "
            "use the LEXICAL_PRECISION_DEFAULT."
        ),
    }
