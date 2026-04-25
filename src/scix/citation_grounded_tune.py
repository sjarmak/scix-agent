"""Shadow-mode threshold tuner for the citation-grounded gate (PRD MH-6).

The PRD calls for a 48-hour shadow-mode run on a 50-paraphrase gold set
to tune the cosine threshold so that the false-positive rate (legitimate
paraphrases incorrectly flagged as ungrounded) stays at or below 15%.
This module implements the offline equivalent: given a list of
``(legitimate_paraphrase, source_quote)`` pairs, return the largest
threshold whose FP rate on those pairs is ≤ ``target_fp_rate``.

The tuner uses the same scoring path as the production gate
(:func:`scix.citation_grounded._score_assertion`) — substring
short-circuit followed by INDUS cosine — so its output is directly
applicable as the production threshold.

Usage::

    from scix.citation_grounded_tune import shadow_tune

    pairs = [
        ("The team measured H0 = 73.0.", "We measure H0 = 73.0 km/s/Mpc."),
        # ... 49 more
    ]
    threshold = shadow_tune(pairs, target_fp_rate=0.15)
    # Plug ``threshold`` into the production grounded_check call.

The tuner exists for the offline tuning workflow; in production a 48-hour
shadow window logs gate decisions without enforcing them, then this
function is run on the captured pairs to set the enforcement threshold.
"""

from __future__ import annotations

import logging

from scix.citation_grounded import (
    DEFAULT_THRESHOLD,
    _get_embedder,
    _score_assertion,
)

logger = logging.getLogger(__name__)

# Threshold sweep range. We probe from 0.99 down to 0.50 in 0.005 steps —
# fine-grained enough to land on a near-optimal threshold without making
# the tuner expensive on a 50-pair gold set.
_PROBE_HIGH: float = 0.99
_PROBE_LOW: float = 0.50
_PROBE_STEP: float = 0.005


def shadow_tune(
    gold_paraphrases: list[tuple[str, str]],
    target_fp_rate: float = 0.15,
) -> float:
    """Pick the largest threshold whose FP rate on legit paraphrases is ≤ target.

    Args:
        gold_paraphrases: List of ``(paraphrase, source_quote)`` pairs.
            Both elements are expected to be legitimate matches: the
            paraphrase restates content present in the source quote.
            FP rate is the fraction of pairs whose grounding score is
            **below** a candidate threshold (would be wrongly flagged).
        target_fp_rate: Maximum acceptable FP rate. Default 0.15 (PRD MH-6).

    Returns:
        The largest threshold in ``[0.50, 0.99]`` (in 0.005 steps) whose
        FP rate is ≤ ``target_fp_rate``. If no threshold satisfies the
        target (e.g., the embedder returns very low scores for these
        pairs), returns :data:`_PROBE_LOW` and logs a warning.

    Raises:
        ValueError: If ``gold_paraphrases`` is empty or
            ``target_fp_rate`` is outside ``[0.0, 1.0]``.
    """
    if not gold_paraphrases:
        raise ValueError("gold_paraphrases must be non-empty")
    if not 0.0 <= target_fp_rate <= 1.0:
        raise ValueError(f"target_fp_rate must be in [0,1], got {target_fp_rate}")

    embedder = _get_embedder()

    # Score every pair once. Each pair is treated as one "assertion"
    # (the paraphrase) with one "tool quote" (the source quote).
    scores: list[float] = []
    for paraphrase, source_quote in gold_paraphrases:
        score = _score_assertion(paraphrase, [source_quote], embedder)
        scores.append(score)

    n = len(scores)
    # Sweep candidate thresholds from high to low. The first one whose FP
    # rate is acceptable is our pick (since we want the largest threshold
    # that satisfies the target).
    candidate = _PROBE_HIGH
    while candidate >= _PROBE_LOW:
        fp_count = sum(1 for s in scores if s < candidate)
        fp_rate = fp_count / n
        if fp_rate <= target_fp_rate:
            logger.info(
                "shadow_tune: threshold=%.3f fp_rate=%.3f (n=%d, target=%.3f)",
                candidate,
                fp_rate,
                n,
                target_fp_rate,
            )
            return round(candidate, 3)
        candidate -= _PROBE_STEP

    logger.warning(
        "shadow_tune: no threshold in [%.2f, %.2f] satisfies fp_rate<=%.2f; "
        "returning floor %.2f. Inspect the gold set or the embedder.",
        _PROBE_LOW,
        _PROBE_HIGH,
        target_fp_rate,
        _PROBE_LOW,
    )
    return _PROBE_LOW


__all__ = ["shadow_tune", "DEFAULT_THRESHOLD"]
