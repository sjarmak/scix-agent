"""Information-theoretic specificity scoring for entities.

Computes IDF-like scores: specificity = log(N / df) where N is the total
number of papers and df is the document frequency of the entity.  Entities
below a configurable threshold are flagged as generic ("filter"); the rest
are marked "keep".

Default threshold: entities appearing in >10% of papers
(specificity < log(10) ~ 2.3) are classified as "filter".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scix.normalize import normalize_entity

# Default: log(10) -- entities in >10% of docs are generic
DEFAULT_THRESHOLD: float = math.log(10)


@dataclass(frozen=True)
class ScoredEntity:
    """Immutable result of specificity scoring for a single entity.

    Attributes:
        entity: Normalized canonical form of the entity.
        score: IDF-like specificity score (higher = more specific).
        classification: 'keep' if score >= threshold, 'filter' otherwise.
    """

    entity: str
    score: float
    classification: str


def specificity_score(
    entity: str,
    *,
    df: int,
    N: int,
) -> float:
    """Compute the IDF-like specificity score for an entity.

    Args:
        entity: Raw entity string (used only for error messages; scoring
            depends on df and N).
        df: Document frequency — number of papers containing this entity.
        N: Total number of papers in the corpus.

    Returns:
        log(N / df) using natural logarithm.

    Raises:
        ValueError: If df <= 0 or N <= 0 or df > N.
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    if df > N:
        raise ValueError(f"df ({df}) cannot exceed N ({N})")
    return math.log(N / df)


def classify_entity(
    score: float,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """Classify an entity based on its specificity score.

    Args:
        score: IDF-like specificity score.
        threshold: Minimum score to be classified as 'keep'.

    Returns:
        'keep' if score >= threshold, 'filter' otherwise.
    """
    return "keep" if score >= threshold else "filter"


def score_entities(
    entity_freqs: Sequence[tuple[str, int]],
    *,
    N: int,
    threshold: float = DEFAULT_THRESHOLD,
    normalize: bool = True,
) -> list[ScoredEntity]:
    """Score and classify a batch of entities by specificity.

    Args:
        entity_freqs: Sequence of (entity_string, document_frequency) pairs.
        N: Total number of papers in the corpus.
        threshold: Minimum specificity score to classify as 'keep'.
        normalize: If True, apply normalize_entity() to each entity string
            before scoring.  Duplicate normalized forms are merged by summing
            their document frequencies.

    Returns:
        List of ScoredEntity sorted by score descending.
    """
    # Aggregate by normalized form if requested
    freq_map: dict[str, int] = {}
    for raw_entity, df in entity_freqs:
        canon = normalize_entity(raw_entity) if normalize else raw_entity
        freq_map[canon] = freq_map.get(canon, 0) + df

    results: list[ScoredEntity] = []
    for entity, df in freq_map.items():
        # Clamp df to N to avoid math domain errors from data quirks
        clamped_df = min(df, N)
        if clamped_df <= 0:
            continue
        score = math.log(N / clamped_df)
        classification = classify_entity(score, threshold=threshold)
        results.append(ScoredEntity(entity=entity, score=score, classification=classification))

    results.sort(key=lambda s: s.score, reverse=True)
    return results
