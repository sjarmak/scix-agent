"""Standard information retrieval evaluation metrics.

Implements nDCG@K, Recall@K, Precision@K, and MRR for ranked retrieval
evaluation with graded or binary relevance judgments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalScore:
    """Aggregated scores for a single query across one retrieval system."""

    query_id: str
    system: str
    ndcg_at_10: float
    recall_at_10: float
    recall_at_20: float
    precision_at_10: float
    mrr: float
    latency_ms: float
    num_retrieved: int
    num_relevant: int


@dataclass(frozen=True)
class EvalReport:
    """Aggregated evaluation report across all queries for one system."""

    system: str
    num_queries: int
    mean_ndcg_at_10: float
    mean_recall_at_10: float
    mean_recall_at_20: float
    mean_precision_at_10: float
    mean_mrr: float
    mean_latency_ms: float
    per_query: tuple[RetrievalScore, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at rank k.

    DCG_k = sum_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)
    """
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += (2.0**rel - 1.0) / math.log2(i + 2)  # i+2 because i is 0-indexed
    return score


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance_map: dict[str, float],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain at rank k.

    Args:
        retrieved_ids: Ordered list of document IDs from the retrieval system.
        relevance_map: Mapping of document ID -> relevance grade (0 = not relevant).
        k: Cutoff rank.

    Returns:
        nDCG@k in [0, 1]. Returns 0.0 if no relevant documents exist.
    """
    if not relevance_map:
        return 0.0

    # Actual DCG from the retrieved ranking
    actual_rels = [relevance_map.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]]
    actual_dcg = dcg_at_k(actual_rels, k)

    # Ideal DCG: sort all relevance grades descending
    ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_rels, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Recall at rank k: fraction of relevant documents found in top-k.

    Returns 0.0 if no relevant documents exist.
    """
    if not relevant_ids:
        return 0.0
    found = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return found / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Precision at rank k: fraction of top-k that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    found = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return found / len(top_k)


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant document.

    Returns 0.0 if no relevant document is found.
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Score computation helper
# ---------------------------------------------------------------------------


def compute_retrieval_score(
    query_id: str,
    system: str,
    retrieved_ids: list[str],
    relevance_map: dict[str, float],
    latency_ms: float = 0.0,
) -> RetrievalScore:
    """Compute all metrics for a single query-system pair.

    Args:
        query_id: Identifier for this query.
        system: Name of the retrieval system (e.g., "specter2", "hybrid_rrf").
        retrieved_ids: Ordered list of retrieved document IDs.
        relevance_map: Doc ID -> relevance grade. Grade > 0 means relevant.
        latency_ms: Query latency in milliseconds.
    """
    relevant_ids = {k for k, v in relevance_map.items() if v > 0}

    return RetrievalScore(
        query_id=query_id,
        system=system,
        ndcg_at_10=round(ndcg_at_k(retrieved_ids, relevance_map, k=10), 4),
        recall_at_10=round(recall_at_k(retrieved_ids, relevant_ids, k=10), 4),
        recall_at_20=round(recall_at_k(retrieved_ids, relevant_ids, k=20), 4),
        precision_at_10=round(precision_at_k(retrieved_ids, relevant_ids, k=10), 4),
        mrr=round(mean_reciprocal_rank(retrieved_ids, relevant_ids), 4),
        latency_ms=round(latency_ms, 2),
        num_retrieved=len(retrieved_ids),
        num_relevant=len(relevant_ids),
    )


def aggregate_scores(
    system: str,
    scores: list[RetrievalScore],
) -> EvalReport:
    """Aggregate per-query scores into a system-level report."""
    n = len(scores)
    if n == 0:
        return EvalReport(
            system=system,
            num_queries=0,
            mean_ndcg_at_10=0.0,
            mean_recall_at_10=0.0,
            mean_recall_at_20=0.0,
            mean_precision_at_10=0.0,
            mean_mrr=0.0,
            mean_latency_ms=0.0,
        )

    return EvalReport(
        system=system,
        num_queries=n,
        mean_ndcg_at_10=round(sum(s.ndcg_at_10 for s in scores) / n, 4),
        mean_recall_at_10=round(sum(s.recall_at_10 for s in scores) / n, 4),
        mean_recall_at_20=round(sum(s.recall_at_20 for s in scores) / n, 4),
        mean_precision_at_10=round(sum(s.precision_at_10 for s in scores) / n, 4),
        mean_mrr=round(sum(s.mrr for s in scores) / n, 4),
        mean_latency_ms=round(sum(s.latency_ms for s in scores) / n, 2),
        per_query=tuple(scores),
    )
