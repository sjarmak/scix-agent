"""Qdrant-backed vector-search helpers.

Feature-flagged: if ``QDRANT_URL`` is not set, none of these helpers should be
invoked and the corresponding MCP tools should not be registered. Postgres +
pgvector remains the source of truth for the full 32M-paper corpus; Qdrant
holds a pilot subset (top-N by PageRank) loaded by
``scripts/qdrant_upsert_pilot.py``.

This module exposes a single capability that pgvector can't cleanly replicate
in SQL: the discovery / recommendation API — "more like these, less like
those" — with optional payload filtering.
"""
from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from typing import Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
except ImportError:  # pragma: no cover
    QdrantClient = None  # type: ignore[assignment]
    qm = None  # type: ignore[assignment]


COLLECTION = "scix_papers_v1"
VECTOR_NAME = "indus"


def is_enabled() -> bool:
    return bool(os.environ.get("QDRANT_URL")) and QdrantClient is not None


def _client(timeout: float = 10.0) -> QdrantClient:
    if not is_enabled():
        raise RuntimeError(
            "Qdrant not configured — set QDRANT_URL and install qdrant-client"
        )
    return QdrantClient(url=os.environ["QDRANT_URL"], timeout=timeout)


def bibcode_to_point_id(bibcode: str) -> int:
    h = hashlib.blake2b(bibcode.encode("utf-8"), digest_size=8).digest()
    return struct.unpack(">Q", h)[0] >> 1


def _filter_from_kwargs(
    year_min: int | None = None,
    year_max: int | None = None,
    doctype: list[str] | None = None,
    community_semantic: int | None = None,
    arxiv_class: list[str] | None = None,
):
    must: list[Any] = []
    if year_min is not None or year_max is not None:
        must.append(qm.FieldCondition(
            key="year",
            range=qm.Range(
                gte=year_min if year_min is not None else None,
                lte=year_max if year_max is not None else None,
            ),
        ))
    if doctype:
        must.append(qm.FieldCondition(
            key="doctype",
            match=qm.MatchAny(any=list(doctype)),
        ))
    if community_semantic is not None:
        must.append(qm.FieldCondition(
            key="community_semantic_coarse",
            match=qm.MatchValue(value=int(community_semantic)),
        ))
    if arxiv_class:
        must.append(qm.FieldCondition(
            key="arxiv_class",
            match=qm.MatchAny(any=list(arxiv_class)),
        ))
    return qm.Filter(must=must) if must else None


@dataclass
class SimilarPaper:
    bibcode: str
    title: str | None
    year: int | None
    first_author: str | None
    score: float
    arxiv_class: list[str]
    community_semantic: int | None
    doctype: str | None


def _row(point) -> SimilarPaper:
    p = point.payload or {}
    return SimilarPaper(
        bibcode=p.get("bibcode", ""),
        title=p.get("title"),
        year=p.get("year"),
        first_author=p.get("first_author"),
        score=float(point.score) if getattr(point, "score", None) is not None else 0.0,
        arxiv_class=list(p.get("arxiv_class") or []),
        community_semantic=p.get("community_semantic_coarse"),
        doctype=p.get("doctype"),
    )


def find_similar_by_examples(
    positive_bibcodes: list[str],
    negative_bibcodes: list[str] | None = None,
    *,
    limit: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
    doctype: list[str] | None = None,
    community_semantic: int | None = None,
    arxiv_class: list[str] | None = None,
    timeout: float = 10.0,
) -> list[SimilarPaper]:
    """Return papers most like ``positive_bibcodes`` and least like ``negative_bibcodes``.

    This is Qdrant's discovery / recommendation API. pgvector can approximate
    this by averaging vectors and subtracting negative-example vectors in SQL,
    but loses the per-example weighting and requires careful hand-rolling of
    every query. Qdrant exposes it as a first-class primitive.

    Args:
        positive_bibcodes: Papers the result should resemble (at least 1).
        negative_bibcodes: Papers the result should avoid (optional).
        limit: Max results.
        year_min/year_max: Filter on paper year.
        doctype: Filter on doctype (e.g. ["article", "review"]).
        community_semantic: Restrict to a single coarse Leiden community.
        arxiv_class: Filter on arXiv class (e.g. ["astro-ph.EP"]).
        timeout: Qdrant request timeout.

    Returns:
        Ranked list of SimilarPaper, most similar first.
    """
    if not positive_bibcodes:
        raise ValueError("positive_bibcodes must be non-empty")
    client = _client(timeout=timeout)

    pos_ids = [bibcode_to_point_id(b) for b in positive_bibcodes]
    neg_ids = [bibcode_to_point_id(b) for b in (negative_bibcodes or [])]

    flt = _filter_from_kwargs(
        year_min=year_min,
        year_max=year_max,
        doctype=doctype,
        community_semantic=community_semantic,
        arxiv_class=arxiv_class,
    )

    # Exclude the positive / negative examples themselves from results.
    exclude_ids = pos_ids + neg_ids
    exclude_filter = qm.Filter(must_not=[
        qm.HasIdCondition(has_id=exclude_ids),
    ])
    combined: qm.Filter
    if flt is None:
        combined = exclude_filter
    else:
        combined = qm.Filter(
            must=flt.must or [],
            must_not=(flt.must_not or []) + (exclude_filter.must_not or []),
            should=flt.should or [],
        )

    # qdrant-client 1.17+: recommendation goes through query_points with a
    # RecommendQuery wrapper. Older .recommend() was removed.
    resp = client.query_points(
        collection_name=COLLECTION,
        query=qm.RecommendQuery(recommend=qm.RecommendInput(
            positive=pos_ids,
            negative=neg_ids or None,
            strategy=qm.RecommendStrategy.AVERAGE_VECTOR,
        )),
        using=VECTOR_NAME,
        limit=limit,
        query_filter=combined,
        with_payload=True,
    )
    return [_row(p) for p in resp.points]


def search_by_text_vector(
    vector: list[float],
    *,
    limit: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
    doctype: list[str] | None = None,
    community_semantic: int | None = None,
    arxiv_class: list[str] | None = None,
    timeout: float = 10.0,
) -> list[SimilarPaper]:
    """Nearest-neighbor search by raw INDUS vector, with payload filters.

    The main win over pgvector here is payload-indexed filtering: Qdrant keeps
    full HNSW speed under restrictive filters, while pgvector's iterative scan
    degrades.
    """
    client = _client(timeout=timeout)
    flt = _filter_from_kwargs(
        year_min=year_min,
        year_max=year_max,
        doctype=doctype,
        community_semantic=community_semantic,
        arxiv_class=arxiv_class,
    )
    resp = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        using=VECTOR_NAME,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    return [_row(h) for h in resp.points]


def collection_info() -> dict[str, Any]:
    """Return basic collection status for health checks."""
    client = _client(timeout=3.0)
    try:
        info = client.get_collection(COLLECTION)
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "error": str(e)}
    return {
        "status": str(info.status),
        "points": info.points_count,
        "segments": getattr(info, "segments_count", None),
        "collection": COLLECTION,
        "vector_name": VECTOR_NAME,
    }
