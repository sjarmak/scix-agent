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

import dataclasses
import hashlib
import json
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
CHUNKS_COLLECTION = "scix_chunks_v1"


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


# ---------------------------------------------------------------------------
# Chunk-level helpers (scix_chunks_v1) — used by the chunk_search MCP tool.
# Kept distinct from the paper-level helpers above because the payload schema
# and filter keys differ (community_id_med list, section_heading_norm list,
# bibcode list).  The chunks collection is single-vector, so query_points calls
# do NOT pass `using=VECTOR_NAME`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkHit:
    """A single hit from the scix_chunks_v1 collection.

    ``snippet`` is left as ``None`` by :func:`chunk_search_by_text`; populate
    it via :func:`fetch_chunk_snippets`, which joins on
    ``papers_fulltext.sections``.

    ``section_heading`` may also be ``None`` because the chunks-collection
    payload schema (per the chunk-embeddings PRD) only indexes
    ``section_heading_norm``; clients can derive a display heading from the
    norm if needed, or fetch the raw heading via the same
    ``papers_fulltext.sections`` join used for snippets.
    """

    bibcode: str
    chunk_id: int
    section_idx: int
    section_heading_norm: str | None
    section_heading: str | None
    score: float
    snippet: str | None
    char_offset: int | None
    n_tokens: int | None


def _chunks_filter_from_kwargs(
    year_min: int | None = None,
    year_max: int | None = None,
    arxiv_class: list[str] | None = None,
    community_id_med: list[int] | None = None,
    section_heading_norm: list[str] | None = None,
    bibcode: list[str] | None = None,
):
    """Compose a qdrant Filter for the chunks collection.

    All filters are AND-combined under ``must=[...]``.  List filters use
    ``MatchAny``; the year window composes into a single ``Range`` condition.
    Returns ``None`` (not an empty Filter) when no filters are supplied so the
    caller can pass it straight through to ``query_points``.
    """
    must: list[Any] = []
    if year_min is not None or year_max is not None:
        must.append(qm.FieldCondition(
            key="year",
            range=qm.Range(
                gte=year_min if year_min is not None else None,
                lte=year_max if year_max is not None else None,
            ),
        ))
    if arxiv_class:
        must.append(qm.FieldCondition(
            key="arxiv_class",
            match=qm.MatchAny(any=list(arxiv_class)),
        ))
    if community_id_med:
        must.append(qm.FieldCondition(
            key="community_id_med",
            match=qm.MatchAny(any=[int(c) for c in community_id_med]),
        ))
    if section_heading_norm:
        must.append(qm.FieldCondition(
            key="section_heading_norm",
            match=qm.MatchAny(any=list(section_heading_norm)),
        ))
    if bibcode:
        must.append(qm.FieldCondition(
            key="bibcode",
            match=qm.MatchAny(any=list(bibcode)),
        ))
    return qm.Filter(must=must) if must else None


def _chunk_row(point) -> ChunkHit:
    """Build a ChunkHit from a qdrant ScoredPoint-like object.

    ``snippet`` is always ``None`` here — it is populated separately by
    :func:`fetch_chunk_snippets` so the network round-trip to Postgres is
    decoupled from the ANN call.
    """
    p = point.payload or {}
    chunk_id_raw = p.get("chunk_id")
    if chunk_id_raw is None:
        chunk_id_raw = getattr(point, "id", 0)
    section_idx_raw = p.get("section_idx", 0)
    char_offset_raw = p.get("char_offset")
    n_tokens_raw = p.get("n_tokens")
    return ChunkHit(
        bibcode=p.get("bibcode", ""),
        chunk_id=int(chunk_id_raw) if chunk_id_raw is not None else 0,
        section_idx=int(section_idx_raw) if section_idx_raw is not None else 0,
        section_heading_norm=p.get("section_heading_norm"),
        section_heading=p.get("section_heading"),
        score=float(point.score) if getattr(point, "score", None) is not None else 0.0,
        snippet=None,
        char_offset=int(char_offset_raw) if char_offset_raw is not None else None,
        n_tokens=int(n_tokens_raw) if n_tokens_raw is not None else None,
    )


def chunk_search_by_text(
    vector: list[float],
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    arxiv_class: list[str] | None = None,
    community_id_med: list[int] | None = None,
    section_heading_norm: list[str] | None = None,
    bibcode: list[str] | None = None,
    limit: int = 20,
    timeout: float = 10.0,
) -> list[ChunkHit]:
    """Run an ANN query against ``scix_chunks_v1`` with payload filters.

    Returns hits with ``snippet=None``; call :func:`fetch_chunk_snippets` to
    populate snippet text from ``papers_fulltext.sections``.
    """
    client = _client(timeout=timeout)
    flt = _chunks_filter_from_kwargs(
        year_min=year_min,
        year_max=year_max,
        arxiv_class=arxiv_class,
        community_id_med=community_id_med,
        section_heading_norm=section_heading_norm,
        bibcode=bibcode,
    )
    resp = client.query_points(
        collection_name=CHUNKS_COLLECTION,
        query=vector,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    return [_chunk_row(h) for h in resp.points]


def fetch_chunk_snippets(
    conn,
    hits: list[ChunkHit],
    *,
    max_snippet_chars: int = 1500,
) -> list[ChunkHit]:
    """Return a NEW list of ChunkHit with ``snippet`` populated from Postgres.

    Issues a single batched ``SELECT bibcode, sections FROM papers_fulltext
    WHERE bibcode = ANY(%s)`` and indexes into each chunk's ``section_idx`` to
    pull ``sections[section_idx]['text']``.  Truncates each snippet to
    ``max_snippet_chars`` and appends ``"..."`` when truncated.

    Hits whose bibcode is missing from ``papers_fulltext`` or whose
    ``section_idx`` is out of range keep ``snippet=None`` and are preserved in
    the output (never dropped).  ChunkHit is frozen, so this returns brand-new
    instances via :func:`dataclasses.replace`.
    """
    if not hits:
        return []
    bibcodes = sorted({h.bibcode for h in hits if h.bibcode})
    sections_by_bibcode: dict[str, list[Any]] = {}
    if bibcodes:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT bibcode, sections FROM papers_fulltext WHERE bibcode = ANY(%s)",
                (bibcodes,),
            )
            for bibcode, sections in cur.fetchall():
                if isinstance(sections, (str, bytes)):
                    try:
                        sections = json.loads(sections)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        sections = []
                if isinstance(sections, list):
                    sections_by_bibcode[bibcode] = sections

    out: list[ChunkHit] = []
    for hit in hits:
        sections = sections_by_bibcode.get(hit.bibcode)
        snippet: str | None = None
        if sections is not None and 0 <= hit.section_idx < len(sections):
            section = sections[hit.section_idx]
            text: str | None = None
            if isinstance(section, dict):
                raw = section.get("text")
                if isinstance(raw, str):
                    text = raw
            if text is not None:
                if len(text) > max_snippet_chars:
                    snippet = text[:max_snippet_chars] + "..."
                else:
                    snippet = text
        out.append(dataclasses.replace(hit, snippet=snippet))
    return out
