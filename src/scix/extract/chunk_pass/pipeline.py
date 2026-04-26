"""Top-level chunk-embedding ingest pipeline.

Streams ``papers_fulltext`` rows in bibcode-sorted batches, joins paper-level
metadata (``year``, ``arxiv_class``, ``community_id_med``, ``doctype``) from
``papers`` and ``paper_metrics``, chunks each paper via
:func:`scix.extract.chunk_pass.chunker.iter_chunks`, encodes the chunks with an
:class:`~scix.extract.chunk_pass.embedder.INDUSEmbedder`, assembles the per-point
payload, upserts to Qdrant, and checkpoints in ``ingest_log``.

Resumability mirrors :mod:`scix.extract.ner_pass`: every batch is keyed in
``ingest_log`` under ``chunk_pass:{first_bibcode}`` (where ``first_bibcode`` is
the first bibcode in the deterministic-ordered batch) and runs that crash or
are killed pick up at the next un-checkpointed batch on rerun.

Idempotency is structural: chunk point ids are derived deterministically from
``(bibcode, parser_version, chunk_id)`` (see
:func:`scix.extract.chunk_pass.collection.chunk_point_id`) so re-running with
the same input data overwrites the existing Qdrant points rather than
duplicating them.

Notes on metadata sources:
  * ``year``, ``arxiv_class``, ``doctype`` come from ``papers``.
  * ``community_id_med`` comes from ``paper_metrics.community_id_medium`` —
    the payload key abbreviates the column name. Joined LEFT OUTER so papers
    without a ``paper_metrics`` row simply get ``community_id_med = None``.
  * ``arxiv_class`` is ``TEXT[]`` in Postgres but the Qdrant collection's
    payload schema indexes it as ``KEYWORD`` (single string). The pipeline
    therefore promotes the first array element to a single string (or
    ``None`` for empty/null arrays), matching the convention used by
    :mod:`scix.extract.chunk_pass.uploader` tests and the existing
    ``scripts/qdrant_upsert_pilot.py`` upserter.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import psycopg

from .chunker import Chunk, iter_chunks
from .collection import CHUNKS_COLLECTION
from .section_norm import normalize_heading
from .uploader import (
    _checkpoint_key,
    assemble_payload,
    is_batch_done,
    record_checkpoint,
    upsert_chunks,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default batch size — number of papers fetched per ``iter_paper_batches``
#: iteration. 200 keeps each batch's GPU work under ~30s on a single 5090
#: with 512-token windows and ``inference_batch=64`` while still amortizing
#: the per-batch checkpoint write.
DEFAULT_BATCH_SIZE = 200

#: Default ``parser_version`` stamp — used both for the ``chunk_point_id``
#: derivation and as the source-version tag downstream code uses to tell
#: which parser produced a chunk. Bump when the chunker semantics change
#: in a way that would invalidate cached vectors.
DEFAULT_PARSER_VERSION = "ads_body_inline_v2"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperWithMeta:
    """One row from the joined ``papers_fulltext`` + ``papers`` + ``paper_metrics`` query.

    ``sections`` is the raw JSONB array (list of ``{heading, level, text, offset}``
    dicts) — passed straight through to :func:`iter_chunks`. The four metadata
    fields are nullable: a missing ``paper_metrics`` row produces
    ``community_id_med = None``; a NULL ``arxiv_class`` array produces
    ``arxiv_class = None``.
    """

    bibcode: str
    sections: list[dict]
    year: int | None
    arxiv_class: str | None
    community_id_med: int | None
    doctype: str | None


@dataclass(frozen=True)
class BatchStats:
    """Per-batch (and aggregated) chunk-pass counters.

    ``elapsed_*_s`` fields measure wall-clock seconds spent in each phase.
    Frozen so callers cannot mutate the returned counters in place — the
    aggregator in :func:`run` builds a fresh instance for the totals.
    """

    papers_seen: int = 0
    papers_with_chunks: int = 0
    chunks_emitted: int = 0
    chunks_uploaded: int = 0
    elapsed_chunk_s: float = 0.0
    elapsed_embed_s: float = 0.0
    elapsed_upload_s: float = 0.0


# ---------------------------------------------------------------------------
# Source-stream
# ---------------------------------------------------------------------------


_SOURCE_SQL = (
    "SELECT pf.bibcode, "
    "       pf.sections, "
    "       p.year, "
    "       p.arxiv_class, "
    "       m.community_id_medium AS community_id_med, "
    "       p.doctype "
    "FROM papers_fulltext pf "
    "JOIN papers p ON p.bibcode = pf.bibcode "
    "LEFT JOIN paper_metrics m ON m.bibcode = pf.bibcode "
    "WHERE pf.bibcode > %s "
    "ORDER BY pf.bibcode ASC "
    "LIMIT %s"
)


def _first_or_none(arr: Any) -> str | None:
    """Promote a Postgres ``TEXT[]`` to a single string (or None).

    Returns ``None`` for ``None`` or empty arrays; otherwise the first
    element coerced to ``str``. Matches the single-string ``KEYWORD``
    schema declared for ``arxiv_class`` in
    :mod:`scix.extract.chunk_pass.collection`.
    """
    if arr is None:
        return None
    if not arr:  # empty list/tuple
        return None
    return str(arr[0])


def _row_to_paper_with_meta(row: tuple[Any, ...]) -> PaperWithMeta:
    """Translate one source-query row tuple into a :class:`PaperWithMeta`."""
    bibcode, sections, year, arxiv_class, community_id_med, doctype = row
    return PaperWithMeta(
        bibcode=bibcode,
        sections=list(sections) if sections is not None else [],
        year=int(year) if year is not None else None,
        arxiv_class=_first_or_none(arxiv_class),
        community_id_med=int(community_id_med) if community_id_med is not None else None,
        doctype=str(doctype) if doctype is not None else None,
    )


def iter_paper_batches(
    conn: psycopg.Connection,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    since_bibcode: str | None = None,
    max_papers: int | None = None,
) -> Iterator[list[PaperWithMeta]]:
    """Stream papers in deterministic ``bibcode`` order via keyset pagination.

    ``since_bibcode`` sets a watermark for resumability — only papers strictly
    greater than that bibcode are yielded. ``max_papers`` caps total yield
    (sample / smoke runs).

    Keyset pagination (``WHERE bibcode > $watermark ORDER BY bibcode LIMIT N``)
    is used instead of a server-side named cursor because the caller commits
    between batches, which would invalidate a named cursor. Each iteration is a
    fresh single-shot query that survives commits and resumes correctly across
    process restarts.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    watermark = since_bibcode or ""
    remaining = max_papers
    while True:
        limit = batch_size if remaining is None else min(batch_size, remaining)
        if limit <= 0:
            return
        with conn.cursor() as cur:
            cur.execute(_SOURCE_SQL, (watermark, limit))
            rows = cur.fetchall()
        if not rows:
            return
        batch = [_row_to_paper_with_meta(r) for r in rows]
        yield batch
        watermark = batch[-1].bibcode
        if remaining is not None:
            remaining -= len(batch)
            if remaining <= 0:
                return


# ---------------------------------------------------------------------------
# Per-batch processing
# ---------------------------------------------------------------------------


def process_batch(
    conn: psycopg.Connection,  # noqa: ARG001 — kept for signature parity with ner_pass
    embedder: Any,
    qdrant_client: Any,
    batch: list[PaperWithMeta],
    *,
    parser_version: str = DEFAULT_PARSER_VERSION,
    dry_run: bool = False,
    collection_name: str = CHUNKS_COLLECTION,
) -> BatchStats:
    """Run chunking + embedding + upload for one batch of papers.

    Returns a :class:`BatchStats` describing what happened. ``conn`` is
    accepted (and mostly unused here) so the signature mirrors
    :func:`scix.extract.ner_pass.process_batch` — the surrounding
    transaction (``record_checkpoint`` + ``conn.commit``) is the caller's
    responsibility (see :func:`run`).

    When ``dry_run=True`` the embedder still encodes (so we can observe
    chunk volume in a smoke run) but the Qdrant upsert is skipped and
    ``chunks_uploaded`` stays zero.
    """
    papers_seen = len(batch)
    if papers_seen == 0:
        return BatchStats()

    # ---- Chunk ---------------------------------------------------------
    t0 = time.monotonic()
    tokenizer = embedder.tokenizer
    flat_chunks: list[Chunk] = []
    chunk_owner: list[PaperWithMeta] = []  # parallel to flat_chunks
    papers_with_chunks = 0
    for paper in batch:
        produced_any = False
        for chunk in iter_chunks(paper.bibcode, paper.sections, tokenizer):
            flat_chunks.append(chunk)
            chunk_owner.append(paper)
            produced_any = True
        if produced_any:
            papers_with_chunks += 1
    elapsed_chunk_s = time.monotonic() - t0

    if not flat_chunks:
        return BatchStats(
            papers_seen=papers_seen,
            papers_with_chunks=0,
            chunks_emitted=0,
            chunks_uploaded=0,
            elapsed_chunk_s=elapsed_chunk_s,
            elapsed_embed_s=0.0,
            elapsed_upload_s=0.0,
        )

    # ---- Embed ---------------------------------------------------------
    t0 = time.monotonic()
    vectors = embedder.encode_batch([c.text for c in flat_chunks])
    elapsed_embed_s = time.monotonic() - t0

    # ---- Assemble payloads --------------------------------------------
    payloads = [
        assemble_payload(
            chunk,
            year=owner.year,
            arxiv_class=owner.arxiv_class,
            community_id_med=owner.community_id_med,
            doctype=owner.doctype,
            section_heading_norm=normalize_heading(chunk.section_heading),
        )
        for chunk, owner in zip(flat_chunks, chunk_owner, strict=True)
    ]

    # ---- Upload --------------------------------------------------------
    chunks_uploaded = 0
    elapsed_upload_s = 0.0
    if not dry_run:
        t0 = time.monotonic()
        chunks_uploaded = upsert_chunks(
            qdrant_client,
            collection_name,
            flat_chunks,
            vectors,
            payloads,
            parser_version=parser_version,
        )
        elapsed_upload_s = time.monotonic() - t0

    return BatchStats(
        papers_seen=papers_seen,
        papers_with_chunks=papers_with_chunks,
        chunks_emitted=len(flat_chunks),
        chunks_uploaded=chunks_uploaded,
        elapsed_chunk_s=elapsed_chunk_s,
        elapsed_embed_s=elapsed_embed_s,
        elapsed_upload_s=elapsed_upload_s,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run(
    conn: psycopg.Connection,
    embedder: Any,
    qdrant_client: Any,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    since_bibcode: str | None = None,
    max_papers: int | None = None,
    parser_version: str = DEFAULT_PARSER_VERSION,
    dry_run: bool = False,
    log_every: int = 1,
    collection_name: str = CHUNKS_COLLECTION,
) -> BatchStats:
    """Stream batches, chunk + embed + upsert, checkpoint, return totals.

    Resumes by skipping any batch whose checkpoint key already shows
    ``status='complete'`` in ``ingest_log``. The first bibcode in each batch
    is the checkpoint key — deterministic because batches are pulled in
    bibcode order.

    With ``dry_run=True`` the pipeline still chunks + embeds (so we can
    observe volume) but skips both the Qdrant upsert and the
    ``ingest_log`` checkpoint, leaving the source store untouched.
    """
    papers_seen = 0
    papers_with_chunks = 0
    chunks_emitted = 0
    chunks_uploaded = 0
    elapsed_chunk_s = 0.0
    elapsed_embed_s = 0.0
    elapsed_upload_s = 0.0

    n_batches = 0

    for batch in iter_paper_batches(
        conn,
        batch_size=batch_size,
        since_bibcode=since_bibcode,
        max_papers=max_papers,
    ):
        first_bibcode = batch[0].bibcode
        key = _checkpoint_key(first_bibcode)

        if is_batch_done(conn, key):
            logger.info("chunk_pass: skip checkpointed batch %s", key)
            continue

        stats = process_batch(
            conn,
            embedder,
            qdrant_client,
            batch,
            parser_version=parser_version,
            dry_run=dry_run,
            collection_name=collection_name,
        )

        if not dry_run:
            record_checkpoint(
                conn,
                key,
                records_loaded=stats.chunks_uploaded,
            )
            conn.commit()

        # Aggregate totals
        papers_seen += stats.papers_seen
        papers_with_chunks += stats.papers_with_chunks
        chunks_emitted += stats.chunks_emitted
        chunks_uploaded += stats.chunks_uploaded
        elapsed_chunk_s += stats.elapsed_chunk_s
        elapsed_embed_s += stats.elapsed_embed_s
        elapsed_upload_s += stats.elapsed_upload_s

        n_batches += 1
        if n_batches % log_every == 0:
            logger.info(
                "chunk_pass: batch %d (%s..) papers=%d chunks=%d uploaded=%d "
                "elapsed_chunk_s=%.2f elapsed_embed_s=%.2f elapsed_upload_s=%.2f",
                n_batches,
                first_bibcode[:18],
                stats.papers_seen,
                stats.chunks_emitted,
                stats.chunks_uploaded,
                stats.elapsed_chunk_s,
                stats.elapsed_embed_s,
                stats.elapsed_upload_s,
            )

    return BatchStats(
        papers_seen=papers_seen,
        papers_with_chunks=papers_with_chunks,
        chunks_emitted=chunks_emitted,
        chunks_uploaded=chunks_uploaded,
        elapsed_chunk_s=elapsed_chunk_s,
        elapsed_embed_s=elapsed_embed_s,
        elapsed_upload_s=elapsed_upload_s,
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_PARSER_VERSION",
    "BatchStats",
    "PaperWithMeta",
    "iter_paper_batches",
    "process_batch",
    "run",
]
