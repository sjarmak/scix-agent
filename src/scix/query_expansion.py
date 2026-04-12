"""Query-time entity expansion (PRD §S3).

Given a free-text query, return the top-``k`` entity ids most similar to it
so that downstream retrieval (hybrid_search, vector_search) can enrich its
filter set or annotate results with expanded entity context.

This module ships a deterministic in-memory numpy implementation for tests
and early integration. The production path — pgvector HNSW over the
``entities`` table backed by INDUS-SDE-ST embeddings — is documented as a
TODO at the bottom of the file and must replace this before §S3 is promoted
to stable.

Determinism: for identical (query, index) inputs the same entity ordering
is always returned. Query vectors are generated from a stable content hash
so tests can rely on exact output.

Performance: ``expand()`` over the default 100-vector fixture completes in
well under 20ms on commodity hardware (numpy argpartition is O(N)).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_DIM = 64


@dataclass(frozen=True)
class EntityIndex:
    """In-memory cosine-similarity index over entity description embeddings.

    Attributes
    ----------
    ids
        ``(N,)`` int64 array of entity ids.
    vectors
        ``(N, D)`` float32 array of L2-normalised embedding vectors.
    """

    ids: np.ndarray
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if self.ids.ndim != 1:
            raise ValueError("ids must be 1-D")
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be 2-D")
        if self.ids.shape[0] != self.vectors.shape[0]:
            raise ValueError(
                f"ids/vectors length mismatch: {self.ids.shape[0]} vs " f"{self.vectors.shape[0]}"
            )

    @property
    def size(self) -> int:
        return int(self.ids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    """L2-normalise rows of ``mat`` (in float32, safe on zero rows)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (mat / norms).astype(np.float32, copy=False)


def _seed_from_text(text: str) -> int:
    """Return a 32-bit deterministic seed derived from ``text``."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _query_vector(query: str, dim: int) -> np.ndarray:
    """Hash ``query`` to a deterministic unit vector of length ``dim``.

    This is deliberately content-addressed so identical queries always
    produce identical vectors — the unit test relies on that. The
    production path will replace this with a real embedding model.
    """
    rng = np.random.default_rng(_seed_from_text(query))
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        vec[0] = 1.0
        norm = 1.0
    return vec / norm


def build_index(
    entity_ids: list[int] | np.ndarray,
    embeddings: np.ndarray,
) -> EntityIndex:
    """Construct an :class:`EntityIndex` from raw ids + embeddings."""
    ids = np.asarray(entity_ids, dtype=np.int64)
    vectors = np.asarray(embeddings, dtype=np.float32)
    vectors = _l2_normalise(vectors)
    return EntityIndex(ids=ids, vectors=vectors)


def build_fixture_index(n: int = 100, dim: int = DEFAULT_DIM, seed: int = 0) -> EntityIndex:
    """Deterministic fixture index for tests and latency benchmarks."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1, dtype=np.int64)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    return build_index(ids.tolist(), vectors)


# Module-level default index — tests override with their own fixture.
_default_index: Optional[EntityIndex] = None


def _get_default_index() -> EntityIndex:
    global _default_index
    if _default_index is None:
        _default_index = build_fixture_index()
    return _default_index


def set_default_index(index: EntityIndex) -> None:
    """Replace the module-level default index (used for test setup)."""
    global _default_index
    _default_index = index


def expand(query: str, k: int = 5, *, index: Optional[EntityIndex] = None) -> list[int]:
    """Return top-``k`` entity ids most similar to ``query``.

    Deterministic for identical ``(query, index)`` inputs. Cosine similarity
    is computed as a dot product on L2-normalised vectors. Ties are broken
    by ascending entity id so ordering is stable.

    Parameters
    ----------
    query
        Free-text user query.
    k
        Maximum number of entity ids to return. Clamped to the index size.
    index
        Optional explicit index; falls back to the module default.
    """
    if k <= 0:
        return []

    idx = index if index is not None else _get_default_index()
    if idx.size == 0:
        return []

    qv = _query_vector(query, idx.dim)
    scores = idx.vectors @ qv  # (N,)

    k_eff = min(k, idx.size)

    # Stable selection: sort by (-score, id).
    # np.lexsort sorts by the *last* key, so pass (ids, -scores).
    order = np.lexsort((idx.ids, -scores))
    top = order[:k_eff]
    return [int(idx.ids[i]) for i in top]


# -----------------------------------------------------------------------------
# Production-grade path (TODO — replace in-memory numpy impl)
# -----------------------------------------------------------------------------
#
# The numpy implementation above is adequate for unit tests and pilot
# evaluation over ~10-100 entity descriptions. Production §S3 must:
#
#   1. Embed the query with INDUS-SDE-ST (or the RRF-fused stack — TBD in u14
#      follow-up) and run an HNSW query against ``entities.description_vec``
#      via pgvector 0.8.2. Use iterative scan + halfvec for memory.
#   2. Respect the M13 resolver contract — query-expansion consumers are
#      JIT-lane only (see docs/mcp_dual_lane_contract.md), so the expansion
#      call must either hit an offline-warmed cache or the resolver service,
#      never write ``document_entities`` directly.
#   3. Cache expansion results by (query, k) for the query_log session TTL.
