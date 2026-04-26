"""Section-aware sliding-window chunker + INDUS embedder + Qdrant uploader.

Public API:

- :class:`Chunk` — frozen dataclass; one chunk = one tokenizer window over a section.
- :func:`iter_chunks` — pure-function generator over ``papers_fulltext.sections``.
- :func:`normalize_heading` — canonicalize a raw section heading.
- :class:`INDUSEmbedder` — INDUS-768 batch encoder.
- :data:`CHUNKS_COLLECTION` / :func:`ensure_collection` / :func:`chunk_point_id`
  — Qdrant collection schema + bootstrap.
- :func:`run` / :class:`BatchStats` — top-level chunk-pass pipeline driver.

The tokenizer is injected, never imported at module load. Production callers
pass a HuggingFace ``AutoTokenizer`` matching INDUS
(``nasa-impact/nasa-smd-ibm-st-v2``); tests pass lightweight stubs.
"""

from __future__ import annotations

from .chunker import Chunk, iter_chunks
from .collection import CHUNKS_COLLECTION, chunk_point_id, ensure_collection
from .embedder import INDUSEmbedder
from .pipeline import BatchStats, run
from .section_norm import normalize_heading

__all__ = [
    "BatchStats",
    "CHUNKS_COLLECTION",
    "Chunk",
    "INDUSEmbedder",
    "chunk_point_id",
    "ensure_collection",
    "iter_chunks",
    "normalize_heading",
    "run",
]
