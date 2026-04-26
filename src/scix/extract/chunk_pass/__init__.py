"""Section-aware sliding-window chunker for paper full-text.

Public API:

- :class:`Chunk` — frozen dataclass; one chunk = one tokenizer window over a section.
- :func:`iter_chunks` — pure-function generator over ``papers_fulltext.sections``.

The tokenizer is injected, never imported at module load. Production callers
pass a HuggingFace ``AutoTokenizer`` matching INDUS
(``nasa-impact/nasa-smd-ibm-st-v2``); tests pass lightweight stubs.
"""

from __future__ import annotations

from .chunker import Chunk, iter_chunks

__all__ = ["Chunk", "iter_chunks"]
