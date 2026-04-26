"""SciX nanopub-inspired claims package.

The ``paper_claims`` table is created by ``migrations/062_paper_claims.sql``.

Submodules:

* :mod:`scix.claims.extract` — extraction pipeline (LLMClient + writer).
* :mod:`scix.claims.retrieval` — query helpers exposed to the MCP server.

Re-exports the public API of both submodules so callers can do
``from scix.claims import extract_claims_for_paper, find_claims``.
"""

from __future__ import annotations

from .extract import (
    ClaimDict,
    ClaudeCliLLMClient,
    LLMClient,
    StubLLMClient,
    classify_section_role,
    extract_claims_for_paper,
    split_paragraphs,
)
from .retrieval import find_claims, read_paper_claims

__all__ = [
    "ClaimDict",
    "ClaudeCliLLMClient",
    "LLMClient",
    "StubLLMClient",
    "classify_section_role",
    "extract_claims_for_paper",
    "find_claims",
    "read_paper_claims",
    "split_paragraphs",
]
