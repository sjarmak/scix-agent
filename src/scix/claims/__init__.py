"""Nanopub-inspired claim extraction pipeline.

Public API:

- :class:`ClaimDict` — TypedDict-like shape returned by an LLMClient and persisted
  into ``paper_claims``.
- :class:`LLMClient` — Protocol; pluggable extractor.
- :class:`ClaudeCliLLMClient` — default LLMClient that shells out to ``claude -p``
  via subprocess (Claude Code OAuth subagent path; no paid-API SDK).
- :class:`StubLLMClient` — test helper returning canned responses.
- :func:`extract_claims_for_paper` — the pipeline entry point.
- :func:`classify_section_role` — heading -> role classifier.
- :func:`split_paragraphs` — section text -> [(paragraph_index, paragraph_text, offset)].
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

__all__ = [
    "ClaimDict",
    "ClaudeCliLLMClient",
    "LLMClient",
    "StubLLMClient",
    "classify_section_role",
    "extract_claims_for_paper",
    "split_paragraphs",
]
