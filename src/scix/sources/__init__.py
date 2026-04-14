"""External-source ingestion package.

Submodules live here for pipelines that pull structured content from third-party
sources (arXiv LaTeX, OpenAlex, S2ORC, etc.). See docs/ADR/006_arxiv_licensing.md
for the licensing posture that governs arXiv LaTeX-derived text.
"""

from __future__ import annotations

from scix.sources.licensing import (
    DEFAULT_SNIPPET_BUDGET,
    SNIPPET_BUDGET_ENV_VAR,
    SnippetPayload,
    enforce_snippet_budget,
)

__all__ = [
    "DEFAULT_SNIPPET_BUDGET",
    "SNIPPET_BUDGET_ENV_VAR",
    "SnippetPayload",
    "enforce_snippet_budget",
]
