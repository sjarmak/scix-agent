"""arXiv LaTeX licensing enforcement helpers.

This module implements the user-facing enforcement side of ADR-006 (see
``docs/ADR/006_arxiv_licensing.md``). arXiv holds only a non-exclusive
distribution license for the papers it hosts, which means SciX can ingest,
parse, index, and embed LaTeX-derived text for internal use — but any
verbatim emission to a user-facing surface must be capped to a small snippet
and must always carry a canonical URL back to the authoritative source.

The module exposes a single pure function, :func:`enforce_snippet_budget`,
plus the :class:`SnippetPayload` return type and the budget defaults. Call
sites that emit LaTeX-derived text wrap their output through this helper;
code review is responsible for verifying the wiring at every emission point.

No side effects, no IO, no network. Pure function.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_SNIPPET_BUDGET: int = 500
"""Default maximum number of characters emitted per user-facing snippet.

Chosen to be roughly 2-3 sentences of English prose — enough to show a
retrieval hit in context, not enough to reconstruct the paper. See
docs/ADR/006_arxiv_licensing.md for the rationale.
"""

SNIPPET_BUDGET_ENV_VAR: str = "SCIX_LATEX_SNIPPET_BUDGET"
"""Environment variable that overrides :data:`DEFAULT_SNIPPET_BUDGET`.

Resolved lazily on every call to :func:`enforce_snippet_budget` so operators
can change the budget at runtime without restarting the service. Invalid
values raise :class:`ValueError` at call time (fail fast, no silent fallback).
"""

_TRUNCATION_MARKER: str = "..."


@dataclass(frozen=True)
class SnippetPayload:
    """Immutable result of a snippet-budget enforcement call.

    Attributes:
        snippet: The emitted text, already trimmed to fit within ``budget``.
            If truncation occurred and the budget is large enough, the tail
            is replaced with an ellipsis marker.
        canonical_url: The URL pointing back to the authoritative arXiv
            record. Guaranteed non-empty by the constructor contract in
            :func:`enforce_snippet_budget`.
        truncated: ``True`` if the original body exceeded ``budget`` and the
            snippet was shortened.
        original_length: Length of the original input body before trimming.
            Preserved for observability and metrics.
        budget: The budget value actually applied to this call (the resolved
            value after explicit-arg / env-var / default precedence).
    """

    snippet: str
    canonical_url: str
    truncated: bool
    original_length: int
    budget: int


def enforce_snippet_budget(
    body: str,
    canonical_url: str,
    budget: int | None = None,
) -> SnippetPayload:
    """Trim ``body`` to the configured snippet budget and return a payload.

    Args:
        body: Raw LaTeX-derived text. May be empty; may exceed the budget.
            Expected to be mostly ASCII/Latin-1 — grapheme-cluster safety is
            intentionally out of scope (see ADR-006).
        canonical_url: Non-empty URL pointing back to the authoritative
            source (e.g. ``https://arxiv.org/abs/{id}``). Required. The
            value is echoed verbatim on the payload — callers that render
            the URL in HTML or another context are responsible for
            context-appropriate escaping.
        budget: Optional explicit budget override. If ``None``, the helper
            reads ``SCIX_LATEX_SNIPPET_BUDGET`` from the environment on each
            call and falls back to :data:`DEFAULT_SNIPPET_BUDGET` when unset
            or empty.

    Returns:
        A frozen :class:`SnippetPayload` with the trimmed snippet, the
        canonical URL, a truncation flag, the original length, and the
        resolved budget.

    Raises:
        ValueError: If ``canonical_url`` is empty or whitespace-only, if
            ``budget`` is negative, or if the env var holds a non-integer or
            negative value.
    """
    if not canonical_url or not canonical_url.strip():
        raise ValueError("canonical_url must be a non-empty string")

    resolved_budget = _resolve_budget(budget)

    original_length = len(body)
    if original_length <= resolved_budget:
        return SnippetPayload(
            snippet=body,
            canonical_url=canonical_url,
            truncated=False,
            original_length=original_length,
            budget=resolved_budget,
        )

    snippet = _truncate(body, resolved_budget)
    return SnippetPayload(
        snippet=snippet,
        canonical_url=canonical_url,
        truncated=True,
        original_length=original_length,
        budget=resolved_budget,
    )


def _resolve_budget(explicit: int | None) -> int:
    """Resolve the effective budget from explicit arg, env var, or default.

    Precedence: explicit argument > env var > :data:`DEFAULT_SNIPPET_BUDGET`.
    Raises :class:`ValueError` for negative or malformed values.
    """
    if explicit is not None:
        if explicit < 0:
            raise ValueError(f"budget must be >= 0, got {explicit}")
        return explicit

    raw = os.environ.get(SNIPPET_BUDGET_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_SNIPPET_BUDGET

    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{SNIPPET_BUDGET_ENV_VAR} must be a non-negative integer, " f"got {raw!r}"
        ) from exc

    if parsed < 0:
        raise ValueError(f"budget from {SNIPPET_BUDGET_ENV_VAR} must be >= 0, got {parsed}")
    return parsed


def _truncate(body: str, budget: int) -> str:
    """Return ``body`` shortened to at most ``budget`` characters.

    If the budget can accommodate the truncation marker, the returned string
    ends with ``"..."`` so readers see that the text was cut. Otherwise a
    plain head slice is returned (still within budget, including budget 0).
    """
    marker_len = len(_TRUNCATION_MARKER)
    if budget <= marker_len:
        return body[:budget]
    return body[: budget - marker_len] + _TRUNCATION_MARKER
