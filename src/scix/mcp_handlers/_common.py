"""Shared helpers used across MCP handler domains (working-set resolution, param errors)."""

from __future__ import annotations

import json
import logging
from typing import Any

from scix.mcp_errors import ErrorCode
from scix.mcp_server import (
    _session_state,
)

logger = logging.getLogger("scix.mcp_server")


def _session_fallthrough_bibcodes() -> list[str]:
    """Session-state fall-through: focused papers, then working set, then [].

    Shared by every tool that scopes to "what the agent has been looking at"
    when no explicit bibcode list is supplied. Used by both
    :func:`_resolve_working_set_bibcodes` and the synthesize / find_gaps
    handlers (which take their explicit-list arg under different names).
    """
    focused = _session_state.get_focused_papers()
    if focused:
        return list(focused)
    return [e.bibcode for e in _session_state.get_working_set()]

def _resolve_working_set_bibcodes(args: dict[str, Any]) -> list[str]:
    """Return the bibcodes to scope a tool call to.

    Resolution order:
        1. ``args["bibcodes"]`` if non-empty (explicit caller intent).
        2. ``_session_state.get_focused_papers()`` (papers inspected via
           ``get_paper`` during the session).
        3. ``_session_state.get_working_set()`` as a final fallback for
           backward compatibility.

    Returns an empty list when no working-set source is available — callers
    decide whether that's an error (e.g. ``temporal_evolution`` with no
    ``bibcode_or_query``) or a no-op signaling full-corpus behaviour
    (e.g. ``facet_counts``).

    This is the canonical pattern referenced by the bead-3uvn working_set
    abstraction; ``find_gaps`` and ``synthesize_findings`` use the same
    fall-through via :func:`_session_fallthrough_bibcodes` (they take the
    explicit-list arg under different names).
    """
    explicit = args.get("bibcodes")
    if isinstance(explicit, list) and explicit:
        return [str(b) for b in explicit]
    return _session_fallthrough_bibcodes()

def _missing_required_params_error(
    *,
    mode: str,
    required: list[str],
    got: list[str],
    message: str,
) -> str:
    """Build the structured validation-error payload for citation_traverse.

    The error envelope addresses bead scix_experiments-zjt9: ``mode``
    overloads ``citation_traverse`` with disjoint required-param sets that
    JSON Schema can't express, so validation runs in the handler and
    returns a machine-readable payload rather than a raw string. Agents
    branch on ``error_code``; humans read ``error``; ``required`` /
    ``got`` make the gap explicit so the agent can self-correct without
    a follow-up round-trip.
    """
    return json.dumps(
        {
            "error": message,
            "error_code": ErrorCode.MISSING_REQUIRED_PARAMS,
            "mode": mode,
            "required": required,
            "got": got,
        }
    )
