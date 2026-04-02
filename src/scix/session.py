"""Session state management for MCP tool orchestration.

Tracks working sets of papers, seen papers, and per-session context
so that MCP tools can share state within a conversation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_WORKING_SET_SOFT_LIMIT = 1000


@dataclass(frozen=True)
class WorkingSetEntry:
    """A single paper in the session working set.

    Frozen to prevent accidental mutation after creation.
    """

    bibcode: str
    added_at: str
    source_tool: str
    source_context: str
    relevance_hint: str
    tags: list[str] = field(default_factory=list)


@dataclass
class _SessionData:
    """Internal per-session storage."""

    working_set: dict[str, WorkingSetEntry] = field(default_factory=dict)
    seen_papers: set[str] = field(default_factory=set)


class SessionState:
    """In-memory session state keyed by session_id.

    For stdio transport a single default session ('_default') is used.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionData] = {}

    def _get(self, session_id: str) -> _SessionData:
        if session_id not in self._sessions:
            self._sessions[session_id] = _SessionData()
        return self._sessions[session_id]

    # -- working set --------------------------------------------------------

    def add_to_working_set(
        self,
        bibcode: str,
        source_tool: str,
        source_context: str = "",
        relevance_hint: str = "",
        tags: list[str] | None = None,
        session_id: str = "_default",
    ) -> WorkingSetEntry:
        """Add a paper to the working set, returning the entry.

        If the bibcode already exists it is replaced (updated).
        A warning is logged when the working set exceeds the soft limit.
        """
        data = self._get(session_id)

        entry = WorkingSetEntry(
            bibcode=bibcode,
            added_at=datetime.now(timezone.utc).isoformat(),
            source_tool=source_tool,
            source_context=source_context,
            relevance_hint=relevance_hint,
            tags=tags if tags is not None else [],
        )
        data.working_set[bibcode] = entry
        data.seen_papers.add(bibcode)

        if len(data.working_set) > _WORKING_SET_SOFT_LIMIT:
            logger.warning(
                "Working set for session '%s' has %d entries (soft limit: %d)",
                session_id,
                len(data.working_set),
                _WORKING_SET_SOFT_LIMIT,
            )

        return entry

    def get_working_set(self, session_id: str = "_default") -> list[WorkingSetEntry]:
        """Return the current working set as a list, ordered by insertion."""
        data = self._get(session_id)
        return list(data.working_set.values())

    def is_in_working_set(self, bibcode: str, session_id: str = "_default") -> bool:
        """Check whether a bibcode is in the working set."""
        data = self._get(session_id)
        return bibcode in data.working_set

    def clear_working_set(self, session_id: str = "_default") -> int:
        """Clear the working set, returning the number of entries removed."""
        data = self._get(session_id)
        count = len(data.working_set)
        data.working_set.clear()
        return count

    # -- seen papers --------------------------------------------------------

    def mark_seen(self, bibcode: str, session_id: str = "_default") -> None:
        """Record that the agent has seen a paper."""
        data = self._get(session_id)
        data.seen_papers.add(bibcode)

    def is_seen(self, bibcode: str, session_id: str = "_default") -> bool:
        """Check whether a paper has been seen in this session."""
        data = self._get(session_id)
        return bibcode in data.seen_papers

    # -- summary ------------------------------------------------------------

    def get_session_summary(self, session_id: str = "_default") -> dict[str, Any]:
        """Return summary statistics for the session."""
        data = self._get(session_id)
        return {
            "session_id": session_id,
            "working_set_size": len(data.working_set),
            "seen_papers_count": len(data.seen_papers),
        }
