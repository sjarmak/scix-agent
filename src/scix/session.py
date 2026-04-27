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

#: Hard cap applied during bulk add (``add_bibcodes_to_working_set``). When a
#: bulk-add operation pushes the working set above this threshold, oldest
#: entries (FIFO) are dropped. The cap exists to keep multi-turn agent flows
#: bounded — an unrestricted append over a long session would degrade
#: downstream tools that scope queries to the working set.
_WORKING_SET_HARD_CAP = 200


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
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class _SessionData:
    """Internal per-session storage."""

    working_set: dict[str, WorkingSetEntry] = field(default_factory=dict)
    seen_papers: set[str] = field(default_factory=set)
    focused_papers: set[str] = field(default_factory=set)


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
            tags=tuple(tags) if tags is not None else (),
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

    def add_bibcodes_to_working_set(
        self,
        bibcodes: list[str],
        source_tool: str,
        source_context: str = "",
        relevance_hint: str = "",
        tags: list[str] | None = None,
        session_id: str = "_default",
    ) -> int:
        """Bulk-add bibcodes to the working set.

        Dedupes against any bibcodes already present (insertion-order dict
        semantics). After appending, applies a FIFO cap of
        :data:`_WORKING_SET_HARD_CAP` — oldest entries are dropped so the
        working set never exceeds the cap. Use this when a tool returns a
        list of papers that should bootstrap subsequent multi-turn analysis.

        Returns the number of bibcodes processed in this call (i.e. the
        ``len(set(bibcodes))`` after intra-call dedupe).
        """
        if not bibcodes:
            return 0

        data = self._get(session_id)
        # Dedupe within the call while preserving order.
        unique: list[str] = []
        seen_in_call: set[str] = set()
        for bib in bibcodes:
            if bib in seen_in_call:
                continue
            seen_in_call.add(bib)
            unique.append(bib)

        for bib in unique:
            self.add_to_working_set(
                bibcode=bib,
                source_tool=source_tool,
                source_context=source_context,
                relevance_hint=relevance_hint,
                tags=tags,
                session_id=session_id,
            )

        # FIFO cap: drop oldest entries until back under the hard cap.
        while len(data.working_set) > _WORKING_SET_HARD_CAP:
            oldest_bibcode = next(iter(data.working_set))
            del data.working_set[oldest_bibcode]

        return len(unique)

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

    # -- implicit tracking ---------------------------------------------------

    def track_seen(
        self,
        bibcodes: list[str],
        session_id: str = "_default",
    ) -> None:
        """Record bibcodes as seen (returned by any tool)."""
        data = self._get(session_id)
        data.seen_papers.update(bibcodes)

    def track_focused(
        self,
        bibcode: str,
        session_id: str = "_default",
    ) -> None:
        """Record a bibcode as focused (inspected via get_paper).

        Also adds to seen and working set for backward compatibility.
        """
        data = self._get(session_id)
        data.focused_papers.add(bibcode)
        data.seen_papers.add(bibcode)
        # Also keep working set in sync so existing find_gaps logic works
        if bibcode not in data.working_set:
            self.add_to_working_set(
                bibcode=bibcode,
                source_tool="get_paper",
                source_context="auto-tracked",
                session_id=session_id,
            )

    def get_focused_papers(self, session_id: str = "_default") -> list[str]:
        """Return the list of focused bibcodes."""
        data = self._get(session_id)
        return list(data.focused_papers)

    def clear_focused(self, session_id: str = "_default") -> int:
        """Clear the focused set, returning the number removed."""
        data = self._get(session_id)
        count = len(data.focused_papers)
        data.focused_papers.clear()
        return count

    # -- summary ------------------------------------------------------------

    def get_session_summary(self, session_id: str = "_default") -> dict[str, Any]:
        """Return summary statistics for the session."""
        data = self._get(session_id)
        return {
            "session_id": session_id,
            "working_set_size": len(data.working_set),
            "focused_papers_count": len(data.focused_papers),
            "seen_papers_count": len(data.seen_papers),
        }
