"""Tests for src/scix/session.py — working set, seen papers, and session summary."""

from __future__ import annotations

import logging

import pytest

from scix.session import (
    _FOCUSED_HARD_CAP,
    _FOCUSED_SOFT_WARN,
    _WORKING_SET_SOFT_LIMIT,
    SessionState,
    WorkingSetEntry,
)


class TestWorkingSetEntry:
    def test_frozen_dataclass(self) -> None:
        entry = WorkingSetEntry(
            bibcode="2024ApJ...1234A",
            added_at="2026-04-01T00:00:00+00:00",
            source_tool="search",
            source_context="keyword search",
            relevance_hint="high",
            tags=["astro"],
        )
        assert entry.bibcode == "2024ApJ...1234A"
        with pytest.raises(AttributeError):
            entry.bibcode = "other"  # type: ignore[misc]

    def test_default_tags(self) -> None:
        entry = WorkingSetEntry(
            bibcode="2024ApJ...1234A",
            added_at="now",
            source_tool="search",
            source_context="",
            relevance_hint="",
        )
        assert entry.tags == ()


class TestSessionStateWorkingSet:
    def test_add_and_get(self) -> None:
        state = SessionState()
        entry = state.add_to_working_set(
            bibcode="2024ApJ...1234A",
            source_tool="search",
            source_context="query: galaxies",
            relevance_hint="top result",
            tags=["astro"],
        )
        assert entry.bibcode == "2024ApJ...1234A"
        assert entry.source_tool == "search"
        ws = state.get_working_set()
        assert len(ws) == 1
        assert ws[0].bibcode == "2024ApJ...1234A"

    def test_add_replaces_duplicate(self) -> None:
        state = SessionState()
        state.add_to_working_set(bibcode="ABC", source_tool="tool1")
        state.add_to_working_set(bibcode="ABC", source_tool="tool2")
        ws = state.get_working_set()
        assert len(ws) == 1
        assert ws[0].source_tool == "tool2"

    def test_is_in_working_set(self) -> None:
        state = SessionState()
        assert state.is_in_working_set("ABC") is False
        state.add_to_working_set(bibcode="ABC", source_tool="t")
        assert state.is_in_working_set("ABC") is True

    def test_clear_working_set(self) -> None:
        state = SessionState()
        state.add_to_working_set(bibcode="A", source_tool="t")
        state.add_to_working_set(bibcode="B", source_tool="t")
        removed = state.clear_working_set()
        assert removed == 2
        assert state.get_working_set() == []

    def test_default_session_id(self) -> None:
        state = SessionState()
        state.add_to_working_set(bibcode="X", source_tool="t")
        summary = state.get_session_summary()
        assert summary["session_id"] == "_default"

    def test_separate_sessions(self) -> None:
        state = SessionState()
        state.add_to_working_set(bibcode="A", source_tool="t", session_id="s1")
        state.add_to_working_set(bibcode="B", source_tool="t", session_id="s2")
        assert len(state.get_working_set(session_id="s1")) == 1
        assert len(state.get_working_set(session_id="s2")) == 1
        assert state.is_in_working_set("A", session_id="s2") is False


class TestSessionSummary:
    def test_summary_counts(self) -> None:
        state = SessionState()
        state.add_to_working_set(bibcode="A", source_tool="t")
        state.add_to_working_set(bibcode="B", source_tool="t")
        summary = state.get_session_summary()
        assert summary["working_set_size"] == 2
        assert summary["seen_papers_count"] == 2  # A, B auto-seen via add


class TestSoftLimit:
    def test_warning_on_exceeding_limit(self, caplog: pytest.LogCaptureFixture) -> None:
        state = SessionState()
        # Fill to the limit — no warning yet
        for i in range(_WORKING_SET_SOFT_LIMIT):
            state.add_to_working_set(bibcode=f"BIB{i:05d}", source_tool="t")

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            state.add_to_working_set(bibcode="OVERFLOW", source_tool="t")

        assert any("soft limit" in r.message for r in caplog.records)
        # Entry is still added (soft limit, not hard)
        assert state.is_in_working_set("OVERFLOW") is True


class TestFocusedPapersCap:
    """``focused_papers`` is bounded by a soft warn at 200 and a hard FIFO cap at 500.

    Mirrors the ``_WORKING_SET_HARD_CAP`` pattern so long-running multi-turn
    sessions don't grow unbounded and degrade downstream tools that scope
    queries to focused papers (see scix_experiments-u0j1).
    """

    def test_constants(self) -> None:
        assert _FOCUSED_SOFT_WARN == 200
        assert _FOCUSED_HARD_CAP == 500

    def test_below_soft_warn_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        state = SessionState()
        with caplog.at_level(logging.WARNING):
            for i in range(_FOCUSED_SOFT_WARN - 1):  # 199
                state.track_focused(f"FOC{i:05d}")
        assert not any("focused" in r.message.lower() for r in caplog.records)
        assert len(state.get_focused_papers()) == _FOCUSED_SOFT_WARN - 1

    def test_soft_warn_emitted_once_at_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        state = SessionState()
        # Add 199 — no warning yet.
        for i in range(_FOCUSED_SOFT_WARN - 1):
            state.track_focused(f"FOC{i:05d}")

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            # 200th entry triggers the warning.
            state.track_focused("THRESHOLD")
            # Subsequent entries must NOT re-emit (warn-once per session).
            for i in range(50):
                state.track_focused(f"AFTER{i:05d}")

        focused_warnings = [r for r in caplog.records if "focused" in r.message.lower()]
        assert len(focused_warnings) == 1
        assert "soft" in focused_warnings[0].message.lower()

    def test_at_hard_cap_no_eviction(self) -> None:
        state = SessionState()
        for i in range(_FOCUSED_HARD_CAP):  # 500
            state.track_focused(f"FOC{i:05d}")
        focused = state.get_focused_papers()
        assert len(focused) == _FOCUSED_HARD_CAP
        # First-inserted entry is still present.
        assert "FOC00000" in focused
        assert f"FOC{_FOCUSED_HARD_CAP - 1:05d}" in focused

    def test_above_hard_cap_evicts_oldest_fifo(self) -> None:
        state = SessionState()
        for i in range(_FOCUSED_HARD_CAP):  # 500
            state.track_focused(f"FOC{i:05d}")
        # 501st entry must evict the oldest (FIFO).
        state.track_focused("OVERFLOW")

        focused = set(state.get_focused_papers())
        assert len(focused) == _FOCUSED_HARD_CAP
        assert "OVERFLOW" in focused
        assert "FOC00000" not in focused  # oldest evicted
        assert "FOC00001" in focused  # second-oldest survives

    def test_eviction_order_strict_fifo(self) -> None:
        """Adding N papers above the cap evicts exactly the N oldest."""
        state = SessionState()
        for i in range(_FOCUSED_HARD_CAP):
            state.track_focused(f"FOC{i:05d}")
        for i in range(5):
            state.track_focused(f"NEW{i:05d}")

        focused = set(state.get_focused_papers())
        assert len(focused) == _FOCUSED_HARD_CAP
        # Five oldest evicted.
        for i in range(5):
            assert f"FOC{i:05d}" not in focused
        # Sixth-oldest is now the oldest survivor.
        assert "FOC00005" in focused
        for i in range(5):
            assert f"NEW{i:05d}" in focused

    def test_re_track_does_not_change_eviction_order(self) -> None:
        """Re-tracking an existing bibcode keeps its original insertion slot.

        This guarantees that "frequently re-touched" papers don't bubble up to
        the front and protect themselves indefinitely from FIFO eviction —
        the cap remains a true bound on session age, not a recency heuristic.
        """
        state = SessionState()
        for i in range(_FOCUSED_HARD_CAP):
            state.track_focused(f"FOC{i:05d}")
        # Re-track the oldest several times — this must NOT move it to the tail.
        for _ in range(10):
            state.track_focused("FOC00000")
        # One more push past the cap — FOC00000 should still be the one evicted.
        state.track_focused("OVERFLOW")
        focused = set(state.get_focused_papers())
        assert "FOC00000" not in focused
        assert "OVERFLOW" in focused
