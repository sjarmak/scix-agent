"""Tests for src/scix/session.py — working set, seen papers, and session summary."""

from __future__ import annotations

import logging

import pytest

from scix.session import SessionState, WorkingSetEntry, _WORKING_SET_SOFT_LIMIT


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
