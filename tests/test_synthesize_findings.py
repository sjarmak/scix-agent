"""Unit tests for synthesize_findings tool (bead scix_experiments-cfh9).

Pure-mechanical aggregation: working set + section shape -> grounded outline.
No LLM is invoked inside the orchestration code (ZFC).

These tests exercise the new module ``scix.synthesize`` directly and the
MCP wiring via ``_dispatch_tool``. They use MagicMock for the database
connection so the test suite stays runnable without ``SCIX_TEST_DSN``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from scix.mcp_server import _dispatch_tool, _session_state
from scix.synthesize import (
    DEFAULT_SECTIONS,
    INTENT_TO_SECTION,
    SectionBucket,
    SynthesisResult,
    synthesize_findings,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_session() -> Iterator[None]:
    """Clear session state between tests (mirrors test_mcp_server.py)."""
    _session_state.clear_working_set()
    _session_state.clear_focused()
    yield
    _session_state.clear_working_set()
    _session_state.clear_focused()


def _mock_conn(rows_by_query: list[list[tuple]]) -> MagicMock:
    """Build a MagicMock psycopg connection that returns the given rows in order.

    Each call to ``cursor()`` (used as a context manager) yields a cursor
    whose ``fetchall()`` returns the next batch of rows. Used to script
    the multi-query behaviour of ``synthesize_findings``.
    """
    conn = MagicMock()
    cursors: list[MagicMock] = []
    for rows in rows_by_query:
        cur = MagicMock()
        cur.fetchall.return_value = rows
        cur.__enter__ = lambda self: self
        cur.__exit__ = MagicMock(return_value=False)
        cursors.append(cur)
    conn.cursor.side_effect = cursors
    return conn


# ---------------------------------------------------------------------------
# Section-mapping invariants
# ---------------------------------------------------------------------------


class TestSectionMapping:
    def test_default_sections_listed_in_canonical_order(self) -> None:
        assert DEFAULT_SECTIONS == ("background", "methods", "results", "open_questions")

    def test_intent_to_section_mapping_is_total(self) -> None:
        # The three known intent labels in citation_contexts.intent.
        assert INTENT_TO_SECTION["background"] == "background"
        assert INTENT_TO_SECTION["method"] == "methods"
        assert INTENT_TO_SECTION["result_comparison"] == "results"


# ---------------------------------------------------------------------------
# Empty / missing input behaviour
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_empty_working_set_returns_empty_structure(self) -> None:
        conn = MagicMock()
        result = synthesize_findings(conn, working_set_bibcodes=[])
        assert isinstance(result, SynthesisResult)
        assert result.sections == []
        assert result.unattributed_bibcodes == []
        assert result.coverage["total_bibcodes"] == 0
        assert "message" in result.metadata

    def test_falls_through_to_focused_papers_when_none(self) -> None:
        # No DB calls should happen if the working set is also empty.
        conn = MagicMock()
        result = synthesize_findings(conn, working_set_bibcodes=None)
        assert result.sections == []
        assert "message" in result.metadata


# ---------------------------------------------------------------------------
# Intent-driven section assignment
# ---------------------------------------------------------------------------


class TestIntentAssignment:
    def test_modal_intent_decides_section(self) -> None:
        """A paper with two 'method' rows and one 'background' row in
        citation_contexts.intent goes to 'methods' (modal intent wins)."""
        # Query 1: papers metadata. (bibcode, title, year, abstract)
        papers_rows = [
            ("2024A", "Method paper", 2024, "An abstract about methods."),
        ]
        # Query 2: intent histogram per target_bibcode.
        # Columns: (target_bibcode, intent, n_rows)
        intent_rows = [
            ("2024A", "method", 2),
            ("2024A", "background", 1),
        ]
        # Query 3: paper_metrics + community labels
        # (bibcode, community_id, label) for working set
        community_rows = [
            ("2024A", 7, "Cosmology"),
        ]

        conn = _mock_conn([papers_rows, intent_rows, community_rows])
        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
        )
        methods = next(s for s in result.sections if s.name == "methods")
        assert any(p["bibcode"] == "2024A" for p in methods.cited_papers)
        # Other sections must NOT contain this paper.
        for s in result.sections:
            if s.name == "methods":
                continue
            assert all(p["bibcode"] != "2024A" for p in s.cited_papers)

    def test_result_comparison_intent_maps_to_results_section(self) -> None:
        papers_rows = [("2024B", "Replication paper", 2024, "abs")]
        intent_rows = [("2024B", "result_comparison", 5)]
        community_rows = [("2024B", 1, "Stellar")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])
        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024B"],
            sections=list(DEFAULT_SECTIONS),
        )
        results_section = next(s for s in result.sections if s.name == "results")
        assert any(p["bibcode"] == "2024B" for p in results_section.cited_papers)


# ---------------------------------------------------------------------------
# Community fall-through (no intent coverage)
# ---------------------------------------------------------------------------


class TestCommunityFallThrough:
    def test_paper_in_modal_community_lands_in_background(self) -> None:
        """No intent coverage; community signal alone decides section."""
        papers_rows = [
            ("2024X", "Paper X", 2024, "abs X"),
            ("2024Y", "Paper Y", 2024, "abs Y"),
            ("2024Z", "Paper Z", 2024, "abs Z"),
        ]
        intent_rows: list[tuple] = []  # no intent coverage at all
        # X and Y in community 5 (modal); Z in community 99 (outlier)
        community_rows = [
            ("2024X", 5, "Galaxies"),
            ("2024Y", 5, "Galaxies"),
            ("2024Z", 99, "Plasma"),
        ]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024X", "2024Y", "2024Z"],
            sections=list(DEFAULT_SECTIONS),
        )

        background = next(s for s in result.sections if s.name == "background")
        open_q = next(s for s in result.sections if s.name == "open_questions")

        bg_bibcodes = {p["bibcode"] for p in background.cited_papers}
        oq_bibcodes = {p["bibcode"] for p in open_q.cited_papers}

        # X and Y are in modal community -> background.
        assert {"2024X", "2024Y"}.issubset(bg_bibcodes)
        # Z is in a minority community -> open_questions (cross-community).
        assert "2024Z" in oq_bibcodes

    def test_no_intent_no_community_lands_in_unattributed(self) -> None:
        papers_rows = [("2024U", "Orphan paper", 2024, "abs")]
        intent_rows: list[tuple] = []
        community_rows: list[tuple] = []  # no metrics row at all
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024U"],
            sections=list(DEFAULT_SECTIONS),
        )
        assert "2024U" in result.unattributed_bibcodes


# ---------------------------------------------------------------------------
# Deterministic structure
# ---------------------------------------------------------------------------


class TestDeterministicStructure:
    def test_returns_all_requested_sections_even_when_empty(self) -> None:
        """The 4 default sections always appear in the output, in canonical
        order, even when some have zero cited papers."""
        papers_rows = [("2024A", "P", 2024, "a")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L1")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
        )
        names = [s.name for s in result.sections]
        assert names == list(DEFAULT_SECTIONS)
        # Each section is a SectionBucket with required fields.
        for s in result.sections:
            assert isinstance(s, SectionBucket)
            assert isinstance(s.cited_papers, list)
            assert isinstance(s.theme_summary, str)

    def test_max_papers_per_section_is_respected(self) -> None:
        # 10 papers all assigned to background via modal community.
        papers_rows = [(f"2024P{i}", f"P{i}", 2024, f"a{i}") for i in range(10)]
        intent_rows: list[tuple] = []
        community_rows = [(f"2024P{i}", 1, "Common") for i in range(10)]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=[f"2024P{i}" for i in range(10)],
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=3,
        )
        background = next(s for s in result.sections if s.name == "background")
        assert len(background.cited_papers) == 3

    def test_coverage_note_reports_section_signal_count(self) -> None:
        papers_rows = [
            ("2024A", "A", 2024, "abs"),
            ("2024B", "B", 2024, "abs"),
        ]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [
            ("2024A", 1, "L"),
            ("2024B", 1, "L"),
        ]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])
        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A", "2024B"],
            sections=list(DEFAULT_SECTIONS),
        )
        # 2 in working set; both have a section signal (intent or community).
        assert result.coverage["total_bibcodes"] == 2
        assert result.coverage["assigned_bibcodes"] == 2
        assert result.coverage["unattributed_bibcodes"] == 0
        # 1 paper had intent coverage; 1 had community-only.
        assert result.coverage["intent_assigned_bibcodes"] == 1
        assert result.coverage["community_assigned_bibcodes"] == 1


# ---------------------------------------------------------------------------
# MCP wiring (dispatched via _dispatch_tool)
# ---------------------------------------------------------------------------


class TestMCPDispatch:
    def test_dispatch_with_working_set_arg(self) -> None:
        papers_rows = [("2024A", "T", 2024, "abs")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "Lbl")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        out = _dispatch_tool(
            conn,
            "synthesize_findings",
            {"working_set_bibcodes": ["2024A"]},
        )
        result = json.loads(out)
        assert "sections" in result
        assert isinstance(result["sections"], list)
        # Default section list -> 4 buckets.
        assert len(result["sections"]) == len(DEFAULT_SECTIONS)
        assert "unattributed_bibcodes" in result
        assert "assignment_coverage" in result
        # Confirm we did NOT regress the wire format to the colliding "coverage"
        # key, which is reserved for claim_blame/find_replications shape.
        assert "coverage" not in result

    def test_dispatch_falls_through_to_session_focused_papers(self) -> None:
        # Seed focused papers via session_state directly (simulating prior
        # get_paper / lit_review calls).
        _session_state.track_focused("2024A")

        papers_rows = [("2024A", "T", 2024, "abs")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "Lbl")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        out = _dispatch_tool(conn, "synthesize_findings", {})
        result = json.loads(out)
        # The methods section should contain the focused paper.
        methods = next(s for s in result["sections"] if s["name"] == "methods")
        assert any(p["bibcode"] == "2024A" for p in methods["cited_papers"])

    def test_dispatch_empty_session_returns_helpful_message(self) -> None:
        conn = MagicMock()
        out = _dispatch_tool(conn, "synthesize_findings", {})
        result = json.loads(out)
        assert result["sections"] == []
        assert "message" in result.get("metadata", {})


# ---------------------------------------------------------------------------
# Acceptance-criterion 5 mirror: 30-paper working set splits across sections
# ---------------------------------------------------------------------------


class TestAcceptanceCoverage:
    def test_30_paper_working_set_splits_into_4_sections_above_50pct(self) -> None:
        """AC5: a 30-paper working set is split into the 4 default sections
        with >50% coverage (i.e. <50% land in unattributed)."""
        bibcodes = [f"2024P{i:02d}" for i in range(30)]
        papers_rows = [(b, f"Title {b}", 2024, f"abs {b}") for b in bibcodes]
        # 5 papers carry intent coverage spanning all 3 intents.
        intent_rows = [
            ("2024P00", "method", 3),
            ("2024P01", "method", 1),
            ("2024P02", "background", 2),
            ("2024P03", "result_comparison", 4),
            ("2024P04", "background", 1),
        ]
        # 25 of the remaining 30 papers have community labels (mixed).
        # 20 in modal community 1, 5 in minority community 2.
        # 5 papers have NO community row (-> unattributed).
        community_rows = []
        for i, b in enumerate(bibcodes):
            if i < 20:
                community_rows.append((b, 1, "Modal"))
            elif i < 25:
                community_rows.append((b, 2, "Minority"))
            # else: skip -> unattributed

        conn = _mock_conn([papers_rows, intent_rows, community_rows])
        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
        )
        assigned = sum(len(s.cited_papers) for s in result.sections)
        # AC5 threshold: > 50% coverage.
        assert assigned > len(bibcodes) * 0.5
        # All 4 sections present.
        assert {s.name for s in result.sections} == set(DEFAULT_SECTIONS)


# ---------------------------------------------------------------------------
# Per-paper section-assignment signals (bead scix_experiments-gtsx)
# ---------------------------------------------------------------------------


class TestPerPaperSignals:
    """AC1: each cited_papers entry exposes the signals that produced its
    section assignment, so an agent can re-bucket papers it disagrees with."""

    def test_signal_used_intent_modal_when_modal_intent_decides(self) -> None:
        papers_rows = [("2024A", "Method paper", 2024, "abs")]
        intent_rows = [
            ("2024A", "method", 2),
            ("2024A", "background", 1),
        ]
        community_rows = [("2024A", 7, "Cosmology")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        assert row["signal_used"] == "intent_modal"
        assert row["section_assigned"] == "methods"

    def test_signal_used_community_fallthrough_when_no_intent_coverage(self) -> None:
        papers_rows = [
            ("2024X", "X", 2024, "abs"),
            ("2024Y", "Y", 2024, "abs"),
        ]
        intent_rows: list[tuple] = []
        community_rows = [
            ("2024X", 5, "Modal"),
            ("2024Y", 5, "Modal"),
        ]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024X", "2024Y"],
            sections=list(DEFAULT_SECTIONS),
        )
        background = next(s for s in result.sections if s.name == "background")
        for row in background.cited_papers:
            assert row["signal_used"] == "community_fallthrough"

    def test_signals_payload_has_full_schema(self) -> None:
        """AC1 schema: signals.{intent_counts, intent_total_rows, community_id,
        community_share, is_modal_community, modal_community_id}."""
        papers_rows = [
            ("2024A", "A", 2024, "abs"),
            ("2024B", "B", 2024, "abs"),
            ("2024C", "C", 2024, "abs"),
        ]
        intent_rows = [
            ("2024A", "method", 3),
            ("2024A", "background", 1),
        ]
        # 2024A & 2024B share modal community 1; 2024C is in community 2.
        community_rows = [
            ("2024A", 1, "Lbl1"),
            ("2024B", 1, "Lbl1"),
            ("2024C", 2, "Lbl2"),
        ]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A", "2024B", "2024C"],
            sections=list(DEFAULT_SECTIONS),
        )

        # Find each paper's row across all sections.
        all_rows = {p["bibcode"]: p for s in result.sections for p in s.cited_papers}

        # 2024A — intent_modal in 'methods'.
        row_a = all_rows["2024A"]
        assert row_a["signals"]["intent_counts"] == {"method": 3, "background": 1}
        assert row_a["signals"]["intent_total_rows"] == 4
        assert row_a["signals"]["community_id"] == 1
        assert row_a["signals"]["modal_community_id"] == 1
        assert row_a["signals"]["is_modal_community"] is True
        # community_share: how many of the 3 working-set bibcodes share community 1?
        assert row_a["signals"]["community_share"] == pytest.approx(2 / 3)
        # alternative_sections: a paper with intent_counts {method,background}
        # AND community evidence has multiple options.
        assert isinstance(row_a["alternative_sections"], list)

        # 2024B — community fall-through to 'background'.
        row_b = all_rows["2024B"]
        assert row_b["signal_used"] == "community_fallthrough"
        assert row_b["signals"]["intent_counts"] == {}
        assert row_b["signals"]["intent_total_rows"] == 0
        assert row_b["signals"]["is_modal_community"] is True

        # 2024C — community fall-through to 'open_questions' (minority).
        row_c = all_rows["2024C"]
        assert row_c["signal_used"] == "community_fallthrough"
        assert row_c["signals"]["community_id"] == 2
        assert row_c["signals"]["is_modal_community"] is False
        assert row_c["signals"]["modal_community_id"] == 1

    def test_alternative_sections_includes_other_options(self) -> None:
        """A paper with intent_counts {method, background} should list both
        'methods' and 'background' as alternatives even though only the modal
        intent decides the assignment."""
        papers_rows = [("2024A", "A", 2024, "abs")]
        intent_rows = [
            ("2024A", "method", 3),
            ("2024A", "background", 1),
            ("2024A", "result_comparison", 1),
        ]
        community_rows = [("2024A", 1, "Lbl")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        # Alternatives should include the other two intent-mapped sections
        # plus 'background' from the community signal (paper is in modal
        # community here since it's the only paper).
        alts = set(row["alternative_sections"])
        assert "background" in alts
        assert "results" in alts
        # The chosen section itself should NOT appear in alternatives.
        assert "methods" not in alts


# ---------------------------------------------------------------------------
# section_overrides kwarg (bead scix_experiments-gtsx AC2/AC3)
# ---------------------------------------------------------------------------


class TestSectionOverrides:
    def test_overrides_pin_papers_regardless_of_signals(self) -> None:
        """AC3: 30-paper working set + section_overrides for 3 of them
        produces a result where those 3 land in the override sections and
        other papers are unchanged."""
        bibcodes = [f"2024P{i:02d}" for i in range(30)]
        papers_rows = [(b, f"T{b}", 2024, f"abs{b}") for b in bibcodes]
        # All 30 papers in modal community 1 -> would all go to 'background'.
        intent_rows: list[tuple] = []
        community_rows = [(b, 1, "L") for b in bibcodes]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        overrides = {
            "2024P00": "methods",
            "2024P01": "results",
            "2024P02": "background",  # already where it would land
        }

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
            section_overrides=overrides,
        )

        # Build {bibcode: section_name} from the result.
        bib_to_section = {
            p["bibcode"]: s.name for s in result.sections for p in s.cited_papers
        }
        assert bib_to_section["2024P00"] == "methods"
        assert bib_to_section["2024P01"] == "results"
        assert bib_to_section["2024P02"] == "background"

        # The other 27 papers should still land in 'background' (modal community).
        other_bibcodes = [b for b in bibcodes if b not in overrides]
        for b in other_bibcodes:
            assert bib_to_section[b] == "background"

        # Overridden papers carry signal_used='override'.
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024P00")
        assert row["signal_used"] == "override"

    def test_override_to_unknown_section_is_ignored(self) -> None:
        """If the override targets a section that isn't in the requested
        sections list, the paper falls through to normal rules."""
        papers_rows = [("2024A", "A", 2024, "abs")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
            section_overrides={"2024A": "not_a_real_section"},
        )
        # Falls back to intent-modal -> methods.
        methods = next(s for s in result.sections if s.name == "methods")
        assert any(p["bibcode"] == "2024A" for p in methods.cited_papers)
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        assert row["signal_used"] == "intent_modal"

    def test_overrides_with_non_string_keys_or_values_skipped(self) -> None:
        """Defensive: malformed override dict entries don't crash."""
        papers_rows = [("2024A", "A", 2024, "abs")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        # Cast to silence type checkers — we're testing runtime defense.
        bad_overrides = {123: "methods", "2024A": 456}  # type: ignore[dict-item]
        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
            section_overrides=bad_overrides,  # type: ignore[arg-type]
        )
        # The bad entries are skipped; paper falls back to intent-modal.
        methods = next(s for s in result.sections if s.name == "methods")
        assert any(p["bibcode"] == "2024A" for p in methods.cited_papers)


# ---------------------------------------------------------------------------
# MCP wiring for section_overrides (bead scix_experiments-gtsx)
# ---------------------------------------------------------------------------


class TestMCPDispatchOverrides:
    def test_dispatch_accepts_section_overrides(self) -> None:
        papers_rows = [("2024A", "T", 2024, "abs")]
        intent_rows: list[tuple] = []
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        out = _dispatch_tool(
            conn,
            "synthesize_findings",
            {
                "working_set_bibcodes": ["2024A"],
                "section_overrides": {"2024A": "methods"},
            },
        )
        result = json.loads(out)
        methods = next(s for s in result["sections"] if s["name"] == "methods")
        assert any(p["bibcode"] == "2024A" for p in methods["cited_papers"])
        row = next(p for p in methods["cited_papers"] if p["bibcode"] == "2024A")
        assert row["signal_used"] == "override"
        # Signals payload survives the JSON round-trip.
        assert "signals" in row
        assert "alternative_sections" in row

    def test_dispatch_rejects_non_dict_section_overrides(self) -> None:
        conn = MagicMock()
        out = _dispatch_tool(
            conn,
            "synthesize_findings",
            {"working_set_bibcodes": ["2024A"], "section_overrides": ["not", "a", "dict"]},
        )
        result = json.loads(out)
        assert "error" in result
