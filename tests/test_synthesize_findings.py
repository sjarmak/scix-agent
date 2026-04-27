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
            ("2024A", "Method paper", 2024, "An abstract about methods.", 0),
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
        papers_rows = [("2024B", "Replication paper", 2024, "abs", 0)]
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
            ("2024X", "Paper X", 2024, "abs X", 0),
            ("2024Y", "Paper Y", 2024, "abs Y", 0),
            ("2024Z", "Paper Z", 2024, "abs Z", 0),
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

    def test_no_intent_no_community_falls_back_or_unattributed(self) -> None:
        """Pre-spj0 contract: orphan paper -> ``unattributed_bibcodes``.

        Post-spj0 contract: orphan paper is eligible for the citation-count
        fallback (Tier 3) when at least one section is empty after Tiers
        1-2. With a non-zero ``max_papers_per_section`` (default 8 -> cap 4)
        the paper lands in the first empty section. With a tiny cap (1)
        the fallback is disabled (``1 // 2 == 0``) and the paper remains
        in unattributed.
        """
        papers_rows = [("2024U", "Orphan paper", 2024, "abs", 0)]
        intent_rows: list[tuple] = []
        community_rows: list[tuple] = []  # no metrics row at all
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        # cap == 0 path: no fallback, paper unattributed.
        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024U"],
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=1,  # 1 // 2 == 0, fallback disabled
        )
        assert "2024U" in result.unattributed_bibcodes
        assert all(not s.cited_papers for s in result.sections)

    def test_no_intent_no_community_with_default_cap_pulls_via_fallback(
        self,
    ) -> None:
        """Companion to the test above: at the default cap, the orphan
        paper is fallback-pulled and marked ``citation_count_fallback``."""
        papers_rows = [("2024U", "Orphan paper", 2024, "abs", 0)]
        intent_rows: list[tuple] = []
        community_rows: list[tuple] = []
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024U"],
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=8,
        )
        # Paper now lives in the first empty section (background).
        assert "2024U" not in result.unattributed_bibcodes
        all_rows = {p["bibcode"]: p for s in result.sections for p in s.cited_papers}
        assert all_rows["2024U"]["signal_used"] == "citation_count_fallback"


# ---------------------------------------------------------------------------
# Deterministic structure
# ---------------------------------------------------------------------------


class TestDeterministicStructure:
    def test_returns_all_requested_sections_even_when_empty(self) -> None:
        """The 4 default sections always appear in the output, in canonical
        order, even when some have zero cited papers."""
        papers_rows = [("2024A", "P", 2024, "a", 0)]
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
        papers_rows = [(f"2024P{i}", f"P{i}", 2024, f"a{i}", 0) for i in range(10)]
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
            ("2024A", "A", 2024, "abs", 0),
            ("2024B", "B", 2024, "abs", 0),
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
        papers_rows = [("2024A", "T", 2024, "abs", 0)]
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

        papers_rows = [("2024A", "T", 2024, "abs", 0)]
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
        papers_rows = [(b, f"Title {b}", 2024, f"abs {b}", 0) for b in bibcodes]
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
        papers_rows = [("2024A", "Method paper", 2024, "abs", 0)]
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
            ("2024X", "X", 2024, "abs", 0),
            ("2024Y", "Y", 2024, "abs", 0),
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
            ("2024A", "A", 2024, "abs", 0),
            ("2024B", "B", 2024, "abs", 0),
            ("2024C", "C", 2024, "abs", 0),
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
        papers_rows = [("2024A", "A", 2024, "abs", 0)]
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
        papers_rows = [(b, f"T{b}", 2024, f"abs{b}", 0) for b in bibcodes]
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
        bib_to_section = {p["bibcode"]: s.name for s in result.sections for p in s.cited_papers}
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
        papers_rows = [("2024A", "A", 2024, "abs", 0)]
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
        papers_rows = [("2024A", "A", 2024, "abs", 0)]
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
        papers_rows = [("2024A", "T", 2024, "abs", 0)]
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


# ---------------------------------------------------------------------------
# Empty-section citation-count fallback (bead scix_experiments-spj0)
# ---------------------------------------------------------------------------


class TestEmptySectionFallback:
    """AC1-4: when a section is empty after intent + community tiers, fill it
    from the unattributed working-set papers sorted by citation_count desc,
    capped at max_papers_per_section // 2.

    Surfaced 2026-04-27 by the lit-review demo: ``citation_contexts.intent``
    only covers ~0.27% of edges, so the ``results`` section frequently came
    out empty even when high-citation results papers were present in the
    working set.
    """

    def test_results_filled_by_citation_count_when_no_intent(self) -> None:
        """AC4: 30-paper working set with no result_comparison intent rows
        still produces a non-empty results section, populated by the
        highest-citation unattributed papers tagged
        ``signal_used='citation_count_fallback'``.

        Realistic shape: tiers 1-2 fill ``background`` (community
        fall-through) and ``methods`` (a few intent rows); ``results``
        and ``open_questions`` are empty until Tier 3 fires.
        """
        bibcodes = [f"2024P{i:02d}" for i in range(30)]
        # citation_count varies — top citers will land in the fallback pull.
        papers_rows = [
            (b, f"Title {b}", 2024, f"abs {b}", 100 - i)  # P00=100, P01=99, ...
            for i, b in enumerate(bibcodes)
        ]
        # No result_comparison intent rows anywhere. P00 -> methods.
        intent_rows = [
            ("2024P00", "method", 1),
        ]
        # P10..P19 attributed via community to 'background' (modal community).
        # P00 (excluding the intent-pinned P00) and P01..P09 + P20..P29 are
        # all unattributed and eligible for Tier 3 fallback. P01 (cit=99)
        # is the highest-citation eligible candidate.
        community_rows = [(f"2024P{i:02d}", 1, "Modal") for i in range(10, 20)]

        conn = _mock_conn([papers_rows, intent_rows, community_rows])
        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=8,
        )

        results_section = next(s for s in result.sections if s.name == "results")
        # AC1: results section is non-empty.
        assert len(results_section.cited_papers) > 0
        # AC2: every paper in the fallback-only section is tagged correctly.
        for row in results_section.cited_papers:
            assert row["signal_used"] == "citation_count_fallback"
        bibs_in_results = {p["bibcode"] for p in results_section.cited_papers}
        # P00 is intent-attributed to methods, must NOT be in the fallback.
        assert "2024P00" not in bibs_in_results
        # Community-attributed papers must NOT be in the fallback either.
        assert bibs_in_results.isdisjoint({f"2024P{i:02d}" for i in range(10, 20)})
        # Highest-citation unattributed paper (P01, citation_count=99) lands
        # in the FIRST empty section processed in canonical order. That's
        # ``results`` here because background was filled by community
        # fall-through and methods by the intent row.
        assert "2024P01" in bibs_in_results

    def test_fallback_capped_at_half_max(self) -> None:
        """AC1: fallback-pulled papers per section <= max_papers_per_section // 2.
        Tests integer floor: 7 // 2 == 3, 8 // 2 == 4.
        """
        bibcodes = [f"2024Q{i:02d}" for i in range(20)]
        papers_rows = [(b, f"T{b}", 2024, f"a{b}", 50 - i) for i, b in enumerate(bibcodes)]
        intent_rows: list[tuple] = []  # nothing attributed via intent
        community_rows: list[tuple] = []  # nothing attributed via community
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=8,
        )

        # All 4 sections were empty after primary bucketing, so all 4 should
        # have fallback pulls — each capped at exactly 8 // 2 == 4 since the
        # pool (20) >= sum-of-caps (16). Asserting equality (not <=) catches
        # both over-pull and under-pull regressions.
        for section in result.sections:
            fallback_rows = [
                p for p in section.cited_papers if p.get("signal_used") == "citation_count_fallback"
            ]
            assert len(fallback_rows) == 8 // 2

    def test_fallback_capped_at_half_for_odd_max(self) -> None:
        """Floor division on odd cap: 7 // 2 == 3."""
        bibcodes = [f"2024R{i:02d}" for i in range(20)]
        papers_rows = [(b, f"T{b}", 2024, f"a{b}", 50 - i) for i, b in enumerate(bibcodes)]
        intent_rows: list[tuple] = []
        community_rows: list[tuple] = []
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=7,
        )
        # Pool (20) >= sum-of-caps (4 sections * 3 each = 12); exact pulls.
        for section in result.sections:
            fallback_rows = [
                p for p in section.cited_papers if p.get("signal_used") == "citation_count_fallback"
            ]
            assert len(fallback_rows) == 7 // 2  # 3

    def test_fallback_excludes_already_attributed(self) -> None:
        """AC3: papers attributed via intent or community must NOT be
        fallback-pulled into other (empty) sections."""
        bibcodes = [f"2024S{i:02d}" for i in range(10)]
        papers_rows = [(b, f"T{b}", 2024, f"a{b}", 100 - i) for i, b in enumerate(bibcodes)]
        # First 3 papers attributed via intent (high citation_count would make
        # them attractive fallback candidates if the rule were broken).
        intent_rows = [
            ("2024S00", "method", 1),
            ("2024S01", "method", 1),
            ("2024S02", "background", 1),
        ]
        # Next 3 attributed via community (modal -> background).
        community_rows = [
            ("2024S03", 1, "Lbl"),
            ("2024S04", 1, "Lbl"),
            ("2024S05", 1, "Lbl"),
        ]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=8,
        )

        # 'results' is empty after primary tiers — should be fallback-filled
        # ONLY from the remaining 4 unattributed papers (S06, S07, S08, S09).
        results_section = next(s for s in result.sections if s.name == "results")
        bibs_in_results = {p["bibcode"] for p in results_section.cited_papers}
        attributed = {f"2024S0{i}" for i in range(6)}  # S00..S05
        assert bibs_in_results.isdisjoint(
            attributed
        ), "fallback must not poach intent/community-attributed papers"
        # All papers in results are fallback-pulled.
        for row in results_section.cited_papers:
            assert row["signal_used"] == "citation_count_fallback"

    def test_coverage_block_has_fallback_pulled_per_section(self) -> None:
        """AC2: coverage dict exposes ``fallback_pulled_per_section`` as a
        ``{section_name: int}`` mapping showing how much of each section is
        secondary signal vs primary."""
        bibcodes = [f"2024T{i:02d}" for i in range(10)]
        papers_rows = [(b, f"T{b}", 2024, f"a{b}", 50 - i) for i, b in enumerate(bibcodes)]
        intent_rows: list[tuple] = []
        community_rows: list[tuple] = []
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=8,
        )

        per_section = result.coverage["fallback_pulled_per_section"]
        assert isinstance(per_section, dict)
        # Every requested section appears as a key (even if 0).
        assert set(per_section.keys()) == set(DEFAULT_SECTIONS)
        # All values are non-negative ints.
        for k, v in per_section.items():
            assert isinstance(v, int)
            assert v >= 0
        # The wire format also surfaces it (via to_dict()).
        wire = result.to_dict()
        assert "fallback_pulled_per_section" in wire["assignment_coverage"]

    def test_overrides_are_not_eligible_for_fallback(self) -> None:
        """AC3 mirror: a paper pinned via section_overrides to one section
        must NOT be fallback-pulled into another (even-empty) section."""
        bibcodes = [f"2024V{i:02d}" for i in range(10)]
        papers_rows = [(b, f"T{b}", 2024, f"a{b}", 100 - i) for i, b in enumerate(bibcodes)]
        intent_rows: list[tuple] = []
        community_rows: list[tuple] = []
        # Pin the highest-citation paper to 'methods' via override.
        overrides = {"2024V00": "methods"}
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=8,
            section_overrides=overrides,
        )

        # 'results' is empty after primary tiers — fallback should NOT pull V00.
        results_section = next(s for s in result.sections if s.name == "results")
        bibs_in_results = {p["bibcode"] for p in results_section.cited_papers}
        assert "2024V00" not in bibs_in_results

        # V00 still appears in 'methods' with signal_used='override'.
        methods = next(s for s in result.sections if s.name == "methods")
        v00_row = next((p for p in methods.cited_papers if p["bibcode"] == "2024V00"), None)
        assert v00_row is not None
        assert v00_row["signal_used"] == "override"

    def test_no_unattributed_means_no_fallback(self) -> None:
        """If all papers are already attributed via tiers 0-2, fallback is
        a no-op (empty pool). Coverage shows 0 for every section."""
        bibcodes = [f"2024W{i:02d}" for i in range(5)]
        papers_rows = [(b, f"T{b}", 2024, f"a{b}", 10) for b in bibcodes]
        # Every paper has a method intent -> all attributed to 'methods'.
        intent_rows = [(b, "method", 1) for b in bibcodes]
        community_rows: list[tuple] = []
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=8,
        )
        per_section = result.coverage["fallback_pulled_per_section"]
        assert all(v == 0 for v in per_section.values())

    def test_fallback_cap_one_pulls_exactly_one(self) -> None:
        """Boundary: max_papers_per_section=2 -> cap=1 -> exactly one paper
        per empty section. Smallest non-zero cap; the most likely place for
        an off-by-one regression in remaining[:cap]."""
        bibcodes = [f"2024X{i:02d}" for i in range(20)]
        papers_rows = [
            (b, f"T{b}", 2024, f"a{b}", 100 - i) for i, b in enumerate(bibcodes)
        ]
        intent_rows: list[tuple] = []
        community_rows: list[tuple] = []
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=2,
        )
        # All 4 sections were empty after primary bucketing; pool (20) >>
        # sum-of-caps (4); each section receives exactly 1.
        for section in result.sections:
            fallback_rows = [
                p
                for p in section.cited_papers
                if p.get("signal_used") == "citation_count_fallback"
            ]
            assert len(fallback_rows) == 1

    def test_coverage_invariant_assigned_equals_sum_of_tiers(self) -> None:
        """Coverage invariant: assigned_bibcodes equals the sum of
        intent + community + override + fallback_pulled. Catches future
        accounting drift where a tier increments its counter but forgets to
        remove the bibcode from `unattributed` (or vice versa)."""
        # Mix of all four tiers in one working set.
        bibcodes = [f"2024Y{i:02d}" for i in range(10)]
        papers_rows = [
            (b, f"T{b}", 2024, f"a{b}", 100 - i) for i, b in enumerate(bibcodes)
        ]
        # Y00 -> intent (methods).
        intent_rows = [("2024Y00", "method", 1)]
        # Y01-Y04 -> community (modal community 1 -> background).
        community_rows = [(f"2024Y{i:02d}", 1, "Modal") for i in range(1, 5)]
        # Y05 -> pinned to 'open_questions' via override.
        # Y06-Y09 -> unattributed; eligible for fallback to fill 'results'.
        overrides = {"2024Y05": "open_questions"}
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=8,
            section_overrides=overrides,
        )
        c = result.coverage
        assert (
            c["intent_assigned_bibcodes"]
            + c["community_assigned_bibcodes"]
            + c["override_assigned_bibcodes"]
            + c["fallback_pulled_bibcodes"]
            == c["assigned_bibcodes"]
        )
