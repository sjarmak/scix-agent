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
    _CORE_SHARE_THRESHOLD,
    _SUPPORTING_SHARE_THRESHOLD,
    _classify_share_tier,
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
    def test_paper_in_core_community_lands_in_background(self) -> None:
        """No intent coverage; community signal alone decides section.

        Post-37wj: a community is 'core' iff its share of the working set
        is >= ``_CORE_SHARE_THRESHOLD`` (0.15). With X papers in community
        5 and Z alone in community 99, the working set is 18 X + 1 Z = 19
        total, so:

        - community 5 has share 18/19 = 0.947 -> core -> background
        - community 99 has share 1/19 = 0.053 -> supporting -> methods
          (intent-free supporting routes to ``methods`` per AC1)
        """
        bibcodes = [f"2024X{i:02d}" for i in range(18)] + ["2024Z"]
        papers_rows = [(b, f"Paper {b}", 2024, f"abs {b}", 0) for b in bibcodes]
        intent_rows: list[tuple] = []  # no intent coverage at all
        # First 18 in modal community 5; Z alone in community 99.
        community_rows = [(b, 5, "Galaxies") for b in bibcodes[:18]]
        community_rows.append(("2024Z", 99, "Plasma"))
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
        )

        background = next(s for s in result.sections if s.name == "background")
        methods = next(s for s in result.sections if s.name == "methods")

        bg_bibcodes = {p["bibcode"] for p in background.cited_papers}
        methods_bibcodes = {p["bibcode"] for p in methods.cited_papers}

        # Core community papers go to background.
        assert set(bibcodes[:18]).issubset(bg_bibcodes)
        # Supporting community paper (Z, share=0.053) routes to methods.
        assert "2024Z" in methods_bibcodes

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
        # Under the post-37wj weighted classifier with 30-paper working set:
        #   community 1: 20/30 = 0.667 share -> core -> background
        #   community 2: 5/30  = 0.167 share -> core -> background (also)
        # Both communities are 'core' here, so open_questions stays empty
        # via Tier-2 — it's only reachable now via override or Tier-3
        # fallback, which Tier-3 may exercise on the 5 community-less papers.
        # 5 papers have NO community row (-> unattributed -> Tier-3 candidates).
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

        # 2024C — sole member of community 2; share = 1/3 = 0.333 (core)
        # under the post-37wj weighted classifier so it routes to
        # 'background' as a core-tier paper (NOT 'open_questions' as the
        # pre-37wj binary modal/non-modal rule would have).
        row_c = all_rows["2024C"]
        assert row_c["signal_used"] == "community_fallthrough"
        assert row_c["signals"]["community_id"] == 2
        assert row_c["signals"]["is_modal_community"] is False
        assert row_c["signals"]["modal_community_id"] == 1
        assert row_c["signals"]["share_tier"] == "core"
        assert row_c["section_assigned"] == "background"

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


# ---------------------------------------------------------------------------
# Weighted core/supporting/peripheral classifier (bead scix_experiments-37wj)
# ---------------------------------------------------------------------------


class TestWeightedShareClassifier:
    """AC1-4: Tier 2 (community fall-through) routes papers by their
    community's share of the working set, not by binary modal/non-modal.

    - share >= 0.15 -> 'core' -> background
    - 0.05 <= share < 0.15 -> 'supporting' -> methods (or background overflow)
    - share < 0.05 -> 'peripheral' -> unattributed (eligible for Tier 3)
    """

    def test_share_thresholds_are_module_constants(self) -> None:
        """AC4: thresholds are constants in the module — easy to tune."""
        assert _CORE_SHARE_THRESHOLD == 0.15
        assert _SUPPORTING_SHARE_THRESHOLD == 0.05

    def test_classify_share_tier_core_boundary(self) -> None:
        """share >= 0.15 -> 'core'."""
        assert _classify_share_tier(0.15) == "core"
        assert _classify_share_tier(1.0) == "core"
        assert _classify_share_tier(0.5) == "core"

    def test_classify_share_tier_supporting_boundary(self) -> None:
        """0.05 <= share < 0.15 -> 'supporting'."""
        assert _classify_share_tier(0.05) == "supporting"
        assert _classify_share_tier(0.14999) == "supporting"
        assert _classify_share_tier(0.10) == "supporting"

    def test_classify_share_tier_peripheral_boundary(self) -> None:
        """share < 0.05 -> 'peripheral'."""
        assert _classify_share_tier(0.04999) == "peripheral"
        assert _classify_share_tier(0.0) == "peripheral"
        assert _classify_share_tier(0.01) == "peripheral"

    def test_supporting_community_routes_to_methods(self) -> None:
        """AC1: a supporting community (share in [0.05, 0.15)) routes to
        ``methods`` rather than ``open_questions``.

        Fixture: 20 papers in community A (core, 19/20=0.95) + 1 paper in
        community B (supporting, 1/20=0.05). Under the binary modal-only
        rule the lone B paper went to ``open_questions``. Under the
        weighted rule it lands in ``methods``.
        """
        bibcodes = [f"2024A{i:02d}" for i in range(19)] + ["2024B"]
        papers_rows = [(b, f"T{b}", 2024, f"abs{b}", 0) for b in bibcodes]
        intent_rows: list[tuple] = []
        community_rows = [(b, 1, "Modal") for b in bibcodes[:19]]
        community_rows.append(("2024B", 2, "Supporting"))
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
        )
        bib_to_section = {p["bibcode"]: s.name for s in result.sections for p in s.cited_papers}
        # Core community papers in background.
        for b in bibcodes[:19]:
            assert bib_to_section[b] == "background"
        # Supporting community paper routes to methods.
        assert bib_to_section["2024B"] == "methods"
        # And carries community_fallthrough as signal_used (wire-stable per AC2).
        methods = next(s for s in result.sections if s.name == "methods")
        b_row = next(p for p in methods.cited_papers if p["bibcode"] == "2024B")
        assert b_row["signal_used"] == "community_fallthrough"

    def test_peripheral_community_lands_in_unattributed_then_tier3(self) -> None:
        """A community below the supporting threshold (share < 0.05) is
        peripheral. Its papers are NOT routed by Tier 2 — they fall into
        the unattributed pool, which makes them eligible for Tier 3
        citation-count fallback.

        Fixture: 25 papers in community A (core, 25/26=0.96) + 1 paper in
        community B (peripheral, 1/26=0.038). With no intent rows on the B
        paper and an empty 'results' section, Tier 3 should fallback-pull
        it from unattributed.
        """
        bibcodes = [f"2024A{i:02d}" for i in range(25)] + ["2024B"]
        papers_rows = [
            (b, f"T{b}", 2024, f"abs{b}", 100 if b == "2024B" else 0)
            for b in bibcodes
        ]
        intent_rows: list[tuple] = []
        community_rows = [(b, 1, "Modal") for b in bibcodes[:25]]
        community_rows.append(("2024B", 2, "Peripheral"))
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=8,
        )
        # 2024B has highest citation_count (100) -> first to be Tier 3
        # fallback-pulled into the first empty section. Section iteration
        # order is DEFAULT_SECTIONS = (background, methods, results,
        # open_questions); background was filled by core community A, so
        # 'methods' is the first empty section and B lands there.
        all_rows = {p["bibcode"]: p for s in result.sections for p in s.cited_papers}
        assert "2024B" in all_rows
        # Tier 3 (fallback) was used, not Tier 2 (community_fallthrough).
        assert all_rows["2024B"]["signal_used"] == "citation_count_fallback"
        assert all_rows["2024B"]["section_assigned"] == "methods"

    def test_share_tier_in_signals_payload(self) -> None:
        """AC2: each fall-through-assigned paper exposes ``share_tier`` in
        its signals payload."""
        bibcodes = [f"2024A{i:02d}" for i in range(19)] + ["2024B"]
        papers_rows = [(b, f"T{b}", 2024, f"abs{b}", 0) for b in bibcodes]
        intent_rows: list[tuple] = []
        community_rows = [(b, 1, "Modal") for b in bibcodes[:19]]
        community_rows.append(("2024B", 2, "Supporting"))
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
        )
        all_rows = {p["bibcode"]: p for s in result.sections for p in s.cited_papers}
        # Core community papers get share_tier='core'.
        for b in bibcodes[:19]:
            assert all_rows[b]["signals"]["share_tier"] == "core"
        # Supporting community paper gets share_tier='supporting'.
        assert all_rows["2024B"]["signals"]["share_tier"] == "supporting"

    def test_share_tier_none_when_no_community(self) -> None:
        """A paper with no community membership (no row in
        ``community_map``) has ``share_tier`` set to ``None`` — not
        omitted, so the schema is uniform across rows."""
        papers_rows = [("2024U", "Orphan", 2024, "abs", 50)]
        intent_rows: list[tuple] = []
        community_rows: list[tuple] = []
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024U"],
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=8,
        )
        # Paper falls to Tier 3 fallback. Signals.share_tier should be None.
        all_rows = {p["bibcode"]: p for s in result.sections for p in s.cited_papers}
        assert all_rows["2024U"]["signals"]["share_tier"] is None

    def test_weighted_classifier_differs_from_modal_only(self) -> None:
        """AC3 verbatim: a working set with two ~equal-share communities
        produces a different distribution than the current modal-only rule
        (specifically: papers from the second-largest community get
        bucketed into supporting tiers, not all dumped into open_questions).

        Fixture: 12 papers — community A (7, share=7/12=0.583, core) and
        community B (5, share=5/12=0.417, also core under weighted rule).
        Wait — both are core under the weighted classifier with the
        defaults, since 0.417 >= 0.15. To produce the AC3 contrast, scale
        community B down to land in [0.05, 0.15). Use community A=12,
        community B=2 (share=2/14=0.143, supporting). Under modal-only,
        B would dump in open_questions; under weighted, B routes to
        methods.
        """
        bibcodes = [f"2024A{i:02d}" for i in range(12)] + [
            "2024B0",
            "2024B1",
        ]
        papers_rows = [(b, f"T{b}", 2024, f"abs{b}", 0) for b in bibcodes]
        intent_rows: list[tuple] = []
        community_rows = [(b, 1, "Modal") for b in bibcodes[:12]]
        community_rows.append(("2024B0", 2, "Supporting"))
        community_rows.append(("2024B1", 2, "Supporting"))
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
        )
        bib_to_section = {p["bibcode"]: s.name for s in result.sections for p in s.cited_papers}
        # Old (modal-only) behavior would have routed both B papers to
        # open_questions. New (weighted) routes them to methods.
        assert bib_to_section["2024B0"] == "methods"
        assert bib_to_section["2024B1"] == "methods"
        # And explicitly: the open_questions section should NOT contain
        # them (would only happen under the old rule).
        oq = next(s for s in result.sections if s.name == "open_questions")
        oq_bibs = {p["bibcode"] for p in oq.cited_papers}
        assert "2024B0" not in oq_bibs
        assert "2024B1" not in oq_bibs

    def test_supporting_falls_back_to_background_when_methods_not_requested(
        self,
    ) -> None:
        """When the requested ``sections`` list does not include
        ``methods``, supporting community papers route to ``background``
        (the "overflow" rung of the AC1 ladder)."""
        bibcodes = [f"2024A{i:02d}" for i in range(19)] + ["2024B"]
        papers_rows = [(b, f"T{b}", 2024, f"abs{b}", 0) for b in bibcodes]
        intent_rows: list[tuple] = []
        community_rows = [(b, 1, "Modal") for b in bibcodes[:19]]
        community_rows.append(("2024B", 2, "Supporting"))
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        # Custom sections list — no 'methods'.
        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=["background", "open_questions"],
            max_papers_per_section=30,
        )
        bib_to_section = {p["bibcode"]: s.name for s in result.sections for p in s.cited_papers}
        # Supporting paper overflows to background.
        assert bib_to_section["2024B"] == "background"

    def test_single_community_full_share_is_core(self) -> None:
        """Backwards-compat: a working set entirely in a single community
        (share=1.0) is 'core' and routes to background — same outcome as
        the old modal=background rule."""
        bibcodes = [f"2024A{i:02d}" for i in range(5)]
        papers_rows = [(b, f"T{b}", 2024, f"abs{b}", 0) for b in bibcodes]
        intent_rows: list[tuple] = []
        community_rows = [(b, 1, "Solo") for b in bibcodes]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
        )
        all_rows = {p["bibcode"]: p for s in result.sections for p in s.cited_papers}
        for b in bibcodes:
            assert all_rows[b]["signals"]["share_tier"] == "core"
        bg = next(s for s in result.sections if s.name == "background")
        bg_bibs = {p["bibcode"] for p in bg.cited_papers}
        assert set(bibcodes).issubset(bg_bibs)


# ---------------------------------------------------------------------------
# Structured theme payload (bead scix_experiments-4la8)
# ---------------------------------------------------------------------------


class TestSectionTheme:
    """AC1-4: each section emits a structured ``theme`` object alongside
    the legacy ``theme_summary`` string.

    The synthesis-writing agent ignored the formatted-label string entirely
    per the lit-review demo (2026-04-27), so this bead exposes the raw
    signals (community membership counts, top arxiv classes, top keywords,
    top-cited papers in section) for the agent to compose its own thematic
    framing.
    """

    def test_theme_communities_sorted_by_paper_count_desc(self) -> None:
        """AC2: communities[] is sorted by paper_count_in_section desc.

        Fixture: 4 communities with sizes 6 / 4 / 3 / 2 in the section.
        Assert the ordering of the first three entries (capped at top 3).
        """
        # 15 papers in one core community (share=15/15=1.0 -> background).
        # All in same section so we can test cross-community aggregation.
        bibcodes = (
            [f"2024C1_{i:02d}" for i in range(6)]
            + [f"2024C2_{i:02d}" for i in range(4)]
            + [f"2024C3_{i:02d}" for i in range(3)]
            + [f"2024C4_{i:02d}" for i in range(2)]
        )
        papers_rows = [
            (b, f"Title {b}", 2024, f"abs {b}", 0, ["astro-ph.GA"], ["galaxies"])
            for b in bibcodes
        ]
        intent_rows: list[tuple] = []
        # All 4 communities are 'core' under the weighted classifier (smallest
        # share is 2/15=0.133, but below threshold 0.15 -> 'supporting').
        # To get all into background, we pin via overrides.
        community_rows = []
        for i, b in enumerate(bibcodes):
            if i < 6:
                community_rows.append((b, 1, "Lbl1"))
            elif i < 10:
                community_rows.append((b, 2, "Lbl2"))
            elif i < 13:
                community_rows.append((b, 3, "Lbl3"))
            else:
                community_rows.append((b, 4, "Lbl4"))
        # Pin everyone to 'background' to guarantee all 4 communities show up
        # in the same section's theme.
        overrides = {b: "background" for b in bibcodes}
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
            section_overrides=overrides,
        )
        bg = next(s for s in result.sections if s.name == "background")
        theme = bg.theme
        comms = theme["communities"]
        # AC2: capped at 3 entries.
        assert len(comms) == 3
        # AC2: sorted desc by paper_count_in_section.
        assert comms[0]["community_id"] == 1 and comms[0]["paper_count_in_section"] == 6
        assert comms[1]["community_id"] == 2 and comms[1]["paper_count_in_section"] == 4
        assert comms[2]["community_id"] == 3 and comms[2]["paper_count_in_section"] == 3

    def test_theme_communities_capped_at_three(self) -> None:
        """AC2: with 5 communities in one section, only the top-3 by
        paper_count_in_section appear in theme.communities."""
        bibcodes = []
        for cid, n in [(1, 8), (2, 5), (3, 4), (4, 3), (5, 2)]:
            bibcodes.extend([f"2024C{cid}_{i:02d}" for i in range(n)])
        papers_rows = [
            (b, f"T{b}", 2024, f"a{b}", 0, ["astro-ph.GA"], ["x"]) for b in bibcodes
        ]
        intent_rows: list[tuple] = []
        community_rows = []
        offset = 0
        for cid, n in [(1, 8), (2, 5), (3, 4), (4, 3), (5, 2)]:
            for i in range(n):
                community_rows.append((bibcodes[offset + i], cid, f"Lbl{cid}"))
            offset += n
        overrides = {b: "background" for b in bibcodes}
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
            section_overrides=overrides,
        )
        bg = next(s for s in result.sections if s.name == "background")
        comms = bg.theme["communities"]
        assert len(comms) == 3
        # Top 3 are communities 1, 2, 3 (highest counts).
        assert [c["community_id"] for c in comms] == [1, 2, 3]

    def test_top_papers_by_citation_has_three_highest(self) -> None:
        """AC1: theme.top_papers_by_citation contains the top-3 papers by
        citation_count (within the section), descending."""
        bibcodes = [f"2024P{i:02d}" for i in range(8)]
        # Citation counts: P00=10, P01=50, P02=30, P03=99, P04=5, P05=20, P06=99, P07=0
        cits = [10, 50, 30, 99, 5, 20, 99, 0]
        papers_rows = [
            (b, f"Title {b}", 2024, f"abs {b}", c, [], []) for b, c in zip(bibcodes, cits)
        ]
        intent_rows: list[tuple] = []
        community_rows = [(b, 1, "Modal") for b in bibcodes]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
        )
        bg = next(s for s in result.sections if s.name == "background")
        top = bg.theme["top_papers_by_citation"]
        assert len(top) == 3
        # Sorted by citation_count desc; ties broken by bibcode asc.
        assert top[0]["bibcode"] == "2024P03" and top[0]["citation_count"] == 99
        assert top[1]["bibcode"] == "2024P06" and top[1]["citation_count"] == 99
        assert top[2]["bibcode"] == "2024P01" and top[2]["citation_count"] == 50
        # Each entry has the expected keys.
        for entry in top:
            assert set(entry.keys()) >= {"bibcode", "title", "citation_count"}

    def test_theme_empty_section_returns_empty_payload(self) -> None:
        """An empty section emits ``theme.communities == []`` and
        ``theme.top_papers_by_citation == []`` — no crash."""
        # Single paper attributed via intent to 'methods'; other sections
        # remain empty after primary tiers; with cap=1, fallback is disabled.
        papers_rows = [("2024A", "T", 2024, "abs", 0, [], [])]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=1,  # disables Tier-3 fallback
        )
        # 'results' is empty under this fixture.
        results_section = next(s for s in result.sections if s.name == "results")
        assert results_section.cited_papers == []
        assert results_section.theme["communities"] == []
        assert results_section.theme["top_papers_by_citation"] == []

    def test_theme_aggregates_arxiv_classes_from_section_papers(self) -> None:
        """AC1: theme.communities[].top_arxiv_classes is derived from the
        section's papers' arxiv_class arrays.

        Fixture: 4 papers in one community, 2 with ['astro-ph.EP'] and 2 with
        ['astro-ph.SR']. Top classes should expose both.
        """
        bibcodes = [f"2024A{i:02d}" for i in range(4)]
        arxiv_classes = [
            ["astro-ph.EP"],
            ["astro-ph.EP"],
            ["astro-ph.SR"],
            ["astro-ph.SR"],
        ]
        papers_rows = [
            (b, f"Planet paper {i}", 2024, "abs", 0, ax, ["jupiter", "saturn"])
            for i, (b, ax) in enumerate(zip(bibcodes, arxiv_classes))
        ]
        intent_rows: list[tuple] = []
        community_rows = [(b, 7, "Planets") for b in bibcodes]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
        )
        bg = next(s for s in result.sections if s.name == "background")
        comms = bg.theme["communities"]
        assert len(comms) == 1
        comm = comms[0]
        assert comm["community_id"] == 7
        assert comm["paper_count_in_section"] == 4
        # Both arxiv classes appear.
        top_arxiv = comm["top_arxiv_classes"]
        assert "astro-ph.EP" in top_arxiv
        assert "astro-ph.SR" in top_arxiv
        # And keywords come through too.
        top_kw = comm["top_keywords"]
        assert "jupiter" in top_kw
        assert "saturn" in top_kw

    def test_theme_summary_string_remains_for_backwards_compat(self) -> None:
        """The legacy ``theme_summary`` string is still emitted for
        backwards compat with pre-4la8 MCP clients. New ``theme`` field is
        additive."""
        papers_rows = [
            ("2024A", "T", 2024, "abs", 0, [], []),
            ("2024B", "T", 2024, "abs", 0, [], []),
        ]
        intent_rows: list[tuple] = []
        community_rows = [
            ("2024A", 1, "Cosmology"),
            ("2024B", 1, "Cosmology"),
        ]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A", "2024B"],
            sections=list(DEFAULT_SECTIONS),
        )
        bg = next(s for s in result.sections if s.name == "background")
        # Legacy field still populated.
        assert bg.theme_summary == "Cosmology"
        # New field also populated.
        assert isinstance(bg.theme, dict)
        assert "communities" in bg.theme
        assert "top_papers_by_citation" in bg.theme

    def test_theme_keyword_fallback_uses_title_tokens(self) -> None:
        """When all section papers' keyword arrays are empty, top_keywords
        falls back to title-token aggregation (per labels pipeline note in
        CLAUDE.md memory community_labels_pipeline.md)."""
        bibcodes = [f"2024K{i:02d}" for i in range(3)]
        # No keywords; titles share tokens "jupiter atmosphere".
        papers_rows = [
            (bibcodes[0], "Jupiter atmosphere dynamics", 2024, "abs", 0, [], []),
            (bibcodes[1], "Atmosphere of jupiter measured", 2024, "abs", 0, [], []),
            (bibcodes[2], "Saturn atmosphere different", 2024, "abs", 0, [], []),
        ]
        intent_rows: list[tuple] = []
        community_rows = [(b, 1, "Planets") for b in bibcodes]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=bibcodes,
            sections=list(DEFAULT_SECTIONS),
            max_papers_per_section=30,
        )
        bg = next(s for s in result.sections if s.name == "background")
        comm = bg.theme["communities"][0]
        # 'atmosphere' should appear (3x) and 'jupiter' (2x).
        top_kw = comm["top_keywords"]
        assert "atmosphere" in top_kw
        assert "jupiter" in top_kw

    def test_theme_in_wire_format_via_to_dict(self) -> None:
        """AC1 wire: ``theme`` is serialised via ``to_dict()`` so MCP clients
        receive it. ``theme_summary`` continues to be emitted alongside."""
        papers_rows = [("2024A", "T", 2024, "abs", 5, ["astro-ph.GA"], ["galaxies"])]
        intent_rows: list[tuple] = []
        community_rows = [("2024A", 1, "Galaxies")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
        )
        wire = result.to_dict()
        bg_wire = next(s for s in wire["sections"] if s["name"] == "background")
        assert "theme" in bg_wire
        assert "theme_summary" in bg_wire
        assert isinstance(bg_wire["theme"], dict)
        assert isinstance(bg_wire["theme"]["communities"], list)
        assert isinstance(bg_wire["theme"]["top_papers_by_citation"], list)


# ---------------------------------------------------------------------------
# Additive grounding fields (bead scix_experiments-tq0t)
# ---------------------------------------------------------------------------


class TestAdditiveGroundingFields:
    """AC1-3: lit-review demo (2026-04-27) revealed three grounding gaps:

    1. ``first_author`` missing from cited_papers — agent had to parse it
       from the abstract and got it wrong once.
    2. Abstract snippets cap at ~280 chars — agent had to call
       ``read_paper`` separately for full text.
    3. No citation context texts in output — bucket assignments via
       ``intent`` are opaque without at least one citing sentence.

    Fix is additive — both new kwargs default ``False`` so the default
    wire format is unchanged.
    """

    # -- AC1: first_author always present -------------------------------------

    def test_first_author_present_in_cited_papers(self) -> None:
        """AC1: every cited_papers row has a ``first_author`` field
        populated from ``papers.first_author``."""
        # 8-tuple fixture: bibcode, title, year, abstract, citation_count,
        # arxiv_class, keywords, first_author
        papers_rows = [
            ("2024A", "T", 2024, "abs", 0, [], [], "Breu, S."),
        ]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        assert row["first_author"] == "Breu, S."

    def test_first_author_none_when_db_value_null(self) -> None:
        """AC1 robustness: NULL ``first_author`` (some old papers lack it)
        surfaces as ``None`` rather than crashing."""
        papers_rows = [("2024A", "T", 2024, "abs", 0, [], [], None)]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        assert row["first_author"] is None

    def test_first_author_in_top_papers_by_citation(self) -> None:
        """AC1 consistency: theme.top_papers_by_citation entries also
        carry ``first_author`` (parallel to bibcode/title/citation_count)."""
        papers_rows = [
            ("2024A", "T", 2024, "abs", 50, [], [], "Smith, J."),
            ("2024B", "T", 2024, "abs", 30, [], [], "Jones, K."),
        ]
        intent_rows: list[tuple] = []
        community_rows = [("2024A", 1, "L"), ("2024B", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A", "2024B"],
            sections=list(DEFAULT_SECTIONS),
        )
        bg = next(s for s in result.sections if s.name == "background")
        top = bg.theme["top_papers_by_citation"]
        # Each entry exposes first_author so the agent can build "Smith
        # 2024" attribution without parsing abstracts.
        first_authors = {entry["bibcode"]: entry["first_author"] for entry in top}
        assert first_authors["2024A"] == "Smith, J."
        assert first_authors["2024B"] == "Jones, K."

    def test_first_author_default_in_legacy_5tuple_fixtures(self) -> None:
        """Backwards compatibility: pre-tq0t 5-tuple test fixtures (no
        first_author column) still produce a row with ``first_author``
        populated as ``None`` (not missing). Mirrors the wave-4 5/6-tuple
        guard pattern."""
        # 5-tuple legacy fixture (bibcode, title, year, abstract, citation_count)
        papers_rows = [("2024A", "T", 2024, "abs", 0)]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        # Legacy fixture missing first_author -> None.
        assert "first_author" in row
        assert row["first_author"] is None

    # -- AC2: include_full_abstracts kwarg ------------------------------------

    def test_include_full_abstracts_off_by_default(self) -> None:
        """AC2 default: without the kwarg, only ``abstract_snippet`` is
        emitted; no ``abstract_full`` field appears."""
        long_abstract = "X" * 1000  # exceeds the 280-char snippet cap
        papers_rows = [("2024A", "T", 2024, long_abstract, 0, [], [], "Smith")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        assert "abstract_snippet" in row
        # Snippet truncates to <= 280 chars.
        assert len(row["abstract_snippet"]) <= 280
        # No abstract_full field by default — preserves wire format.
        assert "abstract_full" not in row

    def test_include_full_abstracts_adds_full_text_field(self) -> None:
        """AC2 opt-in: with the kwarg, every cited_papers row gets an
        ``abstract_full`` field carrying the untruncated abstract."""
        long_abstract = "Y" * 1000
        papers_rows = [("2024A", "T", 2024, long_abstract, 0, [], [], "Smith")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
            include_full_abstracts=True,
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        # abstract_snippet remains (additive change).
        assert "abstract_snippet" in row
        # New abstract_full field carries the full text.
        assert row["abstract_full"] == long_abstract
        assert len(row["abstract_full"]) == 1000

    def test_include_full_abstracts_handles_empty_abstract(self) -> None:
        """AC2 robustness: a paper with NULL/empty abstract gets
        ``abstract_full = ''`` (consistent with snippet behaviour)."""
        papers_rows = [("2024A", "T", 2024, None, 0, [], [], "Smith")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
            include_full_abstracts=True,
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        assert row["abstract_full"] == ""

    # -- AC3: include_citation_contexts kwarg ---------------------------------

    def test_include_citation_contexts_off_by_default(self) -> None:
        """AC3 default: without the kwarg, no ``citation_excerpts`` field
        appears on any cited_papers row."""
        papers_rows = [("2024A", "T", 2024, "abs", 0, [], [], "Smith")]
        intent_rows = [("2024A", "method", 2)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        assert "citation_excerpts" not in row

    def test_include_citation_contexts_attaches_excerpts_for_intent_modal(
        self,
    ) -> None:
        """AC3 opt-in: with the kwarg, papers attributed via intent_modal
        get up to 3 ``citation_excerpts`` rows."""
        papers_rows = [("2024A", "T", 2024, "abs", 0, [], [], "Smith")]
        intent_rows = [("2024A", "method", 3)]
        community_rows = [("2024A", 1, "L")]
        # Excerpts query rows: (target_bibcode, context_text, intent, source_bibcode)
        excerpt_rows = [
            ("2024A", "We use the method of Smith et al.", "method", "2025citerA"),
            ("2024A", "Following Smith's procedure...", "method", "2025citerB"),
        ]
        conn = _mock_conn(
            [papers_rows, intent_rows, community_rows, excerpt_rows]
        )

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
            include_citation_contexts=True,
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        assert "citation_excerpts" in row
        excerpts = row["citation_excerpts"]
        assert len(excerpts) == 2
        # Each excerpt has the documented schema.
        for excerpt in excerpts:
            assert set(excerpt.keys()) == {
                "context_text",
                "intent",
                "citing_bibcode",
            }
        # Excerpt content surfaces faithfully.
        excerpt_texts = {e["context_text"] for e in excerpts}
        assert "We use the method of Smith et al." in excerpt_texts

    def test_include_citation_contexts_only_for_intent_modal(self) -> None:
        """AC3: papers attributed via community_fallthrough, override, or
        citation_count_fallback do NOT get ``citation_excerpts`` even when
        the kwarg is true — only intent_modal-assigned papers do (since
        they're the ones whose section came from a citation_contexts row)."""
        # 2024A -> intent_modal (methods); 2024B -> community_fallthrough (background).
        papers_rows = [
            ("2024A", "TA", 2024, "abs", 0, [], [], "Smith"),
            ("2024B", "TB", 2024, "abs", 0, [], [], "Jones"),
        ]
        intent_rows = [("2024A", "method", 2)]
        community_rows = [("2024A", 1, "L"), ("2024B", 1, "L")]
        # Excerpt query returns rows for BOTH bibcodes, but only A should
        # carry them in the output.
        excerpt_rows = [
            ("2024A", "ctx-A1", "method", "2025citerA"),
            ("2024B", "ctx-B1", "background", "2025citerB"),
        ]
        conn = _mock_conn(
            [papers_rows, intent_rows, community_rows, excerpt_rows]
        )

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A", "2024B"],
            sections=list(DEFAULT_SECTIONS),
            include_citation_contexts=True,
        )
        all_rows = {p["bibcode"]: p for s in result.sections for p in s.cited_papers}
        # 2024A: intent_modal -> excerpts attached.
        assert all_rows["2024A"]["signal_used"] == "intent_modal"
        assert "citation_excerpts" in all_rows["2024A"]
        # 2024B: community_fallthrough -> NO excerpts.
        assert all_rows["2024B"]["signal_used"] == "community_fallthrough"
        assert "citation_excerpts" not in all_rows["2024B"]

    def test_include_citation_contexts_capped_at_three_per_paper(self) -> None:
        """AC3: when more than 3 excerpts exist for a paper, only the
        first 3 (in deterministic order) are surfaced."""
        papers_rows = [("2024A", "T", 2024, "abs", 0, [], [], "Smith")]
        intent_rows = [("2024A", "method", 5)]
        community_rows = [("2024A", 1, "L")]
        # 5 excerpt rows for one paper.
        excerpt_rows = [
            ("2024A", f"ctx-{i}", "method", f"2025citer{i}") for i in range(5)
        ]
        conn = _mock_conn(
            [papers_rows, intent_rows, community_rows, excerpt_rows]
        )

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
            include_citation_contexts=True,
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        assert len(row["citation_excerpts"]) == 3

    def test_include_citation_contexts_empty_result_emits_empty_list(
        self,
    ) -> None:
        """AC3 robustness: when no excerpt rows exist for an intent-modal
        paper, ``citation_excerpts`` is the empty list (not omitted)."""
        papers_rows = [("2024A", "T", 2024, "abs", 0, [], [], "Smith")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        excerpt_rows: list[tuple] = []  # empty
        conn = _mock_conn(
            [papers_rows, intent_rows, community_rows, excerpt_rows]
        )

        result = synthesize_findings(
            conn,
            working_set_bibcodes=["2024A"],
            sections=list(DEFAULT_SECTIONS),
            include_citation_contexts=True,
        )
        methods = next(s for s in result.sections if s.name == "methods")
        row = next(p for p in methods.cited_papers if p["bibcode"] == "2024A")
        assert row["citation_excerpts"] == []

    # -- MCP wire dispatch ----------------------------------------------------

    def test_dispatch_accepts_include_full_abstracts_kwarg(self) -> None:
        """MCP wire: ``include_full_abstracts=true`` flows through the
        handler and surfaces ``abstract_full`` in JSON output."""
        long_abstract = "Z" * 1000
        papers_rows = [("2024A", "T", 2024, long_abstract, 0, [], [], "Smith")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        out = _dispatch_tool(
            conn,
            "synthesize_findings",
            {
                "working_set_bibcodes": ["2024A"],
                "include_full_abstracts": True,
            },
        )
        result = json.loads(out)
        methods = next(s for s in result["sections"] if s["name"] == "methods")
        row = next(p for p in methods["cited_papers"] if p["bibcode"] == "2024A")
        assert row["abstract_full"] == long_abstract

    def test_dispatch_accepts_include_citation_contexts_kwarg(self) -> None:
        """MCP wire: ``include_citation_contexts=true`` flows through and
        surfaces ``citation_excerpts`` for intent_modal papers."""
        papers_rows = [("2024A", "T", 2024, "abs", 0, [], [], "Smith")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        excerpt_rows = [("2024A", "ctx-1", "method", "2025citerA")]
        conn = _mock_conn(
            [papers_rows, intent_rows, community_rows, excerpt_rows]
        )

        out = _dispatch_tool(
            conn,
            "synthesize_findings",
            {
                "working_set_bibcodes": ["2024A"],
                "include_citation_contexts": True,
            },
        )
        result = json.loads(out)
        methods = next(s for s in result["sections"] if s["name"] == "methods")
        row = next(p for p in methods["cited_papers"] if p["bibcode"] == "2024A")
        assert "citation_excerpts" in row
        assert len(row["citation_excerpts"]) == 1
        assert row["citation_excerpts"][0]["context_text"] == "ctx-1"

    def test_dispatch_first_author_in_wire_format(self) -> None:
        """MCP wire: ``first_author`` is part of the default JSON output
        (no kwargs needed; AC1 always-on)."""
        papers_rows = [("2024A", "T", 2024, "abs", 0, [], [], "Breu")]
        intent_rows = [("2024A", "method", 1)]
        community_rows = [("2024A", 1, "L")]
        conn = _mock_conn([papers_rows, intent_rows, community_rows])

        out = _dispatch_tool(
            conn,
            "synthesize_findings",
            {"working_set_bibcodes": ["2024A"]},
        )
        result = json.loads(out)
        methods = next(s for s in result["sections"] if s["name"] == "methods")
        row = next(p for p in methods["cited_papers"] if p["bibcode"] == "2024A")
        assert row["first_author"] == "Breu"
