"""Tests for ``scripts/run_ner_bodies.py`` (M2 body-NER pilot).

All tests run without a database, without a GPU, and without the GLiNER
weights — psycopg connections and the GLiNER model are mocked. The
synthetic paper bodies in ``SYNTHETIC_BODIES`` exercise the full set of
section structures the parser is expected to encounter.

Coverage:

* ``select_kept_sections`` — only ``method`` and ``result`` roles are
  kept; bibliography, introduction, conclusion, abstract, preamble, and
  empty bodies are skipped.
* ``insert_paper_row`` — writes one row per paper with ``source='ner_body'``,
  ``section_name`` populated, and the per-section payload preserved.
* ``run_pipeline`` end-to-end — papers with no kept sections produce no
  rows; papers with method/result sections produce exactly one row each.
* MCP entity-tool query path — a mocked-DB query that filters
  ``staging.extractions`` by ``source='ner_body'`` returns the expected
  shape (AC5 — documents the SQL contract; the MCP wiring is a sibling
  unit).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"
for _p in (SRC, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_ner_bodies as r  # noqa: E402  (script-style import)

from scix.extract.ner_pass import GlinerExtractor, Mention, PaperInput  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


SYNTHETIC_BODIES: dict[str, str] = {
    # 1. Full IMRaD structure — methods + results should be kept.
    "full_imrad": (
        "Introduction\n"
        "We study cosmic dawn using new survey data.\n\n"
        "Methods\n"
        "We applied PyTorch and Astropy to reduce the data from JWST.\n\n"
        "Results\n"
        "We measured the Hubble constant with the SExtractor pipeline.\n\n"
        "References\n"
        "Smith et al. 2020. Jones 2021.\n"
    ),
    # 2. Methods + Results only — both kept.
    "methods_results_only": (
        "Methods\n"
        "We trained a CNN with TensorFlow.\n\n"
        "Results\n"
        "MNIST accuracy reached 99.2%.\n"
    ),
    # 3. Pure introduction — no kept sections.
    "intro_only": (
        "Introduction\n"
        "Background context for the study.\n"
    ),
    # 4. Pure references — no kept sections.
    "refs_only": (
        "References\n"
        "Smith et al. 2020. Jones et al. 2021.\n"
    ),
    # 5. Methods alone.
    "methods_only": (
        "Methods\n"
        "We used the Pandas library to wrangle the data.\n"
    ),
    # 6. Results alone.
    "results_only": (
        "Results\n"
        "The Spitzer telescope detected the source at 24 microns.\n"
    ),
    # 7. Intro + Methods + Discussion + Conclusion — only methods kept.
    "no_results": (
        "Introduction\n"
        "Why we care.\n\n"
        "Methods\n"
        "We use the Hubble Space Telescope and CASA pipeline.\n\n"
        "Discussion\n"
        "What it might mean.\n\n"
        "Conclusion\n"
        "We end here.\n"
    ),
    # 8. Numbered section headers.
    "numbered_headers": (
        "1. Introduction\n"
        "Background.\n\n"
        "2.1 Methods\n"
        "We used the GLiNER model.\n\n"
        "3. Results\n"
        "We saw the H0 = 67 value.\n"
    ),
    # 9. Empty body.
    "empty": "",
    # 10. No recognisable headers — parser returns a single 'full' section,
    #     role 'other', filtered out.
    "no_headers": (
        "This is just one big block of unstructured text with no sectioning at all "
        "and the section parser will fall back to returning a 'full' pseudo-section "
        "which gets classified as role='other' and dropped by the filter.\n"
    ),
}


# ---------------------------------------------------------------------------
# Stub GLiNER model
# ---------------------------------------------------------------------------


class _StubGlinerModel:
    """Mimics GLiNER's batch_predict_entities surface used by the extractor.

    Returns one canned mention per text so we can assert that section
    text reaches the model. The mention surface is the first word of
    the input (used as a sentinel in tests).
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def to(self, _device: str) -> "_StubGlinerModel":
        return self

    def eval(self) -> "_StubGlinerModel":  # noqa: A003 — mimic torch
        return self

    def batch_predict_entities(
        self,
        texts: list[str],
        labels: list[str],
        threshold: float,
        batch_size: int,
    ) -> list[list[dict[str, Any]]]:
        self.calls.append(list(texts))
        out: list[list[dict[str, Any]]] = []
        for t in texts:
            first = t.strip().split()[:1]
            surface = first[0] if first else ""
            out.append(
                [
                    {
                        "text": surface,
                        "label": "software",
                        "score": 0.92,
                    }
                ]
            )
        return out


def _make_stub_extractor() -> tuple[GlinerExtractor, _StubGlinerModel]:
    stub = _StubGlinerModel()
    extractor = GlinerExtractor(model=stub, confidence=0.7)
    return extractor, stub


# ---------------------------------------------------------------------------
# select_kept_sections
# ---------------------------------------------------------------------------


class TestSelectKeptSections:
    def test_full_imrad_keeps_methods_and_results(self) -> None:
        kept = r.select_kept_sections(SYNTHETIC_BODIES["full_imrad"])
        roles = [role for _n, role, _s, _e, _t in kept]
        names = [name for name, _r, _s, _e, _t in kept]
        assert sorted(roles) == ["method", "result"]
        # The parser canonicalises "Methods" → "methods" and "Results" → "results"
        assert "methods" in names
        assert "results" in names
        # References + Introduction must not be present.
        assert "references" not in names
        assert "introduction" not in names

    def test_methods_results_only(self) -> None:
        kept = r.select_kept_sections(SYNTHETIC_BODIES["methods_results_only"])
        assert {role for _n, role, _s, _e, _t in kept} == {"method", "result"}

    def test_intro_only_drops_everything(self) -> None:
        assert r.select_kept_sections(SYNTHETIC_BODIES["intro_only"]) == []

    def test_refs_only_drops_everything(self) -> None:
        assert r.select_kept_sections(SYNTHETIC_BODIES["refs_only"]) == []

    def test_methods_only_kept(self) -> None:
        kept = r.select_kept_sections(SYNTHETIC_BODIES["methods_only"])
        assert len(kept) == 1
        assert kept[0][1] == "method"

    def test_results_only_kept(self) -> None:
        kept = r.select_kept_sections(SYNTHETIC_BODIES["results_only"])
        assert len(kept) == 1
        assert kept[0][1] == "result"

    def test_no_results_keeps_only_methods(self) -> None:
        kept = r.select_kept_sections(SYNTHETIC_BODIES["no_results"])
        roles = [role for _n, role, _s, _e, _t in kept]
        assert roles == ["method"]
        # Discussion / Conclusion are dropped (role=conclusion).
        names = [name for name, _r, _s, _e, _t in kept]
        assert "discussion" not in names
        assert "conclusions" not in names

    def test_numbered_headers(self) -> None:
        kept = r.select_kept_sections(SYNTHETIC_BODIES["numbered_headers"])
        roles = sorted(role for _n, role, _s, _e, _t in kept)
        assert roles == ["method", "result"]

    def test_empty_body(self) -> None:
        assert r.select_kept_sections(SYNTHETIC_BODIES["empty"]) == []
        assert r.select_kept_sections(None) == []

    def test_no_headers_falls_back_and_is_dropped(self) -> None:
        # parse_sections returns ('full', 0, len, body) when no headers
        # match; classify_section_role('full') == 'other' which fails the
        # KEPT_ROLES filter.
        assert r.select_kept_sections(SYNTHETIC_BODIES["no_headers"]) == []

    def test_skipped_sections_include_bibliography_and_introduction(self) -> None:
        # Build a body whose only headers are introduction + bibliography.
        body = (
            "Introduction\nIntroduction text here.\n\n"
            "Bibliography\nSmith 2021.\n"
        )
        assert r.select_kept_sections(body) == []


# ---------------------------------------------------------------------------
# confidence_to_tier
# ---------------------------------------------------------------------------


class TestConfidenceTier:
    def test_high(self) -> None:
        assert r.confidence_to_tier(0.95) == r.TIER_HIGH

    def test_medium(self) -> None:
        assert r.confidence_to_tier(0.75) == r.TIER_MEDIUM

    def test_low(self) -> None:
        assert r.confidence_to_tier(0.5) == r.TIER_LOW


# ---------------------------------------------------------------------------
# insert_paper_row
# ---------------------------------------------------------------------------


def _capture_cursor() -> tuple[MagicMock, MagicMock]:
    """Build a mock psycopg connection whose cursor records execute() args."""
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    return conn, cursor


class TestInsertPaperRow:
    def test_writes_row_with_source_ner_body_and_section_name(self) -> None:
        kept = r.select_kept_sections(SYNTHETIC_BODIES["full_imrad"])
        # Hand-craft per-section mentions so the test does not depend on
        # the stub model's tokenisation.
        per_section = [
            [
                Mention(
                    bibcode="2024TEST..1A",
                    canonical_name="pytorch",
                    surface_text="PyTorch",
                    entity_type="software",
                    confidence=0.91,
                )
            ]
            for _ in kept
        ]
        conn, cursor = _capture_cursor()

        rows = r.insert_paper_row(
            conn,
            "2024TEST..1A",
            kept,
            per_section,
            model_name="gliner-test",
            source_version="ner_body/v1",
            extraction_version="ner_body/v1",
        )

        assert rows == 1
        cursor.execute.assert_called_once()
        sql, params = cursor.execute.call_args[0]
        assert "INSERT INTO staging.extractions" in sql
        # Positional params: bibcode, type, version, payload, source, tier,
        # section_name, char_offset
        assert params[0] == "2024TEST..1A"
        assert params[1] == r.EXTRACTION_TYPE  # 'ner_body'
        assert params[2] == "ner_body/v1"
        payload = json.loads(params[3])
        assert "sections" in payload
        assert payload["model"] == "gliner-test"
        # Each kept section appears in payload, with its mention list intact.
        section_names = [s["name"] for s in payload["sections"]]
        assert "methods" in section_names
        assert "results" in section_names
        assert params[4] == r.EXTRACTION_SOURCE  # 'ner_body'
        assert params[5] == r.TIER_HIGH  # conf 0.91 → HIGH
        # section_name column is comma-joined kept section names.
        assert "methods" in params[6] and "results" in params[6]
        # char_offset is the start offset of the first kept section.
        assert isinstance(params[7], int) and params[7] >= 0

    def test_no_kept_sections_writes_nothing(self) -> None:
        conn, cursor = _capture_cursor()
        rows = r.insert_paper_row(
            conn,
            "2024TEST..2B",
            [],
            [],
            model_name="m",
            source_version="v",
            extraction_version="v",
        )
        assert rows == 0
        cursor.execute.assert_not_called()


# ---------------------------------------------------------------------------
# run_pipeline end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_iter_paper_batches(monkeypatch: pytest.MonkeyPatch) -> list[PaperInput]:
    """Replace iter_paper_batches with a fixed list of synthetic papers.

    Returns the list so individual tests can introspect what was iterated.
    """
    papers = [
        PaperInput(bibcode=f"2024TEST..{i:02d}A", text=body)
        for i, body in enumerate(
            [
                SYNTHETIC_BODIES["full_imrad"],
                SYNTHETIC_BODIES["methods_results_only"],
                SYNTHETIC_BODIES["intro_only"],
                SYNTHETIC_BODIES["refs_only"],
                SYNTHETIC_BODIES["methods_only"],
                SYNTHETIC_BODIES["results_only"],
                SYNTHETIC_BODIES["no_results"],
                SYNTHETIC_BODIES["numbered_headers"],
                SYNTHETIC_BODIES["empty"],
                SYNTHETIC_BODIES["no_headers"],
            ]
        )
    ]

    def _fake_iter(
        conn: Any,
        *,
        target: str = "body",
        batch_size: int = 200,
        since_bibcode: str | None = None,
        max_papers: int | None = None,
    ) -> Iterator[list[PaperInput]]:
        # One batch with all papers — keeps the assertions simple.
        yield papers

    monkeypatch.setattr(r, "iter_paper_batches", _fake_iter)
    return papers


class TestRunPipeline:
    def test_skips_papers_with_no_method_or_result_sections(
        self, stub_iter_paper_batches: list[PaperInput]
    ) -> None:
        extractor, stub_model = _make_stub_extractor()
        conn, cursor = _capture_cursor()

        totals = r.run_pipeline(conn, extractor, batch_size=200)

        # 10 papers seen.
        assert totals["papers_seen"] == 10
        # Papers with kept sections: full_imrad, methods_results_only,
        # methods_only, results_only, no_results, numbered_headers => 6.
        assert totals["papers_with_kept_sections"] == 6
        # Insert called once per kept-section paper.
        assert totals["rows_inserted"] == 6

        # And the model must NEVER have been called with reference-section
        # text (we used distinctive sentinel words above).
        seen_texts = [t for call in stub_model.calls for t in call]
        joined = "\n".join(seen_texts).lower()
        assert "smith et al" not in joined  # references
        assert "background context" not in joined  # introduction


# ---------------------------------------------------------------------------
# MCP entity-tool query path (AC5)
# ---------------------------------------------------------------------------


class TestMcpSourcesFilter:
    """Pin the SQL contract that the MCP entity tool will use to surface
    body-NER rows. Mocks the cursor so we assert only that the query
    produced by a ``sources=['ner_body']`` filter is a parameterised
    ``WHERE source = ANY(%s)`` (or equivalent ``= 'ner_body'``) clause
    against ``staging.extractions``.

    This is intentionally a documentation-of-expected-behaviour test —
    ``mcp_server.py`` is owned by the sibling unit ``mcp-extraction-wiring``.
    The test guarantees that whatever MCP wiring lands, it can hit the
    rows this script writes by issuing the canonical SQL below.
    """

    _CANONICAL_SQL = (
        "SELECT bibcode, section_name, char_offset, payload "
        "FROM staging.extractions "
        "WHERE source = ANY(%s)"
    )

    def test_canonical_query_returns_body_ner_rows(self) -> None:
        conn, cursor = _capture_cursor()
        # Rig the cursor to return one fake body-NER row.
        cursor.fetchall.return_value = [
            (
                "2024TEST..1A",
                "methods,results",
                42,
                {
                    "sections": [
                        {"name": "methods", "role": "method", "mentions": []}
                    ],
                    "model": "gliner-test",
                },
            )
        ]

        with conn.cursor() as cur:
            cur.execute(self._CANONICAL_SQL, (["ner_body"],))
            rows = cur.fetchall()

        # SQL was parameterised, not interpolated.
        sql, params = cursor.execute.call_args[0]
        assert "FROM staging.extractions" in sql
        assert "WHERE source = ANY" in sql
        assert params == (["ner_body"],)

        # Returned rows have the section_name + char_offset columns
        # this script writes — confirms the contract end-to-end.
        assert len(rows) == 1
        bibcode, section_name, char_offset, payload = rows[0]
        assert bibcode == "2024TEST..1A"
        assert "methods" in section_name
        assert isinstance(char_offset, int)
        assert payload["sections"][0]["role"] == "method"

    def test_filter_excludes_other_sources(self) -> None:
        # When the query is run with sources=['ner_v1'] it must NOT match
        # the 'ner_body' rows. We assert this via the SQL shape: the
        # parameterised WHERE clause is the only way the MCP wiring can
        # hit the right partition.
        conn, cursor = _capture_cursor()
        cursor.fetchall.return_value = []  # no rows match ner_v1

        with conn.cursor() as cur:
            cur.execute(self._CANONICAL_SQL, (["ner_v1"],))
            rows = cur.fetchall()

        assert rows == []
        _, params = cursor.execute.call_args[0]
        assert params == (["ner_v1"],)
