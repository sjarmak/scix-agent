from __future__ import annotations

import json
from pathlib import Path

import pytest

from scix.eval.claim_blame_gold import (
    ClaimBlameObservation,
    GoldCase,
    GoldSetError,
    evaluate_claim_blame,
    format_claim_blame_report,
    load_gold_cases,
    parse_agent_result,
)


def _case(**overrides: object) -> GoldCase:
    values: dict[str, object] = {
        "case_id": "h0-tension",
        "claim_text": "Local measurements of H0 disagree with CMB inference.",
        "verified_origin_bibcode": "2011ApJ...730..119R",
        "acceptable_origin_bibcodes": ("2011ApJ...730..119R",),
        "expected_retraction_bibcodes": (),
        "expected_lineage_bibcodes": (
            "2011ApJ...730..119R",
            "2016ApJ...826...56R",
            "2019ApJ...876...85R",
        ),
        "label_status": "verified",
        "source_citations": ("https://ui.adsabs.harvard.edu/abs/2011ApJ...730..119R",),
    }
    values.update(overrides)
    return GoldCase(**values)  # type: ignore[arg-type]


def _observation(**overrides: object) -> ClaimBlameObservation:
    values: dict[str, object] = {
        "case_id": "h0-tension",
        "ranked_origin_bibcodes": (
            "2011ApJ...730..119R",
            "2009ApJ...699..539R",
        ),
        "retraction_warnings": (),
        "lineage_bibcodes": (
            "2011ApJ...730..119R",
            "2016ApJ...826...56R",
            "2019ApJ...876...85R",
        ),
        "covered_seeds": 3,
        "total_seeds": 4,
        "error": None,
    }
    values.update(overrides)
    return ClaimBlameObservation(**values)  # type: ignore[arg-type]


def test_load_gold_cases_rejects_unverified_rows_when_gold_required(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "draft",
                "claim_text": "draft claim",
                "verified_origin_bibcode": "2011ApJ...730..119R",
                "acceptable_origin_bibcodes": ["2011ApJ...730..119R"],
                "expected_retraction_bibcodes": [],
                "expected_lineage_bibcodes": [],
                "label_status": "unverified",
                "source_citations": ["https://ui.adsabs.harvard.edu/abs/2011ApJ...730..119R"],
            }
        )
        + "\n"
    )

    with pytest.raises(GoldSetError, match="unverified"):
        load_gold_cases(path, require_verified=True)

    cases = load_gold_cases(path, require_verified=False)
    assert cases[0].label_status == "unverified"


@pytest.mark.parametrize(
    "field",
    [
        "acceptable_origin_bibcodes",
        "source_citations",
        "expected_retraction_bibcodes",
        "expected_lineage_bibcodes",
    ],
)
def test_gold_case_rejects_string_where_array_is_required(field: str) -> None:
    row: dict[str, object] = {
        "id": "bad-array",
        "claim_text": "claim",
        "verified_origin_bibcode": "2011ApJ...730..119R",
        "acceptable_origin_bibcodes": ["2011ApJ...730..119R"],
        "expected_retraction_bibcodes": [],
        "expected_lineage_bibcodes": [],
        "label_status": "verified",
        "source_citations": ["https://ui.adsabs.harvard.edu/abs/2011ApJ...730..119R"],
    }
    row[field] = "2011ApJ...730..119R"

    with pytest.raises(GoldSetError, match=rf"{field} must be an array"):
        GoldCase.from_mapping(row, location="test")


def test_parse_agent_result_accepts_fenced_json_and_deduplicates() -> None:
    raw = """Result:\n```json
{"ranked_origin_bibcodes":["2011ApJ...730..119R","2011ApJ...730..119R"],
 "retraction_warnings":["2014PhRvL.112x1101B"],
 "lineage_bibcodes":["2011ApJ...730..119R","2016ApJ...826...56R"],
 "coverage":{"covered_seeds":2,"total_seeds":5}}
```"""

    parsed = parse_agent_result("h0-tension", raw)

    assert parsed.ranked_origin_bibcodes == ("2011ApJ...730..119R",)
    assert parsed.retraction_warnings == ("2014PhRvL.112x1101B",)
    assert parsed.covered_seeds == 2
    assert parsed.total_seeds == 5


@pytest.mark.parametrize(
    "raw, message",
    [
        ('{"message":"Tool unavailable"}', "missing required field coverage"),
        (
            '{"ranked_origin_bibcodes":null,"retraction_warnings":[],"lineage_bibcodes":[],"coverage":{"covered_seeds":0,"total_seeds":0}}',
            "ranked_origin_bibcodes must be an array",
        ),
    ],
)
def test_parse_agent_result_rejects_malformed_schema(raw: str, message: str) -> None:
    with pytest.raises(GoldSetError, match=message):
        parse_agent_result("bad", raw)


def test_evaluate_claim_blame_computes_all_requested_metrics() -> None:
    bicep = _case(
        case_id="bicep",
        verified_origin_bibcode="2015PhRvL.114j1301P",
        acceptable_origin_bibcodes=("2015PhRvL.114j1301P",),
        expected_retraction_bibcodes=("2014PhRvL.112x1101B",),
        expected_lineage_bibcodes=(
            "2014PhRvL.112x1101B",
            "2015PhRvL.114j1301P",
        ),
    )
    observations = [
        _observation(),
        _observation(
            case_id="bicep",
            ranked_origin_bibcodes=(
                "2014PhRvL.112x1101B",
                "2015PhRvL.114j1301P",
            ),
            retraction_warnings=(
                "2014PhRvL.112x1101B",
                "2000ApJ...000..000X",
            ),
            lineage_bibcodes=(
                "2014PhRvL.112x1101B",
                "2015PhRvL.114j1301P",
            ),
            covered_seeds=1,
            total_seeds=4,
        ),
    ]

    result = evaluate_claim_blame(
        [_case(), bicep],
        observations,
        covered_edges=821_000,
        total_edges=299_000_000,
    )

    assert result.scored_cases == 2
    assert result.origin_recall_at_1 == pytest.approx(0.5)
    assert result.origin_recall_at_5 == pytest.approx(1.0)
    assert result.retraction_overlay_precision == pytest.approx(0.5)
    assert result.lineage_chronology == pytest.approx(1.0)
    assert result.seed_coverage == pytest.approx(0.5)
    assert result.corpus_edge_coverage == pytest.approx(821_000 / 299_000_000)
    assert result.gate_decision == "hard_intent_filter"


def test_unverified_cases_are_excluded_from_headline_metrics() -> None:
    result = evaluate_claim_blame(
        [_case(label_status="unverified")],
        [_observation()],
        covered_edges=10,
        total_edges=100,
    )

    assert result.scored_cases == 0
    assert result.unverified_cases == 1
    assert result.origin_recall_at_1 is None
    assert result.gate_decision == "insufficient_verified_labels"


def test_failed_verified_run_counts_as_recall_miss_and_blocks_gate() -> None:
    result = evaluate_claim_blame(
        [_case(), _case(case_id="failed")],
        [_observation(), _observation(case_id="failed", error="OAuth failure")],
        covered_edges=10,
        total_edges=100,
    )

    assert result.scored_cases == 2
    assert result.origin_recall_at_1 == pytest.approx(0.5)
    assert result.origin_recall_at_5 == pytest.approx(0.5)
    assert result.failed_observations == 1
    assert result.gate_decision == "incomplete_runs"


def test_report_is_explicit_about_draft_labels_and_gate() -> None:
    result = evaluate_claim_blame(
        [_case()],
        [_observation()],
        covered_edges=821_000,
        total_edges=299_000_000,
    )

    report = format_claim_blame_report(result, generated_at="2026-09-24T00:00:00Z")

    assert "Origin recall@1" in report
    assert "Origin recall@5" in report
    assert "Retraction-overlay precision" in report
    assert "Lineage chronology" in report
    assert "Corpus edge coverage" in report
    assert "hard intent filter" in report
    assert "independently verified" in report


def test_candidate_corpus_has_30_unverified_cited_rows() -> None:
    path = Path(__file__).parent / "eval/claim_blame_gold_v1.candidates.jsonl"

    cases = load_gold_cases(path, require_verified=False)

    assert len(cases) == 30
    assert {case.label_status for case in cases} == {"unverified"}
    assert all(case.source_citations for case in cases)
