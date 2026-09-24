"""Scoring primitives for the real-corpus ``claim_blame`` benchmark.

The candidate file and the benchmark observations are deliberately separate.
Only independently adjudicated rows (``label_status=verified``) contribute to
headline metrics.  This prevents a draft, model-authored candidate list from
silently becoming ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

LabelStatus = Literal["unverified", "verified"]


class GoldSetError(ValueError):
    """Raised when a gold-set row violates the benchmark contract."""


def _unique_strings(values: Iterable[object], *, field_name: str) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise GoldSetError(f"{field_name} must contain non-empty strings")
        normalized = value.strip()
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def _string_array(row: Mapping[str, Any], name: str, *, location: str) -> tuple[str, ...]:
    if name not in row:
        raise GoldSetError(f"{location}: missing required field {name}")
    values = row[name]
    if not isinstance(values, (list, tuple)):
        raise GoldSetError(f"{location}: {name} must be an array")
    return _unique_strings(values, field_name=f"{location}: {name}")


def _optional_string_array(row: Mapping[str, Any], name: str, *, location: str) -> tuple[str, ...]:
    if name not in row:
        return ()
    return _string_array(row, name, location=location)


@dataclass(frozen=True)
class GoldCase:
    """One independently adjudicable claim-origin case."""

    case_id: str
    claim_text: str
    verified_origin_bibcode: str
    acceptable_origin_bibcodes: tuple[str, ...]
    expected_retraction_bibcodes: tuple[str, ...]
    expected_lineage_bibcodes: tuple[str, ...]
    label_status: LabelStatus
    source_citations: tuple[str, ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, location: str) -> GoldCase:
        status = row.get("label_status")
        if status not in {"unverified", "verified"}:
            raise GoldSetError(f"{location}: label_status must be unverified or verified")

        case_id = row.get("id")
        claim_text = row.get("claim_text")
        origin = row.get("verified_origin_bibcode")
        for name, value in (
            ("id", case_id),
            ("claim_text", claim_text),
            ("verified_origin_bibcode", origin),
        ):
            if not isinstance(value, str) or not value.strip():
                raise GoldSetError(f"{location}: {name} must be a non-empty string")

        acceptable = _string_array(row, "acceptable_origin_bibcodes", location=location)
        if not acceptable:
            raise GoldSetError(f"{location}: acceptable_origin_bibcodes cannot be empty")
        citations = _string_array(row, "source_citations", location=location)
        if not citations:
            raise GoldSetError(f"{location}: source_citations cannot be empty")

        return cls(
            case_id=case_id.strip(),
            claim_text=claim_text.strip(),
            verified_origin_bibcode=origin.strip(),
            acceptable_origin_bibcodes=acceptable,
            expected_retraction_bibcodes=_optional_string_array(
                row,
                "expected_retraction_bibcodes",
                location=location,
            ),
            expected_lineage_bibcodes=_optional_string_array(
                row,
                "expected_lineage_bibcodes",
                location=location,
            ),
            label_status=status,
            source_citations=citations,
        )


@dataclass(frozen=True)
class ClaimBlameObservation:
    """Normalized structured output from one investigator run."""

    case_id: str
    ranked_origin_bibcodes: tuple[str, ...]
    retraction_warnings: tuple[str, ...]
    lineage_bibcodes: tuple[str, ...]
    covered_seeds: int
    total_seeds: int
    error: str | None = None

    @classmethod
    def from_mapping(cls, case_id: str, row: Mapping[str, Any]) -> ClaimBlameObservation:
        error = row.get("error")
        if error is not None and not isinstance(error, str):
            raise GoldSetError(f"{case_id}: error must be a string or null")
        if error:
            return cls(
                case_id=case_id,
                ranked_origin_bibcodes=(),
                retraction_warnings=(),
                lineage_bibcodes=(),
                covered_seeds=0,
                total_seeds=0,
                error=error,
            )

        if "coverage" not in row:
            raise GoldSetError(f"{case_id}: missing required field coverage")
        coverage = row["coverage"]
        if not isinstance(coverage, Mapping):
            raise GoldSetError(f"{case_id}: coverage must be an object")
        if "covered_seeds" not in coverage or "total_seeds" not in coverage:
            raise GoldSetError(f"{case_id}: coverage requires covered_seeds and total_seeds")
        covered = coverage["covered_seeds"]
        total = coverage["total_seeds"]
        if not isinstance(covered, int) or not isinstance(total, int):
            raise GoldSetError(f"{case_id}: coverage counts must be integers")
        if covered < 0 or total < 0 or covered > total:
            raise GoldSetError(f"{case_id}: invalid coverage counts {covered}/{total}")

        return cls(
            case_id=case_id,
            ranked_origin_bibcodes=_string_array(row, "ranked_origin_bibcodes", location=case_id),
            retraction_warnings=_string_array(row, "retraction_warnings", location=case_id),
            lineage_bibcodes=_string_array(row, "lineage_bibcodes", location=case_id),
            covered_seeds=covered,
            total_seeds=total,
            error=error,
        )


@dataclass(frozen=True)
class ClaimBlameEvaluation:
    total_cases: int
    scored_cases: int
    unverified_cases: int
    missing_observations: int
    failed_observations: int
    origin_recall_at_1: float | None
    origin_recall_at_5: float | None
    retraction_overlay_precision: float | None
    lineage_chronology: float | None
    seed_coverage: float | None
    corpus_edge_coverage: float | None
    covered_edges: int
    total_edges: int
    gate_decision: str


def load_gold_cases(path: Path, *, require_verified: bool = True) -> list[GoldCase]:
    """Load and validate JSONL cases, optionally rejecting draft labels."""

    cases: list[GoldCase] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        location = f"{path}:{line_no}"
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GoldSetError(f"{location}: invalid JSON: {exc}") from exc
        if not isinstance(row, Mapping):
            raise GoldSetError(f"{location}: row must be an object")
        case = GoldCase.from_mapping(row, location=location)
        if case.case_id in seen:
            raise GoldSetError(f"{location}: duplicate id {case.case_id!r}")
        if require_verified and case.label_status != "verified":
            raise GoldSetError(f"{location}: unverified row cannot be used as gold")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise GoldSetError(f"{path}: no cases found")
    return cases


def _first_json_object(raw: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    raise GoldSetError("agent output did not contain a JSON object")


def parse_agent_result(case_id: str, raw: str) -> ClaimBlameObservation:
    """Extract the first structured result object from OAuth-agent output."""

    return ClaimBlameObservation.from_mapping(case_id, _first_json_object(raw))


def _lineage_pair_score(expected: Sequence[str], observed: Sequence[str]) -> tuple[int, int]:
    """Return correctly ordered expected pairs and the total expected pairs."""

    total = len(expected) * (len(expected) - 1) // 2
    if total == 0:
        return 0, 0
    positions = {bibcode: index for index, bibcode in enumerate(observed)}
    correct = 0
    for left_index, left in enumerate(expected):
        for right in expected[left_index + 1 :]:
            if left in positions and right in positions and positions[left] < positions[right]:
                correct += 1
    return correct, total


def _gate_decision(recall_at_5: float | None, *, missing: int, failed: int) -> str:
    if recall_at_5 is None:
        return "insufficient_verified_labels"
    if missing or failed:
        return "incomplete_runs"
    if recall_at_5 >= 0.70:
        return "hard_intent_filter"
    if recall_at_5 < 0.40:
        return "keep_weights"
    return "collect_more_evidence"


def evaluate_claim_blame(
    cases: Sequence[GoldCase],
    observations: Sequence[ClaimBlameObservation],
    *,
    covered_edges: int,
    total_edges: int,
) -> ClaimBlameEvaluation:
    """Compute benchmark metrics over verified, successful cases only."""

    if covered_edges < 0 or total_edges < 0 or covered_edges > total_edges:
        raise ValueError("invalid corpus edge coverage counts")
    by_id = {observation.case_id: observation for observation in observations}
    if len(by_id) != len(observations):
        raise GoldSetError("duplicate observation case_id")

    verified = [case for case in cases if case.label_status == "verified"]
    successful: list[tuple[GoldCase, ClaimBlameObservation]] = []
    missing = 0
    failed = 0
    for case in verified:
        observation = by_id.get(case.case_id)
        if observation is None:
            missing += 1
        elif observation.error:
            failed += 1
        else:
            successful.append((case, observation))

    scored = len(verified)
    recall_1: float | None = None
    recall_5: float | None = None
    if scored:
        hits_1 = 0
        hits_5 = 0
        for case, observation in successful:
            acceptable = set(case.acceptable_origin_bibcodes)
            hits_1 += bool(set(observation.ranked_origin_bibcodes[:1]) & acceptable)
            hits_5 += bool(set(observation.ranked_origin_bibcodes[:5]) & acceptable)
        # Failed and missing runs remain in the denominator. Excluding them
        # would turn execution failures into an artificial accuracy gain.
        recall_1 = hits_1 / scored
        recall_5 = hits_5 / scored

    predicted_retractions = 0
    correct_retractions = 0
    correct_pairs = 0
    total_pairs = 0
    covered_seeds = 0
    total_seeds = 0
    for case, observation in successful:
        expected_retractions = set(case.expected_retraction_bibcodes)
        predicted = set(observation.retraction_warnings)
        predicted_retractions += len(predicted)
        correct_retractions += len(predicted & expected_retractions)
        pair_correct, pair_total = _lineage_pair_score(
            case.expected_lineage_bibcodes, observation.lineage_bibcodes
        )
        correct_pairs += pair_correct
        total_pairs += pair_total
        covered_seeds += observation.covered_seeds
        total_seeds += observation.total_seeds

    retraction_precision = (
        correct_retractions / predicted_retractions if predicted_retractions else None
    )
    chronology = correct_pairs / total_pairs if total_pairs else None
    seed_coverage = covered_seeds / total_seeds if total_seeds else None
    corpus_coverage = covered_edges / total_edges if total_edges else None

    return ClaimBlameEvaluation(
        total_cases=len(cases),
        scored_cases=scored,
        unverified_cases=len(cases) - len(verified),
        missing_observations=missing,
        failed_observations=failed,
        origin_recall_at_1=recall_1,
        origin_recall_at_5=recall_5,
        retraction_overlay_precision=retraction_precision,
        lineage_chronology=chronology,
        seed_coverage=seed_coverage,
        corpus_edge_coverage=corpus_coverage,
        covered_edges=covered_edges,
        total_edges=total_edges,
        gate_decision=_gate_decision(recall_5, missing=missing, failed=failed),
    )


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def format_claim_blame_report(result: ClaimBlameEvaluation, *, generated_at: str) -> str:
    """Render a compact, publication-shaped Markdown benchmark report."""

    decision_text = {
        "hard_intent_filter": "Enable the hard intent filter.",
        "keep_weights": "Keep the current intent weights.",
        "collect_more_evidence": "Collect more evidence before changing intent handling.",
        "insufficient_verified_labels": "No gate decision: insufficient verified labels.",
        "incomplete_runs": "No gate decision: one or more benchmark runs are incomplete.",
    }[result.gate_decision]
    return "\n".join(
        [
            "> **In-house evaluation.** Candidate labels must be independently verified "
            "before they enter headline metrics.",
            "",
            "# claim_blame Gold v1",
            "",
            f"**Generated:** {generated_at}",
            f"**Cases:** {result.total_cases} total; {result.scored_cases} scored; "
            f"{result.unverified_cases} unverified; {result.missing_observations} missing; "
            f"{result.failed_observations} failed",
            "",
            "| Metric | Result |",
            "| --- | ---: |",
            f"| Origin recall@1 | {_percent(result.origin_recall_at_1)} |",
            f"| Origin recall@5 | {_percent(result.origin_recall_at_5)} |",
            f"| Retraction-overlay precision | {_percent(result.retraction_overlay_precision)} |",
            f"| Lineage chronology | {_percent(result.lineage_chronology)} |",
            f"| Seed coverage | {_percent(result.seed_coverage)} |",
            f"| Corpus edge coverage | {_percent(result.corpus_edge_coverage)} "
            f"({result.covered_edges:,}/{result.total_edges:,}) |",
            "",
            "## Gate",
            "",
            f"**{result.gate_decision}:** {decision_text}",
            "",
            "The gate uses origin recall@5: ≥70% enables the hard intent filter; "
            "<40% keeps the existing weights; the middle band requires more evidence.",
            "",
        ]
    )


__all__ = [
    "ClaimBlameEvaluation",
    "ClaimBlameObservation",
    "GoldCase",
    "GoldSetError",
    "evaluate_claim_blame",
    "format_claim_blame_report",
    "load_gold_cases",
    "parse_agent_result",
]
