"""OAuth execution seam for the ``claim_blame`` gold benchmark."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scix.eval.claim_blame_gold import (
    ClaimBlameObservation,
    GoldCase,
    GoldSetError,
    parse_agent_result,
)

DEFAULT_TIMEOUT_S = 600.0
DEFAULT_MAX_TURNS = 25
ALLOWED_TOOLS = (
    "Agent(deep_search_investigator)",
    "mcp__scix__search",
    "mcp__scix__concept_search",
    "mcp__scix__get_paper",
    "mcp__scix__read_paper",
    "mcp__scix__citation_traverse",
    "mcp__scix__citation_similarity",
    "mcp__scix__entity",
    "mcp__scix__graph_context",
    "mcp__scix__find_gaps",
    "mcp__scix__temporal_evolution",
    "mcp__scix__facet_counts",
    "mcp__scix__claim_blame",
    "mcp__scix__forward_citations",
)
PAID_AUTH_ENV_VARS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_VERTEX",
    }
)


def oauth_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment that cannot select paid API authentication."""

    values = os.environ if source is None else source
    return {key: value for key, value in values.items() if key not in PAID_AUTH_ENV_VARS}


def build_investigator_prompt(case: GoldCase) -> str:
    """Build the strict-output prompt used by the OAuth subagent path."""

    return f"""Use the deep_search_investigator subagent to trace this claim's origin.
You must call claim_blame and may use the investigator's other SciX tools to
validate candidate origins, retractions, and chronological lineage.

Claim: {case.claim_text}

Do not use the candidate label or source citations as evidence; they are hidden
benchmark annotations. Return exactly one JSON object with this schema:
{{
  "ranked_origin_bibcodes": ["up to five ADS bibcodes, best first"],
  "retraction_warnings": ["ADS bibcodes flagged by the overlay"],
  "lineage_bibcodes": ["ADS bibcodes in chronological order"],
  "coverage": {{"covered_seeds": 0, "total_seeds": 0}}
}}
Use integer coverage counts copied from claim_blame. Do not add prose or fences.
"""


def _empty_observation(case_id: str, error: str) -> ClaimBlameObservation:
    return ClaimBlameObservation(
        case_id=case_id,
        ranked_origin_bibcodes=(),
        retraction_warnings=(),
        lineage_bibcodes=(),
        covered_seeds=0,
        total_seeds=0,
        error=error,
    )


@dataclass(frozen=True)
class OAuthClaimBlameRunner:
    """Run one case through ``claude -p`` using OAuth, never a paid SDK."""

    repo_root: Path
    claude_binary: str = "claude"
    timeout_s: float = DEFAULT_TIMEOUT_S
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run

    def __call__(self, case: GoldCase) -> ClaimBlameObservation:
        command = [
            self.claude_binary,
            "-p",
            build_investigator_prompt(case),
            "--output-format",
            "json",
            "--max-turns",
            str(DEFAULT_MAX_TURNS),
            "--tools",
            "Agent",
            "--allowedTools",
            ",".join(ALLOWED_TOOLS),
        ]
        try:
            completed = self.run_command(
                command,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
                env=oauth_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _empty_observation(case.case_id, f"claude -p failed: {exc}")

        if completed.returncode != 0:
            detail = completed.stderr.strip()[-500:] or "no stderr"
            return _empty_observation(
                case.case_id,
                f"claude -p exited {completed.returncode}: {detail}",
            )

        raw_result = completed.stdout
        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError:
            envelope = None
        if isinstance(envelope, Mapping) and "result" in envelope:
            result = envelope["result"]
            raw_result = result if isinstance(result, str) else json.dumps(result)

        try:
            return parse_agent_result(case.case_id, raw_result)
        except GoldSetError as exc:
            return _empty_observation(case.case_id, str(exc))


def load_observations(path: Path) -> list[ClaimBlameObservation]:
    """Load normalized observation JSONL written by the benchmark CLI."""

    observations: list[ClaimBlameObservation] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GoldSetError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, Mapping):
            raise GoldSetError(f"{path}:{line_no}: row must be an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise GoldSetError(f"{path}:{line_no}: case_id must be a non-empty string")
        observations.append(ClaimBlameObservation.from_mapping(case_id, row))
    return observations


def observation_to_mapping(observation: ClaimBlameObservation) -> dict[str, Any]:
    """Return the stable JSON representation of one observation."""

    return {
        "case_id": observation.case_id,
        "ranked_origin_bibcodes": list(observation.ranked_origin_bibcodes),
        "retraction_warnings": list(observation.retraction_warnings),
        "lineage_bibcodes": list(observation.lineage_bibcodes),
        "coverage": {
            "covered_seeds": observation.covered_seeds,
            "total_seeds": observation.total_seeds,
        },
        "error": observation.error,
    }


def write_observations(path: Path, observations: Sequence[ClaimBlameObservation]) -> None:
    """Write observations atomically enough for resumable batch evaluation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(observation_to_mapping(observation), sort_keys=True) + "\n"
        for observation in observations
    )
    path.write_text(payload, encoding="utf-8")


__all__ = [
    "OAuthClaimBlameRunner",
    "build_investigator_prompt",
    "load_observations",
    "observation_to_mapping",
    "write_observations",
]
