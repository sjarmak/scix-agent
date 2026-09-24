from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scix.eval.claim_blame_gold import GoldCase
from scix.eval.claim_blame_runner import (
    ALLOWED_TOOLS,
    OAuthClaimBlameRunner,
    build_investigator_prompt,
    load_observations,
    oauth_environment,
)
from scripts.eval_claim_blame_gold import main


def _case() -> GoldCase:
    return GoldCase(
        case_id="h0-tension",
        claim_text="Local measurements of H0 disagree with CMB inference.",
        verified_origin_bibcode="2011ApJ...730..119R",
        acceptable_origin_bibcodes=("2011ApJ...730..119R",),
        expected_retraction_bibcodes=(),
        expected_lineage_bibcodes=(),
        label_status="verified",
        source_citations=("https://ui.adsabs.harvard.edu/abs/2011ApJ...730..119R",),
    )


def test_prompt_requires_investigator_claim_blame_and_structured_output() -> None:
    prompt = build_investigator_prompt(_case())

    assert "deep_search_investigator" in prompt
    assert "claim_blame" in prompt
    assert "ranked_origin_bibcodes" in prompt
    assert "Do not use the candidate label" in prompt


def test_oauth_runner_parses_claude_json_envelope(tmp_path: Path) -> None:
    calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd, env))
        payload = {
            "result": json.dumps(
                {
                    "ranked_origin_bibcodes": ["2011ApJ...730..119R"],
                    "retraction_warnings": [],
                    "lineage_bibcodes": ["2011ApJ...730..119R"],
                    "coverage": {"covered_seeds": 1, "total_seeds": 1},
                }
            )
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    runner = OAuthClaimBlameRunner(repo_root=tmp_path, run_command=fake_run)
    observation = runner(_case())

    assert calls[0][0][:2] == ["claude", "-p"]
    tools_index = calls[0][0].index("--tools")
    assert calls[0][0][tools_index + 1] == "Agent"
    allowed_index = calls[0][0].index("--allowedTools")
    assert calls[0][0][allowed_index + 1].split(",") == list(ALLOWED_TOOLS)
    assert "Agent(deep_search_investigator)" in ALLOWED_TOOLS
    assert "mcp__scix__claim_blame" in ALLOWED_TOOLS
    assert calls[0][1] == tmp_path
    assert "ANTHROPIC_API_KEY" not in calls[0][2]
    assert observation.ranked_origin_bibcodes == ("2011ApJ...730..119R",)
    assert observation.error is None


def test_oauth_environment_removes_paid_provider_configuration() -> None:
    source = {
        "PATH": "/bin",
        "ANTHROPIC_API_KEY": "paid-key",
        "ANTHROPIC_AUTH_TOKEN": "paid-token",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_USE_VERTEX": "1",
        "CLAUDE_CODE_USE_FOUNDRY": "1",
        "AWS_BEARER_TOKEN_BEDROCK": "paid-token",
    }

    assert oauth_environment(source) == {"PATH": "/bin"}


def test_oauth_runner_surfaces_subprocess_failure(tmp_path: Path) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["claude"], 2, "", "authentication failed")

    observation = OAuthClaimBlameRunner(repo_root=tmp_path, run_command=fake_run)(_case())

    assert observation.error == "claude -p exited 2: authentication failed"
    assert observation.ranked_origin_bibcodes == ()


def test_oauth_runner_records_malformed_agent_json_as_failure(tmp_path: Path) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["claude"], 0, json.dumps({"result": '{"message":"Tool unavailable"}'}), ""
        )

    observation = OAuthClaimBlameRunner(repo_root=tmp_path, run_command=fake_run)(_case())

    assert observation.error == "h0-tension: missing required field coverage"


def test_load_observations_round_trips_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "h0-tension",
                "ranked_origin_bibcodes": ["2011ApJ...730..119R"],
                "retraction_warnings": [],
                "lineage_bibcodes": ["2011ApJ...730..119R"],
                "coverage": {"covered_seeds": 1, "total_seeds": 2},
                "error": None,
            }
        )
        + "\n"
    )

    observations = load_observations(path)

    assert observations[0].case_id == "h0-tension"
    assert observations[0].total_seeds == 2


def test_offline_cli_writes_report(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    observations_path = tmp_path / "observations.jsonl"
    report_path = tmp_path / "report.md"
    cases_path.write_text(
        json.dumps(
            {
                "id": "h0-tension",
                "claim_text": _case().claim_text,
                "verified_origin_bibcode": _case().verified_origin_bibcode,
                "acceptable_origin_bibcodes": list(_case().acceptable_origin_bibcodes),
                "expected_retraction_bibcodes": [],
                "expected_lineage_bibcodes": [],
                "label_status": "verified",
                "source_citations": list(_case().source_citations),
            }
        )
        + "\n"
    )
    observations_path.write_text(
        json.dumps(
            {
                "case_id": "h0-tension",
                "ranked_origin_bibcodes": ["2011ApJ...730..119R"],
                "retraction_warnings": [],
                "lineage_bibcodes": ["2011ApJ...730..119R"],
                "coverage": {"covered_seeds": 1, "total_seeds": 2},
                "error": None,
            }
        )
        + "\n"
    )

    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--observations",
            str(observations_path),
            "--report",
            str(report_path),
            "--covered-edges",
            "821000",
            "--total-edges",
            "299000000",
        ]
    )

    assert exit_code == 0
    assert "Origin recall@1 | 100.00%" in report_path.read_text()


def test_oauth_cli_requires_explicit_prod_flag(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "h0-tension",
                "claim_text": _case().claim_text,
                "verified_origin_bibcode": _case().verified_origin_bibcode,
                "acceptable_origin_bibcodes": list(_case().acceptable_origin_bibcodes),
                "expected_retraction_bibcodes": [],
                "expected_lineage_bibcodes": [],
                "label_status": "verified",
                "source_citations": list(_case().source_citations),
            }
        )
        + "\n"
    )

    with pytest.raises(SystemExit, match="--allow-prod"):
        main(
            [
                "--cases",
                str(cases_path),
                "--run-oauth",
                "--covered-edges",
                "1",
                "--total-edges",
                "1",
            ]
        )
