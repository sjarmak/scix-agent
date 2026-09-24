#!/usr/bin/env python3
"""Run and score the real-corpus ``claim_blame`` gold benchmark.

Real investigator calls use ``claude -p`` and the
``deep_search_investigator`` OAuth subagent.  They are opt-in because they read
the production SciX corpus; run them through ``scix-batch`` with
``--allow-prod``.  Existing normalized observations can be scored offline.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scix.eval.claim_blame_gold import (  # noqa: E402
    evaluate_claim_blame,
    format_claim_blame_report,
    load_gold_cases,
)
from scix.eval.claim_blame_runner import (  # noqa: E402
    OAuthClaimBlameRunner,
    load_observations,
    write_observations,
)

DEFAULT_CASES = REPO_ROOT / "tests/eval/claim_blame_gold_v1.candidates.jsonl"
DEFAULT_OBSERVATIONS = REPO_ROOT / "results/claim_blame_gold_v1_runs.jsonl"
DEFAULT_REPORT = REPO_ROOT / "results/claim_blame_gold_v1.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-oauth",
        action="store_true",
        help="Run cases through the deep_search_investigator OAuth subagent.",
    )
    source.add_argument(
        "--observations",
        type=Path,
        help="Score an existing normalized observation JSONL file.",
    )
    parser.add_argument("--output-observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--covered-edges", type=int, required=True)
    parser.add_argument("--total-edges", type=int, required=True)
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Run draft cases for diagnostics; they remain excluded from headline metrics.",
    )
    parser.add_argument(
        "--allow-prod",
        action="store_true",
        help="Required with --run-oauth; confirms read-only production corpus access.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args(argv)


def _validate_oauth_execution(args: argparse.Namespace) -> None:
    if not args.allow_prod:
        raise SystemExit("--run-oauth requires --allow-prod")
    if not os.environ.get("SYSTEMD_SCOPE"):
        raise SystemExit("--run-oauth must run inside scix-batch (SYSTEMD_SCOPE is unset)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_gold_cases(args.cases, require_verified=False)

    if args.run_oauth:
        _validate_oauth_execution(args)
        runnable = [
            case for case in cases if case.label_status == "verified" or args.include_unverified
        ]
        if not runnable:
            raise SystemExit(
                "no verified cases to run; adjudicate labels or pass --include-unverified "
                "for a diagnostic run"
            )
        runner = OAuthClaimBlameRunner(repo_root=REPO_ROOT, timeout_s=args.timeout)
        observations = []
        for case in runnable:
            observations.append(runner(case))
            write_observations(args.output_observations, observations)
    else:
        observations = load_observations(args.observations)

    result = evaluate_claim_blame(
        cases,
        observations,
        covered_edges=args.covered_edges,
        total_edges=args.total_edges,
    )
    report = format_claim_blame_report(
        result,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
