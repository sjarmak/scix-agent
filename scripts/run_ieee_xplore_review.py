#!/usr/bin/env python3
"""Run the bounded, publisher-native IEEE review plan.

The private checkpoint contains DOI and IEEE article identifiers only.  The
public summary contains aggregate counts and query provenance, with no record
identities or IEEE-returned descriptive content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from scix.review_sources.ieee_xplore import (
    IeeeProtocolError,
    IeeeSearchResult,
    IeeeSearchSpec,
    IeeeXploreClient,
    build_public_summary,
    load_search_plan,
    write_private_manifest,
)


def _write_public_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _checkpoint_payload(
    *,
    run_date: str,
    plan_path: Path,
    results: list[IeeeSearchResult],
    status: str,
    api_calls_attempted: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_date": run_date,
        "source_lane": "publisher_native_ieee",
        "status": status,
        "api_calls_attempted": api_calls_attempted,
        "retention_boundary": (
            "Local DOI and IEEE article identifiers only. No IEEE-returned title, abstract, "
            "author, or full-text content is retained."
        ),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "results": [result.to_private_dict() for result in results],
    }


def _restore_checkpoint(
    *,
    path: Path,
    plan_path: Path,
    plan_specs: tuple[IeeeSearchSpec, ...],
    run_date: str,
    client: IeeeXploreClient,
) -> list[IeeeSearchResult]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise IeeeProtocolError("Could not read the IEEE review checkpoint") from exc
    if not isinstance(raw, dict):
        raise IeeeProtocolError("IEEE review checkpoint must be an object")
    expected_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if raw.get("plan_sha256") != expected_hash:
        raise IeeeProtocolError("IEEE review checkpoint belongs to a different search plan")
    if raw.get("source_lane") != "publisher_native_ieee":
        raise IeeeProtocolError("IEEE review checkpoint has the wrong source lane")
    raw_results = raw.get("results")
    if not isinstance(raw_results, list):
        raise IeeeProtocolError("IEEE review checkpoint results must be a list")
    if len(raw_results) > len(plan_specs):
        raise IeeeProtocolError("IEEE review checkpoint contains too many results")
    restored = [
        IeeeSearchResult.from_private_dict(item, expected_spec=plan_specs[index])
        for index, item in enumerate(raw_results)
    ]
    if raw.get("run_date") == run_date:
        prior_calls = raw.get("api_calls_attempted", sum(item.calls for item in restored))
        if isinstance(prior_calls, bool) or not isinstance(prior_calls, int):
            raise IeeeProtocolError("IEEE review checkpoint has an invalid call count")
        client.account_for_prior_calls(prior_calls)
    return restored


def run_review(
    *,
    plan_path: Path | str,
    private_output: Path | str,
    public_output: Path | str,
    run_date: str,
    client: IeeeXploreClient,
    max_pages: int = 2,
) -> dict[str, object]:
    """Run all searches, checkpointing each completed topic-by-venue cell."""
    resolved_plan = Path(plan_path)
    resolved_private = Path(private_output)
    resolved_public = Path(public_output)
    plan = load_search_plan(resolved_plan)
    results = _restore_checkpoint(
        path=resolved_private,
        plan_path=resolved_plan,
        plan_specs=plan.specs,
        run_date=run_date,
        client=client,
    )

    write_private_manifest(
        resolved_private,
        _checkpoint_payload(
            run_date=run_date,
            plan_path=resolved_plan,
            results=results,
            status="incomplete",
            api_calls_attempted=client.calls_used,
        ),
    )
    try:
        for spec in plan.specs[len(results) :]:
            results.append(client.search(spec, max_pages=max_pages))
            write_private_manifest(
                resolved_private,
                _checkpoint_payload(
                    run_date=run_date,
                    plan_path=resolved_plan,
                    results=results,
                    status="incomplete",
                    api_calls_attempted=client.calls_used,
                ),
            )
    except BaseException:
        write_private_manifest(
            resolved_private,
            _checkpoint_payload(
                run_date=run_date,
                plan_path=resolved_plan,
                results=results,
                status="incomplete",
                api_calls_attempted=client.calls_used,
            ),
        )
        raise

    write_private_manifest(
        resolved_private,
        _checkpoint_payload(
            run_date=run_date,
            plan_path=resolved_plan,
            results=results,
            status="complete",
            api_calls_attempted=client.calls_used,
        ),
    )
    summary = build_public_summary(results, run_date=run_date)
    summary["api_calls"] = client.calls_used
    summary["plan_sha256"] = hashlib.sha256(resolved_plan.read_bytes()).hexdigest()
    _write_public_json(resolved_public, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--run-date", default=date.today().isoformat())
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--daily-call-limit", type=int, default=180)
    parser.add_argument("--requests-per-second", type=float, default=9.0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    client = IeeeXploreClient(
        daily_call_limit=args.daily_call_limit,
        requests_per_second=args.requests_per_second,
    )
    summary = run_review(
        plan_path=args.plan,
        private_output=args.private_output,
        public_output=args.public_output,
        run_date=args.run_date,
        client=client,
        max_pages=args.max_pages,
    )
    print(
        "IEEE review complete: "
        f"queries={summary['query_count']} "
        f"calls={summary['api_calls']} "
        f"unique_identifiers={summary['unique_identifier_records']} "
        f"truncated={summary['truncated_searches']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
