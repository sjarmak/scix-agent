"""CLI-level tests for the bounded IEEE review runner."""

from __future__ import annotations

import json
from pathlib import Path

from run_ieee_xplore_review import run_review

from scix.review_sources.ieee_xplore import IeeeXploreClient


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, article_number: str) -> None:
        self._article_number = article_number

    def json(self) -> object:
        return {
            "total_records": 1,
            "articles": [
                {
                    "doi": f"10.1109/example.{self._article_number}",
                    "article_number": self._article_number,
                    "title": "provider content must not reach the manifest",
                    "abstract": "provider content must not reach the manifest",
                }
            ],
        }


def _write_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_lane": "publisher_native_ieee",
                "years": [2018, 2026],
                "topics": [{"id": "evaluation", "querytext": "agent evaluation"}],
                "venues": [
                    {"id": "tse", "publication_title": "Transactions on Software Engineering"},
                    {"id": "icse", "publication_title": "Software Engineering"},
                ],
            }
        )
    )


def test_run_review_writes_private_checkpoint_and_public_summary(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    private_output = tmp_path / "private.json"
    public_output = tmp_path / "public.json"
    _write_plan(plan)
    counter = 0

    def fetcher(_url: str, **_kwargs: object) -> FakeResponse:
        nonlocal counter
        counter += 1
        return FakeResponse(str(counter))

    client = IeeeXploreClient(api_key="secret", fetcher=fetcher)

    summary = run_review(
        plan_path=plan,
        private_output=private_output,
        public_output=public_output,
        run_date="2026-08-06",
        client=client,
        max_pages=2,
    )

    private_payload = json.loads(private_output.read_text())
    public_payload = json.loads(public_output.read_text())
    assert private_payload["status"] == "complete"
    assert len(private_payload["results"]) == 2
    assert private_payload["results"][0]["identifiers"][0]["article_number"] == "1"
    assert public_payload == summary
    rendered_public = public_output.read_text()
    assert "10.1109/example" not in rendered_public
    assert "provider content" not in private_output.read_text()
    assert private_output.stat().st_mode & 0o777 == 0o600


def test_run_review_checkpoints_completed_search_before_failure(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    private_output = tmp_path / "private.json"
    public_output = tmp_path / "public.json"
    _write_plan(plan)
    counter = 0

    def fetcher(_url: str, **_kwargs: object) -> FakeResponse:
        nonlocal counter
        counter += 1
        if counter == 2:
            response = FakeResponse("2")
            response.status_code = 403
            response.text = "Developer Inactive"
            return response
        return FakeResponse(str(counter))

    client = IeeeXploreClient(api_key="secret", fetcher=fetcher)

    try:
        run_review(
            plan_path=plan,
            private_output=private_output,
            public_output=public_output,
            run_date="2026-08-06",
            client=client,
        )
    except Exception:
        pass

    private_payload = json.loads(private_output.read_text())
    assert private_payload["status"] == "incomplete"
    assert len(private_payload["results"]) == 1
    assert "error" not in private_payload
    assert not public_output.exists()


def test_run_review_resumes_completed_cells_without_requerying(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    private_output = tmp_path / "private.json"
    public_output = tmp_path / "public.json"
    _write_plan(plan)
    first_counter = 0

    def interrupted_fetcher(_url: str, **_kwargs: object) -> FakeResponse:
        nonlocal first_counter
        first_counter += 1
        response = FakeResponse(str(first_counter))
        if first_counter == 2:
            response.status_code = 503
            response.text = "temporarily unavailable"
        return response

    try:
        run_review(
            plan_path=plan,
            private_output=private_output,
            public_output=public_output,
            run_date="2026-08-06",
            client=IeeeXploreClient(api_key="secret", fetcher=interrupted_fetcher),
        )
    except Exception:
        pass

    resumed_calls = 0

    def resumed_fetcher(_url: str, **_kwargs: object) -> FakeResponse:
        nonlocal resumed_calls
        resumed_calls += 1
        return FakeResponse("2")

    summary = run_review(
        plan_path=plan,
        private_output=private_output,
        public_output=public_output,
        run_date="2026-08-06",
        client=IeeeXploreClient(api_key="secret", fetcher=resumed_fetcher),
    )

    private_payload = json.loads(private_output.read_text())
    assert resumed_calls == 1
    assert private_payload["status"] == "complete"
    assert [item["search"]["key"] for item in private_payload["results"]] == [
        "evaluation--tse",
        "evaluation--icse",
    ]
    assert summary["query_count"] == 2
