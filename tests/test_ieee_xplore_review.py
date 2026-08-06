"""Tests for the bounded IEEE Xplore review client.

The fixtures are synthetic.  No IEEE-returned titles, abstracts, or API keys
belong in this test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from scix.review_sources.ieee_xplore import (
    IeeeAccessError,
    IeeeCallBudgetExceeded,
    IeeeProtocolError,
    IeeeSearchSpec,
    IeeeXploreClient,
    build_public_summary,
    load_search_plan,
    write_private_manifest,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_search_pages_and_retains_identifiers_without_content() -> None:
    calls: list[dict[str, object]] = []
    pages = [
        {
            "total_records": 3,
            "articles": [
                {
                    "doi": "https://doi.org/10.1109/EXAMPLE.1",
                    "article_number": "1001",
                    "title": "must not be retained",
                    "abstract": "must not be retained either",
                },
                {"doi": "10.1109/example.2", "article_number": "1002"},
            ],
        },
        {
            "total_records": 3,
            "articles": [
                {"doi": "10.1109/example.2", "article_number": "1002"},
                {"article_number": "1003"},
            ],
        },
    ]

    def fetcher(url: str, **kwargs: object) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, pages.pop(0))

    client = IeeeXploreClient(
        api_key="secret-test-key",
        fetcher=fetcher,
        page_size=2,
        daily_call_limit=10,
    )
    result = client.search(
        IeeeSearchSpec(
            key="retrieval-tse",
            querytext='("code search" OR "repository retrieval")',
            publication_title="IEEE Transactions on Software Engineering",
            start_year=2018,
            end_year=2026,
        )
    )

    assert result.total_records == 3
    assert result.calls == 2
    assert result.pages == 2
    assert result.truncated is False
    assert [item.doi for item in result.identifiers] == [
        "10.1109/example.1",
        "10.1109/example.2",
        None,
    ]
    assert [item.article_number for item in result.identifiers] == ["1001", "1002", "1003"]
    assert "title" not in result.to_private_dict()["identifiers"][0]
    assert "abstract" not in result.to_private_dict()["identifiers"][0]
    assert calls[0]["params"]["apikey"] == "secret-test-key"  # type: ignore[index]
    assert calls[1]["params"]["start_record"] == "3"  # type: ignore[index]


def test_search_stops_at_max_pages_and_marks_truncation() -> None:
    def fetcher(_url: str, **_kwargs: object) -> FakeResponse:
        return FakeResponse(
            200,
            {
                "total_records": 9,
                "articles": [
                    {"article_number": "1"},
                    {"article_number": "2"},
                ],
            },
        )

    client = IeeeXploreClient(api_key="key", fetcher=fetcher, page_size=2)
    result = client.search(
        IeeeSearchSpec(key="bounded", querytext="coding agent"),
        max_pages=1,
    )

    assert result.calls == 1
    assert result.truncated is True


def test_access_error_redacts_key_from_provider_body() -> None:
    key = "do-not-leak-this-key"

    def fetcher(_url: str, **_kwargs: object) -> FakeResponse:
        return FakeResponse(403, {}, text=f"Developer Inactive: {key}")

    client = IeeeXploreClient(api_key=key, fetcher=fetcher)

    with pytest.raises(IeeeAccessError) as exc_info:
        client.search(IeeeSearchSpec(key="smoke", querytext="software engineering"))

    message = str(exc_info.value)
    assert key not in message
    assert "Developer Inactive" in message


def test_transport_error_redacts_key_from_exception() -> None:
    key = "do-not-leak-this-key"

    def fetcher(_url: str, **_kwargs: object) -> FakeResponse:
        raise requests.ConnectionError(f"failed URL ?apikey={key}")

    client = IeeeXploreClient(api_key=key, fetcher=fetcher)

    with pytest.raises(IeeeAccessError) as exc_info:
        client.search(IeeeSearchSpec(key="smoke", querytext="software engineering"))

    assert key not in str(exc_info.value)


def test_call_budget_blocks_request_before_overrun() -> None:
    call_count = 0

    def fetcher(_url: str, **_kwargs: object) -> FakeResponse:
        nonlocal call_count
        call_count += 1
        return FakeResponse(200, {"total_records": 1, "articles": [{"article_number": "1"}]})

    client = IeeeXploreClient(api_key="key", fetcher=fetcher, daily_call_limit=1)
    client.search(IeeeSearchSpec(key="first", querytext="first"))

    with pytest.raises(IeeeCallBudgetExceeded):
        client.search(IeeeSearchSpec(key="second", querytext="second"))

    assert call_count == 1


def test_prior_calls_count_against_daily_budget() -> None:
    client = IeeeXploreClient(
        api_key="key",
        fetcher=lambda _url, **_kwargs: FakeResponse(200, {"total_records": 0, "articles": []}),
        daily_call_limit=2,
    )
    client.account_for_prior_calls(2)

    with pytest.raises(IeeeCallBudgetExceeded):
        client.search(IeeeSearchSpec(key="blocked", querytext="blocked"))


def test_invalid_payload_is_a_protocol_error() -> None:
    client = IeeeXploreClient(
        api_key="key",
        fetcher=lambda _url, **_kwargs: FakeResponse(200, {"articles": "not-a-list"}),
    )

    with pytest.raises(IeeeProtocolError):
        client.search(IeeeSearchSpec(key="bad", querytext="bad"))


def test_load_search_plan_expands_topics_across_venues(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_lane": "publisher_native_ieee",
                "years": [2018, 2026],
                "topics": [
                    {"id": "evaluation", "querytext": "agent evaluation"},
                    {"id": "retrieval", "querytext": "code search"},
                ],
                "venues": [
                    {"id": "tse", "publication_title": "Transactions on Software Engineering"},
                    {"id": "icse", "publication_title": "Software Engineering"},
                ],
            }
        )
    )

    plan = load_search_plan(plan_path)

    assert [spec.key for spec in plan.specs] == [
        "evaluation--tse",
        "evaluation--icse",
        "retrieval--tse",
        "retrieval--icse",
    ]
    assert all(spec.start_year == 2018 for spec in plan.specs)
    assert all(spec.end_year == 2026 for spec in plan.specs)


def test_public_summary_excludes_identifiers_and_api_key() -> None:
    client = IeeeXploreClient(
        api_key="secret-key",
        fetcher=lambda _url, **_kwargs: FakeResponse(
            200,
            {
                "total_records": 1,
                "articles": [{"doi": "10.1109/example.1", "article_number": "1001"}],
            },
        ),
    )
    result = client.search(IeeeSearchSpec(key="one", querytext="coding agent"))

    summary = build_public_summary([result], run_date="2026-08-06")
    rendered = json.dumps(summary)

    assert summary["source_lane"] == "publisher_native_ieee"
    assert summary["unique_identifier_records"] == 1
    assert "10.1109/example.1" not in rendered
    assert "1001" not in rendered
    assert "secret-key" not in rendered


def test_private_manifest_is_owner_readable_only(tmp_path: Path) -> None:
    output = tmp_path / "private.json"
    write_private_manifest(output, {"records": []})

    assert output.stat().st_mode & 0o777 == 0o600
