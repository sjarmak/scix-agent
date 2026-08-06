"""Tests for the bounded Scopus review source lane.

All provider responses are synthetic.  The suite contains no Scopus metadata,
credentials, abstracts, or full text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from scix.review_sources.scopus import (
    ScopusAccessError,
    ScopusCallBudgetExceeded,
    ScopusClient,
    ScopusProtocolError,
    ScopusSearchSpec,
    build_public_summary,
    load_search_plan,
)
from scripts.run_scopus_review import run_review


class FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def result_page(total: int, entries: list[dict[str, str]]) -> dict[str, object]:
    return {
        "search-results": {
            "opensearch:totalResults": str(total),
            "entry": entries,
        }
    }


def test_client_requires_key_and_valid_limits() -> None:
    with pytest.raises(ScopusAccessError):
        ScopusClient(api_key="")
    with pytest.raises(ValueError):
        ScopusClient(api_key="key", page_size=201)
    with pytest.raises(ValueError):
        ScopusClient(api_key="key", call_limit=0)
    with pytest.raises(ValueError):
        ScopusClient(api_key="key", requests_per_second=0)


def test_search_pages_and_retains_only_identifiers() -> None:
    pages = [
        result_page(
            3,
            [
                {
                    "prism:doi": "https://doi.org/10.1000/ONE",
                    "eid": "2-s2.0-1",
                    "dc:title": "must not be retained",
                    "dc:description": "must not be retained",
                },
                {"prism:doi": "10.1000/two", "dc:identifier": "SCOPUS_ID:2"},
            ],
        ),
        result_page(3, [{"eid": "2-s2.0-3"}]),
    ]
    calls: list[dict[str, object]] = []

    def fetcher(url: str, **kwargs: object) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(200, pages.pop(0))

    client = ScopusClient(
        api_key="secret-key",
        institution_token="secret-token",
        fetcher=fetcher,
        page_size=2,
        call_limit=10,
    )
    result = client.search(
        ScopusSearchSpec(
            key="retrieval--tse",
            topic_query='TITLE-ABS-KEY("code search")',
            venue_filter='EXACTSRCTITLE("Transactions")',
            start_year=2018,
            end_year=2026,
        )
    )

    assert result.total_records == 3
    assert result.calls == 2
    assert result.pages == 2
    assert result.truncated is False
    assert [item.doi for item in result.identifiers] == [
        "10.1000/one",
        "10.1000/two",
        None,
    ]
    assert [item.eid for item in result.identifiers] == ["2-s2.0-1", "2", "2-s2.0-3"]
    assert "dc:title" not in result.to_private_dict()["identifiers"][0]
    assert calls[0]["headers"]["X-ELS-APIKey"] == "secret-key"  # type: ignore[index]
    assert calls[0]["headers"]["X-ELS-Insttoken"] == "secret-token"  # type: ignore[index]
    assert calls[1]["params"]["start"] == "2"  # type: ignore[index]


def test_search_stops_at_max_pages_and_marks_truncation() -> None:
    client = ScopusClient(
        api_key="key",
        fetcher=lambda _url, **_kwargs: FakeResponse(
            200,
            result_page(9, [{"eid": "1"}, {"eid": "2"}]),
        ),
        page_size=2,
    )
    result = client.search(
        ScopusSearchSpec(
            key="bounded",
            topic_query="TITLE-ABS-KEY(agent)",
            venue_filter="SRCTITLE(software)",
            start_year=2018,
            end_year=2026,
        ),
        max_pages=1,
    )
    assert result.calls == 1
    assert result.truncated is True


def test_access_error_redacts_key_and_institution_token() -> None:
    key = "do-not-leak-key"
    token = "do-not-leak-token"
    client = ScopusClient(
        api_key=key,
        institution_token=token,
        fetcher=lambda _url, **_kwargs: FakeResponse(
            403,
            {},
            text="Insufficient entitlement " + key + " " + token,
        ),
    )
    with pytest.raises(ScopusAccessError) as exc_info:
        client.search(
            ScopusSearchSpec(
                key="smoke",
                topic_query="TITLE-ABS-KEY(agent)",
                venue_filter="SRCTITLE(software)",
                start_year=2018,
                end_year=2026,
            )
        )
    message = str(exc_info.value)
    assert key not in message
    assert token not in message
    assert "Insufficient entitlement" in message


def test_transport_error_redacts_credentials() -> None:
    key = "secret-key"

    def fetcher(_url: str, **_kwargs: object) -> FakeResponse:
        raise requests.ConnectionError("failed with " + key)

    client = ScopusClient(api_key=key, fetcher=fetcher)
    with pytest.raises(ScopusAccessError) as exc_info:
        client.search(
            ScopusSearchSpec(
                key="smoke",
                topic_query="TITLE-ABS-KEY(agent)",
                venue_filter="SRCTITLE(software)",
                start_year=2018,
                end_year=2026,
            )
        )
    assert key not in str(exc_info.value)


def test_call_budget_blocks_before_overrun() -> None:
    calls = 0

    def fetcher(_url: str, **_kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(200, result_page(0, []))

    client = ScopusClient(api_key="key", fetcher=fetcher, call_limit=1)
    spec = ScopusSearchSpec(
        key="one",
        topic_query="TITLE-ABS-KEY(agent)",
        venue_filter="SRCTITLE(software)",
        start_year=2018,
        end_year=2026,
    )
    client.search(spec)
    with pytest.raises(ScopusCallBudgetExceeded):
        client.search(
            ScopusSearchSpec(
                key="two",
                topic_query="TITLE-ABS-KEY(review)",
                venue_filter="SRCTITLE(software)",
                start_year=2018,
                end_year=2026,
            )
        )
    assert calls == 1


def test_prior_calls_count_against_budget() -> None:
    client = ScopusClient(
        api_key="key",
        fetcher=lambda _url, **_kwargs: FakeResponse(200, result_page(0, [])),
        call_limit=2,
    )
    client.account_for_prior_calls(2)
    with pytest.raises(ScopusCallBudgetExceeded):
        client.search(
            ScopusSearchSpec(
                key="blocked",
                topic_query="TITLE-ABS-KEY(agent)",
                venue_filter="SRCTITLE(software)",
                start_year=2018,
                end_year=2026,
            )
        )
    with pytest.raises(ValueError):
        client.account_for_prior_calls(-1)


def test_invalid_payload_is_protocol_error() -> None:
    client = ScopusClient(
        api_key="key",
        fetcher=lambda _url, **_kwargs: FakeResponse(200, {"unexpected": {}}),
    )
    with pytest.raises(ScopusProtocolError):
        client.search(
            ScopusSearchSpec(
                key="bad",
                topic_query="TITLE-ABS-KEY(agent)",
                venue_filter="SRCTITLE(software)",
                start_year=2018,
                end_year=2026,
            )
        )


def test_invalid_json_and_non_object_entries_are_protocol_errors() -> None:
    invalid_json = ScopusClient(
        api_key="key",
        fetcher=lambda _url, **_kwargs: FakeResponse(
            200,
            ValueError("invalid"),
        ),
    )
    with pytest.raises(ScopusProtocolError, match="invalid JSON"):
        invalid_json.search(
            ScopusSearchSpec(
                key="bad-json",
                topic_query="TITLE-ABS-KEY(agent)",
                venue_filter="SRCTITLE(software)",
                start_year=2018,
                end_year=2026,
            )
        )

    bad_entry = ScopusClient(
        api_key="key",
        fetcher=lambda _url, **_kwargs: FakeResponse(
            200,
            {
                "search-results": {
                    "opensearch:totalResults": "1",
                    "entry": ["not-an-object"],
                }
            },
        ),
    )
    with pytest.raises(ScopusProtocolError, match="non-object entry"):
        bad_entry.search(
            ScopusSearchSpec(
                key="bad-entry",
                topic_query="TITLE-ABS-KEY(agent)",
                venue_filter="SRCTITLE(software)",
                start_year=2018,
                end_year=2026,
            )
        )


def test_total_change_during_pagination_is_protocol_error() -> None:
    pages = [
        result_page(2, [{"eid": "1"}]),
        result_page(3, [{"eid": "2"}]),
    ]
    client = ScopusClient(
        api_key="key",
        fetcher=lambda _url, **_kwargs: FakeResponse(200, pages.pop(0)),
        page_size=1,
    )
    with pytest.raises(ScopusProtocolError, match="changed during pagination"):
        client.search(
            ScopusSearchSpec(
                key="changing-total",
                topic_query="TITLE-ABS-KEY(agent)",
                venue_filter="SRCTITLE(software)",
                start_year=2018,
                end_year=2026,
            )
        )


def test_load_search_plan_expands_topics_and_venues(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_lane": "index_native_scopus",
                "years": [2018, 2026],
                "topics": [
                    {"id": "evaluation", "query": "TITLE-ABS-KEY(evaluation)"},
                    {"id": "retrieval", "query": "TITLE-ABS-KEY(retrieval)"},
                ],
                "venues": [
                    {"id": "tse", "filter": "EXACTSRCTITLE(TSE)"},
                    {"id": "icse", "filter": "CONFNAME(ICSE)"},
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
    assert "PUBYEAR > 2017" in plan.specs[0].query
    assert "PUBYEAR < 2027" in plan.specs[0].query


def test_release_plan_has_64_bounded_cells() -> None:
    plan = load_search_plan("docs/eval/erca_scopus_search_plan_2026-08.json")
    assert len(plan.specs) == 64
    assert len({spec.fingerprint() for spec in plan.specs}) == 64


def test_public_summary_excludes_identifiers_and_credentials() -> None:
    client = ScopusClient(
        api_key="secret-key",
        fetcher=lambda _url, **_kwargs: FakeResponse(
            200,
            result_page(1, [{"prism:doi": "10.1000/private", "eid": "2-s2.0-private"}]),
        ),
    )
    result = client.search(
        ScopusSearchSpec(
            key="one",
            topic_query="TITLE-ABS-KEY(agent)",
            venue_filter="SRCTITLE(software)",
            start_year=2018,
            end_year=2026,
        )
    )
    summary = build_public_summary([result], run_date="2026-08-06")
    rendered = json.dumps(summary)
    assert summary["source_lane"] == "index_native_scopus"
    assert summary["unique_identifier_records"] == 1
    assert "10.1000/private" not in rendered
    assert "2-s2.0-private" not in rendered
    assert "secret-key" not in rendered


def test_runner_writes_private_checkpoint_and_public_aggregate(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_lane": "index_native_scopus",
                "years": [2018, 2026],
                "topics": [{"id": "evaluation", "query": "TITLE-ABS-KEY(evaluation)"}],
                "venues": [{"id": "tse", "filter": "EXACTSRCTITLE(TSE)"}],
            }
        )
    )
    private = tmp_path / "private.json"
    public = tmp_path / "public.json"
    client = ScopusClient(
        api_key="secret-key",
        fetcher=lambda _url, **_kwargs: FakeResponse(
            200,
            result_page(
                1,
                [
                    {
                        "prism:doi": "10.1000/private",
                        "eid": "2-s2.0-private",
                        "dc:title": "not retained",
                    }
                ],
            ),
        ),
    )
    summary = run_review(
        plan_path=plan_path,
        private_output=private,
        public_output=public,
        run_date="2026-08-06",
        client=client,
    )
    assert private.stat().st_mode & 0o777 == 0o600
    assert summary["unique_identifier_records"] == 1
    private_text = private.read_text()
    public_text = public.read_text()
    assert "10.1000/private" in private_text
    assert "not retained" not in private_text
    assert "10.1000/private" not in public_text
