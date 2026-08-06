"""Bounded, identifier-only IEEE Xplore metadata review client.

The client queries the publisher-native Metadata Search API but deliberately
does not retain returned titles, abstracts, author lists, or full text.  It is
intended for reproducible review provenance, not for building a publisher
mirror or adding another SciX serving lane.

IEEE requires the API key as a query parameter.  This module therefore never
logs prepared URLs, response URLs, request parameters, or raw exception text
without first redacting the key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

IEEE_XPLORE_ENDPOINT = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
SOURCE_LANE = "publisher_native_ieee"


class _Response(Protocol):
    status_code: int
    text: str

    def json(self) -> object: ...


Fetcher = Callable[..., _Response]


class IeeeReviewError(RuntimeError):
    """Base error for the bounded IEEE review client."""


class IeeeAccessError(IeeeReviewError):
    """Raised when credentials, transport, or provider access fails."""


class IeeeProtocolError(IeeeReviewError):
    """Raised when the API response does not match the documented contract."""


class IeeeCallBudgetExceeded(IeeeReviewError):
    """Raised before a request would exceed the configured daily call budget."""


def _normalize_doi(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    lowered = value.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix):
            value = value[len(prefix) :]
            break
    return value.strip().lower() or None


def _clean_identifier(raw: object) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _redact(message: object, secret: str) -> str:
    rendered = str(message)
    if secret:
        rendered = rendered.replace(secret, "<redacted>")
    return rendered


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise IeeeProtocolError(f"IEEE Xplore returned an invalid {field}")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise IeeeProtocolError(f"IEEE Xplore returned an invalid {field}") from exc
    if parsed < 0:
        raise IeeeProtocolError(f"IEEE Xplore returned a negative {field}")
    return parsed


@dataclass(frozen=True)
class IeeeSearchSpec:
    """One provider-native topic-by-venue search."""

    key: str
    querytext: str
    publication_title: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    content_type: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9]+(?:-{1,2}[a-z0-9]+)*", self.key):
            raise ValueError("Search key must be lowercase, hyphenated, and stable")
        if not self.querytext.strip():
            raise ValueError("querytext must not be empty")
        if self.start_year is not None and self.start_year < 1900:
            raise ValueError("start_year is outside the supported review range")
        if self.end_year is not None and self.end_year < 1900:
            raise ValueError("end_year is outside the supported review range")
        if (
            self.start_year is not None
            and self.end_year is not None
            and self.start_year > self.end_year
        ):
            raise ValueError("start_year must not be after end_year")

    def request_params(self, *, start_record: int, page_size: int) -> dict[str, str]:
        params = {
            "querytext": self.querytext,
            "start_record": str(start_record),
            "max_records": str(page_size),
        }
        if self.publication_title:
            params["publication_title"] = self.publication_title
        if self.start_year is not None:
            params["start_year"] = str(self.start_year)
        if self.end_year is not None:
            params["end_year"] = str(self.end_year)
        if self.content_type:
            params["content_type"] = self.content_type
        return params

    def to_public_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "querytext": self.querytext,
            "publication_title": self.publication_title,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "content_type": self.content_type,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class IeeeIdentifier:
    """Minimum identity retained from an IEEE result."""

    doi: str | None
    article_number: str | None

    def to_private_dict(self) -> dict[str, str | None]:
        return {"doi": self.doi, "article_number": self.article_number}


@dataclass(frozen=True)
class IeeeSearchResult:
    """Bounded result for one search specification."""

    spec: IeeeSearchSpec
    total_records: int
    identifiers: tuple[IeeeIdentifier, ...]
    calls: int
    pages: int
    truncated: bool
    query_fingerprint: str

    def to_private_dict(self) -> dict[str, object]:
        return {
            "source_lane": SOURCE_LANE,
            "search": self.spec.to_public_dict(),
            "query_fingerprint": self.query_fingerprint,
            "total_records": self.total_records,
            "retrieved_identifier_records": len(self.identifiers),
            "calls": self.calls,
            "pages": self.pages,
            "truncated": self.truncated,
            "identifiers": [item.to_private_dict() for item in self.identifiers],
        }

    @classmethod
    def from_private_dict(
        cls,
        raw: object,
        *,
        expected_spec: IeeeSearchSpec,
    ) -> IeeeSearchResult:
        """Restore one identifier-only checkpoint after validating its plan cell."""
        if not isinstance(raw, dict):
            raise IeeeProtocolError("IEEE checkpoint result must be an object")
        if raw.get("source_lane") != SOURCE_LANE:
            raise IeeeProtocolError("IEEE checkpoint result has the wrong source lane")
        if raw.get("search") != expected_spec.to_public_dict():
            raise IeeeProtocolError("IEEE checkpoint search does not match the current plan")
        fingerprint = raw.get("query_fingerprint")
        if fingerprint != expected_spec.fingerprint():
            raise IeeeProtocolError("IEEE checkpoint query fingerprint does not match the plan")
        raw_identifiers = raw.get("identifiers")
        if not isinstance(raw_identifiers, list):
            raise IeeeProtocolError("IEEE checkpoint identifiers must be a list")
        identifiers: list[IeeeIdentifier] = []
        for item in raw_identifiers:
            if not isinstance(item, dict):
                raise IeeeProtocolError("IEEE checkpoint identifier must be an object")
            identifier = IeeeIdentifier(
                doi=_normalize_doi(item.get("doi")),
                article_number=_clean_identifier(item.get("article_number")),
            )
            if identifier.doi is None and identifier.article_number is None:
                raise IeeeProtocolError("IEEE checkpoint identifier is empty")
            identifiers.append(identifier)
        return cls(
            spec=expected_spec,
            total_records=_integer(raw.get("total_records"), field="total_records"),
            identifiers=tuple(identifiers),
            calls=_integer(raw.get("calls"), field="calls"),
            pages=_integer(raw.get("pages"), field="pages"),
            truncated=bool(raw.get("truncated")),
            query_fingerprint=fingerprint,
        )


@dataclass(frozen=True)
class IeeeSearchPlan:
    """Expanded topic-by-venue review plan."""

    schema_version: int
    source_lane: str
    specs: tuple[IeeeSearchSpec, ...]


class IeeeXploreClient:
    """Rate-limited IEEE client that retains identifiers, not content."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        fetcher: Fetcher = requests.get,
        endpoint: str = IEEE_XPLORE_ENDPOINT,
        page_size: int = 200,
        daily_call_limit: int = 200,
        requests_per_second: float = 9.0,
        timeout: float = 30.0,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("IEEE_XPLORE_API_KEY")
        self._api_key = (resolved_key or "").strip()
        if not self._api_key:
            raise IeeeAccessError("IEEE_XPLORE_API_KEY is missing or empty")
        if not 1 <= page_size <= 200:
            raise ValueError("page_size must be between 1 and 200")
        if daily_call_limit < 1:
            raise ValueError("daily_call_limit must be positive")
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._fetcher = fetcher
        self._endpoint = endpoint
        self._page_size = page_size
        self._daily_call_limit = daily_call_limit
        self._requests_per_second = requests_per_second
        self._timeout = timeout
        self._calls = 0
        self._last_request_at = 0.0

    @property
    def calls_used(self) -> int:
        return self._calls

    def account_for_prior_calls(self, calls: int) -> None:
        """Reserve same-day calls recorded by a resumable checkpoint."""
        if calls < 0:
            raise ValueError("prior call count must not be negative")
        if self._calls + calls > self._daily_call_limit:
            raise IeeeCallBudgetExceeded(
                f"IEEE Xplore prior calls exceed the {self._daily_call_limit}-request budget"
            )
        self._calls += calls

    def _request(self, params: dict[str, str]) -> dict[str, Any]:
        if self._calls >= self._daily_call_limit:
            raise IeeeCallBudgetExceeded(
                f"IEEE Xplore daily call budget exhausted at {self._daily_call_limit} requests"
            )

        minimum_interval = 1.0 / self._requests_per_second
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)

        request_params = dict(params)
        request_params["apikey"] = self._api_key
        self._calls += 1
        self._last_request_at = time.monotonic()
        try:
            response = self._fetcher(
                self._endpoint,
                params=request_params,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            safe = _redact(exc, self._api_key)
            raise IeeeAccessError(f"IEEE Xplore transport failed: {safe}") from None

        if response.status_code != 200:
            body = _redact(response.text, self._api_key).strip().replace("\n", " ")
            detail = body[:240] if body else "no provider detail"
            raise IeeeAccessError(
                f"IEEE Xplore request returned HTTP {response.status_code}: {detail}"
            )

        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise IeeeProtocolError("IEEE Xplore returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise IeeeProtocolError("IEEE Xplore returned a non-object response")
        return payload

    def search(
        self,
        spec: IeeeSearchSpec,
        *,
        max_pages: int | None = None,
    ) -> IeeeSearchResult:
        """Execute a bounded search and return DOI/article identities only."""
        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be positive")

        start_record = 1
        raw_records_seen = 0
        total_records: int | None = None
        pages = 0
        calls_before = self._calls
        identifiers: list[IeeeIdentifier] = []
        seen: set[tuple[str | None, str | None]] = set()

        while True:
            payload = self._request(
                spec.request_params(start_record=start_record, page_size=self._page_size)
            )
            pages += 1
            raw_total = payload.get("total_records", payload.get("totalfound"))
            current_total = _integer(raw_total, field="total_records")
            if total_records is None:
                total_records = current_total
            elif current_total != total_records:
                raise IeeeProtocolError("IEEE Xplore total_records changed during pagination")

            raw_articles = payload.get("articles", [])
            if not isinstance(raw_articles, list):
                raise IeeeProtocolError("IEEE Xplore returned an invalid articles collection")

            for raw_article in raw_articles:
                if not isinstance(raw_article, dict):
                    raise IeeeProtocolError("IEEE Xplore returned a non-object article record")
                identifier = IeeeIdentifier(
                    doi=_normalize_doi(raw_article.get("doi")),
                    article_number=_clean_identifier(raw_article.get("article_number")),
                )
                if identifier.doi is None and identifier.article_number is None:
                    continue
                dedupe_key = (identifier.doi, identifier.article_number)
                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    identifiers.append(identifier)

            page_count = len(raw_articles)
            raw_records_seen += page_count
            if raw_records_seen >= total_records or page_count == 0:
                break
            if max_pages is not None and pages >= max_pages:
                break
            start_record += page_count

        return IeeeSearchResult(
            spec=spec,
            total_records=total_records or 0,
            identifiers=tuple(identifiers),
            calls=self._calls - calls_before,
            pages=pages,
            truncated=raw_records_seen < (total_records or 0),
            query_fingerprint=spec.fingerprint(),
        )


def load_search_plan(path: Path | str) -> IeeeSearchPlan:
    """Load and expand a public topic-by-venue IEEE search plan."""
    plan_path = Path(path)
    try:
        raw = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise IeeeProtocolError(f"Could not load IEEE search plan: {plan_path}") from exc
    if not isinstance(raw, dict):
        raise IeeeProtocolError("IEEE search plan must be a JSON object")
    schema_version = _integer(raw.get("schema_version"), field="schema_version")
    source_lane = raw.get("source_lane")
    if source_lane != SOURCE_LANE:
        raise IeeeProtocolError(f"IEEE search plan source_lane must be {SOURCE_LANE}")

    years = raw.get("years")
    if not isinstance(years, list) or len(years) != 2:
        raise IeeeProtocolError("IEEE search plan years must contain [start, end]")
    start_year = _integer(years[0], field="start_year")
    end_year = _integer(years[1], field="end_year")

    topics = raw.get("topics")
    venues = raw.get("venues")
    if not isinstance(topics, list) or not topics:
        raise IeeeProtocolError("IEEE search plan topics must be non-empty")
    if not isinstance(venues, list) or not venues:
        raise IeeeProtocolError("IEEE search plan venues must be non-empty")

    specs: list[IeeeSearchSpec] = []
    for topic in topics:
        if not isinstance(topic, dict):
            raise IeeeProtocolError("IEEE search plan topic must be an object")
        topic_id = topic.get("id")
        querytext = topic.get("querytext")
        if not isinstance(topic_id, str) or not isinstance(querytext, str):
            raise IeeeProtocolError("IEEE search plan topic requires id and querytext")
        for venue in venues:
            if not isinstance(venue, dict):
                raise IeeeProtocolError("IEEE search plan venue must be an object")
            venue_id = venue.get("id")
            publication_title = venue.get("publication_title")
            if not isinstance(venue_id, str) or not isinstance(publication_title, str):
                raise IeeeProtocolError("IEEE search plan venue requires id and publication_title")
            content_type = venue.get("content_type")
            if content_type is not None and not isinstance(content_type, str):
                raise IeeeProtocolError("IEEE search plan content_type must be a string")
            specs.append(
                IeeeSearchSpec(
                    key=f"{topic_id}--{venue_id}",
                    querytext=querytext,
                    publication_title=publication_title,
                    start_year=start_year,
                    end_year=end_year,
                    content_type=content_type,
                )
            )

    return IeeeSearchPlan(
        schema_version=schema_version,
        source_lane=source_lane,
        specs=tuple(specs),
    )


def build_public_summary(
    results: list[IeeeSearchResult] | tuple[IeeeSearchResult, ...],
    *,
    run_date: str,
) -> dict[str, object]:
    """Build a publishable aggregate that excludes record identities."""
    unique: set[tuple[str | None, str | None]] = set()
    for result in results:
        unique.update((item.doi, item.article_number) for item in result.identifiers)
    return {
        "schema_version": 1,
        "run_date": run_date,
        "source_lane": SOURCE_LANE,
        "retention_boundary": (
            "Aggregate query provenance only; no IEEE title, abstract, author, or full-text "
            "content is included."
        ),
        "query_count": len(results),
        "api_calls": sum(result.calls for result in results),
        "provider_reported_records_across_queries": sum(result.total_records for result in results),
        "unique_identifier_records": len(unique),
        "truncated_searches": sum(1 for result in results if result.truncated),
        "searches": [
            {
                **result.spec.to_public_dict(),
                "query_fingerprint": result.query_fingerprint,
                "total_records": result.total_records,
                "retrieved_identifier_records": len(result.identifiers),
                "calls": result.calls,
                "truncated": result.truncated,
            }
            for result in results
        ],
    }


def write_private_manifest(path: Path | str, payload: object) -> None:
    """Atomically write an owner-only local manifest."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, output)
        output.chmod(0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise


__all__ = [
    "IEEE_XPLORE_ENDPOINT",
    "SOURCE_LANE",
    "IeeeAccessError",
    "IeeeCallBudgetExceeded",
    "IeeeIdentifier",
    "IeeeProtocolError",
    "IeeeReviewError",
    "IeeeSearchPlan",
    "IeeeSearchResult",
    "IeeeSearchSpec",
    "IeeeXploreClient",
    "build_public_summary",
    "load_search_plan",
    "write_private_manifest",
]
