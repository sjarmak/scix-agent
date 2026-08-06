"""Bounded, identifier-only Scopus Search API review client.

This source lane is a publisher/index-native supplement for a literature
review.  It is deliberately separate from SciX ingest and stores only DOI and
Scopus EID identities in private checkpoints.  Public summaries contain query
provenance and aggregate counts, not provider-returned descriptive metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

SCOPUS_SEARCH_ENDPOINT = "https://api.elsevier.com/content/search/scopus"
SOURCE_LANE = "index_native_scopus"


class _Response(Protocol):
    status_code: int
    text: str

    def json(self) -> object: ...


Fetcher = Callable[..., _Response]


class ScopusReviewError(RuntimeError):
    """Base error for the bounded Scopus review client."""


class ScopusAccessError(ScopusReviewError):
    """Raised when credentials, entitlements, transport, or access fail."""


class ScopusProtocolError(ScopusReviewError):
    """Raised when a response does not match the documented API contract."""


class ScopusCallBudgetExceeded(ScopusReviewError):
    """Raised before a request would exceed the configured run budget."""


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
    if value.upper().startswith("SCOPUS_ID:"):
        value = value.split(":", 1)[1]
    return value or None


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ScopusProtocolError(f"Scopus returned an invalid {field}")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ScopusProtocolError(f"Scopus returned an invalid {field}") from exc
    if parsed < 0:
        raise ScopusProtocolError(f"Scopus returned a negative {field}")
    return parsed


def _redact(message: object, secrets: tuple[str, ...]) -> str:
    rendered = str(message)
    for secret in secrets:
        if secret:
            rendered = rendered.replace(secret, "<redacted>")
    return rendered


@dataclass(frozen=True)
class ScopusSearchSpec:
    """One topic-by-venue Scopus query."""

    key: str
    topic_query: str
    venue_filter: str
    start_year: int
    end_year: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9]+(?:-{1,2}[a-z0-9]+)*", self.key):
            raise ValueError("Search key must be lowercase, hyphenated, and stable")
        if not self.topic_query.strip() or not self.venue_filter.strip():
            raise ValueError("topic_query and venue_filter must not be empty")
        if self.start_year < 1900 or self.end_year < 1900:
            raise ValueError("Search years are outside the supported range")
        if self.start_year > self.end_year:
            raise ValueError("start_year must not be after end_year")

    @property
    def query(self) -> str:
        return (
            f"({self.topic_query}) AND ({self.venue_filter}) "
            f"AND PUBYEAR > {self.start_year - 1} AND PUBYEAR < {self.end_year + 1}"
        )

    def request_params(self, *, start: int, page_size: int) -> dict[str, str]:
        return {
            "query": self.query,
            "start": str(start),
            "count": str(page_size),
            "view": "STANDARD",
            "sort": "coverDate",
        }

    def to_public_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "topic_query": self.topic_query,
            "venue_filter": self.venue_filter,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "query": self.query,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ScopusIdentifier:
    """Minimum record identity retained from Scopus."""

    doi: str | None
    eid: str | None

    def to_private_dict(self) -> dict[str, str | None]:
        return {"doi": self.doi, "eid": self.eid}


@dataclass(frozen=True)
class ScopusSearchResult:
    """Bounded result for one search specification."""

    spec: ScopusSearchSpec
    total_records: int
    identifiers: tuple[ScopusIdentifier, ...]
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
        expected_spec: ScopusSearchSpec,
    ) -> "ScopusSearchResult":
        """Restore one identifier-only checkpoint and validate its plan cell."""
        if not isinstance(raw, dict):
            raise ScopusProtocolError("Scopus checkpoint result must be an object")
        if raw.get("source_lane") != SOURCE_LANE:
            raise ScopusProtocolError("Scopus checkpoint result has the wrong source lane")
        if raw.get("search") != expected_spec.to_public_dict():
            raise ScopusProtocolError("Scopus checkpoint search does not match the plan")
        fingerprint = raw.get("query_fingerprint")
        if fingerprint != expected_spec.fingerprint():
            raise ScopusProtocolError("Scopus checkpoint fingerprint does not match the plan")
        raw_identifiers = raw.get("identifiers")
        if not isinstance(raw_identifiers, list):
            raise ScopusProtocolError("Scopus checkpoint identifiers must be a list")
        identifiers: list[ScopusIdentifier] = []
        for item in raw_identifiers:
            if not isinstance(item, dict):
                raise ScopusProtocolError("Scopus checkpoint identifier must be an object")
            identifier = ScopusIdentifier(
                doi=_normalize_doi(item.get("doi")),
                eid=_clean_identifier(item.get("eid")),
            )
            if identifier.doi is None and identifier.eid is None:
                raise ScopusProtocolError("Scopus checkpoint identifier is empty")
            identifiers.append(identifier)
        return cls(
            spec=expected_spec,
            total_records=_integer(raw.get("total_records"), field="total_records"),
            identifiers=tuple(identifiers),
            calls=_integer(raw.get("calls"), field="calls"),
            pages=_integer(raw.get("pages"), field="pages"),
            truncated=bool(raw.get("truncated")),
            query_fingerprint=str(fingerprint),
        )


@dataclass(frozen=True)
class ScopusSearchPlan:
    """Expanded topic-by-venue review plan."""

    schema_version: int
    source_lane: str
    specs: tuple[ScopusSearchSpec, ...]


class ScopusClient:
    """Rate-limited Scopus Search API client retaining identifiers only."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        institution_token: str | None = None,
        fetcher: Fetcher = requests.get,
        endpoint: str = SCOPUS_SEARCH_ENDPOINT,
        page_size: int = 200,
        call_limit: int = 180,
        requests_per_second: float = 8.0,
        timeout: float = 30.0,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("ELSEVIER_API_KEY")
        resolved_token = (
            institution_token
            if institution_token is not None
            else os.environ.get("ELSEVIER_INST_TOKEN")
        )
        self._api_key = (resolved_key or "").strip()
        self._institution_token = (resolved_token or "").strip()
        if not self._api_key:
            raise ScopusAccessError("ELSEVIER_API_KEY is missing or empty")
        if not 1 <= page_size <= 200:
            raise ValueError("page_size must be between 1 and 200 for STANDARD view")
        if call_limit < 1:
            raise ValueError("call_limit must be positive")
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._fetcher = fetcher
        self._endpoint = endpoint
        self._page_size = page_size
        self._call_limit = call_limit
        self._requests_per_second = requests_per_second
        self._timeout = timeout
        self._calls = 0
        self._last_request_at = 0.0

    @property
    def calls_used(self) -> int:
        return self._calls

    def account_for_prior_calls(self, calls: int) -> None:
        if calls < 0:
            raise ValueError("prior call count must not be negative")
        if self._calls + calls > self._call_limit:
            raise ScopusCallBudgetExceeded(
                f"Scopus prior calls exceed the {self._call_limit}-request run budget"
            )
        self._calls += calls

    def _request(self, params: dict[str, str]) -> dict[str, Any]:
        if self._calls >= self._call_limit:
            raise ScopusCallBudgetExceeded(
                f"Scopus run call budget exhausted at {self._call_limit} requests"
            )
        minimum_interval = 1.0 / self._requests_per_second
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)

        headers = {
            "Accept": "application/json",
            "X-ELS-APIKey": self._api_key,
        }
        if self._institution_token:
            headers["X-ELS-Insttoken"] = self._institution_token
        self._calls += 1
        self._last_request_at = time.monotonic()
        secrets = (self._api_key, self._institution_token)
        try:
            response = self._fetcher(
                self._endpoint,
                params=params,
                headers=headers,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            safe = _redact(exc, secrets)
            raise ScopusAccessError(f"Scopus transport failed: {safe}") from None

        if response.status_code != 200:
            body = _redact(response.text, secrets).strip().replace("\n", " ")
            detail = body[:240] if body else "no provider detail"
            raise ScopusAccessError(
                f"Scopus request returned HTTP {response.status_code}: {detail}"
            )
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ScopusProtocolError("Scopus returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ScopusProtocolError("Scopus returned a non-object response")
        return payload

    def search(
        self,
        spec: ScopusSearchSpec,
        *,
        max_pages: int | None = None,
    ) -> ScopusSearchResult:
        """Execute a bounded search and return DOI/EID identities only."""
        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be positive")
        start = 0
        raw_records_seen = 0
        total_records: int | None = None
        pages = 0
        calls_before = self._calls
        identifiers: list[ScopusIdentifier] = []
        seen: set[tuple[str | None, str | None]] = set()

        while True:
            payload = self._request(spec.request_params(start=start, page_size=self._page_size))
            raw_results = payload.get("search-results")
            if not isinstance(raw_results, dict):
                raise ScopusProtocolError("Scopus response has no search-results object")
            current_total = _integer(
                raw_results.get("opensearch:totalResults"),
                field="opensearch:totalResults",
            )
            if total_records is None:
                total_records = current_total
            elif total_records != current_total:
                raise ScopusProtocolError("Scopus totalResults changed during pagination")
            raw_entries = raw_results.get("entry", [])
            if not isinstance(raw_entries, list):
                raise ScopusProtocolError("Scopus returned an invalid entry collection")
            pages += 1

            for raw_entry in raw_entries:
                if not isinstance(raw_entry, dict):
                    raise ScopusProtocolError("Scopus returned a non-object entry")
                identifier = ScopusIdentifier(
                    doi=_normalize_doi(raw_entry.get("prism:doi")),
                    eid=_clean_identifier(raw_entry.get("eid", raw_entry.get("dc:identifier"))),
                )
                if identifier.doi is None and identifier.eid is None:
                    continue
                key = (identifier.doi, identifier.eid)
                if key not in seen:
                    seen.add(key)
                    identifiers.append(identifier)

            page_count = len(raw_entries)
            raw_records_seen += page_count
            if raw_records_seen >= current_total or page_count == 0:
                break
            if max_pages is not None and pages >= max_pages:
                break
            start += page_count

        return ScopusSearchResult(
            spec=spec,
            total_records=total_records or 0,
            identifiers=tuple(identifiers),
            calls=self._calls - calls_before,
            pages=pages,
            truncated=raw_records_seen < (total_records or 0),
            query_fingerprint=spec.fingerprint(),
        )


def load_search_plan(path: Path | str) -> ScopusSearchPlan:
    """Load and expand a public topic-by-venue Scopus plan."""
    plan_path = Path(path)
    try:
        raw = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopusProtocolError(f"Could not load Scopus search plan: {plan_path}") from exc
    if not isinstance(raw, dict):
        raise ScopusProtocolError("Scopus search plan must be a JSON object")
    schema_version = _integer(raw.get("schema_version"), field="schema_version")
    if raw.get("source_lane") != SOURCE_LANE:
        raise ScopusProtocolError(f"Scopus search plan source_lane must be {SOURCE_LANE}")
    years = raw.get("years")
    topics = raw.get("topics")
    venues = raw.get("venues")
    if not isinstance(years, list) or len(years) != 2:
        raise ScopusProtocolError("Scopus search plan years must contain [start, end]")
    if not isinstance(topics, list) or not topics:
        raise ScopusProtocolError("Scopus search plan topics must be non-empty")
    if not isinstance(venues, list) or not venues:
        raise ScopusProtocolError("Scopus search plan venues must be non-empty")
    start_year = _integer(years[0], field="start_year")
    end_year = _integer(years[1], field="end_year")

    specs: list[ScopusSearchSpec] = []
    for topic in topics:
        if not isinstance(topic, dict):
            raise ScopusProtocolError("Scopus topic must be an object")
        topic_id = topic.get("id")
        topic_query = topic.get("query")
        if not isinstance(topic_id, str) or not isinstance(topic_query, str):
            raise ScopusProtocolError("Scopus topic requires id and query")
        for venue in venues:
            if not isinstance(venue, dict):
                raise ScopusProtocolError("Scopus venue must be an object")
            venue_id = venue.get("id")
            venue_filter = venue.get("filter")
            if not isinstance(venue_id, str) or not isinstance(venue_filter, str):
                raise ScopusProtocolError("Scopus venue requires id and filter")
            specs.append(
                ScopusSearchSpec(
                    key=f"{topic_id}--{venue_id}",
                    topic_query=topic_query,
                    venue_filter=venue_filter,
                    start_year=start_year,
                    end_year=end_year,
                )
            )
    return ScopusSearchPlan(
        schema_version=schema_version,
        source_lane=SOURCE_LANE,
        specs=tuple(specs),
    )


def build_public_summary(
    results: list[ScopusSearchResult] | tuple[ScopusSearchResult, ...],
    *,
    run_date: str,
) -> dict[str, object]:
    """Build a publishable aggregate that excludes record identities."""
    unique: set[tuple[str | None, str | None]] = set()
    for result in results:
        unique.update((item.doi, item.eid) for item in result.identifiers)
    return {
        "schema_version": 1,
        "run_date": run_date,
        "source_lane": SOURCE_LANE,
        "retention_boundary": (
            "Aggregate query provenance only; no Scopus title, abstract, author, "
            "or full-text content is included."
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


__all__ = [
    "SCOPUS_SEARCH_ENDPOINT",
    "SOURCE_LANE",
    "ScopusAccessError",
    "ScopusCallBudgetExceeded",
    "ScopusClient",
    "ScopusIdentifier",
    "ScopusProtocolError",
    "ScopusReviewError",
    "ScopusSearchPlan",
    "ScopusSearchResult",
    "ScopusSearchSpec",
    "build_public_summary",
    "load_search_plan",
]
