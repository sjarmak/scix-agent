"""Tests for src/scix/http_client.py — ResilientClient."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from scix.http_client import CachedResponse, CircuitBreakerOpen, ResilientClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(
    status_code: int = 200, text: str = '{"ok": true}', url: str = "http://example.com"
) -> MagicMock:
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    resp.url = url
    resp.headers = {"Content-Type": "application/json"}
    resp.json.return_value = json.loads(text) if text else None
    return resp


# ---------------------------------------------------------------------------
# Retry on 503
# ---------------------------------------------------------------------------


class TestRetry:
    """Retry with exponential backoff on 5xx errors."""

    @patch("scix.http_client.requests.get")
    @patch("scix.http_client.time.sleep")
    def test_retries_on_503_then_succeeds(self, mock_sleep: MagicMock, mock_get: MagicMock) -> None:
        """Two 503s followed by a 200 — should succeed after 3 calls total."""
        mock_get.side_effect = [
            _mock_response(503, '{"error": "unavailable"}'),
            _mock_response(503, '{"error": "unavailable"}'),
            _mock_response(200, '{"ok": true}'),
        ]
        client = ResilientClient(max_retries=3, backoff_base=1.0)
        resp = client.get("http://example.com/api")

        assert resp.status_code == 200
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2
        # Backoff delays should be increasing
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays[0] < delays[1]

    @patch("scix.http_client.requests.get")
    @patch("scix.http_client.time.sleep")
    def test_exhausts_retries_on_503(self, mock_sleep: MagicMock, mock_get: MagicMock) -> None:
        """All attempts return 503 — should raise HTTPError after max_retries."""
        mock_get.return_value = _mock_response(503, '{"error": "down"}')
        client = ResilientClient(max_retries=3, backoff_base=1.0)

        with pytest.raises(requests.HTTPError):
            client.get("http://example.com/api")

        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2  # sleeps between attempts, not after last

    @patch("scix.http_client.requests.get")
    @patch("scix.http_client.time.sleep")
    def test_retries_on_connection_error(self, mock_sleep: MagicMock, mock_get: MagicMock) -> None:
        """Connection errors also trigger retry."""
        mock_get.side_effect = [
            requests.ConnectionError("refused"),
            _mock_response(200, '{"ok": true}'),
        ]
        client = ResilientClient(max_retries=2, backoff_base=1.0)
        resp = client.get("http://example.com/api")

        assert resp.status_code == 200
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Per-host rate limiting enforcement."""

    @patch("scix.http_client.requests.get")
    def test_rate_limit_delays_second_request(self, mock_get: MagicMock) -> None:
        """Two rapid requests with rate_limit=1/s — second request is delayed."""
        mock_get.return_value = _mock_response(200)
        client = ResilientClient(rate_limit=2.0)  # 2 req/s = 0.5s interval

        t0 = time.monotonic()
        client.get("http://example.com/a")
        client.get("http://example.com/b")
        elapsed = time.monotonic() - t0

        # Should have waited ~0.5s for the second request
        assert elapsed >= 0.4
        assert mock_get.call_count == 2

    @patch("scix.http_client.requests.get")
    def test_different_hosts_not_rate_limited(self, mock_get: MagicMock) -> None:
        """Requests to different hosts should not rate-limit each other."""
        mock_get.return_value = _mock_response(200)
        client = ResilientClient(rate_limit=1.0)  # 1 req/s

        t0 = time.monotonic()
        client.get("http://host-a.com/x")
        client.get("http://host-b.com/y")
        elapsed = time.monotonic() - t0

        # Different hosts — no delay between them
        assert elapsed < 0.5


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Circuit breaker opens after N failures, closes after cooldown."""

    @patch("scix.http_client.requests.get")
    @patch("scix.http_client.time.sleep")
    def test_circuit_opens_after_threshold(
        self, mock_sleep: MagicMock, mock_get: MagicMock
    ) -> None:
        """After N consecutive failures, circuit breaker opens."""
        mock_get.return_value = _mock_response(503)
        client = ResilientClient(
            max_retries=1,
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=60.0,
        )

        # Exhaust 3 consecutive failures
        for _ in range(3):
            with pytest.raises(requests.HTTPError):
                client.get("http://failing.com/api")

        # Next request should hit circuit breaker without making HTTP call
        mock_get.reset_mock()
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            client.get("http://failing.com/api")

        assert exc_info.value.host == "failing.com"
        assert mock_get.call_count == 0  # No HTTP call made

    @patch("scix.http_client.requests.get")
    @patch("scix.http_client.time.sleep")
    def test_circuit_closes_after_cooldown(
        self, mock_sleep: MagicMock, mock_get: MagicMock
    ) -> None:
        """Circuit breaker resets after cooldown period (half-open state)."""
        mock_get.return_value = _mock_response(503)
        client = ResilientClient(
            max_retries=1,
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown=1.0,
        )

        # Open the circuit
        for _ in range(2):
            with pytest.raises(requests.HTTPError):
                client.get("http://failing.com/api")

        # Verify it's open
        with pytest.raises(CircuitBreakerOpen):
            client.get("http://failing.com/api")

        # Manually advance the last_failure_time to simulate cooldown
        host_state = client._get_host_state("failing.com")
        host_state.last_failure_time = time.monotonic() - 2.0  # 2s ago, cooldown is 1s

        # Now it should allow the request again (half-open)
        mock_get.return_value = _mock_response(200)
        resp = client.get("http://failing.com/api")
        assert resp.status_code == 200

    @patch("scix.http_client.requests.get")
    @patch("scix.http_client.time.sleep")
    def test_success_resets_failure_count(self, mock_sleep: MagicMock, mock_get: MagicMock) -> None:
        """A successful request resets the consecutive failure counter."""
        client = ResilientClient(
            max_retries=1,
            circuit_breaker_threshold=3,
        )
        mock_get.return_value = _mock_response(503)

        # 2 failures (below threshold)
        for _ in range(2):
            with pytest.raises(requests.HTTPError):
                client.get("http://example.com/api")

        # 1 success resets counter
        mock_get.return_value = _mock_response(200)
        client.get("http://example.com/api")

        state = client._get_host_state("example.com")
        assert state.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Disk caching
# ---------------------------------------------------------------------------


class TestDiskCache:
    """Response caching to disk with TTL."""

    @patch("scix.http_client.requests.get")
    def test_cache_hit_no_http_call(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """Second identical request is served from cache without HTTP call."""
        mock_get.return_value = _mock_response(200, '{"data": "value"}')
        client = ResilientClient(cache_dir=tmp_path, cache_ttl=3600)

        # First request — HTTP call made, response cached
        resp1 = client.get("http://example.com/data", params={"q": "test"})
        assert mock_get.call_count == 1
        assert resp1.status_code == 200

        # Second request — served from cache
        mock_get.reset_mock()
        resp2 = client.get("http://example.com/data", params={"q": "test"})
        assert mock_get.call_count == 0
        assert isinstance(resp2, CachedResponse)
        assert resp2.status_code == 200
        assert resp2.json() == {"data": "value"}

    @patch("scix.http_client.requests.get")
    def test_cache_expired_refetches(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """Expired cache entries trigger a fresh HTTP request."""
        mock_get.return_value = _mock_response(200, '{"fresh": true}')
        client = ResilientClient(cache_dir=tmp_path, cache_ttl=0.0)  # immediate expiry

        client.get("http://example.com/data")
        assert mock_get.call_count == 1

        # Even though file exists, TTL=0 means it's expired
        resp2 = client.get("http://example.com/data")
        assert mock_get.call_count == 2

    @patch("scix.http_client.requests.get")
    def test_different_params_different_cache(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """Different params produce different cache keys."""
        mock_get.return_value = _mock_response(200, '{"v": 1}')
        client = ResilientClient(cache_dir=tmp_path, cache_ttl=3600)

        client.get("http://example.com/api", params={"a": "1"})
        client.get("http://example.com/api", params={"a": "2"})
        assert mock_get.call_count == 2  # Both are cache misses


# ---------------------------------------------------------------------------
# User-Agent header
# ---------------------------------------------------------------------------


class TestUserAgent:
    """User-Agent header set on all requests."""

    @patch("scix.http_client.requests.get")
    def test_default_user_agent(self, mock_get: MagicMock) -> None:
        """Default User-Agent is 'scix-harvester/1.0'."""
        mock_get.return_value = _mock_response(200)
        client = ResilientClient()
        client.get("http://example.com/api")

        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"]["User-Agent"] == "scix-harvester/1.0"

    @patch("scix.http_client.requests.get")
    def test_custom_user_agent(self, mock_get: MagicMock) -> None:
        """Custom User-Agent is passed through."""
        mock_get.return_value = _mock_response(200)
        client = ResilientClient(user_agent="my-agent/2.0")
        client.get("http://example.com/api")

        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"]["User-Agent"] == "my-agent/2.0"

    @patch("scix.http_client.requests.get")
    def test_user_agent_not_overridden_by_caller(self, mock_get: MagicMock) -> None:
        """Caller-provided User-Agent takes precedence."""
        mock_get.return_value = _mock_response(200)
        client = ResilientClient()
        client.get("http://example.com/api", headers={"User-Agent": "custom/1.0"})

        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"]["User-Agent"] == "custom/1.0"
