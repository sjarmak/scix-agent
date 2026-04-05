"""Resilient HTTP client with retry, rate limiting, circuit breaker, and caching.

Provides a drop-in replacement for ``requests.get()`` with production-grade
resilience patterns suitable for harvesting external APIs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open for a host."""

    def __init__(self, host: str, failures: int, cooldown_remaining: float) -> None:
        self.host = host
        self.failures = failures
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            f"Circuit breaker open for {host}: "
            f"{failures} consecutive failures, "
            f"{cooldown_remaining:.1f}s until reset"
        )


@dataclass(frozen=True)
class CachedResponse:
    """Immutable representation of a cached HTTP response."""

    status_code: int
    text: str
    headers: dict[str, str]
    url: str

    def json(self) -> Any:
        """Parse response text as JSON."""
        return json.loads(self.text)


@dataclass
class _HostState:
    """Mutable per-host tracking for rate limiting and circuit breaker."""

    last_request_time: float = 0.0
    consecutive_failures: int = 0
    last_failure_time: float = 0.0


class ResilientClient:
    """HTTP client with retry, rate limiting, circuit breaker, and disk caching.

    Args:
        max_retries: Number of retries on 5xx or connection errors.
        backoff_base: Base delay in seconds for exponential backoff.
        rate_limit: Maximum requests per second per host.
        circuit_breaker_threshold: Consecutive failures before opening circuit.
        circuit_breaker_cooldown: Seconds to wait before resetting circuit breaker.
        cache_dir: Directory for disk cache. None disables caching.
        cache_ttl: Cache entry time-to-live in seconds.
        user_agent: User-Agent header value.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        *,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        rate_limit: float = 10.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown: float = 60.0,
        cache_dir: Path | str | None = None,
        cache_ttl: float = 3600.0,
        user_agent: str = "scix-harvester/1.0",
        timeout: float = 60.0,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.rate_limit = rate_limit
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_cooldown = circuit_breaker_cooldown
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_ttl = cache_ttl
        self.user_agent = user_agent
        self.timeout = timeout
        self._host_states: dict[str, _HostState] = {}

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_host(self, url: str) -> str:
        """Extract host from URL."""
        return urlparse(url).netloc

    def _get_host_state(self, host: str) -> _HostState:
        """Get or create per-host state."""
        if host not in self._host_states:
            self._host_states[host] = _HostState()
        return self._host_states[host]

    def _check_circuit_breaker(self, host: str) -> None:
        """Raise CircuitBreakerOpen if circuit is open for this host."""
        state = self._get_host_state(host)
        if state.consecutive_failures >= self.circuit_breaker_threshold:
            elapsed = time.monotonic() - state.last_failure_time
            if elapsed < self.circuit_breaker_cooldown:
                raise CircuitBreakerOpen(
                    host=host,
                    failures=state.consecutive_failures,
                    cooldown_remaining=self.circuit_breaker_cooldown - elapsed,
                )
            # Cooldown elapsed — reset and allow the request (half-open)
            state.consecutive_failures = 0

    def _apply_rate_limit(self, host: str) -> None:
        """Sleep if needed to enforce per-host rate limit."""
        if self.rate_limit <= 0:
            return
        state = self._get_host_state(host)
        min_interval = 1.0 / self.rate_limit
        now = time.monotonic()
        elapsed = now - state.last_request_time
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug("Rate limiting %s: sleeping %.3fs", host, sleep_time)
            time.sleep(sleep_time)
        state.last_request_time = time.monotonic()

    def _cache_key(self, url: str, params: dict[str, str] | None) -> str:
        """Generate a deterministic cache key from URL and params."""
        parts = [url]
        if params:
            sorted_params = sorted(params.items())
            parts.append(json.dumps(sorted_params, sort_keys=True))
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _read_cache(self, key: str) -> CachedResponse | None:
        """Read a cached response if it exists and hasn't expired."""
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text())
            cached_at = data.get("cached_at", 0)
            if time.time() - cached_at > self.cache_ttl:
                logger.debug("Cache expired for key %s", key[:12])
                cache_file.unlink(missing_ok=True)
                return None
            logger.debug("Cache hit for key %s", key[:12])
            return CachedResponse(
                status_code=data["status_code"],
                text=data["text"],
                headers=data.get("headers", {}),
                url=data.get("url", ""),
            )
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Failed to read cache file %s: %s", cache_file, exc)
            return None

    def _write_cache(self, key: str, response: requests.Response) -> None:
        """Write a response to the disk cache."""
        if self.cache_dir is None:
            return
        cache_file = self.cache_dir / f"{key}.json"
        data = {
            "cached_at": time.time(),
            "status_code": response.status_code,
            "text": response.text,
            "headers": dict(response.headers),
            "url": response.url,
        }
        try:
            cache_file.write_text(json.dumps(data))
        except OSError as exc:
            logger.warning("Failed to write cache file %s: %s", cache_file, exc)

    def get(
        self,
        url: str,
        params: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> requests.Response | CachedResponse:
        """Send a GET request with retry, rate limiting, circuit breaker, and caching.

        Args:
            url: The URL to request.
            params: Optional query parameters.
            **kwargs: Additional keyword arguments passed to ``requests.get()``.

        Returns:
            A ``requests.Response`` or ``CachedResponse`` object.

        Raises:
            CircuitBreakerOpen: If the circuit breaker is open for the host.
            requests.RequestException: If all retries are exhausted.
        """
        host = self._get_host(url)

        # Circuit breaker check
        self._check_circuit_breaker(host)

        # Cache check
        cache_key = self._cache_key(url, params)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        # Rate limiting
        self._apply_rate_limit(host)

        # Set User-Agent header
        headers = kwargs.pop("headers", {}) or {}
        headers.setdefault("User-Agent", self.user_agent)
        timeout = kwargs.pop("timeout", self.timeout)

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    **kwargs,
                )

                if response.status_code >= 500:
                    last_exc = requests.HTTPError(
                        f"{response.status_code} Server Error", response=response
                    )
                    if attempt < self.max_retries:
                        delay = self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                        logger.warning(
                            "Request to %s returned %d (attempt %d/%d), retrying in %.1fs",
                            url,
                            response.status_code,
                            attempt,
                            self.max_retries,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    # Final attempt also failed with 5xx
                    state = self._get_host_state(host)
                    state.consecutive_failures += 1
                    state.last_failure_time = time.monotonic()
                    raise last_exc

                # Success — reset circuit breaker and cache
                state = self._get_host_state(host)
                state.consecutive_failures = 0
                self._write_cache(cache_key, response)
                return response

            except requests.ConnectionError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.warning(
                        "Connection error to %s (attempt %d/%d): %s, retrying in %.1fs",
                        url,
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                state = self._get_host_state(host)
                state.consecutive_failures += 1
                state.last_failure_time = time.monotonic()
                raise

        # Should not reach here, but satisfy type checker
        assert last_exc is not None
        raise last_exc
