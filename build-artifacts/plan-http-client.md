# Plan: http-client

## Step 1: Create src/scix/http_client.py

1. Define `CircuitBreakerOpen` exception
2. Define `CachedResponse` frozen dataclass (status_code, text, headers, json)
3. Define `ResilientClient` class with constructor params:
   - max_retries (default 3), backoff_base (default 1.0)
   - rate_limit (default 10.0 req/s per host)
   - circuit_breaker_threshold (default 5), circuit_breaker_cooldown (default 60.0)
   - cache_dir (optional Path), cache_ttl (default 3600s)
   - user_agent (default 'scix-harvester/1.0')
4. Implement `.get(url, params=None, **kwargs)` method:
   - Check circuit breaker state for host
   - Apply rate limiting (sleep if needed)
   - Check disk cache (if cache_dir set)
   - Execute request with retry+backoff on 5xx/connection errors
   - Update circuit breaker state on success/failure
   - Cache successful responses to disk

## Step 2: Create tests/test_http_client.py

1. Test retry on 503 (mock requests.get to return 503, verify 3 attempts)
2. Test rate limiting (two rapid requests, verify delay)
3. Test circuit breaker open/close transitions
4. Test disk caching (second call returns cached, no HTTP)
5. Test User-Agent header
