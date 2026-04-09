# Research: http-client

## Current HTTP patterns in codebase

- Harvesters (`harvest_gcmd.py`, `harvest_pds4.py`) use `urllib.request` with manual retry+backoff
- Common pattern: 3 retries, exponential wait (2^attempt seconds), User-Agent header
- No rate limiting, no circuit breaker, no caching in existing code
- `requests` library is installed (2.33.1) but not used by harvesters yet

## Conventions observed

- NamedTuples and frozen dataclasses for immutable data
- Type annotations on all signatures
- Logging via `logging` module
- Error handling with explicit retry loops
- User-Agent: `scix-experiments/1.0` (harvesters), spec says `scix-harvester/1.0`

## Design decisions

- Use `requests` library as the HTTP backend (spec requirement)
- Custom exceptions: `CircuitBreakerOpen` for circuit breaker state
- Token bucket approximation via per-host last-request-time tracking
- Disk cache: JSON files keyed by SHA256(url+sorted_params), with TTL metadata
- All config via constructor params with sensible defaults
