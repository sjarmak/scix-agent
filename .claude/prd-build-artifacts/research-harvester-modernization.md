# Research: Harvester Modernization

## Reference Pattern (harvest_gcmd.py)

- Lazy `_get_client()` returning module-level `ResilientClient`
- Imports: `from scix.http_client import ResilientClient` and `from scix.harvest_utils import HarvestRunLog`
- `run_harvest()` creates `HarvestRunLog(conn, source_name)`, calls `.start()`, `.complete()`, `.fail()`
- Uses `client.get(url)` with `.json()` and `.text` on response
- Still writes to entity_dictionary via `bulk_load()` for backward compat

## Harvesters to Modernize

### 1. harvest_ascl.py

- **urllib usage**: `urllib.request.Request`, `urllib.request.urlopen`, `urllib.error.URLError`
- **download function**: `download_ascl_catalog()` — manual retry loop with urllib
- **run_harvest**: Simple: download -> parse -> bulk_load. No HarvestRunLog.
- **Change**: Replace urllib in download_ascl_catalog with \_get_client().get(), add HarvestRunLog to run_harvest

### 2. harvest_aas_facilities.py

- **urllib usage**: `urllib.request.Request`, `urllib.request.urlopen`, `urllib.error.URLError`
- **download function**: `download_aas_facilities()` — returns HTML string. Uses manual retry.
- **run_harvest**: download -> parse -> bulk_load. No HarvestRunLog.
- **Note**: Downloads HTML (not JSON), so need `response.text` instead of `.json()`

### 3. harvest_vizier.py

- **urllib usage**: `urllib.request.Request`, `urllib.request.urlopen`, `urllib.error.URLError`, `urllib.parse.urlencode`
- **download function**: `query_tap_vizier()` — uses POST method with urlencode'd params. Returns raw bytes.
- **Challenge**: ResilientClient only has `.get()`. This harvester uses POST. Must use `requests.post()` directly or adapt. Looking at the code: the TAP endpoint accepts params as POST body. We can use the `requests` library's `post()` directly but ResilientClient doesn't expose `.post()`. Alternative: use GET with query params, but TAP sync standard uses POST. Decision: keep using `requests.post()` through a helper but still add ResilientClient import for consistency. Actually, better approach: convert to GET with URL params since TAP sync endpoints accept GET too, or use requests directly. Looking more carefully: the ResilientClient wraps `requests.get()`. For this harvester we need to use the client for the get-based resilience but this endpoint uses POST. Simplest: use ResilientClient's retry/rate-limit patterns but call `requests.post()` directly. Or: Many TAP endpoints accept GET. Let's keep it simple: use `_get_client().get(url, params=params_dict)` since ResilientClient.get() accepts params.
- **run_harvest**: query -> parse -> build_entries -> bulk_load. No HarvestRunLog.

### 4. harvest_pwc_methods.py

- **urllib usage**: `urllib.request.Request`, `urllib.request.urlopen`, `urllib.error.URLError`
- **download function**: `download_methods()` — downloads gzip file to disk. Skips if exists.
- **run_pipeline**: download -> parse -> load_methods. No HarvestRunLog. Note: function name is `run_pipeline` not `run_harvest`.
- **Note**: Downloads binary (gzip) — need `response.content` (raw bytes). ResilientClient returns `requests.Response` which has `.content`.

### 5. harvest_physh.py

- **urllib usage**: `urllib.request.Request`, `urllib.request.urlopen`, `urllib.error.URLError`
- **download function**: `download_physh()` — downloads gzip JSON, with optional cache_dir.
- **run_harvest**: download -> parse -> bulk_load. No HarvestRunLog.
- **Note**: Downloads binary gzip data, then decompresses.

### 6. harvest_astromlab.py

- **urllib usage**: `urllib.request.Request`, `urllib.request.urlopen`, `urllib.error.URLError`
- **download function**: `download_concepts()` — downloads CSV to disk. Skips if exists.
- **run_pipeline**: download -> parse -> load. No HarvestRunLog. Function name is `run_pipeline`.

## Test Changes Needed

Each test file currently mocks `urllib.request.urlopen`. After modernization:

- Download function tests mock `_get_client` or `ResilientClient.get`
- run_harvest/run_pipeline tests need to mock `HarvestRunLog` class
- Remove `import urllib.error` from tests where used for exception types
