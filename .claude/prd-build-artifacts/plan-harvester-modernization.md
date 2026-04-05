# Plan: Harvester Modernization

## For each harvester, apply these changes:

### Script changes:

1. Remove `import urllib.error`, `import urllib.request`, `import urllib.parse`
2. Remove `import time` if only used for retry sleep (ResilientClient handles retries)
3. Add `from scix.http_client import ResilientClient`
4. Add `from scix.harvest_utils import HarvestRunLog`
5. Add module-level `_client` and `_get_client()` lazy init pattern
6. Replace download function internals: remove manual retry loop, use `_get_client().get(url)`
7. Wrap run_harvest/run_pipeline DB section with HarvestRunLog lifecycle

### Test changes:

1. Remove `import urllib.error`
2. Change download mocks from `patch("harvest_X.urllib.request.urlopen")` to `patch("harvest_X._get_client")`
3. Add `HarvestRunLog` mock to run_harvest tests
4. Update retry/failure tests to use `requests.RequestException` instead of `urllib.error.URLError`

## Execution Order:

1. harvest_ascl.py + test
2. harvest_aas_facilities.py + test
3. harvest_vizier.py + test (special: POST -> GET conversion)
4. harvest_pwc_methods.py + test (special: binary download)
5. harvest_physh.py + test (special: binary gzip + cache)
6. harvest_astromlab.py + test (special: binary download to file)
