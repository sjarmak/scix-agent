# Plan: Deterministic Normalization

## Step 1: Create src/scix/normalize.py

### Data structures

- `NormalizationResult` frozen dataclass: `canonical: str`, `original: str`
- `ALIAS_MAP`: `MappingProxyType[str, str]` — hand-curated abbreviation→expansion dict
  (keys are already normalized lowercase forms)

### Pipeline stages (in order)

1. Unicode NFKC + lowercase + strip
2. Punctuation normalization: hyphens/dashes to spaces, remove possessives ('s, s')
3. Alias resolution via ALIAS_MAP lookup
4. Whitespace collapse (multiple spaces → single space, strip)

### Functions

- `normalize_entity(entity: str) -> str` — returns canonical form
- `normalize_batch(entities: list[str]) -> tuple[list[str], dict[str, str]]`
  - Returns (normalized list, denorm_map: normalized→original first-seen)

### Alias dictionary coverage (astronomy-focused)

- Instruments: HST, JWST, ALMA, VLT, Chandra, Spitzer, Fermi, etc.
- Methods: MCMC, PCA, SED, CNN, MLE, etc.
- Surveys/datasets: SDSS, 2MASS, LSST, etc.
- Missions: HST→hubble space telescope, etc.

## Step 2: Create tests/test_normalize.py

- Test each stage independently
- Test acceptance criteria explicitly
- Test batch with >30% dedup on 100 entities
- Test determinism (run twice, same result)
- Test edge cases: empty string, None-like, already normalized

## Step 3: Run tests, fix failures

## Step 4: Format with black, lint with ruff
