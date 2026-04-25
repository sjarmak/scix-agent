# Plan — quant-claim-extractor (M4)

## File scope

- `src/scix/claim_extractor.py` — module
- `scripts/run_claim_extractor.py` — CLI driver
- `tests/test_claim_extractor.py` — unit + integration tests
- `tests/fixtures/quant_claims_cosmology_50.jsonl` — 50-line gold

## Module: src/scix/claim_extractor.py

```python
@dataclass(frozen=True)
class ClaimSpan:
    quantity: str          # canonical name e.g. "H0"
    value: float
    uncertainty: float | None
    unit: str | None
    span: tuple[int, int]  # (start, end) char offsets in body
    # optional asymmetric uncertainty
    uncertainty_pos: float | None = None
    uncertainty_neg: float | None = None
    surface: str = ""       # raw matched text

EXTRACTION_TYPE = "quant_claim"
EXTRACTION_VERSION = "quant_claim_regex_v1"

QUANTITY_DICT: dict[str, list[str]] = {
    "H0":            ["H0", "H_0", "H_{0}", "Hubble constant", "Hubble parameter"],
    "Omega_m":       ["Omega_m", "Omega_M", "\\Omega_m", "\\Omega_{m}", "Omega_matter", "Ω_m"],
    "Omega_b":       ["Omega_b", "\\Omega_b", "\\Omega_{b}", "Ω_b"],
    "Omega_Lambda":  ["Omega_Lambda", "\\Omega_\\Lambda", "\\Omega_{\\Lambda}", "Ω_Λ"],
    "sigma_8":       ["sigma_8", "sigma8", "\\sigma_8", "\\sigma_{8}", "σ_8"],
    "n_s":           ["n_s", "\\n_s", "n_{s}"],
    "w0":            ["w_0", "w0", "w_{0}"],
}

def extract_claims(body: str) -> list[ClaimSpan]: ...
def llm_disambiguate(span: ClaimSpan) -> ClaimSpan:  # hook only
    raise NotImplementedError(...)
def to_payload(claims: list[ClaimSpan]) -> dict: ...  # for staging insert
```

### Regex catalog

Build per-quantity surface pattern (alternation, longest-first, regex
escape). For each candidate surface, attempt to match assignment forms:

1. `<surf>\s*=\s*<value>\s*(\\pm|±|\+/-)\s*<uncertainty>(\s*<unit>)?`
2. `<surf>\s*=\s*<value>\s*\^\{\+<a>\}_\{-<b>\}(\s*<unit>)?`
3. `<surf>\s*=\s*<value>(\s*<unit>)?`  (no uncertainty)

Where:
- `<value>` = `-?\d+(\.\d+)?(?:[eE][+-]?\d+)?`
- `<unit>` = a permissive `[A-Za-z][A-Za-z0-9/^\-\\{}_]*` (1-30 chars), optional

Apply per-canonical-quantity and de-duplicate overlapping matches by
preferring longer surface forms first.

## CLI: scripts/run_claim_extractor.py

argparse flags:
- `--max-papers INT`
- `--dry-run`
- `--allow-prod`
- `--since-bibcode TEXT`
- `--dsn TEXT` (defaults to SCIX_DSN)
- `--batch-size INT` (default 200)
- `--target {body,abstract}` (default body)

Behavior:
1. Refuse to run against production DSN unless `--allow-prod`.
2. If `--dry-run`, do not write to DB; print a summary.
3. Otherwise, iterate papers (bibcode > since-bibcode) ORDER BY bibcode,
   extract claims, INSERT into staging.extractions on conflict update.

## Test plan (tests/test_claim_extractor.py)

1. `test_extract_basic_unicode` — `Omega_m = 0.315 ± 0.007`
2. `test_extract_basic_ascii` — `H0 = 73.4 +/- 1.1 km/s/Mpc`
3. `test_extract_latex_pm` — `\sigma_8 = 0.811 \pm 0.006`
4. `test_extract_asymmetric` — `H_0 = 67.4 ^{+1.2}_{-1.5}`
5. `test_quantity_canonicalization` — surface variants all map to canonical name
6. `test_recall_per_quantity_on_cosmology_fixture` — recall ≥ 0.80 for H0,
   Omega_m, sigma_8 on the 50-line fixture.
7. `test_llm_disambiguate_raises` — hook raises NotImplementedError.
8. `test_to_payload_shape` — payload contains `{quantity, value,
   uncertainty, unit, span}` per claim.
9. `test_staging_insert_calls_psycopg_with_quant_claim_type` — mocked
   psycopg, asserting `extraction_type='quant_claim'` is in the SQL params.

## Fixture distribution

50 lines:
- 20 H0 (variety: km/s/Mpc, kms-1Mpc-1, no unit, asymmetric, LaTeX)
- 15 Omega_m (Omega_m, \Omega_m, Ω_m, with/without uncertainty)
- 10 sigma_8 (sigma_8, sigma8, \sigma_8, asymmetric, LaTeX)
- 3 Omega_b
- 2 Omega_Lambda

Each line is a JSON object: `{"text": "...", "expected": {"quantity":
"H0", "value": 73.4, "uncertainty": 1.1, "unit": "km/s/Mpc"}}`.

## Out of scope

- LLM disambiguation (hook only, raises NotImplementedError).
- Promotion to public.extractions (handled by promote_staging_extractions.py).
- Cross-paper aggregation / Hubble tension clustering (acceptance criterion of
  the MCP entity tool, not this work unit).
