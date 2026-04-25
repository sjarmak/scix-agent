# Research — quant-claim-extractor (M4)

## Schema (staging.extractions)

Migration 049 (latest) defines:

```
CREATE TABLE IF NOT EXISTS staging.extractions (
    id                  SERIAL PRIMARY KEY,
    bibcode             TEXT NOT NULL,
    extraction_type     TEXT NOT NULL,
    extraction_version  TEXT NOT NULL,
    payload             JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    source              TEXT,           -- added 049
    confidence_tier     SMALLINT,       -- added 049
    CONSTRAINT uq_staging_extractions_bibcode_type_version
        UNIQUE (bibcode, extraction_type, extraction_version)
);
```

- The schema uses a JSONB `payload` column (not `evidence_text`). We will
  store `{quantity, value, uncertainty, unit, span}` there.
- `extraction_type` will be `'quant_claim'`.
- `extraction_version` will be a stamp like `quant_claim_regex_v1`.
- Multiple claims per paper share `(bibcode, extraction_type, extraction_version)`,
  which collides with the UNIQUE constraint. The promotion pattern used by
  M4 NER stores a single payload per paper. To remain compatible we will
  store one row per paper with `payload = {"claims": [ClaimSpan, ...]}` or
  use a unique extraction_version per claim. The PRD says one row per
  claim with the JSONB payload `{quantity, value, uncertainty, unit,
  span}` — to satisfy the unique constraint, we ship one row per paper
  whose payload aggregates all claim spans (`claims` array). Tests assert
  the payload shape directly.

## --allow-prod guard pattern (scripts/refresh_fusion_mv.py)

```python
from scix.db import DEFAULT_DSN, is_production_dsn, redact_dsn

if is_production_dsn(args.dsn) and not args.allow_prod:
    logger.error(
        "Refusing to run against production DSN %s — pass --allow-prod to override",
        redact_dsn(args.dsn),
    )
    return 2
```

Same shape used by `link_tier2.py`, `report_community_coverage.py`. We
will copy this pattern verbatim.

## Quantity vocabulary (cosmology)

Canonical names with surface variants observed in the literature:

- H0  : `H0`, `H_0`, `H_{0}`, `Hubble constant`, `Hubble parameter` (when given a value)
- Omega_m : `Omega_m`, `Omega_M`, `\Omega_m`, `\Omega_{m}`, `Omega_matter`, `Ω_m`
- Omega_b : `Omega_b`, `\Omega_b`, `\Omega_{b}`, `Ω_b`
- Omega_Lambda : `Omega_Lambda`, `\Omega_\Lambda`, `\Omega_{\Lambda}`, `Ω_Λ`
- sigma_8 : `sigma_8`, `sigma8`, `\sigma_8`, `\sigma_{8}`, `σ_8`
- ns : `n_s`, `\n_s`, `n_{s}` (scalar spectral index)
- w0 : `w`, `w_0`, `w0` (dark energy EoS)

## Uncertainty forms

The extractor must handle:

1. ASCII: `value +/- uncertainty`
2. Unicode: `value ± uncertainty`
3. LaTeX: `value \pm uncertainty`
4. Asymmetric LaTeX: `value^{+a}_{-b}` (or `value^{+a}_{-b}` with various brace styles)
5. Plain "plus or minus": `value plus or minus uncertainty` (low priority)

## Hook design

Per CLAUDE.md (`feedback_no_paid_apis`), the LLM-tier disambiguation pass
is documented but unimplemented:

```python
def llm_disambiguate(span: ClaimSpan) -> ClaimSpan:
    raise NotImplementedError(
        "Requires paid API; see CLAUDE.md feedback_no_paid_apis"
    )
```

## Test strategy

- 50-snippet hand-curated cosmology fixture in
  `tests/fixtures/quant_claims_cosmology_50.jsonl` (~20 H0, ~15 Omega_m,
  ~10 sigma_8, ~5 other).
- Parametrized unit tests for each uncertainty form.
- Recall test: `assert recall(quantity) >= 0.80` for H0, Omega_m, sigma_8.
- Mocked psycopg test for the staging insert path.
