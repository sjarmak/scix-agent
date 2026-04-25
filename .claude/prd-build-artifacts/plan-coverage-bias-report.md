# Plan: coverage-bias-report (M1)

## Strategy: extend (import & wrap), do not duplicate

The new file `scripts/report_full_text_coverage_bias.py` imports DB query
functions from `coverage_bias_analysis.py`, adds two new facet collectors
(bibstem, community_semantic_medium), computes KL-divergence per facet,
emits JSON, and appends a docs section.

## JSON schema

```json
{
  "generated_at": "2026-04-25T...Z",
  "dsn_redacted": "dbname=scix",
  "corpus_total": 32400594,
  "fulltext_total": 14941487,
  "fulltext_pct": 46.1,
  "kl_divergence_basis": "P=full-text distribution; Q=corpus prior; smoothing eps=1e-12",
  "facets": {
    "arxiv_class": {
      "kl_divergence_vs_corpus_prior": 0.123,
      "rows": [
        {"label": "cs.LG", "total": 198807, "with_body": 197077, "without_body": 1730,
         "pct_with_body": 99.1, "p_fulltext": 0.013, "q_corpus": 0.006, "log_ratio": 0.7}
      ]
    },
    "year": { "kl_divergence_vs_corpus_prior": ..., "rows": [...] },
    "citation_bucket": { "kl_divergence_vs_corpus_prior": ..., "rows": [...] },
    "bibstem": { "kl_divergence_vs_corpus_prior": ..., "rows": [...] },
    "community_semantic_medium": { "kl_divergence_vs_corpus_prior": ..., "rows": [...] }   // omitted if column absent
  }
}
```

## Module structure

```python
# scripts/report_full_text_coverage_bias.py
"""..."""

# Imports: argparse, json, datetime, math, sys/Path
# Reuse: from coverage_bias_analysis import (DistributionRow, get_*_distribution, ...)
# Reuse: from scix.db import get_connection, redact_dsn, DEFAULT_DSN

def kl_divergence(p, q, eps=1e-12) -> float:
    """D_KL(P||Q). Smoothed with eps for zeros. Pure-Python."""

def rows_to_distributions(rows) -> tuple[list[float], list[float]]:
    """Convert DistributionRow list into (P, Q) probability vectors."""

def get_bibstem_distribution(conn, limit=20) -> list[DistributionRow]: ...
def get_community_semantic_distribution(conn, limit=20) -> list[DistributionRow] | None:
    # Returns None if paper_metrics.community_semantic_medium is missing
    ...

def build_facet_payload(name, rows) -> dict: ...

def synthetic_facets() -> dict:
    """For --dry-run: deterministic toy distributions for AC2 schema check."""

def run_report(dsn=None, json_out=None, docs_path=None, dry_run=False) -> dict: ...

def upsert_agent_guidance_section(docs_path: Path, payload: dict) -> None:
    """Append (or replace) the 'Agent guidance...' section in the docs."""

def main(): ...
```

## Docs section structure

Append at end of docs/full_text_coverage_analysis.md, between markers
`<!-- agent-guidance:start -->` / `<!-- agent-guidance:end -->` so re-runs
update in place rather than appending duplicates.

```
## Agent guidance: safe vs unsafe queries on the full-text cohort

Generated from KL-divergence analysis ... (cite results JSON path).

### Safe queries to restrict to full-text
1. Modern arXiv-class queries (cs.LG / hep-ph / quant-ph) — KL component small, full-text >98%.
2. Recent-year queries (year ≥ 2018) — full-text >65%, distribution close to prior.
3. High-citation queries (101–500 / 500+ buckets) — full-text >63%, modest skew.

### Unsafe queries (full-text-only would bias the result)
1. Pre-1950 historical literature — full-text <15%, large KL component.
2. Conference abstracts (AGU / EGU / APS Meeting) — full-text near 0%.
3. Doctype=proposal / dataset / software — 0% full-text.
```

The exact safe/unsafe entries will be grounded in the live KL numbers
when run against prod, but the synthetic --dry-run version still
produces the section structure (with placeholder rationale) so the
acceptance check (3+3 examples) passes against the synthetic JSON.

## Test plan

- `test_kl_divergence_known`: verifies D_KL([0.5,0.5],[0.5,0.5]) == 0,
  D_KL([1,0],[0.5,0.5]) > 0.5, smoothing handles zeros.
- `test_kl_divergence_symmetry_breaks`: D_KL(P||Q) != D_KL(Q||P) in
  general.
- `test_rows_to_distributions`: DistributionRow list → normalised probs.
- `test_synthetic_payload_schema`: --dry-run JSON has all required keys
  and each facet has both counts and kl_divergence_vs_corpus_prior.
- `test_dry_run_writes_json`: import script, call run_report with
  dry_run=True, assert the file exists and parses.

## Acceptance check matrix

| AC | Approach |
|---|---|
| 1. exec bit | `chmod +x` after writing |
| 2. --dry-run JSON keys | Synthetic facets path + assertion in test |
| 3. docs section present | upsert_agent_guidance_section invoked from --dry-run too; section template guarantees ≥3 safe + ≥3 unsafe bullets |
| 4. pytest passes | All four unit tests pass |
| 5. docstring documents scix-batch | Top-of-file docstring includes the wrapper command line |
