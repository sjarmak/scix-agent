# Entity-link audit report (M9)

- Total candidates sampled: **24**
- Total labeled: **24**

Judge labels sourced from deterministic stub — set `ANTHROPIC_API_KEY` + `use_real=True` for real judging.

## Per-tier precision (Wilson 95% CI)

| tier | correct | total | precision | CI low | CI high |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 6 | 0.333 | 0.097 | 0.700 |
| 2 | 2 | 6 | 0.333 | 0.097 | 0.700 |
| 4 | 2 | 6 | 0.333 | 0.097 | 0.700 |
| 5 | 2 | 6 | 0.333 | 0.097 | 0.700 |

_Worked example `wilson_95_ci(95, 100)` → **[0.888, 0.978]**_
