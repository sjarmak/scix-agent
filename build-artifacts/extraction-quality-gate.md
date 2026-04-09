# Extraction Quality Gate

**Decision**: `PROCEED`

**Overall F1**: 0.85 (threshold 0.60)
**Papers evaluated**: 50

## Metrics by Extraction Type

| Type | TP | FP | FN | Precision | Recall | F1 |
|------|---:|---:|---:|----------:|-------:|---:|
| datasets | 5 | 5 | 1 | 0.50 | 0.83 | 0.62 |
| instruments | 27 | 7 | 0 | 0.79 | 1.00 | 0.89 |
| materials | 102 | 29 | 0 | 0.78 | 1.00 | 0.88 |
| methods | 76 | 27 | 3 | 0.74 | 0.96 | 0.84 |
| **overall** | 210 | 68 | 4 | 0.76 | 0.98 | 0.85 |

## Failure Modes Observed

- Materials over-extraction: pipeline extracts theoretical physics constructs (scalar fields, Brownian particles, quantum states) as materials
- Hallucinated materials: entities not present in abstract (e.g. NaHCO3, organometal trihalide perovskite)
- Instruments misidentified: source names extracted as instruments (VLA 1623 is a YSO, not the VLA instrument)
- Methods over-extraction: phenomena and concepts extracted as methods (Adler-Bell-Jackiw anomaly, divisibility property)

## Notes

Annotator: Claude (structured rubric per type). Sample: 50 papers from extraction pipeline. Rubric: datasets=named collections/catalogs; instruments=named physical instruments; materials=chemical compounds/minerals/substances (NOT theoretical constructs or quantum states); methods=named analytical/computational/experimental techniques (NOT phenomena or concepts).
