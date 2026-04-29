# Tier-3 Designation-Anchored Target Linker — Findings (xz4.9)

**Date:** 2026-04-29
**Branch:** `bd/xz4.9-designation-anchored`
**Predecessor:** xz4.p2v (failed, ~0.25-0.40 precision; rolled back)

## Outcome

**Precision gate (≥0.90) PASS.** 100-row random sample over 540 rows
emitted from 10,418 cohort papers (bibcode_prefix=`2024`):

| Metric | Value |
| --- | --- |
| Sample size | 100 |
| TP | 96 |
| FP | 4 |
| SKIP | 0 |
| **Precision** | **0.960** |
| Gate (0.90) | PASS |

## Approach

Pivoted from name-anchored (xz4.p2v) to **designation-anchored** matching:

1. Two Aho-Corasick automata:
   * `name_automaton`: name-shaped surfaces (`Spitzer`, `Apollo`, `Ceres`).
   * `designation_automaton`: designation-shaped surfaces classified by
     `_DESIGNATION_RE` — `(NNNN)`, `(NNNN) Name`, `YYYY LL[NN]`,
     `NP/Name`, `[CDPIXA]/YYYY LL[NN]`.
2. Per cohort paper, scan with both. For each entity:
   * **Named entities** (canonical is name-shape): emit only if both a
     name AND a designation hit are present.
   * **Designation-only entities** (canonical is itself a designation,
     e.g. `2014 KZ113`): emit if a designation hit is present.
3. Designation case filter: AC scanning is case-insensitive, so
   `YYYY LL` and `[CDPIXA]/YYYY LL` candidates are re-checked against the
   original-case text and dropped unless the letter portion is all
   upper-case. This kills false positives like `2020 by`, `2023 to`,
   `2024 UT` (lowercase) that the AC automaton would otherwise match.

## Yield (preview)

| Slice | Cohort papers | Linked papers | Rows |
| --- | ---: | ---: | ---: |
| 2024Icar (smoke) | 399 | 27 | 38 |
| 2024 (precision sample) | 10,418 | 339 | 540 |

Linker speed (after entity load): ~30-50 papers/sec on a single core.
Full-corpus projection: 100K-300K cohort papers × 30 pps ≈ 1-3 hours
(plus ~1 minute setup).

## False-positive analysis

All 4 FPs are designation-only matches where the regex shape collides
with non-asteroid uses:

| FP | matched | confounder |
| --- | --- | --- |
| `2024EPSC...17..757L` / `2020 AJ` | designation-only | journal-citation token (`...2020, AJ, ...`) |
| `2024ApJ...976...11P` / `2020 CE` | designation-only | year + Common-Era marker on a millennium-timescale paper |
| `2024RNAAS...8..235K` / `2010 RA` | designation-only | substring of `2010 RA78` (real designation in the same paper) |
| `2024EPSC...17..371B` / `2024 UT` | designation-only | UT-time / observation-date token |

**Common thread:** all four are short `YYYY LL` designations matching a
2-letter token (`AJ`, `CE`, `RA`, `UT`) that has a non-asteroid English
or scientific meaning. Possible follow-up tightenings (out of scope —
already past the 0.90 gate):

1. Forbid an immediately-following digit on the matched span — kills
   the `2010 RA` ⊂ `2010 RA78` case.
2. For designation-only short forms (`YYYY LL` with letter ≤ 2 chars),
   require an asteroid-context cue (`asteroid`, `minor planet`,
   `(NNNN)`, `NEA`, `TNO`, etc.) co-present in the abstract.

## Acceptance status

* ≥0.90 precision over 100-row random sample → **0.96 PASS**.
* Yield expectation: revised down from 100-300K to 5-30K linked papers
  by the bead. The 2024-only sample produced 339 linked papers from
  10,418 cohort papers (3.3% link rate); extrapolating to ~150K cohort
  papers across the full corpus → ~5,000 linked papers, consistent
  with the bead's revised expectation.

## Deliverables landed (committed on this branch)

* `scripts/link_targets_designation_anchored.py` — designation-anchored
  driver. Reuses cohort SQL + paper iterator from
  `link_targets_discipline_gated.py`.
* `tests/test_link_targets_designation_anchored.py` — 49 tests covering
  the designation regex, surface partitioning, co-presence rule, case
  filter, and confidence scoring.
* `scripts/eval_target_gated_precision.py` — extended with `--link-type`
  parameter so the same eval works for both p2v and xz4.9.
* `build-artifacts/tier3_target_designation_summary.md` — last-run yield
  report.
* `build-artifacts/tier3_target-designation-anchored_eval_100.tsv` and
  `.md` — annotated 100-row sample + cards.
* This findings document.

## Database state

`document_entities` contains 540 tier-3 rows with
`link_type='target_designation_anchored'` from the `--bibcode-prefix=2024`
sample run. Full-corpus run is the next operational step.
