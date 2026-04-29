# Tier-3 Discipline-Gated Target Linker — Findings (xz4.p2v)

**Date:** 2026-04-29
**Branch:** `bd/xz4.p2v-discipline-gated-targets`

## Outcome

**Precision gate (≥0.90) NOT MET.** Rows from all three production runs
were rolled back; database `document_entities` no longer contains any
`link_type='target_gated_match'` rows.

## Summary of three runs

| Run | Filters active | Rows | Papers linked | Estimated precision |
| --- | --- | ---: | ---: | ---: |
| R1  | Cohort gate only (no stop-words, min_len=6) | 531,244 | 160,633 | ≪ 0.10 |
| R2  | + English dictionary stop-words (~75k), min_len=7 | 47,411 | 30,539 | ~0.25–0.30 |
| R3  | + 7,684 author-surname overlap, designation case filter | 19,718 | 14,472 | ~0.25–0.40 |

(Tier-1+Tier-2 baseline before any tier-3 work: **15,381 papers**.)

## Why precision stays below the gate

Asteroid namesakes draw from precisely the categories most heavily
referenced in planetary-science cohort papers:

1. **Spacecraft / mission names** that share a name with an asteroid
   namesake — Hayabusa, BepiColombo, Apollo, Spirit, Opportunity,
   Cassini, Spitzer, Chang'E. These dominate observed FPs.

2. **Observatory / facility names** that also have an asteroid
   namesake — La Silla, Kitt Peak, Paranal, Mauna Kea, Haute-Provence,
   Skalnate Pleso, Siding Spring, Canarias.

3. **Scientific concepts / effects / theorems** named after the same
   astronomers as their asteroid namesakes — Yarkovsky, Poynting (–
   Robertson), Vaisala, Lyapunov, Bouguer, Cassegrain, Hadamard,
   van der Waals, von Zeipel.

4. **Place names** that are also asteroid names — San Juan, San Pedro,
   Yakutia, Shandong, Wakkanai, Xinjiang.

5. **Author-citation patterns** — astronomer surnames are heavily
   cited in planetary papers and many of those astronomers have
   asteroids named after them. The 7,684-name surname overlap filter
   removes the highest-frequency surnames but the long tail keeps
   leaking through.

The discipline gate (`arxiv_class`/`bibstem`/keyword sentinels) and
stop-word filtering succeed at suppressing the obvious common-word
matches ("The", "NOT", "May", "Field"), but the long tail of
mission/observatory/concept/place collisions cannot be filtered with a
static list at scale.

## Recommended next iteration

Pivot from name-anchored to **designation-anchored** matching. Every
SsODNet entity has at least one designation alias of shape `(NNNN)
Name`, `NNNN Name`, `YYYY LL`, `YYYY LLNN`, or
`Comet/Pp/Cp Name`. Require co-presence in the same paper of:

* the canonical name OR a non-designation alias, AND
* a designation-shaped alias of the SAME entity_id

This effectively requires the paper to cite the asteroid by its
catalog number, which is what disambiguates "Spitzer Space Telescope"
(no `(2160)` co-present) from "asteroid (2160) Spitzer" (yes).

Implementation sketch:
- Build TWO automata: one over names/aliases, one over designation-
  shaped aliases.
- Per paper: scan with both, intersect entity_ids.
- Only emit candidates whose entity_id appears in the designation
  scan.

Yield expectation: lower than the 100–300K target in the bead, but
likely much closer to the 0.90 precision gate. The bead's recall goal
should be revisited under a designation-anchored model.

## Deliverables landed (committed on the branch)

* `scripts/link_targets_discipline_gated.py` — driver with cohort SQL,
  three-stage entity filter, and confidence scoring
* `config/planetary_cohort.yaml` — cohort gate + stop-word knobs
* `config/ssodnet_author_surname_overlap.txt` — 7,684 SsODNet/author
  surname overlaps
* `scripts/eval_target_gated_precision.py` — sampler + TSV scorer
* `tests/test_link_targets_discipline_gated.py` — 20 tests, all
  passing
* `build-artifacts/tier3_target_gated_summary.md` — last-run yield
  report
* This findings document.
