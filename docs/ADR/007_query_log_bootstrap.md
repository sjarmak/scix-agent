# ADR-007: query_log seed bootstrap (M3.5.0 unblock)

**Status**: Accepted
**Date**: 2026-04-15
**Bead**: scix_experiments-xz4.1.18

## Context

`scripts/curate_entity_core.py` (M3.5.1) depends on `query_log` having
enough realistic astronomy-query traffic to produce a useful
`curated_entity_core`. The original PRD acceptance criterion called for
"≥500 non-test rows spanning ≥14 days". In practice the MCP server
accumulated only **13 rows from one day** because it was used
interactively in just two ad hoc sessions between 2026-04-09 and
2026-04-15, and `_log_query()` was not populating the migration-031
instrumentation columns until xz4.1.17.

`curate_entity_core.py` binds queries to entities via **exact
case-insensitive match** on `canonical_name` / `entity_aliases.alias`:

```sql
JOIN entities e ON lower(e.canonical_name) = z.q
```

Misspellings, typos, and paraphrases cannot substitute for missing
organic traffic — they bind to nothing and contribute zero rows to
pass1/pass3.

## Decision

Bootstrap `query_log` with a seeded run of `keyword_search` against the
**exact canonical names** of `ambiguity_class='unique'` entities. The
seed script is `scripts/seed_query_log.py`, driven by the pinned
manifest `scripts/seed_bootstrap_plan.json`. Every row is tagged with a
deterministic `session_id` derived from `sha256(manifest)`:

```
seed-bootstrap-v1-0fddfe19101c
```

`curate_entity_core.py` accepts a new `--session-id` flag that restricts
pass1/pass3 to the tagged rows, so the bootstrap run is reproducible and
separable from concurrent or future organic traffic.

## Result (2026-04-15)

- **query_log**: 617 non-test rows written under
  `seed-bootstrap-v1-0fddfe19101c`, bind-rate 1.000.
- **curated_entity_core**: 617 rows promoted via
  `core_lifecycle.promote`.
- **Pass distribution**: 37 pass1 (zero-result gap) + 580 pass3 (unique
  - ≥1 hit).
- **Source stratification**:

  | source  | count | pct   |
  | ------- | ----- | ----- |
  | gcmd    | 500   | 81.0% |
  | pwc     | 99    | 16.0% |
  | physh   | 13    | 2.1%  |
  | ssodnet | 5     | 0.8%  |

## Why only 4 sources (not 6+)

The original PRD acceptance called for stratification across ≥6 of 7
live harvesters. The current `entities` table holds 1,487 rows with
`ambiguity_class='unique'` distributed across only **4 sources**:
gcmd (1370), pwc (99), physh (13), ssodnet (5). The other 5 sources
(vizier, ascl, aas, spase, ads_data) have zero unique entities — every
entity from those sources is classified as `domain_safe`, `homograph`,
or `banned`.

Pass3 is strictly filtered by `ambiguity_class='unique'` by design, so
no matter how many queries we seed, pass3 can only produce the 4
sources that have unique entities. Pass1 has no ambiguity filter but is
limited by the `result_count = 0` selection.

The ≥6-source target is therefore a **data constraint**, not a seeding
shortfall. It will only be resolvable by one of:

1. Running the ambiguity classifier against the missing sources so more
   entities earn `unique` classification.
2. Relaxing pass3 to accept `domain_safe` entities (requires an M-level
   PRD decision — not in scope for this bootstrap).
3. Allowing organic traffic to accumulate over months and
   opportunistically include entities from the missing sources.

## Reproducibility

```bash
# Re-run exactly:
python scripts/seed_query_log.py --plan scripts/seed_bootstrap_plan.json

# Curate from a specific seed run only:
python scripts/curate_entity_core.py \
    --session-id seed-bootstrap-v1-0fddfe19101c \
    --populate

# Roll back the seeded rows (query_log):
python scripts/seed_query_log.py \
    --plan scripts/seed_bootstrap_plan.json \
    --rollback

# Demote the seeded entities from curated_entity_core:
psql "dbname=scix" -c "
    DELETE FROM curated_entity_core
     WHERE entity_id IN (
        SELECT entity_id FROM core_promotion_log
         WHERE reason IN ('curate_pass1_gap', 'curate_pass3_unique_with_hits')
           AND promoted_at >= '2026-04-15'
     );
"
```

## Cleanup trigger

When organic (non-seeded) `query_log` traffic for the 14-day curation
window exceeds **200 non-test rows**, re-run curation without
`--session-id` and optionally roll back the seeded rows. The curated
core should then reflect actual user intent rather than the seed
manifest's bias. Until then, the seeded rows dominate pass3 and the
curated core is a **bootstrap snapshot**, not an organic signal.

## Trade-offs

- **Pro**: unblocks M3.5.1, produces a usable curated core today, keeps
  seed traffic reproducible and demote-able.
- **Pro**: exact-canonical-name seeding guarantees 100% bind rate
  (confirmed in run) — no wasted traffic.
- **Con**: curated core is biased toward gcmd (81%) because gcmd has
  the largest `unique` pool. Stratification imbalance is real but
  honest.
- **Con**: the initial core does not represent user intent. Downstream
  consumers (Tier 2 linker, load tests) should treat v1 as a
  bootstrap, not a ground truth.

## Alternatives considered and rejected

- **Synthetic misspellings for pass1** — would bind to zero entities
  because pass1 uses exact match, not fuzzy match.
- **Widen pass3 to `domain_safe`** — out of scope; changes curation
  semantics.
- **Defer until organic traffic accumulates** — blocks M3.5.1 and
  downstream M6 / M11 for weeks or months with no forcing function.
- **Broader session-log extraction across `~/.claude/projects/`** —
  yielded ≤60 organic rows from real tool invocations (most of the
  grep matches are tool-list echoes). Insufficient for the AC.
