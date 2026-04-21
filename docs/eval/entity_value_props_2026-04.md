# Entity Enrichment Value Props Eval — 2026-04

> **NOTE:** This file is a template. Run
> `python scripts/eval_entity_value_props.py --props all --write-report`
> (after pointing `--db` at a populated database and ensuring the
> `claude` CLI is on PATH) to populate it with real numbers.

## Summary

Overall score: _n/a — not yet run_ / 3.0

| Prop | N | Mean | StdErr |
|---|---|---|---|
| alias_expansion | _pending_ | _pending_ | _pending_ |
| ontology_expansion | _pending_ | _pending_ | _pending_ |
| disambiguation | _pending_ | _pending_ | _pending_ |
| type_filter | _pending_ | _pending_ | _pending_ |
| specific_entity | _pending_ | _pending_ | _pending_ |
| community_expansion | _pending_ | _pending_ | _pending_ |

## Methodology

- **Gold sets:** `data/eval/entity_value_props/*.yaml`, at least 10
  curated queries per prop. YAML format is designed to scale to the
  PRD's ~50-per-prop target without schema changes.
- **Retrieval:** `scix.search.hybrid_search` with
  `enable_alias_expansion=True` and `enable_ontology_parser=True`. Top
  10 hits per query are forwarded to the judge.
- **Judge:** a Claude Code subagent invoked via `claude -p
  --output-format=json` as a subprocess. No paid Anthropic API; no
  `anthropic` SDK import. Each query is scored on an ordinal 0-3
  rubric (0 = fails, 3 = works correctly).
- **Aggregation:** per-prop mean + standard error
  (sample_stddev / √n). Overall score is the mean of per-prop means —
  props are weighted equally regardless of query count so one prop
  with many queries can't dominate another with fewer.

## Per-prop results

### alias_expansion

_Populated on report run._

### ontology_expansion

_Populated on report run._

### disambiguation

_Populated on report run._

### type_filter

_Populated on report run._

### specific_entity

_Populated on report run._

### community_expansion

_Populated on report run._
