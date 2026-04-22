# `retrieval_50q.jsonl` — 50-query retrieval eval queries

Hand-curated query set for the title+abstract-only vs.
title+abstract+section-level retrieval comparison
(PRD §D5 — `docs/prd/prd_section_embeddings_mcp_consolidation.md`).

**Path note:** PRD references this file as `eval/retrieval_50q.jsonl`;
repo convention puts checked-in eval artifacts under `data/eval/`
(see `.gitignore` carve-outs). Downstream consumers should resolve
against `data/eval/retrieval_50q.jsonl`.

Unblocks the full eval run (`wqr.9.5`) once `section_retrieval`
(`wqr.9.3` / PRD §D3) lands. Gold-standard relevance labels are not
included here — they are generated in `wqr.9.5` during the full run
by pooling top-k results from each method and judging via the
existing `scripts/eval_retrieval_50q.py` citation-based pipeline plus
a manual review pass.

## Format

One JSON object per line:

```jsonc
{
  "query_id": "m07",           // prefix = bucket letter (t/c/m/a) + 2-digit index
  "bucket": "method",          // title | concept | method | author
  "discipline": "planetary",   // astrophysics | planetary | earth_science | heliophysics
  "query": "N body simulation methods for protoplanetary disk dynamics",
  "notes": "Simulation methodology."
}
```

## Bucket rationale (PRD §D5)

| Bucket | n | What it tests |
|---|---|---|
| title | 12 | Easy baseline — every method should hit; catches regressions |
| concept | 16 | Semantic retrieval — INDUS-trained dense should dominate |
| method | 14 | Methods-section content — `section_retrieval` should lift nDCG@10 by ≥ 5 points (PRD acceptance) |
| author | 8 | Author-filter behaviour — tests mixed author+topic queries |

## Discipline balance

Astrophysics 13 · Planetary 12 · Earth science 12 · Heliophysics 13.
Chosen to keep each bucket × discipline cell at 2–4 queries so
sub-bucket metrics remain meaningful but not overfit to a single
subfield.

## Notes for downstream label generation (`wqr.9.5`)

- Pool the top-k from all methods (lexical, INDUS, nomic,
  `section_retrieval`, fused variants) per query, dedupe, and judge.
- For `author` queries, gold-set size should be kept small
  (≤ 20 relevant per query) — these test filter precision, not
  recall.
- For `method` queries, the gold set should prefer papers whose
  **methods section** materially describes the technique; an
  abstract-only mention is a weaker label.
