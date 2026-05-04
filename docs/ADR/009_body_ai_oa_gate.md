# ADR-009: Body-AI Pipelines Gate on `papers_is_oa_or_preprint`

- **Status**: Accepted (2026-05-04)
- **Deciders**: SciX maintainers
- **Scope**: All AI pipelines (NER, embeddings, citation-context extraction) that read `papers.body` or `papers_fulltext.sections`
- **Supersedes**: none
- **Related**: ADR-006 (arXiv LaTeX licensing — internal use only), bead `scix_experiments-8584`

## Context

`papers.body` carries 14.95M paper bodies harvested from a mixture of upstream sources. Distribution by openness:

- ~5.4M papers (~207 GB) carry an OA flag — `'OPENACCESS' = ANY(property)` — the union of EPRINT_OPENACCESS, PUB_OPENACCESS, ADS_OPENACCESS, PMC_OPENACCESS, AUTHOR_OPENACCESS.
- ~5.4M additional matches via non-empty `arxiv_class` (preprints).
- ~9.5M papers (~235 GB) closed or unknown — the risk substrate for Wiley / Springer / Elsevier / etc. text-and-data-mining (TDM) clauses that constrain or prohibit AI use of body text without per-publisher licensing.

Three body-AI pipelines were built but not yet running on prod when this ADR was written:

- `scripts/run_ner_bodies.py` — section-aware NER (GLiNER) over method/result sections.
- `scripts/extract_citation_contexts.py` — citation-context window extraction for the citation-grounded retrieval lane.
- `python -m scix.embeddings.section_pipeline` — section-level embeddings (nomic-embed-text-v1.5 at 1024d Matryoshka).

Running any of these on closed-access body text without explicit policy clearance creates publisher-agreement liability. Per-publisher policy (e.g. a `paper_publisher_policy` table keyed on bibstem or DOI prefix) waits on actual contract review with ADS and is out of scope here. We need a single, technical gate that defaults the safe direction.

## Decision

Install one source of truth for the safety predicate at the SQL layer; all body-AI pipelines call it in their `WHERE` clauses; an explicit operator opt-in flag (`--include-closed`) is the only way to bypass it.

Implementation (migration `068_papers_is_oa_or_preprint.sql`):

```sql
CREATE OR REPLACE FUNCTION papers_is_oa_or_preprint(p papers)
RETURNS BOOLEAN
LANGUAGE SQL
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT COALESCE('OPENACCESS' = ANY(p.property), FALSE)
        OR COALESCE(array_length(p.arxiv_class, 1) > 0, FALSE);
$$;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_is_oa
    ON papers ((papers_is_oa_or_preprint(papers)))
    WHERE body IS NOT NULL;
```

Body-AI scripts call the function in their WHERE clauses. The PG planner inlines simple SQL functions, so callers writing `WHERE papers_is_oa_or_preprint(p)` hit the partial expression index.

Default behaviour for every body-AI pipeline is `oa_only=True`. Operators wanting closed-access papers must pass `--include-closed`, which:

- flips `oa_only=False` through the call chain,
- emits a `WARNING`-level startup log (visible in journal/syslog) confirming the policy gate is OFF,
- leaves an explicit audit trail in operator-edited cron / shell history.

Abstract-only AI (INDUS title+abstract embeddings, GLiNER abstract pass) is **not** gated — abstracts are universally indexable under the ADS metadata terms.

## Alternatives Considered

1. **`GENERATED ALWAYS AS (...) STORED` column on `papers`.** Rejected. PG16 `ADD COLUMN ... STORED` requires a full table rewrite holding `AccessExclusiveLock` for the duration. On 32M rows with ~40-column metadata + JSONB raw + GIN indexes, multi-hour blocking. Migration 053 (halfvec shadow column) chose a nullable column + out-of-band backfill specifically to avoid this; an expression index goes one step further by avoiding the column entirely.

2. **Per-call WHERE predicate (no shared function).** Rejected. Easy to drift; new body-AI pipelines would have to know the exact boolean expression to reproduce. Single source of truth at the SQL function level prevents the drift.

3. **Per-publisher policy table.** Deferred. Depends on actual contract terms from ADS and from individual publishers; premature to schema-design. The OA-or-preprint gate is the safe default that doesn't depend on contract review.

## Consequences

Positive:

- One predicate. Adding a new body-AI pipeline means calling `papers_is_oa_or_preprint(p)` in its WHERE clause; no per-pipeline policy logic.
- Index-backed lookup (~150 MB partial index) makes the gate effectively free at query time.
- Closed-access opt-in is auditable in logs.

Negative / acceptable losses:

- Edge case: ~36 papers out of 14.95M are arxiv preprints flagged via `identifier LIKE 'arXiv:%'` only (no `arxiv_class` array). Excluded from the gate. Acceptable loss given the scale.
- The `array_length` branch returns NULL on empty arrays in raw SQL; we wrap each branch in `COALESCE(..., FALSE)` so the function is BOOLEAN-not-tri-valued. Means `WHERE NOT papers_is_oa_or_preprint(p)` returns the same population as `WHERE papers_is_oa_or_preprint(p) = FALSE` (matters for diagnostic queries on closed-access coverage).
- Operators running closed-access workloads must remember `--include-closed`. A loud `WARNING` log is the mitigation.

## Compliance

Body-AI scripts (gated):

- `scripts/run_ner_bodies.py`
- `scripts/extract_citation_contexts.py`
- `python -m scix.embeddings.section_pipeline`

Out-of-scope here, gated separately or not body-AI:

- Abstract-only AI pipelines (universally indexable).
- `src/scix/extract/chunk_pass/pipeline.py` — additional body-AI pipeline (INDUS body-chunk → Qdrant) found during 8584 review; tracked under follow-up bead `scix_experiments-6eix`.
