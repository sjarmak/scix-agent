# Runbook — `scripts/run_ner_bodies.py`

Section-aware GLiNER NER pass over paper bodies. Companion to
`scripts/run_ner_pass.py` (which only operates on abstracts).

This is the M2 deliverable of `docs/prd/prd_full_text_applications_v2.md`.

---

## What it does

For every paper in `papers.body`, the script:

1. Splits the body into sections via `scix.section_parser.parse_sections`.
2. Runs `scix.section_role.classify_section_role` on each section header
   and **keeps only sections with role `method` or `result`**. Everything
   else — abstract, introduction, conclusion, acknowledgments,
   references/bibliography, appendix, preamble — is **skipped**.
3. Calls `GlinerExtractor.predict` on the kept section text only.
4. Writes one row per paper to `staging.extractions` with
   `extraction_type='ner_body'`, `source='ner_body'`,
   `section_name` (comma-joined kept sections), `char_offset` (start of
   first kept section), and a structured `payload` containing the per-
   section mention breakdown.

### Why section-aware

- Bibliography sections inflate every entity type with author surnames
  mis-typed as `location` or `organism`.
- Introductions are dominated by background prose and citation chains
  that resemble named entities to GLiNER.
- Method and result sections concentrate the actual `software`,
  `dataset`, `instrument`, and `method` mentions we want.

The S2 section-role classifier (commit 62d8740 + f6ba52e) is the
prerequisite that makes this possible at scale.

### Schema dependency

Migration **060** (`migrations/060_staging_extractions_section_columns.sql`)
must be applied before running this script. It adds the `section_name`
and `char_offset` columns to `staging.extractions` and a partial index
keyed on `section_name`.

```bash
psql "$SCIX_DSN" -f migrations/060_staging_extractions_section_columns.sql
```

The migration is idempotent — safe to re-run.

---

## CLI

```
python scripts/run_ner_bodies.py [--max-papers N] [--since-bibcode BIBCODE]
                                 [--batch-size N] [--dry-run] [--allow-prod]
                                 [--confidence FLOAT] [--model HF_ID]
                                 [--source-version STR]
                                 [--inference-batch N] [--max-text-chars N]
                                 [--compile] [--dsn DSN] [-v]
```

Mirrors `scripts/run_ner_pass.py` exactly except `--target` is hard-wired
to `body` (this script is body-only by design).

`--allow-prod` is **required** if `--dsn` resolves to a production
database (`dbname=scix` or any DSN flagged by `scix.db.is_production_dsn`).
Without it, the script exits 2 before opening a connection.

---

## scix-batch wrapping

Body NER loads the GLiNER model into VRAM, accumulates section text in
RAM, and walks bibcodes in lexicographic order with keyset pagination.
Steady-state RSS sits around 8 GB; transient peaks during inference
batches can hit 12-15 GB. Per CLAUDE.md, **always wrap heavy runs in
`scix-batch`** so systemd-oomd kills the job (not the gascity supervisor)
on overshoot.

```bash
scix-batch --mem-high 16G --mem-max 24G \
    python scripts/run_ner_bodies.py --allow-prod
```

For a smoke / dev run that bounds the cost, drop the `--allow-prod` and
either point at `SCIX_TEST_DSN` or use `--dry-run`:

```bash
SCIX_DSN="$SCIX_TEST_DSN" \
    scix-batch --mem-high 8G --mem-max 12G \
    python scripts/run_ner_bodies.py --max-papers 100
```

---

## Worked example: 1000-paper smoke run

The smoke run is the primary "did this thing work" check. It picks the
first 1000 bibcodes (lexicographic order), runs section-aware NER, and
writes a per-paper row to `staging.extractions`.

**Pre-flight**

```bash
# 1. Confirm the migration is applied.
psql "$SCIX_DSN" -c "\d staging.extractions" | grep -E "section_name|char_offset"
# Expect two rows: section_name | text | ... and char_offset | integer | ...

# 2. Confirm GPU is available + GLiNER weights are cached.
python -c "import torch; print('cuda:', torch.cuda.is_available())"
ls -lh ~/.cache/huggingface/hub/models--gliner-community--gliner_large-v2.5/ \
    || echo 'will download ~430 MB on first run'
```

**Dry run first** (no DB writes; emits TSV per mention to stdout):

```bash
scix-batch --mem-high 12G --mem-max 16G \
    python scripts/run_ner_bodies.py \
        --max-papers 1000 \
        --dry-run \
        -v \
    | tee /tmp/ner_bodies_smoke.tsv
```

Sanity-check the output:

```bash
# Distribution by entity_type — expect software / instrument / method to dominate.
awk -F'\t' '{print $4}' /tmp/ner_bodies_smoke.tsv | sort | uniq -c | sort -rn

# Distribution by section_name — should be only methods / results / observations / data.
awk -F'\t' '{print $2}' /tmp/ner_bodies_smoke.tsv | sort | uniq -c | sort -rn
```

If the section breakdown shows `references`, `introduction`, `bibliography`,
or `acknowledgments`, the role-classifier filter is broken — STOP and
investigate `src/scix/section_role.py` before any production write.

**Real write** (1000-paper smoke into `scix_test`):

```bash
SCIX_DSN="$SCIX_TEST_DSN" \
    scix-batch --mem-high 16G --mem-max 24G \
    python scripts/run_ner_bodies.py --max-papers 1000 -v
```

Verify rows landed:

```bash
psql "$SCIX_TEST_DSN" -c "
    SELECT count(*),
           count(DISTINCT bibcode) AS papers,
           min(char_offset) AS min_off,
           max(char_offset) AS max_off
      FROM staging.extractions
     WHERE source = 'ner_body';
"

psql "$SCIX_TEST_DSN" -c "
    SELECT section_name, count(*)
      FROM staging.extractions
     WHERE source = 'ner_body'
     GROUP BY 1
     ORDER BY 2 DESC
     LIMIT 20;
"
```

---

## Staging → public promotion

Body NER rows live in `staging.extractions` until the operator runs the
existing promotion driver:

```bash
scix-batch python scripts/promote_staging_extractions.py \
    --source-filter ner_body \
    --batch-size 5000
```

The promotion script copies rows to `public.extractions` (and, when
entity-link rows are present, to `public.extraction_entity_links`) using
the `(bibcode, extraction_type, extraction_version)` unique key for
ON CONFLICT DO NOTHING. Promotion is read-only against staging — re-running
the body NER pass after a promotion is safe; the staging upsert refreshes
the row in place.

For the body-NER subset specifically, the per-section payload survives
the promotion intact (`payload` is JSONB and copied verbatim), so the
MCP entity tool and any downstream consumers see the full section
breakdown without re-querying staging.

---

## MCP entity tool integration

Once promoted, body NER rows are queryable through the existing entity
tool surface. The query path filters on `source='ner_body'`:

```sql
SELECT bibcode, section_name, char_offset, payload
  FROM staging.extractions   -- or public.extractions after promotion
 WHERE source = 'ner_body'
   AND section_name LIKE '%methods%'
 ORDER BY bibcode
 LIMIT 100;
```

The MCP wiring that exposes this filter through `mcp__scix__entity` is
landed by the sibling unit `mcp-extraction-wiring`. This runbook documents
the SQL contract; the MCP tool is responsible for translating its
`sources=['ner_body']` parameter to the WHERE clause above.

---

## Operational deferral: the 100K stratified pilot

The PRD calls for a 100K-paper stratified pilot run as the M2 evaluation
artifact. **This script ships the pipeline; it does not run the pilot.**
The pilot requires a dedicated GPU window, `--allow-prod`, and operator
sign-off on the stratification recipe.

When the operator is ready, the pilot is invoked as:

```bash
# 1. Generate a stratified bibcode sample (per-discipline + per-year).
#    Recipe lives in scripts/sample_stratified_bibcodes.py (TODO — not in
#    scope for this unit). For now the operator can hand-curate via:
psql "$SCIX_DSN" -c "
    \\copy (
        SELECT bibcode FROM (
            SELECT bibcode,
                   row_number() OVER (
                       PARTITION BY arxiv_class, year ORDER BY random()
                   ) AS rn
              FROM papers
             WHERE body IS NOT NULL AND body <> ''
        ) s
        WHERE rn <= 100   -- 100 per (discipline, year) cell
    ) TO '/tmp/pilot_bibcodes.txt'
"

# 2. Run the pilot inside a scix-batch scope, using --since-bibcode +
#    bash to walk the sample. 100K papers at ~50 papers/s on a 5090 ≈ 35 min.
scix-batch --mem-high 16G --mem-max 24G \
    python scripts/run_ner_bodies.py \
        --allow-prod \
        --max-papers 100000 \
        --batch-size 200 \
        -v \
        2>&1 | tee /tmp/ner_bodies_pilot.log

# 3. Promote into public.extractions when satisfied with the sample.
scix-batch python scripts/promote_staging_extractions.py \
    --source-filter ner_body --allow-prod
```

The pilot output is the per-discipline coverage table + a hand-validation
sample (50 papers, 3 entities each), to be appended to
`docs/eval/body_ner_pilot.md` (created by a follow-up evaluation unit, not
this one).

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ProgrammingError: column "section_name" of relation "extractions" does not exist` | Migration 060 not applied | Run the `psql -f migrations/060_*.sql` command from the schema-dependency section |
| Script exits 2 with "Refusing to run against production DSN" | `--allow-prod` missing | Add `--allow-prod` if you really mean it; otherwise re-run with `SCIX_TEST_DSN` |
| OOM kill of the gascity supervisor | Forgot `scix-batch` | Always wrap the command per CLAUDE.md memory rule on systemd-oomd |
| Empty `section_name` distribution / 0 rows inserted | Section parser sees no headers (paper bodies are pure text) | Inspect `papers.body` shape — the parser falls back to a single `('full', ...)` section that is filtered out |
| Same paper inserted twice with different mentions | `--source-version` was bumped between runs | Expected — the unique key includes `extraction_version`, so multiple versions coexist |

---

## Related work units

- **M1** `coverage-bias-report` (commit 130f4c6) — provides the discipline
  coverage baseline this pipeline's output will be compared against.
- **S2** `section-role-classifier` (commits 62d8740, f6ba52e) — provides
  the `classify_section_role` function this script depends on.
- **mcp-extraction-wiring** (sibling, parallel) — exposes the
  `sources=['ner_body']` filter through the MCP entity tool.
