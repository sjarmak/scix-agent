# Co-mention edge table — design and refresh strategy

> Status: shipped (migration 063, 2026-04-29). Tracks
> [scix_experiments-dbl.5](#) — *Co-mention edge table (entity↔entity within
> paper)*. Parent epic: `scix_experiments-dbl` (cross-discipline expansion).

## What this table is

`co_mentions` is the entity↔entity co-occurrence summary derived from
`document_entities`. One row per *unordered* pair `(entity_a_id, entity_b_id)`
with `entity_a_id < entity_b_id`, carrying:

| Column | Type | Meaning |
|---|---|---|
| `entity_a_id` | `INTEGER` | Smaller entity id of the pair (canonical-order half of the symmetry). |
| `entity_b_id` | `INTEGER` | Larger entity id. |
| `n_papers`    | `INTEGER` | Distinct bibcodes where both entities are linked via `document_entities`, any `match_method`. Always `>= 2` (CHECK + populator HAVING). |
| `first_year`  | `SMALLINT` | Earliest `papers.year` across the support set (`NULL` when no supporting paper has a year). |
| `last_year`   | `SMALLINT` | Latest `papers.year` across the support set. |

Indexes (migration 063):

- `co_mentions_pkey (entity_a_id, entity_b_id)` — point lookup of pair strength.
- `ix_co_mentions_a_npapers (entity_a_id, n_papers DESC)` — top-k partners
  when the queried entity is the smaller side of a pair.
- `ix_co_mentions_b_npapers (entity_b_id, n_papers DESC)` — top-k partners
  when the queried entity is the larger side. Symmetric lookup is a
  `UNION ALL` over both arms with an outer `LIMIT k`.

## What it is *not*

- **Not a paper-level join table.** We do not store `(paper, entity_a, entity_b)`
  triples — that would be O(150M+) rows. Per-paper context lives in
  `document_entities`, paper-paper context lives via `citations`, and this
  table is strictly the marginal-counted (entity, entity) summary.
- **Not a confidence-aware edge.** We pool across all `match_method` values
  (gliner, keyword_exact_lower, aho_corasick_abstract, …) without filtering
  by confidence. Confidence-filtered variants can be derived at query time
  by re-aggregating against `document_entities` if a use case demands it;
  shipping a confidence-stratified table here would multiply storage with
  uncertain payoff.
- **Not a PMI / NPMI table.** We store the support count `n_papers` only.
  Marginal frequency per entity comes from
  `agent_entity_context.citing_paper_count` (or a fresh count against
  `document_entities`). Compute PMI / NPMI / Jaccard at query time when
  needed; encoding a single normalization here would lock callers into one
  scoring choice.

## Min-support floor (`n_papers >= 2`)

The HAVING clause and CHECK constraint together drop singleton
co-mentions. Reasons:

1. **Storage.** Singleton pairs are the long tail — typically 4–6× the row
   count of `n_papers >= 2` pairs. Dropping them is the largest single
   storage win.
2. **Signal.** A pair that co-occurs in exactly one paper carries no
   association evidence beyond "two entities happened to land in the same
   document at least once" — much of which is incidental (a method paper
   listing dozens of comparison datasets, etc.).
3. **Recoverability.** A caller that genuinely needs n=1 pairs can compute
   them on demand from `document_entities` for a specific entity.

## Refresh strategy

`scripts/populate_co_mentions.py` rebuilds the table by year-chunked
aggregation:

```
SCIX_TEST_DSN="dbname=scix_test" python scripts/populate_co_mentions.py
scix-batch python scripts/populate_co_mentions.py --allow-prod
```

The script:

1. Discovers the distinct year set in `papers` (plus a `NULL`-year
   bucket if any).
2. For each year, generates pair instances via the canonical
   self-join on `document_entities` and aggregates them within the year
   into a TEMP staging table (`tmp_co_mentions_partial`).
3. After all years are processed, performs a final
   `INSERT INTO co_mentions … FROM tmp_co_mentions_partial GROUP BY a, b
   HAVING SUM(n_papers_partial) >= min_n_papers` merge into the live
   table.

`n_papers` across years is additive because each
`(paper, entity_a, entity_b)` tuple appears in exactly one year-chunk —
no double counting.

The script raises `work_mem` to 4 GB for its session and runs with
`synchronous_commit = OFF` (recoverable: a crash mid-run leaves
`co_mentions` empty after TRUNCATE, which is fine because the next
rebuild will repopulate it).

Each run is logged to `co_mention_runs`:

```sql
SELECT id, started_at, finished_at - started_at AS dur,
       refresh_kind, n_papers_input, n_pairs_output
  FROM co_mention_runs
  ORDER BY id DESC LIMIT 5;
```

### Full vs incremental refresh

| Refresh kind | When to use | What it does | Cost (estimate) |
|---|---|---|---|
| `full` (default) | After NER backfill finishes; weekly cron after that. | TRUNCATE + rebuild from `document_entities`. | `O(D + P)` where D = `document_entities` rows (currently 100M) and P = pair instances (~150M). Expected runtime: a few hours on the production scix DB at the time of writing. |
| `pilot` | Smoke test or runtime estimation on a single year. | Inserts pairs for one year only into the live table without TRUNCATE. | Minutes for a small year. Pilot rows are wiped by the next `full` run. |
| Incremental | Not implemented. | — | — |

#### Why no incremental refresh

Adding incremental refresh sounds attractive — re-process only papers
whose `document_entities` changed since the last run — but the pair
table is *summary-shaped*, and even one removed `document_entities` row
can affect O(k²) pair rows for a paper with k entities. Concretely:

- An incremental refresh for "papers updated since last run" would
  need to *subtract* the pair-row contributions of those papers' prior
  state, then *add* their new state, requiring a snapshot of the prior
  state. We don't keep that snapshot.
- Alternatively the incremental flow could regenerate the full pair
  set for each touched paper and `UPSERT … ON CONFLICT (a, b) DO
  UPDATE SET n_papers = co_mentions.n_papers + EXCLUDED.n_papers`.
  That works for adds but mishandles deletes (cannot subtract).
- Catching deletes generically requires either a `document_entities`
  audit log or a row-level trigger, both of which would slow the NER
  backfill (which is the heaviest write-path against
  `document_entities` today).

The fall-back is straightforward: rebuild on a cadence that matches
the rate of change in `document_entities`. While the NER backfill is
running, daily rebuilds keep co-mentions fresh enough for agent
exploration. Once NER backfill finishes and `document_entities` becomes
stable, weekly is appropriate.

Operators should consider `co_mentions` rows *implicitly* lower-bounded
in `n_papers` by what `document_entities` showed at the most recent
rebuild. Use `co_mention_runs` to compute staleness, and re-run when
the gap exceeds the staleness budget for a given consumer.

## Schema invariants (CHECK constraints)

Migration 063 enforces three CHECK constraints so the population script
isn't the only place these rules live:

- `co_mentions_a_lt_b      CHECK (entity_a_id < entity_b_id)`
- `co_mentions_n_papers_ge CHECK (n_papers >= 2)`
- `co_mentions_year_order  CHECK (first_year IS NULL OR last_year IS NULL OR first_year <= last_year)`

Tests in `tests/test_co_mentions.py::TestCoMentionsSchemaConstraints`
exercise each of the three.

## Foreign keys (deferred)

The initial migration intentionally omits FKs to `entities(id)`. FK
creation requires `ShareRowExclusive` on the parent table, which
conflicts with long-running `CREATE INDEX CONCURRENTLY` builds on
`entities` (a multi-hour trigram-index rebuild was in flight when this
table was first created). Referential integrity is maintained instead
by the rebuild semantics of `populate_co_mentions.py`: the table is
TRUNCATEd and re-INSERTed from a JOIN against `entities` (via
`document_entities`), so any deleted/superseded entity is dropped on
the next refresh.

A follow-up migration can attach FKs once the entities table is quiet.
The acceptance bar is "no concurrent DDL on `entities`."

## MCP surface (`entity_context.co_mentions`)

The `entity_context` MCP tool (port of `scix.search.get_entity_context`)
gains a new top-level `co_mentions` field on its response. By default it
returns the top-10 partners sorted by `n_papers DESC`; the
`co_mentions_limit` argument (range `[0, 100]`) lets callers widen or
disable the surface.

A standalone `scix.search.get_top_co_mentions` helper exists for
internal use (e.g. composing co-mention results into other Python
analyses). The MCP surface intentionally does **not** expose it as a
separate tool to keep the documented ≤15-tool cap from sliding further
above its current 21-tool count (see `docs/mcp_tool_audit_2026-04.md`).

## Future work

- **Confidence-stratified variant.** A view `v_co_mentions_high_conf`
  that only counts pair-instances where both `match_method`s have
  `confidence >= 0.9` might give a higher-precision exploration surface
  for downstream PMI/lift scoring. Defer until a real consumer asks.
- **Symmetric matview wrapper.** A `CREATE MATERIALIZED VIEW
  v_co_mentions_symmetric` that doubles each row into both directions
  and adds `entity_marginal_n` columns would simplify some agent
  queries at the cost of 2× storage. Probably not worth it; the
  `UNION ALL` query is already fast and adds zero storage.
- **Cross-discipline bridge ranker.** With per-entity discipline tags
  (already on `entities.discipline`), it's straightforward to filter to
  pairs whose two entities sit in different disciplines and rank by
  PMI. That's the intended hero query for the parent epic and should
  be a first follow-up.
