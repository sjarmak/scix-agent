# Research — unit-v3-projection (UMAP projection runner)

## Inputs reviewed

- `migrations/055_paper_umap_2d.sql` — paper_umap_2d schema:
  - `bibcode text PK FK -> papers(bibcode)`
  - `x, y double precision NOT NULL`
  - `community_id integer NULL`
  - `resolution text NOT NULL`
  - `projected_at timestamptz DEFAULT now()`
  - Index: `(resolution, community_id)` for community filter lookups.
- `migrations/053_paper_embeddings_halfvec.sql` + `054_*` — paper_embeddings
  has a `halfvec(768)` shadow column `embedding_hv` populated for
  `model_name='indus'` rows. The legacy 768-d `embedding` column still holds
  INDUS for rows not yet backfilled; halfvec is canonical going forward.
  Primary key on the table is `(bibcode, model_name)`.
- `migrations/051_community_semantic_columns.sql` — `paper_metrics` has
  `community_semantic_coarse/medium/fine` (INT) with btree indexes on each.
- `scripts/viz/build_temporal_sankey_data.py` — Layer-0 sibling. Reference
  style: `Config` frozen dataclass, `load_synthetic` generator, lazy psycopg
  import, server-side named cursor with `fetchmany`, allowlist-based SQL
  column interpolation (via `COMMUNITY_COLUMNS`), pure `main()` entrypoint
  returning an int, logging via the `logging` module.
- `src/scix/db.py` — provides `DEFAULT_DSN`, `is_production_dsn`, `redact_dsn`.
- `tests/test_build_temporal_sankey_data.py` — companion test layout to
  mirror: imports from `scripts.viz...`, schema validator round-trip, tmp_path
  fixture for output files.

## Dependency status

- `numpy 2.4.4` — installed.
- `psycopg 3.3.3` — installed.
- `umap-learn 0.5.12` — installed via
  `/home/ds/projects/scix_experiments/.venv/bin/python -m pip install umap-learn`
  during the Phase-1 research step.
- `cuml` — not installed. `pick_backend("auto")` must fall back silently to
  umap-learn when `import cuml` raises ImportError.

## Column / type decisions

- We will read from `paper_embeddings.embedding_hv` filtered by
  `model_name = 'indus'` (canonical INDUS column post-halfvec migration).
  psycopg returns halfvec as a Python list of floats (pgvector's Python
  adapter), which `np.asarray(...).astype(np.float32)` converts cleanly.
  (If the adapter is missing, we parse string form.)
- `paper_metrics.community_semantic_coarse/medium/fine` — resolution maps to
  column via an allowlist constant `COMMUNITY_COLUMNS`, identical to the
  Sankey script's pattern.
- `paper_umap_2d` upsert shape: `(bibcode, x, y, community_id, resolution)`
  with ON CONFLICT (bibcode) DO UPDATE — the migration's PK is on bibcode
  alone, so a paper only has one projection row at a time.

## Sampling strategy

Stratified sample by community. Two candidate SQL approaches:

1. **Window-function cap** — deterministic cap of `ceil(sample_size /
   n_communities)` per community using
   `ROW_NUMBER() OVER (PARTITION BY community_id ORDER BY random())`.
   Pros: exact per-community cap, reproducible.
   Cons: random() sort is O(N log N) per partition.
2. **TABLESAMPLE BERNOULLI (p)** — probability-based sample at DB level.
   Pros: fast on huge tables.
   Cons: not stratified; small communities may be under-sampled.

**Chosen**: Window-function cap — stratification is core to the unit's
"stratified by community_semantic_coarse" requirement, and the window
predicate lets us tune per-community cap directly. Documented in code.

## --dry-run vs --synthetic

Per the spec: `--synthetic N` = skip DB, generate N synthetic 768-d vectors +
community assignments. `--dry-run` = skip BOTH DB write AND file write;
only print summary. Combinations:

- `--synthetic 200 --output /tmp/out.json` → writes file, no DB.
- `--synthetic 200 --dry-run` → no file, no DB.
- `--dry-run` alone (DB path) → hits DB read-only, no write, no file.

## CLI flags (all required by spec)

`--sample-size` (default 100_000), `--resolution` (default `coarse`),
`--backend {auto,cuml,umap-learn}` (default `auto`), `--dsn`, `--output`
(default `data/viz/umap.json`), `--dry-run`, `--synthetic N`.

## Output JSON schema

```
[
  {"bibcode": "2020ApJ...900..100X",
   "x": 1.234, "y": -5.678,
   "community_id": 17, "resolution": "coarse"},
  ...
]
```

Top-level is a JSON array (not an envelope) — matches the unit spec verbatim.
Validator will accept a list of objects and return a tuple of ProjectedPoint.

## Risks / gotchas

- cuML vs umap-learn interface parity — both expose `.fit_transform(X)` but
  cuML returns a cupy array. Wrapper must call `.get()` on cuML result.
- UMAP on 200 points is fast (<1s); we pin `n_neighbors=min(15, n-1)` so tiny
  synthetic datasets don't blow up UMAP's kNN-graph construction.
- halfvec decode: pgvector's psycopg3 integration registers an adapter at
  connection open. If absent, values come back as strings like
  `"[0.1, 0.2, ...]"` — we handle both defensively (parse via `ast.literal_eval`
  fallback), but prefer relying on the adapter.
- Resolution allowlist check BEFORE SQL interpolation — defense in depth.
