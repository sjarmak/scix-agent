# Plan — unit-v3-projection (UMAP projection runner)

## Target files

- `scripts/viz/project_embeddings_umap.py` (new, +x)
- `tests/test_project_embeddings_umap.py` (new)

## Module layout (project_embeddings_umap.py)

1. **Imports** (stdlib first, then numpy, then lazy-imported psycopg / umap / cuml).
2. **Logger setup** — `logging.basicConfig(INFO, ...)` mirror sankey script.
3. **Constants**
   - `DEFAULT_SAMPLE_SIZE = 100_000`
   - `DEFAULT_RESOLUTION = "coarse"`
   - `DEFAULT_OUTPUT_REL = Path("data/viz/umap.json")`
   - `DEFAULT_BACKEND = "auto"`
   - `RESOLUTIONS = ("coarse", "medium", "fine")`
   - `COMMUNITY_COLUMNS = {"coarse": "community_semantic_coarse", ...}`
   - `_EMBEDDING_DIM = 768`
   - `_DB_FETCH_BATCH = 10_000`
4. **Data model** (frozen dataclasses)
   - `ProjectedPoint(bibcode, x, y, community_id, resolution)`
   - `Config(sample_size, resolution, backend, dsn, output, dry_run, synthetic_n)`.
     `synthetic_n: int | None` — None means DB path.
5. **Protocol**
   - `class UMAPBackend(Protocol): def fit_transform(self, X: np.ndarray) -> np.ndarray: ...`
6. **Helpers**
   - `load_embeddings_synthetic(n, seed=42) -> (bibcodes, np.ndarray[n,768], community_ids)`
     - Deterministic numpy RNG, float32 embeddings.
     - `community_ids[i] = i % 20` (20 synthetic communities) — simulate stratification.
   - `load_embeddings_from_db(dsn, resolution, sample_size) -> (bibcodes, embeddings, community_ids)`
     - Allowlist-check resolution.
     - Query:
       ```sql
       WITH ranked AS (
         SELECT pe.bibcode, pe.embedding_hv, pm.<col> AS community_id,
                ROW_NUMBER() OVER (PARTITION BY pm.<col> ORDER BY random()) AS rn
         FROM paper_embeddings pe
         JOIN paper_metrics pm USING (bibcode)
         WHERE pe.model_name = 'indus'
           AND pe.embedding_hv IS NOT NULL
           AND pm.<col> IS NOT NULL
       )
       SELECT bibcode, embedding_hv, community_id
       FROM ranked
       WHERE rn <= :per_community
       LIMIT :sample_size
       ```
     - `per_community = max(1, ceil(sample_size / 20))` heuristic, documented.
     - Stream via server-side named cursor, accumulate into numpy array.
     - Decode halfvec via pgvector adapter; fallback to string parse.
   - `pick_backend(name) -> (backend_instance, label)` — logs the choice.
   - `project(embeddings, backend) -> np.ndarray[(n,2)]` — tiny wrapper.
   - `build_points(bibcodes, xy, community_ids, resolution) -> tuple[ProjectedPoint, ...]`
   - `serialize(points, path) -> None` — JSON array write.
   - `validate_projection_payload(obj) -> tuple[ProjectedPoint, ...]` — schema check.
   - `write_to_db(dsn, points) -> int` — psycopg executemany UPSERT.
7. **CLI**
   - `_parse_args(argv) -> argparse.Namespace`
   - `_config_from_args(args) -> Config`
   - `main(argv=None) -> int`

## main() flow

1. Parse args, build Config.
2. Validate resolution against allowlist; return 2 on failure.
3. If `synthetic_n` is not None → `load_embeddings_synthetic`.
   Else → guard log for prod DSN, call `load_embeddings_from_db`.
4. Pick backend, `fit_transform`, build points.
5. Log summary (n points, backend, output path, dry_run flag).
6. If `dry_run`: return 0 without writing.
7. Else: `serialize(points, config.output)` + optional `write_to_db`.
   (DB write only when NOT synthetic AND NOT dry_run.)
8. Return 0.

## Tests (tests/test_project_embeddings_umap.py)

1. `test_synthetic_end_to_end(tmp_path)` — `main(["--synthetic", "200",
   "--output", str(tmp_path/"umap.json"), "--backend", "umap-learn"])` →
   returns 0, file exists, `validate_projection_payload` accepts it, len=200.
2. `test_cli_parsing()` — exercise `_parse_args` with every flag; assert
   Namespace values (including defaults).
3. `test_output_schema_validator()` — build a valid payload (accept) and an
   invalid one (missing key, wrong type) — assert ValueError.
4. `test_dry_run_writes_no_file(tmp_path)` — `main(["--synthetic", "50",
   "--dry-run", "--output", str(tmp_path/"x.json"), "--backend",
   "umap-learn"])` → returns 0, file does NOT exist.
5. `test_synthetic_loader_shape()` — `load_embeddings_synthetic(50)` returns
   (list[50], ndarray[50,768] float32, list[50] with ints).
6. `test_resolution_allowlist_rejected()` — DB loader with unknown resolution
   raises ValueError (defense-in-depth coverage).

All tests use `umap-learn` backend only — no DB, no GPU.

## Style / quality gates

- black line-length 100, ruff-clean.
- Type annotations on every public function.
- `from __future__ import annotations`.
- `logging.basicConfig` at module level mirroring sankey script.
- `sys.path` insert for `src/` import so tests/CLI work from any CWD.

## Commit sequencing

1. Write research + plan artifacts.
2. Implement script + test in one patch.
3. Run pytest → fix any failures.
4. Write test-unit-v3-projection.md with pytest output summary.
5. Stage files, set +x, commit.
