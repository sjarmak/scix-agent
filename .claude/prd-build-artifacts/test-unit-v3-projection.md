# Test results — unit-v3-projection

## Command

```
/home/ds/projects/scix_experiments/.venv/bin/python -m pytest \
    tests/test_project_embeddings_umap.py -q
```

## Result

**24 passed, 0 failed, 3 warnings, 12.21 s.**

The 3 warnings are from umap-learn itself (``n_jobs value 1 overridden to 1
by setting random_state``) — a benign upstream notice that we deliberately
accept in exchange for reproducible projections with a fixed ``random_state``.

## Acceptance-criterion coverage

| Criterion | Test(s) | Outcome |
|-----------|---------|---------|
| 6(a) synthetic end-to-end (umap-learn, ~200 768-d vectors) | `test_synthetic_end_to_end` | PASS — 200 points written, validator round-trip accepts output |
| 6(b) CLI flag parsing | `test_cli_parsing_defaults`, `test_cli_parsing_all_flags`, `test_cli_parsing_rejects_unknown_resolution`, `test_cli_parsing_rejects_unknown_backend` | PASS |
| 6(c) output-JSON schema validator | `test_validate_projection_payload_accepts_valid`, `test_validate_projection_payload_rejects_missing_key`, `test_validate_projection_payload_rejects_wrong_type`, `test_validate_projection_payload_rejects_unknown_resolution`, `test_validate_projection_payload_rejects_non_list` | PASS |
| 6(d) --dry-run writes no file | `test_dry_run_writes_no_file` | PASS |

## Supporting coverage

| Concern | Test |
|---------|------|
| Synthetic loader shape + dtype | `test_synthetic_loader_shape` |
| Synthetic loader determinism | `test_synthetic_loader_is_deterministic` |
| Synthetic loader rejects n<=0 | `test_synthetic_loader_rejects_nonpositive_n` |
| Resolution allowlist gates SQL builder | `test_stratified_sample_sql_allowlist_rejects_unknown` |
| SQL uses expected paper_metrics column | `test_stratified_sample_sql_uses_expected_column` |
| DB loader rejects unknown resolution | `test_load_embeddings_from_db_rejects_unknown_resolution` |
| Backend selection (umap-learn) | `test_pick_backend_umap_learn_small_n` |
| Backend auto falls back without cuML | `test_pick_backend_auto_falls_back_without_cuml` |
| Backend rejects unknown name | `test_pick_backend_rejects_unknown` |
| project + build_points integration | `test_project_and_build_points_roundtrip` |
| build_points length mismatch | `test_build_points_rejects_length_mismatch` |
| build_points unknown resolution | `test_build_points_rejects_unknown_resolution` |
| serialize + validate round-trip | `test_serialize_writes_valid_json` |

## CLI smoke tests (manual)

```
$ .venv/bin/python scripts/viz/project_embeddings_umap.py \
      --synthetic 200 --output /tmp/umap_test_v3.json --backend umap-learn
...
INFO: wrote /tmp/umap_test_v3.json (200 points)
$ head -c 300 /tmp/umap_test_v3.json
[
  {
    "bibcode": "synthetic-000000",
    "x": 10.995...,
    "y": 3.460...,
    "community_id": 0,
    "resolution": "coarse"
  },
  ...
```

```
$ .venv/bin/python scripts/viz/project_embeddings_umap.py \
      --synthetic 100 --dry-run --backend umap-learn \
      --output /tmp/should_not_exist.json
INFO: --dry-run set: skipping file write and DB upsert
$ ls -la /tmp/should_not_exist.json
ls: cannot access /tmp/should_not_exist.json: No such file or directory
```

Both behaviors match the spec (6(a) and 6(d) verified at the CLI layer as
well as through pytest).

## Dependency install notes

``umap-learn 0.5.12`` was installed into the project venv during Phase 1 via
``.venv/bin/python -m pip install umap-learn``. ``cuml`` is not available on
this machine (expected — the RTX 5090 does not ship with the RAPIDS stack
by default), so ``pick_backend("auto")`` silently falls back to umap-learn.
This fallback is covered by ``test_pick_backend_auto_falls_back_without_cuml``.
