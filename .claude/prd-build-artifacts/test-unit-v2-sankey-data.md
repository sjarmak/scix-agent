# Test — unit-v2-sankey-data

## Command

```
pytest tests/test_build_temporal_sankey_data.py -q
```

## Result

```
.....                                                                    [100%]
5 passed in 0.19s
```

All five tests pass:

1. `test_validate_sankey_data_accepts_minimal_valid_payload` — schema validator
   accepts a hand-built minimal dataset and returns a frozen `SankeyData`.
2. `test_validate_sankey_data_rejects_missing_keys` — three negative cases:
   missing top-level key, missing link field, wrong node-field type.
3. `test_decade_of_boundary_years` — confirms `decade_of(1999) == 1990`,
   `decade_of(2000) == 2000`, plus neighbouring boundaries up to 2025.
4. `test_top_flows_cap_limits_links` — builds 50 communities × 5 decades
   (200 candidate links) and asserts `len(result.links) == 10` when
   `top_flows=10`. Also asserts descending-value ordering.
5. `test_dry_run_does_not_write_output` — invokes `main(["--dry-run",
   "--synthetic", "--output", str(tmp_path/"sankey.json")])`, asserts
   return code 0 and that the output file does not exist.

## CLI smoke test

```
python scripts/viz/build_temporal_sankey_data.py --dry-run --synthetic
```

Succeeds. Emits a summary log:

```
aggregated: 80 nodes, 60 links (top_flows=500)
summary: nodes=80 links=60 papers=10000 decades=[1990, 2000, 2010, 2020]
         communities=20 dry_run=True synthetic=True
--dry-run set, skipping write of .../data/viz/sankey.json
```

## Non-dry-run verification

Running without `--dry-run` writes the JSON and the output round-trips through
`validate_sankey_data` without error (80 nodes, 60 links).

## Mapping to acceptance criteria

| AC | Evidence |
| -- | -------- |
| 1. Script exists + executable | `scripts/viz/build_temporal_sankey_data.py` created; `chmod +x` applied; staged with `git update-index --chmod=+x`. |
| 2. `--dry-run --synthetic` prints summary | See CLI smoke test above. |
| 3. CLI flags present | `argparse` defines `--resolution`, `--top-flows`, `--output`, `--dsn`, `--dry-run`, `--synthetic` with the required defaults. |
| 4. JSON schema + validator | `Node`/`Link`/`SankeyData` frozen dataclasses; `validate_sankey_data(dict) -> SankeyData`. |
| 5. Pytest suite | All 5 tests pass; 4 required sub-points covered, with 1 extra negative-case test for 5(1) robustness. |
