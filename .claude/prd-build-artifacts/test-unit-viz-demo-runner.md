# Test — unit-viz-demo-runner

## Unit tests

File: `tests/test_viz_demo_runner.py`

Covers all acceptance criteria that map to automatable checks:

1. `test_run_sh_exists` — file present at `scripts/viz/run.sh`.
2. `test_run_sh_executable` — file has user-exec bit.
3. `test_run_sh_syntax` — `bash -n scripts/viz/run.sh` returns 0.
4. `test_run_sh_help` — `--help` exits 0 and emits `--port`, `--host`,
   `--no-build` in the usage text.
5. `test_build_only_synthetic_produces_json` (marked `integration`) —
   removes any existing `data/viz/{sankey,umap}.json`, runs
   `./scripts/viz/run.sh --build-only --synthetic` with a 600s timeout,
   asserts both JSON files exist, parse as JSON, and have the expected
   top-level shape.
   - The test auto-sets `VENV_PY` to the current test interpreter when
     `REPO_ROOT/.venv/bin/python` does not exist (so pytest run from a
     git worktree that shares the main repo's venv works without extra
     env setup).

## Results — `pytest tests/test_viz_demo_runner.py -q`

```
.....                                                                    [100%]
5 passed in 14.70s
```

## Regression suite

Command:

```
.venv/bin/python -m pytest \
  tests/test_viz_server.py \
  tests/test_sankey_frontend.py \
  tests/test_umap_frontend.py \
  tests/test_trace_stream.py \
  tests/test_agent_trace_frontend.py \
  tests/test_mcp_trace_instrumentation.py \
  -q
```

Result:

```
..........................................                               [100%]
42 passed in 4.18s
```

No existing tests in the viz stack regressed. The new Makefile target
plus run.sh do not import into any existing module, so the blast radius
is limited to the operator entrypoint.

## Manual checks (acceptance spec crosswalk)

| AC | Check | Outcome |
|----|-------|---------|
| 1 | `scripts/viz/run.sh` exists + exec + `bash -n` ok | PASS (tests 1-3) |
| 2 | `--help` prints `--port`, `--host`, `--no-build` | PASS (test 4) |
| 3 | `--build-only --synthetic` produces both JSON files, exits 0 | PASS (test 5 + manual run) |
| 4 | Top-level Makefile has `viz-demo` target | PASS — `make -n viz-demo` prints `./scripts/viz/run.sh` |
| 5 | `docs/viz/DEMO.md` exists with three views + three V4 scenarios | PASS — sections V2, V3, V4 with Scenario 1/2/3 |
| 6 | `pytest tests/test_viz_demo_runner.py -q` passes | PASS (5 passed) |

## Known caveats

- The integration test writes `data/viz/sankey.json` and
  `data/viz/umap.json` and leaves them on disk. These files are
  regenerable artifacts and are gitignored at the `data/` level. Runs
  are idempotent.
- Running under a pure fresh checkout without a `.venv` will require
  either `uv sync` / `pip install -e .[viz]` first, or
  `VENV_PY=python3 ./scripts/viz/run.sh`. Documented in `DEMO.md`.
