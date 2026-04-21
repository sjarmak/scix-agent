# Research — unit-viz-demo-runner

## Scope

Create `scripts/viz/run.sh` — a single entrypoint that builds the Sankey and
UMAP JSON data files (via the existing build scripts, optionally with
`--synthetic`), then launches the viz FastAPI app with uvicorn. Add a
top-level `Makefile` with a `viz-demo` target, write `docs/viz/DEMO.md`
with the three-view narrative and V4 scenarios, and a pytest smoke test.

## Inputs confirmed after rebase

After `git rebase prd-build/kg-visualization`, all required upstream files
are present:

- `scripts/viz/build_temporal_sankey_data.py` — Sankey data builder
- `scripts/viz/project_embeddings_umap.py` — UMAP projection runner
- `src/scix/viz/server.py` — FastAPI app (`scix.viz.server:app`)
- `web/viz/sankey.html`, `web/viz/umap_browser.html`, `web/viz/agent_trace.html`

## CLI shapes (confirmed)

### `build_temporal_sankey_data.py`

- Flags: `--resolution {coarse,medium,fine}`, `--top-flows N`, `--output PATH`,
  `--dsn DSN`, `--dry-run`, `--synthetic`.
- `--synthetic` loads a deterministic 10k-row dataset (no DB required),
  `--dry-run` does everything except writing the output file.
- Default output path: `data/viz/sankey.json` (relative to repo root).
- Writes pretty-printed JSON with `nodes`, `links` arrays.

### `project_embeddings_umap.py`

- Flags: `--sample-size N`, `--resolution {coarse,medium,fine}`,
  `--backend {auto,cuml,umap-learn}`, `--dsn DSN`, `--output PATH`,
  `--dry-run`, `--synthetic N`.
- `--synthetic N` generates `N` deterministic 768-d synthetic embeddings
  (no DB required). In synthetic mode the script skips the `paper_umap_2d`
  DB upsert — JSON file is still written unless `--dry-run`.
- Default output path: `data/viz/umap.json` (relative to repo root).
- Backend `auto` falls back to umap-learn (installed in `.venv`) when cuML
  is unavailable (we have no cuML in this env, so `auto` -> umap-learn).

## Dependency availability (.venv)

All Python deps are already installed in the main repo `.venv/` (worktree
does not have its own):

- `uvicorn` 0.44.0
- `fastapi` 0.136.0
- `umap-learn` 0.5.12

The `VENV_PY` default in `run.sh` is `.venv/bin/python`. When invoked from
a worktree (where `.venv/` does not exist at the worktree root), we must
still target the canonical repo root. Simplest robust approach: accept
override via `VENV_PY` env var; recommend pointing it at the main repo's
`.venv/bin/python` when running from a worktree. For the normal use case
(running from `/home/ds/projects/scix_experiments/`), `.venv/bin/python`
and `.venv/bin/uvicorn` resolve correctly.

## Viz server launch command

From `src/scix/viz/server.py`:

    uvicorn scix.viz.server:app --host 0.0.0.0 --port 8765

The server mounts `web/viz/` as static files and adds JSON API + trace-stream
routes under `/viz/*`. It has no DB dependency until routes are hit, so
launching with synthetic JSON data is fine for a demo.

## Data files policy

The acceptance criteria require `data/viz/sankey.json` and
`data/viz/umap.json` to exist after `--build-only --synthetic`. The pytest
smoke test removes any pre-existing files first (to isolate from previous
runs), runs the command, asserts the files exist, then restores the
originals if it captured them.

Both build scripts resolve relative `--output` paths against the repo root
using `_REPO_ROOT / p` — so even if run from inside a worktree, output
lands at the absolute `data/viz/*.json` at the main repo root. For the
run.sh we `cd "$REPO_ROOT"` first so any relative defaults resolve
correctly and the behaviour is identical regardless of cwd.

## Test strategy

Four tests in `tests/test_viz_demo_runner.py`:

1. `test_run_sh_executable` — `os.access("scripts/viz/run.sh", os.X_OK)`
2. `test_run_sh_syntax` — `bash -n scripts/viz/run.sh` returns 0
3. `test_run_sh_help` — stdout contains `--port`, `--host`, `--no-build`
4. `test_build_only_synthetic_produces_json` (marked `integration`) —
   runs `./scripts/viz/run.sh --build-only --synthetic`, asserts both
   JSON files now exist, cleans up afterwards.

Regression suite to rerun: `tests/test_viz_server.py`,
`tests/test_sankey_frontend.py`, `tests/test_umap_frontend.py`,
`tests/test_trace_stream.py`, `tests/test_agent_trace_frontend.py`,
`tests/test_mcp_trace_instrumentation.py`.

## Makefile

Top-level Makefile does not exist yet. We add a minimal one with two
targets: `viz-demo` (launches server) and `viz-demo-build` (build-only,
synthetic). Both delegate to `./scripts/viz/run.sh`. Tabs, not spaces.

## DEMO.md

`docs/viz/` does not exist yet. We create it with a single `DEMO.md` that
covers:

- Overview narrative (why three views, the common thread of
  agent-navigable scientific knowledge).
- Quickstart (`make viz-demo`).
- V2 Temporal Sankey — three talking-point flows.
- V3 UMAP browser — interaction cheat sheet.
- V4 Agent trace overlay — three prepared scenarios (literature survey,
  related methods, entity disambiguation).
- Tips (port conflicts, data regeneration, known limitations).
