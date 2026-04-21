# Plan — unit-viz-demo-runner

## Files to create

1. `scripts/viz/run.sh` — executable bash entrypoint.
2. `Makefile` (top-level, new).
3. `docs/viz/DEMO.md` — demo narrative + V4 scenarios.
4. `tests/test_viz_demo_runner.py` — pytest smoke tests.

## `scripts/viz/run.sh`

Bash script with `set -euo pipefail`:

- Resolve `SCRIPT_DIR` and `REPO_ROOT`; `cd "$REPO_ROOT"` so relative
  paths (including `data/viz/*.json`) always resolve against the repo
  root regardless of cwd.
- Defaults: `HOST=127.0.0.1`, `PORT=8765`, `NO_BUILD=0`, `BUILD_ONLY=0`,
  `SYNTHETIC=0`, `VENV_PY="${VENV_PY:-.venv/bin/python}"`.
- Flag parsing via manual `while` loop on `$1`:
  - `--host HOST` / `--host=HOST`
  - `--port PORT` / `--port=PORT`
  - `--no-build` — skip the JSON build step (assume files already exist).
  - `--build-only` — build JSON files but do NOT launch the server.
  - `--synthetic` — force synthetic mode (also forces a rebuild so the
    output matches the expected synthetic dataset).
  - `--help | -h` — print usage then exit 0.
  - Unknown flag -> print error to stderr + usage -> exit 2.
- Build step (unless `--no-build`):
  - If `data/viz/sankey.json` missing OR `--synthetic` set: run
    `$VENV_PY scripts/viz/build_temporal_sankey_data.py [--synthetic]`.
  - If `data/viz/umap.json` missing OR `--synthetic` set: run
    `$VENV_PY scripts/viz/project_embeddings_umap.py [--synthetic 2000]`.
- If `--build-only`: log success, exit 0.
- Launch: `exec "$VENV_PY" -m uvicorn scix.viz.server:app --host "$HOST" --port "$PORT"`.
  - Using `python -m uvicorn` instead of `.venv/bin/uvicorn` makes the
    `VENV_PY` override consistent for both phases.

Usage string (must include `--port`, `--host`, `--no-build`):

```
Usage: scripts/viz/run.sh [--host HOST] [--port PORT] [--no-build]
                          [--build-only] [--synthetic] [-h|--help]

Options:
  --host HOST     Bind address (default: 127.0.0.1)
  --port PORT     Listen port (default: 8765)
  --no-build      Skip regenerating data/viz/*.json even if files are missing.
  --build-only    Regenerate data files then exit without starting the server.
  --synthetic     Force synthetic (no-DB) rebuild of both data files.
  -h, --help      Print this help and exit.

Env overrides:
  VENV_PY         Python interpreter (default: .venv/bin/python).
```

## `Makefile`

Tabs for recipe indent:

```
.PHONY: viz-demo viz-demo-build

viz-demo:
<TAB>./scripts/viz/run.sh

viz-demo-build:
<TAB>./scripts/viz/run.sh --build-only --synthetic
```

## `docs/viz/DEMO.md`

Sections:

- **Overview** — the narrative (agent-navigable knowledge / hybrid
  retrieval / three complementary views).
- **Quickstart** — `make viz-demo` or `./scripts/viz/run.sh`.
- **V2: Temporal community Sankey** — what it shows, three flows to
  point at during a talk.
- **V3: UMAP embedding browser** — zoom/hover/click interactions.
- **V4: Agent trace overlay** — three prepared scenarios with prompt
  template, expected tool sequence, observable behaviour each:
  1. Literature survey
  2. Related methods discovery
  3. Entity disambiguation
- **Tips** — port conflicts, regenerating data, known limitations.

## `tests/test_viz_demo_runner.py`

- `test_run_sh_executable` — `os.access` check.
- `test_run_sh_syntax` — `bash -n scripts/viz/run.sh`.
- `test_run_sh_help` — stdout contains `--port`, `--host`, `--no-build`.
- `test_build_only_synthetic_produces_json` (marked `integration`):
  - Locate repo root from `Path(__file__).resolve().parents[1]`.
  - Save current mtime of `data/viz/{sankey,umap}.json` if present;
    `unlink` them to force regeneration.
  - Run `./scripts/viz/run.sh --build-only --synthetic` with `cwd=repo`.
  - Assert both JSON files now exist and parse as JSON.
  - Assert exit code 0.
  - (No restore — the test leaves freshly-generated synthetic files
    behind; they are regenerable idempotent artifacts.)

## Commit

Single commit:

```
prd-build: unit-viz-demo-runner — One-command demo runner + Makefile + DEMO.md
```

Stage the executable bit on `scripts/viz/run.sh` with
`git update-index --chmod=+x`.

## Verification matrix

| Criterion | Verification |
|---|---|
| run.sh exists + executable | `ls -l`, test_run_sh_executable |
| `bash -n` syntax ok | test_run_sh_syntax |
| `--help` shows flags | test_run_sh_help |
| `--build-only --synthetic` produces JSON | test_build_only_synthetic_produces_json |
| Makefile `viz-demo` target | `make -n viz-demo` |
| DEMO.md exists with 3 views + 3 scenarios | manual |
| Regression suite passes | pytest on listed tests |
