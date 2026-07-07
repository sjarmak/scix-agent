---
name: scix-build-test-ci
description: >
  Build, test, and CI runbook for SciX Experiments: recreate the venv and the
  optional-extras matrix (dev/qdrant/graph/mcp/embed/ner_pass/viz/...), run
  make check vs make check-ci, understand the "not integration and not
  network" marker filter, the optional-dependency collection guards in
  tests/conftest.py, the ci/scix_test_schema.sql snapshot the CI Postgres
  loads, the MCP contract-drift test, and the pinned black/ruff formatters.
  Load when setting up a fresh checkout, when CI is red and local is green (or
  vice versa), when tests skip silently or whole test modules vanish, when
  pre-commit or fmt-check fails, or when test_mcp_contract_conformance fails.
  NOT for prod-DSN/write-test safety semantics — use scix-db-safety-and-telemetry.
  NOT for changing the MCP tool surface itself — use scix-mcp-tool-surface.
  NOT for running heavy jobs on this host — use scix-memory-and-batch-discipline.
  NOT for what the project is — use scix-orientation.
---

# SciX build, test, and CI

Get a fresh checkout green **the same way CI is green**. Everything below was
verified by source-reading this repo at commit `452ab86` (branch
`bd/0yp5-external-copy-accuracy-audit`, an ancestor of `origin/main`'s tip
`e59d89d` and **5 commits behind** it as of 2026-07-07 — those 5 commits add
a LikeC4 `architecture/` model + a `likec4-pages.yml` Pages workflow, so on
current `origin/main` `check.yml` is no longer the only workflow; re-verify
with `git log --oneline HEAD..origin/main`). The working tree also carries
UNCOMMITTED in-flight material (the s7cy
embed-pipeline remediation); this skill teaches **committed HEAD**, and marks
in-flight divergences explicitly. PROVISIONAL pending Stephanie (discovery Q2):
none of the uncommitted material is canonized here.

**When NOT to use this skill:**

| You actually want                                              | Go to                              |
| -------------------------------------------------------------- | ---------------------------------- |
| DSN guards, `is_production_dsn`, why write tests skip silently | `scix-db-safety-and-telemetry`     |
| Adding/renaming an MCP tool, the 15-tool cap rationale         | `scix-mcp-tool-surface`            |
| Running anything multi-GB or >1 min on this host               | `scix-memory-and-batch-discipline` |
| What SciX is, repo layout, doc-drift map                       | `scix-orientation`                 |
| What counts as evidence, gold sets, eval harness               | `scix-eval-and-evidence`           |
| How changes are gated (ADRs, HALT-branch-ready)                | `scix-change-control`              |

---

## 1. Recreate the environment

Prerequisites (README, verified 2026-07-07): Python **3.11+** (CI runs 3.11;
this installation's `.venv` symlinks to system Python **3.12.3**), PostgreSQL 16
with pgvector for DB-backed tests, GPU only for the embedding pipeline.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'                    # baseline: pytest + pinned formatters
pip install -e '.[dev,qdrant,graph,mcp]'   # the exact set CI installs
```

### The extras matrix (pyproject.toml, committed HEAD)

| Extra      | Pulls in                                                                                     | Unlocks                                                                                          | Heavy?          |
| ---------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | --------------- |
| `dev`      | pytest, **black==25.1.0**, **ruff==0.15.8**, libcst, hypothesis, pre-commit, pandas, pyarrow | the test runner + formatters; PWC-harvester parser tests                                         | no              |
| `qdrant`   | **qdrant-client==1.17.1** (exact pin), fastembed, pgvector                                   | `test_qdrant_contract.py`, `test_backfill_qdrant_filter_fields.py`, `test_qdrant_outbox_sync.py` | no              |
| `graph`    | python-igraph, leidenalg                                                                     | `test_graph_experiment.py`, `test_recompute_citation_communities.py`                             | no              |
| `mcp`      | mcp>=1.2,<2.0                                                                                | the MCP server + contract tests                                                                  | no              |
| `embed`    | transformers>=4.36, torch>=2.1                                                               | `test_somd_detect.py`, `test_chunk_pass_embedder.py`                                             | **yes (torch)** |
| `ner_pass` | gliner>=0.2.26, **transformers>=5.0, torch>=2.10**                                           | GLiNER dbl.3 NER pass                                                                            | **yes**         |
| `ner_eval` | seqeval, datasets, transformers, torch                                                       | NER evaluation                                                                                   | **yes**         |
| `viz`      | fastapi, uvicorn, orjson                                                                     | the 12 viz/frontend test modules                                                                 | no              |
| `search`   | sentence-transformers                                                                        | sentence-transformer paths                                                                       | **yes**         |
| `analysis` | matplotlib                                                                                   | plotting                                                                                         | no              |
| `docling`  | docling>=2.0                                                                                 | document conversion                                                                              | **yes**         |

Traps:

- **`ner_pass` upgrades the whole venv.** It requires transformers>=5.0 and
  torch>=2.10 while `embed` only floors at 4.36/2.1 — installing `ner_pass`
  moves shared deps forward for everything (the pyproject comment records this
  happened on first install). Don't add it to a venv you need stable without
  checking what resolves.
- **`qdrant-client` is an exact pin (1.17.1)**, per PRD MH-16 + ADR-013
  (REST-only transport; gRPC 1.17.x fails on 1.18 responses). Bump procedure is
  written next to the pin: re-run `tests/test_qdrant_contract.py` against the
  candidate, then update the pin AND `PINNED_VERSION` in that test together.
- Committed HEAD has **no `[project.scripts]` entry** — there is no `scix`
  console command at HEAD. A `scix = "scix.cli:main"` entry plus
  `src/scix/cli.py` exist only as uncommitted working-tree changes (in-flight;
  PROVISIONAL pending Stephanie, discovery Q2). Do not teach or depend on it.
- Installing torch-pulling extras and any heavy test run on this installation
  must respect the host memory discipline — see
  `scix-memory-and-batch-discipline` before running anything that loads a model.

### Environment variables

```bash
cp .env.example .env       # ADS_API_KEY etc.; .env is gitignored
export SCIX_TEST_DSN="dbname=scix_test"   # REQUIRED before pytest — see §4
```

---

## 2. `make check` vs `make check-ci`

Both live in the top-level `Makefile`. **They are not the same gate; a branch
can pass one and fail the other.**

|             | `make check`                                                           | `make check-ci`                                                               |
| ----------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Lint        | `ruff check src/ scripts/ tests/` (whole tree)                         | `lint-changed`: ruff on files changed vs `BASE` only                          |
| Format      | `black --check` whole tree                                             | `fmt-check-changed`: black on changed files only                              |
| Tests       | `pytest -q $(PYTEST_ARGS)` — **all markers** unless you set the filter | same target, but CI sets `PYTEST_ARGS='-m "not integration and not network"'` |
| Who runs it | eventual full-tree target                                              | the actual CI gate (`.github/workflows/check.yml`)                            |

Why they differ: the tree carries **pre-existing black/ruff format debt**.
The CI gate lints/format-checks only files changed vs `BASE`
(`git diff --name-only --diff-filter=ACMR $(BASE)...HEAD -- '*.py'`,
default `BASE ?= origin/main`) so debt is paid down as files are touched —
the same philosophy as the pre-commit hooks, which run on staged files only.
Consequences:

- `make check` can fail on formatting **you never touched**. That is tree
  debt, not your bug. CI will not fail on it.
- `make check-ci` on a stale local clone diffs against a stale `origin/main`.
  Run `git fetch origin main` first (CI checks out with `fetch-depth: 0` for
  exactly this reason).
- In CI, `BASE` is `origin/<base_ref>` on PRs and `origin/<default_branch>` on
  pushes.

### Reproduce CI locally (copy-paste)

```bash
git fetch origin main
export SCIX_TEST_DSN="dbname=scix_test"
export SCIX_DSN="dbname=scix_test"     # CI sets BOTH — never let a test fall back to prod
PYTEST_ARGS='-m "not integration and not network"' make check-ci
```

**Do not run `make check` (unfiltered) casually on this installation.** With
no marker filter it runs the `integration`-marked tests, and any test that
resolves the default DSN with `SCIX_DSN`/`SCIX_TEST_DSN` unset points at the
production `scix` database (32M live papers). Full DSN semantics:
`scix-db-safety-and-telemetry`.

---

## 3. The marker filter

Markers are registered in `pyproject.toml` `[tool.pytest.ini_options]`:

| Marker        | Meaning                                              | In CI?     |
| ------------- | ---------------------------------------------------- | ---------- |
| `integration` | needs a running scix database (live corpus for many) | deselected |
| `network`     | needs outbound network                               | deselected |
| `unit`        | fast, dependency-free                                | runs       |

Counts as of 2026-07-07 (they drift — recount, don't trust): 268 test files in
`tests/`; 104 files reference `pytest.mark.integration`, 1 references
`pytest.mark.network`, 6 reference `pytest.mark.unit`. Unmarked tests run in
CI too — the filter only _removes_ the two marked classes; it is not an
allowlist. `pythonpath = ["src", "tests", "scripts"]` and
`testpaths = ["tests"]` come from the same section.

---

## 4. DSN discipline for tests (summary — sibling owns the detail)

- `SCIX_DSN` unset → `dbname=scix` = **production** (`src/scix/db.py`,
  `DEFAULT_DSN`).
- Destructive/write tests guard themselves and **skip silently** when
  `SCIX_TEST_DSN` is unset or the DSN resolves to prod (pattern:
  `_skip_destructive` in `tests/test_db.py`). A green `pytest tests/` without
  `SCIX_TEST_DSN` means the destructive suite never ran — not that it passed.
- `is_production_dsn(None)` returns `False` — resolve the `DEFAULT_DSN`
  fallback before calling it, or an unset DSN slips the guard.

Everything deeper (guard internals, `--allow-prod`, telemetry):
`scix-db-safety-and-telemetry`.

---

## 5. Optional-dependency collection guards

`tests/conftest.py` skips **collecting whole test modules** when their
optional import is absent, so `ModuleNotFoundError` at import time never fails
the run. The map is `_OPTIONAL_DEP_MODULES` (dep → files it gates), checked
via `importlib.util.find_spec`; a `warnings.warn` names every skipped module.
As of 2026-07-07:

| Missing module                                  | Test modules skipped                                                             |
| ----------------------------------------------- | -------------------------------------------------------------------------------- |
| `fastapi`                                       | 12 (all viz/frontend: `test_viz_*`, `test_*_frontend`, `test_trace_stream`, ...) |
| `qdrant_client`                                 | 2 (`test_backfill_qdrant_filter_fields`, `test_qdrant_contract`)                 |
| `torch`                                         | 2 (`test_somd_detect`, `test_chunk_pass_embedder`)                               |
| `defusedxml`, `pgvector`, `igraph`, `leidenalg` | 1 each                                                                           |

Two consequences:

1. **A dependency-free sandbox silently loses whole test modules.** If you are
   "testing a viz change" without `pip install -e '.[viz]'`, your tests are
   not running. Check the collection warnings, or count:
   `pytest --collect-only -q 2>/dev/null | tail -1`.
2. CI installs `dev,qdrant,graph,mcp` and NOT the heavy set, so torch/fastapi
   modules self-skip **by design** (bead o835 "CI green-up"). Local
   disagreement with CI on collected-test count usually means an extras
   mismatch, not a bug.

Modules where only _some_ tests need the optional dep guard in-file with
`pytest.importorskip` instead (e.g. `test_project_embeddings_umap.py`), so
their dependency-free tests still run in CI. When adding a test with an
optional import: gate the whole module in `_OPTIONAL_DEP_MODULES` if every
test needs it, otherwise `importorskip` in-file.

---

## 6. `ci/scix_test_schema.sql` — the CI database

CI (`.github/workflows/check.yml`, the only workflow at this commit; note
`origin/main` also carries `likec4-pages.yml`, see the provenance header) runs a
`pgvector/pgvector:pg16` service container, database `scix_test`, and loads
the schema **once** from the snapshot:

```bash
psql "$SCIX_TEST_DSN" -v ON_ERROR_STOP=1 -f ci/scix_test_schema.sql
```

Facts about the snapshot (verified 2026-07-07):

- It is a `pg_dump` schema dump (header says PostgreSQL 16.14), 5457 lines.
  Schema only — CI has **no corpus data**; that is why data-dependent tests
  are marked `integration` and deselected.
- There is **no regeneration script in-repo**; the only reference to the file
  is the workflow itself. Migrations (`migrations/001`–`072`, append-only,
  **no auto-runner** — applied by hand) and the snapshot are updated
  independently. If your change adds schema a CI-visible test needs, you must
  refresh the snapshot (schema-only `pg_dump` of a fully-migrated `scix_test`)
  in the same commit, or CI fails on missing relations.
- The snapshot currently still contains `paper_embeddings` and
  `embedding_outbox` — objects that were dropped out-of-process from
  production (bead s7cy) — and does **not** contain the `indus_qdrant_synced`
  watermark table (migration 072 is uncommitted). This is consistent:
  the snapshot matches **committed HEAD's** code expectations, which is what
  CI tests. The prod/CI schema divergence is in-flight material — PROVISIONAL
  pending Stephanie (discovery Q2); do not "fix" the snapshot toward the
  uncommitted state.

To build a local `scix_test` mirror of CI —
**do not run casually on this installation**: the local Postgres instance is
shared with the production `scix` database; per project CLAUDE.md a
fully-migrated `scix_test` already exists here (claim not re-verified this
session — verifying requires a DB connection):

```bash
# shown with its guard: only on a machine/instance you own, never against prod
createdb scix_test
psql "dbname=scix_test" -v ON_ERROR_STOP=1 -f ci/scix_test_schema.sql
```

(`scripts/setup_db.sh` is the **prod** bootstrap — it creates/targets
`scix` from root-level `schema.sql`, not the test DB. Different tool,
different blast radius.)

---

## 7. The MCP contract test (the drift gate)

`tests/test_mcp_contract_conformance.py` is DB-less and model-less, so it
**always runs in CI**. Two layers:

1. **Surface conformance** — the agent-visible tool count stays within
   `VISIBLE_TOOL_CAP` (15, imported from `scix.mcp_server`), every tool has a
   valid JSON-schema `inputSchema`, and every error code is in the closed
   catalog (`scix.mcp_errors.CATALOG`; the scan follows the handlers into
   `src/scix/mcp_handlers/`).
2. **Published-artifact drift** — `contract/scix_mcp_v1.json` must equal the
   live `build_contract()` output byte-for-byte.

If it fails after you touched the tool surface **intentionally**:

```bash
python scripts/gen_mcp_contract.py    # rewrites contract/scix_mcp_v1.json
```

then commit the regenerated artifact with the change. A **breaking** change
bumps `CONTRACT_VERSION` in `scix.mcp_contract` (currently "1") so a new
`scix_mcp_v2.json` is published alongside the old one. If it fails and you did
NOT intend a surface change, you drifted — do not regenerate to make it green.
A 16th visible tool fails at **import time** (`RuntimeError` in
`mcp_server`), before any test runs. The whole add/consolidate-a-tool
checklist and the cap's rationale: `scix-mcp-tool-surface`.

---

## 8. Pinned formatters

`black==25.1.0` and `ruff==0.15.8` are **exact pins** in the `dev` extra, and
`.pre-commit-config.yaml` mirrors them (`v0.15.8` / `25.1.0`). Rationale
(recorded in pyproject): unpinned, CI installs the latest formatter and
`fmt-check` fails on version-drift restyling of the whole tree — noise, not
debt. **Bump the pyproject pins and the pre-commit revs together.**

Config (both `[tool.black]` and `[tool.ruff]`): `line-length = 100`. Ruff
selects `E,F,I,W` and ignores `E501` (black owns line length). Per-file
ignores: `E402` is allowed under `scripts/` and `tests/` (imports after
`sys.path` setup / `pytest.importorskip` are intentional there); `src/` keeps
`E402` enforced.

```bash
pip install pre-commit && pre-commit install   # hooks run on staged files only
make fmt                                       # auto-fix: ruff --fix + black
```

Pre-commit also runs `check-added-large-files --maxkb=10000` — the corpus
guard (ADS JSONL shards must never be committed).

---

## 9. Symptom → cause table

| Symptom                                                   | Cause                                                                  | Fix                                                                                           |
| --------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Green locally, CI red on lint/format                      | you ran `make check-ci` against a stale `origin/main`, or never ran it | `git fetch origin main`, rerun §2 parity block                                                |
| Red locally on files you never touched                    | full-tree `make check` hitting pre-existing format debt                | use `make check-ci`; debt is paid as files are touched                                        |
| `pytest` green but suspiciously fast / write tests absent | `SCIX_TEST_DSN` unset → destructive suite skipped **silently**         | `export SCIX_TEST_DSN="dbname=scix_test"`; see §4                                             |
| Whole test modules missing from collection                | optional dep absent → `conftest.py` collect_ignore                     | install the matching extra (§5 table); read the collection warnings                           |
| `test_mcp_contract_conformance` fails                     | tool surface changed without regenerating the artifact                 | intentional → `python scripts/gen_mcp_contract.py` + commit; unintentional → revert the drift |
| MCP server won't even import (`RuntimeError`)             | 16th visible tool                                                      | remove/consolidate; `scix-mcp-tool-surface`                                                   |
| CI red on missing relation/column                         | migration landed without refreshing `ci/scix_test_schema.sql`          | refresh snapshot in the same commit (§6)                                                      |
| `fmt-check` fails only in CI after a formatter bump       | pyproject pins and `.pre-commit-config.yaml` revs diverged             | bump both together (§8)                                                                       |
| Local test run OOM-kills other sessions on this host      | heavy extras + default cgroup                                          | `scix-memory-and-batch-discipline`                                                            |

---

## Provenance and maintenance

Authored 2026-07-07 against commit `452ab86` on branch
`bd/0yp5-external-copy-accuracy-audit` (an ancestor of `origin/main` `e59d89d`,
5 commits behind at authoring time; note: **not** authored from a checkout
named `main`, and not tip-of-`origin/main`). All
claims verified by source-reading only; no test suite, DB connection, or
install was executed. Re-verify before trusting drift-prone facts:

```bash
git rev-parse --short HEAD && git branch --show-current       # is this still 452ab86-era?
sed -n '1,70p' Makefile                                        # check vs check-ci targets, BASE default
grep -n 'PYTEST_ARGS\|pip install\|BASE=' .github/workflows/check.yml   # CI marker filter + extras set
grep -n 'black==\|ruff==\|qdrant-client==' pyproject.toml      # formatter + client pins
grep -n 'rev:' .pre-commit-config.yaml                         # pre-commit mirrors the pins?
grep -n 'markers' -A4 pyproject.toml                           # registered markers
sed -n '27,53p' tests/conftest.py                              # _OPTIONAL_DEP_MODULES map
ls migrations/*.sql | wc -l                                    # migration count (72 at authoring)
wc -l ci/scix_test_schema.sql                                  # snapshot size (5457 at authoring)
grep -c 'paper_embeddings' ci/scix_test_schema.sql             # snapshot still pre-s7cy? (>0 at authoring)
git show HEAD:pyproject.toml | grep -c 'project.scripts'       # 0 at authoring — scix CLI still uncommitted?
ls tests/*.py | wc -l                                          # test-file count (268 at authoring)
```
