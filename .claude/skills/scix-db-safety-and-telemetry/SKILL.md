---
name: scix-db-safety-and-telemetry
description: >
  Production-database protection and query_log telemetry for SciX. Load this
  BEFORE any command that opens a PostgreSQL connection in this repo: running
  pytest, setting SCIX_DSN / SCIX_TEST_DSN, calling is_production_dsn, passing
  --allow-prod to a script, wiping/seeding test tables, or analysing the
  query_log table (success/error_msg semantics, failure rates, guard-block
  counts). Triggers: "which database am I pointed at", "tests all passed but
  wrote nothing", "SCIX_TEST_DSN", "--allow-prod refused", "query_log says
  success", "failure rate by tool". NOT for memory/cgroup sizing or the
  scix-batch wrapper itself — use scix-memory-and-batch-discipline. NOT for
  schema-migration or index-build discipline — use
  scix-index-and-storage-discipline. NOT for what gets ADR-gated — use
  scix-change-control.
---

# SciX DB safety and telemetry

**The single most important fact in this repo: with no environment variables
set, every connection goes to the production database.** `scix` is the live
32M-paper corpus, co-hosted on this machine. There is no staging tier between
"your shell" and "production" except the guards documented here — and each
guard has a documented hole. Read the traps before you touch a DSN.

Jargon, defined once:

- **DSN** — libpq connection string, either key=value form (`dbname=scix_test
host=localhost`) or URI form (`postgresql://user@host/scix_test`).
- **prod scix** — the local PostgreSQL database named `scix` (the 32M-paper
  corpus). "Production" in this repo means _that dbname_, nothing else.
- **guard** — a code check that refuses/skips an operation when the DSN looks
  like production.
- **query_log** — the PostgreSQL table where every MCP tool call is recorded
  (telemetry, not application data).

When NOT to use this skill: sizing heavy jobs or invoking the `scix-batch`
cgroup wrapper (→ `scix-memory-and-batch-discipline`); deciding whether a DB
change needs an ADR (→ `scix-change-control`); index builds and migrations
(→ `scix-index-and-storage-discipline`); MCP tool-surface changes
(→ `scix-mcp-tool-surface`).

---

## 1. How the DSN resolves

Source of truth: `src/scix/db.py` (verified 2026-07-07).

```python
DEFAULT_DSN = os.environ.get("SCIX_DSN", "dbname=scix")   # db.py:16 — read ONCE at import
_PRODUCTION_DB_NAMES = frozenset({"scix"})                # db.py:18
```

| Context                                       | DSN used                                         | Notes                                                                                         |
| --------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| Library code (`get_connection()` with no arg) | `SCIX_DSN`, else `dbname=scix`                   | **Default = prod.**                                                                           |
| MCP server pool (`mcp_server._get_pool`)      | `SCIX_DSN`, else `DEFAULT_DSN`                   | `SCIX_TEST_DSN` does **NOT** redirect the server — it only flips the `is_test` log flag (§5). |
| Destructive tests                             | `SCIX_TEST_DSN` only                             | Unset → tests skip (§3).                                                                      |
| Read-only tests (`tests/helpers.py DSN`)      | `SCIX_DSN`, else `dbname=scix`                   | Read-only tests DO run against prod by default.                                               |
| `scix.query_log.log_query` (no explicit dsn)  | explicit `dsn` > `SCIX_TEST_DSN` > `DEFAULT_DSN` | The one module that _prefers_ the test DSN when set.                                          |
| `--dsn` flags on `scripts/*.py`               | `default=DEFAULT_DSN`                            | So the argparse default is prod too.                                                          |

Because `DEFAULT_DSN` is evaluated at import time, `export SCIX_DSN=...` after
a Python process (or the MCP server) has started has no effect on it.

Safe-by-default shell for any test/dev session in this repo:

```bash
export SCIX_TEST_DSN="dbname=scix_test"
export SCIX_DSN="dbname=scix_test"     # belt AND suspenders — CI does exactly this
```

CI parity: `.github/workflows/check.yml` sets **both** `SCIX_TEST_DSN` and
`SCIX_DSN` to the `scix_test` service database and loads the schema from
`ci/scix_test_schema.sql`. The local `scix_test` database has the full schema
(all migrations applied), no data. Setting only `SCIX_TEST_DSN` enables the
write tests but leaves read-only tests and any library default pointed at prod.

---

## 2. `is_production_dsn` — semantics and the None trap

`src/scix/db.py::is_production_dsn(dsn)` parses the DSN with libpq
(`psycopg.conninfo.conninfo_to_dict`, so key=value AND URI forms both work) and
returns True iff `dbname` is `scix` (case-insensitive).

**Three ways it silently answers False for a connection that will hit prod:**

1. **`None`/empty input returns False.** The docstring says it plainly: callers
   must resolve the effective DSN _before_ calling. This is wrong:

   ```python
   is_production_dsn(os.environ.get("SCIX_DSN"))   # env unset -> None -> False
   psycopg.connect()                                # ...but libpq/DEFAULT still hits prod
   ```

   This is correct:

   ```python
   from scix.db import DEFAULT_DSN, is_production_dsn
   is_production_dsn(dsn or DEFAULT_DSN)
   ```

2. **Malformed DSNs return False** (`psycopg.ProgrammingError` is caught). A
   typo'd DSN passes the guard, then the actual connect fails — usually
   harmless, but never treat `is_production_dsn(...) == False` as proof of
   safety on its own.

3. **Only the dbname `scix` counts as production.** `scix_test`, `postgres`,
   or any other database on the same host is "non-production" to this guard.
   The guard protects the corpus, not the host.

Check what a shell will resolve to (read-only, safe):

```bash
python -c "from scix.db import DEFAULT_DSN, is_production_dsn; \
print(repr(DEFAULT_DSN), is_production_dsn(DEFAULT_DSN))"
```

---

## 3. Write-test skip semantics — green is not proof

Destructive/write tests across the suite (72 test files reference
`SCIX_TEST_DSN`, counted 2026-07-07) follow this pattern
(`tests/helpers.py::get_test_dsn`, `tests/test_db.py`, `tests/test_ingest.py`,
`tests/test_migrations.py`, ...):

- `SCIX_TEST_DSN` unset → **skip** (reason string names the variable).
- `SCIX_TEST_DSN` set but pointing at prod (`dbname=scix`) → **skip**
  (helpers `get_test_dsn()` returns None).
- Otherwise → run destructively against that DSN (table wipes, index drops).

**The trap:** `pytest tests/` with no env exits green while the entire
destructive suite was deselected. A green run is NOT evidence that write paths
work. Prove which lane you ran:

```bash
# How many tests skipped for missing SCIX_TEST_DSN? (should be 0 in a real write run)
pytest tests/ -q -rs -m "not integration and not network" 2>&1 | grep -c "SCIX_TEST_DSN"
```

(That command executes the test suite — unit lanes are light, but on this
co-hosted installation run anything heavier under the `scix-batch` wrapper;
see `scix-memory-and-batch-discipline`.)

Known residual gap (source-verified 2026-07-07): `tests/test_db.py` and
`tests/test_ingest.py` each carry a **local** `_is_production_dsn` that parses
only key=value tokens. A URI-form DSN (`postgresql://user@host/scix`) contains
no `dbname=` token, so those two files' guards would NOT recognize it as
production — the exact bypass class `tests/helpers.py`'s header comment says
was centralized away on 2026-04-13, surviving in these two local copies.
Defensive rule until that gets a bead: **only ever set `SCIX_TEST_DSN` in
key=value form, and never to anything but a dedicated test database.**

---

## 4. `--allow-prod` and its enforcement markers

Prod-writing scripts refuse production by default. The base pattern
(verified in `scripts/refresh_fusion_mv.py`, `scripts/qdrant_outbox_sync.py`,
`scripts/run_ner_bodies.py`; 27 committed scripts under `scripts/` define the
flag as of 2026-07-07):

```python
if is_production_dsn(args.dsn) and not args.allow_prod:
    # log "Refusing to run against production DSN ... pass --allow-prod" and exit
```

`args.dsn` defaults to `DEFAULT_DSN` (already resolved), so the §2 None trap
does not fire inside these scripts — it fires when _you_ hand-roll a guard.

**Two different self-enforcement markers exist. Do not conflate them.** This
table is reproduced here because `--allow-prod` is a prod-DB-protection concern
(this skill's domain); the canonical home for the batch-scope guard mechanics
(`--require-batch-scope` / `SYSTEMD_SCOPE`, the not-auto-set trap, `scix-batch`
sizing) is `scix-memory-and-batch-discipline` — go there when the question is
"why did my heavy job's scope guard fire", come here when it is "will this
touch prod".

| Guard                                 | Env marker checked | Who sets the marker                                                                         | Scripts (2026-07-07)                                                                                                                                                                                                                                                                      |
| ------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--allow-prod` + systemd-scope check  | `INVOCATION_ID`    | `systemd-run --scope` sets it automatically (the `scix-batch` wrapper on this installation) | `populate_papers_fulltext.py`, `recompute_citation_communities.py`, `backfill_part_of_inheritance.py`, `extract_citation_contexts.py`, `run_lafia_pass.py` (skipped on `--dry-run`); `coldtext_swap_papers_fulltext.py` requires it unconditionally; `seal_fulltext_to_nas.py` only warns |
| `--require-batch-scope` (opt-in flag) | `SYSTEMD_SCOPE`    | **Not** set by `systemd-run` — must be exported by the caller/wrapper                       | `audit_paper_embeddings.py`, `run_chunk_pass.py`, `run_negative_results.py`, `run_ner_pass.py`                                                                                                                                                                                            |

`scripts/populate_papers_fulltext.py:264` states the discriminating fact:
_"`systemd-run --scope` sets `INVOCATION_ID` but not `SYSTEMD_SCOPE`."_
Consequently `--require-batch-scope` only passes if the invoking environment
exports `SYSTEMD_SCOPE` itself. Note the repo `CLAUDE.md` "Do" section says
`--allow-prod` self-checks `SYSTEMD_SCOPE`; the source says `INVOCATION_ID`.
Trust the source lines cited above.

**The `--allow-prod` guard is NOT universal.** `refresh_fusion_mv.py` and most
of the 27 check only the DSN, not the systemd scope. Before running any prod
script, read its `main()` and confirm which guards it actually has:

```bash
grep -n "allow_prod\|INVOCATION_ID\|SYSTEMD_SCOPE" scripts/<script>.py
```

Canonical prod invocation shape on this installation (**do not run casually —
prod write; PROVISIONAL pending Stephanie, Q5: treat every prod-DB write as
HALT-branch-ready requiring operator sign-off**):

```bash
scix-batch python scripts/populate_papers_fulltext.py --allow-prod ...
```

In-flight note (PROVISIONAL pending Stephanie, Q2): the working tree may carry
an untracked `scripts/seed_indus_qdrant_synced.py` plus modified
`scripts/embed.py` / `scripts/daily_sync.sh` — the un-landed s7cy dense-ingest
remediation. It follows the same `--allow-prod` pattern but is **not committed
reality**; do not treat it as the standard path (see
`scix-embedding-pipeline`).

---

## 5. query_log telemetry — the `success=TRUE` trap

Schema (migration `016_query_log.sql` base + `031_query_log.sql`
instrumentation columns; snapshot in `ci/scix_test_schema.sql`):

```
id, tool_name (NOT NULL), params_json, latency_ms, success (NOT NULL),
error_msg, created_at, ts, tool, query, result_count, session_id, is_test
```

There is **no result-payload column** — only the columns above survive.

Two writers, verified 2026-07-07:

1. **`mcp_server.call_tool` → `mcp_runtime._log_query`** (one row per MCP tool
   call). `success` starts `True` and flips to `False` **only if
   `_dispatch_tool` raises** (`src/scix/mcp_server.py:917-926`). Every
   _structured_ error — guard blocks, validation failures, `qdrant_disabled`,
   any `{"error": ..., "error_code": ...}` envelope returned as JSON — logs
   `success=TRUE`, because the function returned instead of raising.
2. **`scix.query_log.log_query`** hardcodes `success=TRUE` in its INSERT
   (`src/scix/query_log.py:79`).

One partial rescue exists: when the `search` tool's unscoped-broad-query guard
blocks a request, `_log_query` lifts the payload's `unscoped_broad_blocked`
flag into `error_msg='unscoped_broad_query'`
(`src/scix/mcp_runtime.py:293-294`, tag from the closed catalog in
`src/scix/mcp_errors.py`). That row is `success=TRUE, error_msg NOT NULL`.

### The predicates

```sql
-- WRONG: returns only raised exceptions; silently drops every guard-blocked
-- request (they are success=TRUE):
SELECT count(*) FROM query_log WHERE success = FALSE AND error_msg IS NOT NULL;

-- CORRECT problem-row predicate:
SELECT count(*) FROM query_log WHERE success = FALSE OR error_msg IS NOT NULL;

-- Guard-block rate specifically:
SELECT count(*) FROM query_log WHERE error_msg = 'unscoped_broad_query';
```

(Read-only SELECTs; against prod, run them in a session you have verified is
read-only, and keep anything long-running under the batch wrapper.)

### The residual blind spot

Structured errors **other than** the unscoped-broad block get no `error_msg`
lift. They land as `success=TRUE, error_msg NULL` — indistinguishable in
`query_log` from a genuine success except (weakly) by `result_count=0`. No
predicate over this table can recover them; that failure class is measurable
only at the response layer. Do not claim "error rate" from `query_log` without
stating this boundary.

### Shipped analytics inherit the trap

`scripts/analyze_query_log.py::failure_rate_by_tool` filters
`WHERE NOT success` — its "failure rate" counts raised exceptions only,
excluding guard blocks and all structured errors. Treat its output as an
exception rate, not an error rate.

### `is_test` semantics

The MCP server stamps `is_test` on every row from
`_is_test_session = bool(os.environ.get("SCIX_TEST_DSN"))`
(`src/scix/mcp_server.py:147`) — the mere _presence_ of `SCIX_TEST_DSN` in the
server's environment, independent of the DSN the pool actually connects with
(`SCIX_DSN`/default, §1). A server started with `SCIX_TEST_DSN` set but
`SCIX_DSN` unset writes `is_test=TRUE` telemetry **into prod**. When filtering
prod telemetry, `WHERE NOT is_test` is therefore approximately right; when
generating traffic, unset `SCIX_TEST_DSN` for the server process if you intend
the rows to count as real.

---

## 6. Pre-flight checklist (any DB-touching command)

1. `echo "${SCIX_DSN:-<unset -> dbname=scix PROD>}"` — know your default lane.
2. Test/dev work: `export SCIX_TEST_DSN="dbname=scix_test"` and
   `export SCIX_DSN="dbname=scix_test"` (CI parity, §1).
3. After a "green" pytest run, count `SCIX_TEST_DSN` skips (§3) before
   claiming write coverage.
4. Hand-rolled guard? Resolve the DSN first: `is_production_dsn(dsn or
DEFAULT_DSN)` — never pass a raw `os.environ.get` (§2).
5. Prod script? Read its `main()` for which guards it actually has, then
   invoke under the batch wrapper with `--allow-prod` (§4) — operator
   sign-off first (PROVISIONAL pending Stephanie, Q5).
6. `query_log` analysis? `success=FALSE OR error_msg IS NOT NULL`, state the
   structured-error blind spot, filter `is_test` knowingly (§5).

---

## Provenance and maintenance

Authored 2026-07-07 against branch `bd/0yp5-external-copy-accuracy-audit`
(HEAD `452ab86` — **not `main`**; the working tree additionally carried the
uncommitted s7cy remediation, PROVISIONAL Q2). Every cited file
(`src/scix/db.py`, `src/scix/mcp_server.py`, `src/scix/mcp_runtime.py`,
`src/scix/query_log.py`, `src/scix/mcp_errors.py`, `tests/helpers.py`,
`tests/test_db.py`, `tests/test_ingest.py`, `scripts/analyze_query_log.py`,
the four `--allow-prod` scripts quoted) was clean vs HEAD at authoring time,
so the content above is committed reality.

Re-verify (all read-only):

```bash
git rev-parse --short HEAD && git branch --show-current
grep -n "DEFAULT_DSN\|_PRODUCTION_DB_NAMES" src/scix/db.py | head -3
grep -n "if not dsn" src/scix/db.py                                    # None-returns-False still present?
grep -rln -- --allow-prod scripts/*.py | wc -l                          # was 27 committed (28 incl. untracked) on 2026-07-07
grep -rn "INVOCATION_ID" scripts/*.py | wc -l                           # systemd-scope-gated subset drifts
grep -n "success = True" src/scix/mcp_server.py                         # the trap's origin line
grep -n "unscoped_broad" src/scix/mcp_runtime.py | head -3              # error_msg lift still keyed on the flag
grep -n "VALUES (%s, TRUE" src/scix/query_log.py                        # log_query still hardcodes TRUE
grep -n "_is_test_session" src/scix/mcp_server.py | head -2             # is_test still keyed on SCIX_TEST_DSN presence
grep -c "SCIX_TEST_DSN" .github/workflows/check.yml                     # CI still sets both DSNs
grep -rln "SCIX_TEST_DSN" tests/*.py | wc -l                            # was 72 on 2026-07-07
grep -n "_is_production_dsn" tests/test_db.py tests/test_ingest.py      # local URI-gap copies still present?
```
