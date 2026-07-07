---
name: scix-memory-and-batch-discipline
description: >-
  Load BEFORE running any heavy or long-running command in this repo on the
  production host: embedding/ingest passes, NER (GLiNER) runs, chunk passes,
  index or graph recomputes, eval/fusion sweeps, backfills, big-table
  migrations, model training, or anything multi-GB or over ~1 minute. Covers
  the non-negotiable scix-batch wrapper, the systemd-oomd co-hosting hazard
  (an unwrapped job gets the gascity supervisor killed as collateral),
  MemoryHigh/MemoryMax sizing, the two self-enforcement guards
  (--require-batch-scope/SYSTEMD_SCOPE and --allow-prod/INVOCATION_ID), the
  SYSTEMD_SCOPE-is-not-auto-set trap, and what the scope does NOT bound
  (docker, Postgres, Qdrant). NOT for DSN/telemetry semantics
  (scix-db-safety-and-telemetry), index-build validation
  (scix-index-and-storage-discipline), what the embed scripts do
  (scix-embedding-pipeline), or test/CI invocation (scix-build-test-ci).
---

# SciX memory and batch discipline

Facts below verified 2026-07-07 against branch `bd/0yp5-external-copy-accuracy-audit`
@ `452ab86` (not `main`; every file cited here is byte-identical between this
HEAD and `origin/main` @ `e59d89d`, checked with `git diff --quiet e59d89d 452ab86 -- <file>`).

## The rule

**Every heavy job on this host runs under `scix-batch`. No exceptions.**

"Heavy" per the repo's own line (CLAUDE.md / AGENTS.md, "Memory isolation"):
**multi-GB resident memory OR longer than ~1 minute.** When unsure, wrap it;
the wrapper costs nothing on a well-behaved job.

Why this is the single highest-damage trap here: this machine is not a
dedicated SciX box. It co-hosts the **Gas City agent-fleet supervisor** (the
process that runs the mayor and every worker session) inside the same
`user@1000.service` slice as your shell. That slice has systemd-oomd
kill-on-pressure enabled:

```text
$ systemctl show user@1000.service -p ManagedOOMMemoryPressure,ManagedOOMMemoryPressureLimit
ManagedOOMMemoryPressure=kill
ManagedOOMMemoryPressureLimit=2147483648    # raw fraction of 2^32 = 50%
```

(Shipped by the distro drop-in
`/usr/lib/systemd/system/user@.service.d/10-oomd-user-service-defaults.conf`,
`ManagedOOMMemoryPressureLimit=50%`, systemd 255. Host fact, dated 2026-07-07.)

So: an unwrapped multi-GB script pushes the whole user slice past 50% memory
pressure, and oomd kills a _descendant cgroup of its choosing_, not
necessarily yours. In practice it frequently picks the gascity supervisor,
taking down the mayor and every worker session at once. Your job survives;
everyone else dies. `scix-batch` puts your job in its own bounded transient
cgroup so runaway growth OOMs (or throttles) **inside the scope** instead of
pressuring the shared slice.

## Jargon, defined once

| Term                         | Meaning here                                                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| cgroup                       | Linux control group; the unit systemd uses to account and bound memory per process tree.                                                                           |
| systemd-oomd                 | Userspace OOM killer; watches memory _pressure_ (PSI) per cgroup and kills whole cgroups, pre-empting the kernel OOM killer.                                       |
| scope (transient)            | A systemd unit created on the fly around an already-running process tree (`systemd-run --scope`). Shows up in `systemctl --user`.                                  |
| `MemoryHigh`                 | Soft ceiling: above it the kernel throttles/reclaims the cgroup hard (job slows, does not die).                                                                    |
| `MemoryMax`                  | Hard ceiling: above it the job is OOM-killed _inside its own scope only_.                                                                                          |
| `ManagedOOMPreference=avoid` | Tells oomd to prefer other victims when the _parent_ slice is under pressure (a capped, well-behaved batch job should not be the casualty of someone else's leak). |
| `INVOCATION_ID`              | Env var systemd sets for processes it manages. `systemd-run --scope` sets it in the child; a plain interactive shell does not have it. Guard marker #2.            |
| `SYSTEMD_SCOPE`              | Env var some scripts check. **Nothing sets it automatically** (see the trap below); you export it yourself. Guard marker #1.                                       |

## The wrapper: `scix-batch` (now `mem-batch`)

Lives outside the repo at `~/.local/bin/mem-batch`, with a backward-compat
symlink `~/.local/bin/scix-batch -> mem-batch` (renamed 2026-07-07 because it
is useful beyond SciX; all repo prose still says `scix-batch`, and both names
work). It is a thin `exec` around:

```bash
systemd-run --user --scope \
    --unit="mem-batch-$(date +%s)-$$" \
    --description="mem-batch job: $*" \
    --expand-environment=no \
    --property="MemoryHigh=$MEM_HIGH"  \   # default 20G
    --property="MemoryMax=$MEM_MAX"    \   # default 30G
    --property="ManagedOOMMemoryPressure=kill" \
    --property="ManagedOOMPreference=avoid" \
    -- "$@"
```

Properties of the wrapper you rely on:

- **Defaults `MemoryHigh=20G`, `MemoryMax=30G`**; override per invocation with
  `--mem-high SIZE` / `--mem-max SIZE` (before the command, optionally
  followed by `--`).
- **Environment is inherited.** A `--scope` child keeps your shell's exported
  env, so `QDRANT_URL`, `SCIX_DSN`, `SCIX_TEST_DSN`, `SYSTEMD_SCOPE`, etc.
  pass through. (`--expand-environment=no` only means literal `${VAR}` text in
  arguments is not re-expanded by systemd-run.)
- **Synchronous**: `scix-batch CMD` returns when CMD exits, with its exit code.
- The unit name is printed at start; use it to inspect or stop the job.

## Invocation patterns (copy-pasteable)

Basic form, the default for anything heavy:

```bash
scix-batch python scripts/whatever.py --flags
```

Raise the ceiling for a known-large job (CLAUDE.md's own example):

```bash
scix-batch --mem-high 40G --mem-max 60G python scripts/big_job.py
```

Wrap a command that starts with `env` or a shell (use `--` to end flag
parsing):

```bash
scix-batch --mem-high 6G --mem-max 8G -- env FOO=bar bash run.sh
```

Script that self-enforces via `--require-batch-scope` (SYSTEMD_SCOPE is NOT
set for you; export it — see the trap below). Heavy NER pass; **do not run
casually**, it is a prod-DB GPU workload:

```bash
SYSTEMD_SCOPE=1 scix-batch python scripts/run_ner_pass.py --require-batch-scope ...
```

Prod-writing script guarded by `--allow-prod` + `INVOCATION_ID` (the wrapper
satisfies the INVOCATION_ID check by itself). **Do not run casually** — this
writes the production database:

```bash
scix-batch python scripts/populate_papers_fulltext.py --allow-prod ...
```

Cron / shell-script form: the repo convention is a `$SCIX_BATCH` variable with
a PATH fallback so CI (no wrapper installed) still runs, with a loud warning.
Committed example, `scripts/run_citation_contexts_shard.sh:39`:

```bash
SCIX_BATCH="${SCIX_BATCH:-scix-batch}"
if ! command -v "$SCIX_BATCH" >/dev/null 2>&1; then
    echo "WARN: $SCIX_BATCH not found on PATH — running without memory cgroup." >&2
    SCIX_BATCH=""
fi
$SCIX_BATCH "$PYTHON" scripts/extract_citation_contexts.py ...
```

Committed `scripts/daily_sync.sh:120` uses the same idiom
(`${SCIX_BATCH:-} $PYTHON scripts/refresh_v_claim_edges.py --allow-prod`).
Note: `daily_sync.sh` is currently _modified in the working tree_ as part of
the in-flight s7cy embedding remediation — treat the committed version as
reality and the working-tree diff as PROVISIONAL pending Stephanie (discovery
Q2); see scix-embedding-pipeline for that fire.

## The two self-enforcement guards

Scripts that can hurt the host or prod refuse to start unless they can see
evidence they are wrapped. There are **two different markers**, and they are
NOT interchangeable:

| Guard                                | Env var checked                 | Who sets the var                               | Scripts (committed, verified by grep 2026-07-07)                                                                                                                                                                                                 |
| ------------------------------------ | ------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--require-batch-scope` flag         | `SYSTEMD_SCOPE` (presence only) | **You** (`SYSTEMD_SCOPE=1 scix-batch ...`)     | `scripts/run_ner_pass.py:137`, `scripts/run_chunk_pass.py:129`, `scripts/run_negative_results.py:134`, `scripts/audit_paper_embeddings.py:461`                                                                                                   |
| `--allow-prod` scope check           | `INVOCATION_ID` (presence only) | **systemd-run** (automatic under `scix-batch`) | `scripts/populate_papers_fulltext.py:276`, `scripts/extract_citation_contexts.py:88`, `scripts/backfill_part_of_inheritance.py:656`, `scripts/recompute_citation_communities.py:446`, `scripts/run_lafia_pass.py:203` (waived under `--dry-run`) |
| `--allow-prod` scope check (outlier) | `SYSTEMD_SCOPE`                 | **You**                                        | `scripts/link_section_entities.py:475` checks SYSTEMD_SCOPE, not INVOCATION_ID — bare `scix-batch ... --allow-prod` fails here; prepend `SYSTEMD_SCOPE=1`                                                                                        |
| Unconditional scope check            | `INVOCATION_ID`, hard exit      | systemd-run                                    | `scripts/coldtext_swap_papers_fulltext.py:71`                                                                                                                                                                                                    |
| Unconditional scope check            | `INVOCATION_ID`, warn only      | systemd-run                                    | `scripts/seal_fulltext_to_nas.py:150`                                                                                                                                                                                                            |

(The `--allow-prod` DSN half of these guards — `is_production_dsn()`, its
empty-DSN `False` trap, `SCIX_TEST_DSN` — belongs to
**scix-db-safety-and-telemetry**. Here only the scope half matters: passing
`--allow-prod` _outside_ a systemd unit is refused, so prod writes are forced
into a memory-bounded cgroup.)

### The trap: SYSTEMD_SCOPE is not set automatically

`systemd-run --scope` sets `INVOCATION_ID` in the child environment but does
**not** set `SYSTEMD_SCOPE`. This is stated in-repo
(`scripts/populate_papers_fulltext.py:264`: "`systemd-run --scope` sets
`INVOCATION_ID` but _not_ `SYSTEMD_SCOPE`") and corroborated by the
`INVOCATION_ID=%02x...` setenv template visible in
`strings /usr/bin/systemd-run`. Not runtime-tested this session (read-only
campaign); if a `--require-batch-scope` run refuses under a bare wrapper
call, this is why.

Consequences:

1. `scix-batch python scripts/run_ner_pass.py --require-batch-scope` **fails**
   with `ERROR: --require-batch-scope set but SYSTEMD_SCOPE not in environment.`
   The working invocation is `SYSTEMD_SCOPE=1 scix-batch python ...` (the
   scope inherits your exported env). The guard checks presence, not value.
2. **Known doc drift**: CLAUDE.md / AGENTS.md's "Do" bullet ("The script
   self-checks `SYSTEMD_SCOPE` env (set automatically by `systemd-run
--scope`)") is wrong on the "automatically" part and names only one of the
   two markers. Trust the script sources cited above. AGENTS.md also lists a
   `compass-memory-isolation` compass and `docs/conventions/` playbooks that
   do not exist in the tree — this skill is the actual home for that content.
3. Yes, `SYSTEMD_SCOPE=1 python heavy.py` (no wrapper) would fool guard #1.
   Don't. The guard is a seatbelt reminder, not the protection; the cgroup is
   the protection.

## What the scope does NOT bound

`scix-batch` bounds **the launched process tree only**. Three co-resident
memory consumers live outside it:

| Consumer                  | Why it escapes                                                                                                                | How to bound it instead                                                                                                                                                                                                      |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rootful Docker containers | Containers run under the docker daemon's cgroup slice, not your scope (wrapper header calls this out explicitly).             | Per-container limits: `docker run --memory=...` or compose `deploy.resources.limits.memory`. Use scix-batch for the host-side driver process only.                                                                           |
| Postgres postmaster       | System service (`postgresql`), its own cgroup. A query you _send_ from a wrapped script does its heavy lifting in the server. | Bound server-side per session, the committed pattern in `scripts/eval_lexical_rank_flag.py:641`: `SET work_mem = '256MB'` and `SET max_parallel_workers_per_gather = 0` ("scix-batch's cgroup does NOT cap the postmaster"). |
| Qdrant server             | Serves the dense lane as its own long-running process/container (see `docs/runbooks/qdrant.md`), outside your scope.          | Qdrant-side config / container limits; see scix-vector-serving-qdrant.                                                                                                                                                       |

So a "wrapped" eval that fires huge parallel queries can still pressure the
host through Postgres. Wrapping is necessary, not sufficient: also cap
server-side work when your SQL is heavy.

## Sizing

- Defaults (20G high / 30G max) fit most passes. Committed examples of
  explicit sizing: `scripts/run_chunk_pass.py` docstring uses
  `--mem-high 16G --mem-max 24G`; CLAUDE.md shows `--mem-high 40G --mem-max 60G`
  for known-large jobs.
- Symptom of hitting `MemoryHigh`: the job slows to a crawl but lives
  (reclaim/throttle). Symptom of hitting `MemoryMax`: the job is OOM-killed
  inside its scope; the rest of the host is untouched. Both are the wrapper
  working as designed — resize deliberately, don't unwrap.
- Host RAM is shared with the agent fleet; do not casually push `--mem-max`
  toward total RAM. Check headroom first: `free -g`.

## Monitor, inspect, stop

The wrapper prints its unit name (`mem-batch-<epoch>-<pid>`) at start.

```bash
systemctl --user list-units 'mem-batch-*'          # live batch jobs
systemctl --user status mem-batch-<epoch>-<pid>.scope
systemd-cgtop --user -1 | head -20                 # who is using memory, one shot
systemctl --user stop mem-batch-<epoch>-<pid>.scope   # kill a runaway job cleanly
```

`scripts/check_batch_context.sh` in this skill's directory is a read-only
diagnostic: run it (optionally inside a wrapper) to see which guard markers
are present, your current cgroup, the host oomd settings, and live scopes.

## Failure modes, triage

| Symptom                                                                            | Cause                                                                       | Fix                                                                                                       |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Gascity supervisor / mayor / worker sessions die while your script runs            | You ran heavy work unwrapped; oomd killed a co-tenant at 50% slice pressure | Stop the job; rerun under `scix-batch`; report the kill (supervisor recovery is fleet ops, not this repo) |
| `ERROR: --require-batch-scope set but SYSTEMD_SCOPE not in environment.`           | Marker #1 is user-set; bare `scix-batch` doesn't set it                     | `SYSTEMD_SCOPE=1 scix-batch python ...`                                                                   |
| `Refusing to run --allow-prod outside a systemd scope. Invoke via: scix-batch ...` | No `INVOCATION_ID`: you ran it in a plain shell                             | Wrap in `scix-batch` (do not export INVOCATION_ID by hand)                                                |
| `link_section_entities.py --allow-prod` refuses even under scix-batch              | Outlier guard checks SYSTEMD_SCOPE                                          | `SYSTEMD_SCOPE=1 scix-batch python scripts/link_section_entities.py --allow-prod ...`                     |
| `WARN: scix-batch not found on PATH — running without memory cgroup.`              | Wrapper missing (CI sandbox, fresh host)                                    | Fine in CI; on the prod host, stop and install/point `$SCIX_BATCH` at `~/.local/bin/mem-batch`            |
| Job suddenly very slow, not dead                                                   | `MemoryHigh` throttling                                                     | Deliberate resize: `--mem-high/--mem-max`, after checking `free -g`                                       |
| Job killed, host fine, scope shows `oom-kill`                                      | `MemoryMax` ceiling hit                                                     | Same as above; the containment worked                                                                     |
| `mem-batch: unknown flag: ...`                                                     | Wrapper flags must precede the command                                      | `scix-batch [--mem-high S] [--mem-max S] [--] CMD ARGS...`                                                |

## When NOT to use this skill

- Sub-minute, sub-GB commands (`git`, `grep`, `ls`, `--help`, single quick
  queries, `pytest --collect-only`) need no wrapper.
- Deciding whether a command may touch the production DB at all, DSN
  resolution, `SCIX_TEST_DSN`, `is_production_dsn` semantics, query_log
  analysis → **scix-db-safety-and-telemetry**.
- Whether a heavy job is _allowed_ (ADR-pinned axes, operator-gated windows,
  what needs sign-off) → **scix-change-control**.
- What the embed/ingest scripts actually do, and the current broken/in-flight
  embed path → **scix-embedding-pipeline**.
- Index build validation ritual (50k scratch build, forced-index-scan smoke
  test) → **scix-index-and-storage-discipline**.
- Test/CI invocation and env recreation → **scix-build-test-ci**.

## Host-specificity note

The memory-isolation _mechanism_ (a bounded transient scope) is portable and
harmless anywhere. The _urgency_ is an operational requirement of this
installation: the production host co-runs the Gas City supervisor in the same
user slice, and the guards in `scripts/` assume the wrapper exists. On a
clone of this repo on a dedicated machine, the `--require-batch-scope` /
`--allow-prod` scope guards still demand a systemd scope; the honest options
there are to run under `systemd-run --user --scope` equivalently, or change
the guards through change control — never to hand-export `INVOCATION_ID`.

## Provenance and maintenance

Authored 2026-07-07 from branch `bd/0yp5-external-copy-accuracy-audit` @
`452ab86` (not main; all cited files verified identical to `origin/main` @
`e59d89d`). The wrapper (`~/.local/bin/mem-batch`) lives outside the repo and
was last modified 2026-07-07; it will drift independently of git.

Re-verify each volatile fact in one line:

```bash
ls -la ~/.local/bin/scix-batch ~/.local/bin/mem-batch && sed -n '1,70p' ~/.local/bin/mem-batch   # wrapper: defaults, flags, properties, docker caveat
systemctl show user@1000.service -p ManagedOOMMemoryPressure,ManagedOOMMemoryPressureLimit        # oomd kill-at-50% still in force
grep -rln "require-batch-scope" scripts/                                                          # marker-#1 scripts
grep -rn "INVOCATION_ID\|SYSTEMD_SCOPE" scripts/*.py | grep -c "os.environ\|env.get"              # guard-site count
grep -n "SYSTEMD_SCOPE" scripts/link_section_entities.py                                          # the outlier guard
grep -n "INVOCATION_ID" scripts/populate_papers_fulltext.py                                       # the not-auto-set statement
grep -n "SCIX_BATCH" scripts/daily_sync.sh scripts/run_citation_contexts_shard.sh                 # cron fallback idiom
grep -n "work_mem" scripts/eval_lexical_rank_flag.py                                              # postmaster-not-capped pattern
```
