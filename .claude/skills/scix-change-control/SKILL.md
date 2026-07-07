---
name: scix-change-control
description: >
  How change is gated in SciX: which changes require an ADR
  (retrieval/vector/storage/dimensionality/quantization axes), the 15-tool MCP
  cap and contract regeneration, migration discipline (append-only, no
  auto-runner, the hand-applied 069-072 gap), which changes HALT at
  branch-ready for sign-off, where project truth lives (beads/ADRs, not git
  history), and what is intentionally retired vs parked. Also documents the
  Gas City bead/dispatch machinery (internal-orchestration). Load BEFORE
  proposing or landing any change to retrieval, vectors, storage, the tool
  surface, the schema, or prod data, or before re-landing anything from an old
  branch. NOT for how the retrieval stack works (scix-retrieval-architecture),
  DSN/prod-DB guards (scix-db-safety-and-telemetry), running heavy jobs safely
  (scix-memory-and-batch-discipline), or tool-surface internals
  (scix-mcp-tool-surface).
---

# SciX Change Control

How change is classified, gated, and landed in this repo — the
non-negotiables, the incident that bought each one, and the current
operational gates. Date-stamped facts are as of **2026-07-07**.

**When NOT to use this skill:** you want to _understand_ a subsystem, not
_change_ one. Orientation → `scix-orientation`. Retrieval internals →
`scix-retrieval-architecture`. Vector serving → `scix-vector-serving-qdrant`.
Embedding/ingest → `scix-embedding-pipeline`. Index/storage DDL mechanics →
`scix-index-and-storage-discipline`. Tool-surface internals →
`scix-mcp-tool-surface`. CI/tests → `scix-build-test-ci`. Evidence standards →
`scix-eval-and-evidence`.

## Jargon (defined once)

| Term                 | Meaning here                                                                                                                                              |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ADR                  | Architecture Decision Record, `docs/ADR/NNN_*.md`. The binding record for pinned axes. Numbered 006–016 as of 2026-07-07.                                 |
| bead                 | A work item in the `bd` issue tracker (`.beads/`, Dolt-backed). IDs look like `scix_experiments-s7cy`, usually cited by suffix (`s7cy`).                  |
| ADR-pinned axis      | A design axis whose value is fixed by an accepted ADR or feedback doc; changing it requires a new/amended ADR, never a casual commit.                     |
| HALT-branch-ready    | Discipline for gated work: implement fully on a branch, tests included, then STOP. No prod execution, no merge, until the operator (Stephanie) signs off. |
| operator             | The human maintainer (Stephanie / sjarmak). Solo committer; everything ultimately routes to her.                                                          |
| prod DB              | The local PostgreSQL database `scix` (32.4M papers). It is the DEFAULT DSN when `SCIX_DSN` is unset — see `scix-db-safety-and-telemetry`.                 |
| rig / mayor / worker | Gas City fleet terms — see the internal-orchestration section at the end.                                                                                 |

## 1. Change classification — what gate applies

Classify every proposed change with this table BEFORE writing code.

| Change touches…                                               | Gate                                                                     | Artifact required                                               |
| ------------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------- |
| An ADR-pinned axis (table in §2)                              | New or amended ADR + operator sign-off                                   | `docs/ADR/NNN_*.md`, Accepted status                            |
| MCP tool surface (add/remove/rename/schema)                   | 15-tool cap + contract regen (§3)                                        | Regenerated `contract/scix_mcp_v1.json`; ADR if raising the cap |
| Prod DB schema                                                | Migration discipline (§4) + HALT-branch-ready                            | New `migrations/NNN_*.sql`, applied only by operator            |
| Prod DB data (writes, drops, ingests, index builds)           | HALT-branch-ready (§5)                                                   | Branch + tests + runbook; no execution                          |
| Ordinary code (`src/`, `scripts/`, `tests/`) not in the above | Normal review; tests ship with the fix; `make check-ci` green            | Commit on a `bd/<id>-*` branch                                  |
| Docs known-stale (README, CHANGELOG, `scix-mcp` skill)        | Correct only under an explicit bead — never silently                     | A bead ID in the commit                                         |
| Anything previously retired/parked (§7)                       | Check bead + branch state first; re-land only with evidence it is wanted | Bead citation                                                   |

PROVISIONAL pending Stephanie (discovery Q5): the exact HALT list is
conservatively broad. Until she narrows it, ALSO treat corpus/repo repins,
NER-label additions, and the sealed cold-text tier (ADR-016) as
HALT-branch-ready.

## 2. ADR-pinned axes — the non-negotiables and their incidents

Source of truth: `AGENTS.md` (= `CLAUDE.md`, a symlink) section "Retrieval
architecture (decisions, change only via ADR)", plus the ADRs themselves.
Verify any of these with a one-line grep before relying on it (commands in
the Provenance section).

| Pinned axis                 | Pinned value                                                         | Why (the incident/evidence)                                                                                                                                                                                                                            |
| --------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Dense ANN serving substrate | Qdrant (`scix_indus_v2_papers_s1`), never pgvector/pgvectorscale     | ADR-013. pgvectorscale DiskANN could not scan the expression index forced by the dimensionless `embedding` column (`assertion failed: attnum > 0`); a 56 h index build was lost 2026-06-11 and the dense lane was down ~2 weeks (bead `12rp`, closed). |
| Vector dimensionality       | 768d                                                                 | PG block-size/TOAST limits; matches INDUS native output (AGENTS.md pin).                                                                                                                                                                               |
| Quantization                | `halfvec` (float16) safe; **binary quantization banned** for storage | >40% nDCG@10 loss on scientific retrieval (AGENTS.md pin). Binary allowed only as a first-pass filter.                                                                                                                                                 |
| Embedding lanes             | No paid-API lane (e.g. `text-embedding-3-large`)                     | `feedback_no_paid_apis.md`; any second dense lane must be local-weight + ADR-approved. The OpenAI lane was removed from live code.                                                                                                                     |
| Data-dir placement          | Qdrant and Postgres data dirs NEVER on NAS (`/mnt`)                  | NFS is unsafe for live-write workloads (`docs/prd/qdrant_nas_migration.md`).                                                                                                                                                                           |
| Body-AI access              | Body-text AI must gate on `papers_is_oa_or_preprint`                 | Publisher TDM-clause risk (Wiley/Springer). Abstract-only AI is universally indexable.                                                                                                                                                                 |
| MCP visible tools           | ≤ 15                                                                 | Premortem-driven tool-selection accuracy; see §3.                                                                                                                                                                                                      |

### The index-validation rules (binding for ALL future index/lane work)

ADR-013, section "Validation rules this failure bought" — bought by the
DiskANN catastrophe above:

1. **No index is trusted until one query has returned from it.** Any new
   index type/config gets a ≤50k-row scratch build + forced-index-scan smoke
   test BEFORE any multi-hour build.
2. **Benchmark DDL must be byte-identical to production DDL** (opclass,
   column type, storage params) or the benchmark is fiction. (Prod built
   `vector_cosine_ops` while the validated benchmark used
   `halfvec_cosine_ops` — the go/no-go numbers described a config prod never
   ran.)
3. **Never drop a serving index before its replacement is validated
   scannable.** Disk pressure gets solved some other way first.
4. **Re-examine an architecture pin when it starts selecting immature
   components.** "The only option that preserves the pin" is a trigger to
   question the pin.

DDL mechanics live in `scix-index-and-storage-discipline`; this skill owns
the _gate_: any change on these axes without an ADR is a change-control
violation regardless of how good the benchmark looks.

## 3. The 15-tool cap and the contract gate

The agent-visible MCP tool surface is capped at **15**, triple-enforced:

1. **Import-time guard** — `VISIBLE_TOOL_CAP = 15`
   (`src/scix/mcp_server.py:625`); exceeding it raises `RuntimeError` at
   import, so the server will not boot. The guard evaluates the DEFAULT
   hidden set, not live `SCIX_HIDDEN_TOOLS` overrides.
2. **Contract conformance test** — `tests/test_mcp_contract_conformance.py`
   asserts `contract/scix_mcp_v1.json` matches the live
   `build_contract()`; any surface change without regeneration fails CI.
3. **Startup self-test** — the server verifies its own tool count at startup.

History: the surface went 30 → 13 → 15 (consolidation bead `9afa`; governed
by `docs/prd/prd_v1_tool_consolidation.md` and
`docs/mcp_tool_audit_2026-04.md` / `_2026-06.md`). A silent 15→17 drift once
went unnoticed for two months (bead `xjqi`) — that is why the guard fails
loudly at import.

Checklist for an intentional tool-surface change:

- [ ] Read `docs/prd/prd_v1_tool_consolidation.md` +
      `docs/mcp_tool_audit_2026-04.md`; budget against the cap.
- [ ] If the change would exceed 15: consolidate (merge real overlaps) or
      raise the cap **via ADR** — see `docs/mcp_tool_audit_2026-06.md` §6.
      Never widen `SCIX_HIDDEN_TOOLS` to smuggle a 16th tool past the guard.
- [ ] Regenerate the contract: `python scripts/gen_mcp_contract.py`
- [ ] Breaking change? Bump `CONTRACT_VERSION` in `src/scix/mcp_contract.py`
      (currently `"1"`, line 32) → new `scix_mcp_v2.json`.
- [ ] `pytest tests/test_mcp_contract_conformance.py` green.

Tool internals, alias routing, error-code catalog →
`scix-mcp-tool-surface`.

## 4. Migration discipline — append-only, no auto-runner

Facts (verified 2026-07-07):

- `migrations/` holds numbered SQL files `001`–`072`, **append-only**.
- **There is no auto-migration runner.** Migrations are applied by hand by
  the operator. Nothing in CI or cron applies them to prod.
- Bookkeeping is manual and inconsistent: `schema_migrations` (created in
  `migrations/019_schema_migrations.sql`) exists, but only 5 of 72 migration
  files self-record into it (056, 059, 063, 064, 070). Recording the rest is
  a by-hand step that is sometimes skipped.
- **The 069–072 gap:** migrations 069–072 were applied to prod by hand and
  not recorded; the highest recorded version in prod `schema_migrations` is
  reported as **68** (discovery/audit finding, 2026-07-07 — NOT re-verified
  here; checking requires a prod DB read, operator-only).
  Consequence: **never infer applied-schema state from `schema_migrations`.**
  Confirm against the actual catalog (operator runs the check), or against
  the migration files + ADR/bead prose.
- `migrations/072_indus_qdrant_synced.sql` is **untracked/uncommitted** in
  the working tree as of 2026-07-07 (part of the in-flight s7cy fix, §6).
- `ci/scix_test_schema.sql` is the consolidated snapshot CI loads — a new
  migration that changes schema shape must be reflected there or CI tests
  will run against a stale schema.

Checklist for adding a migration:

- [ ] Next free number, descriptive name: `migrations/073_<what>.sql`.
- [ ] Idempotent where possible (`IF NOT EXISTS` / `IF EXISTS`).
- [ ] Include the `INSERT INTO schema_migrations (version, filename) …` row
      so the gap does not grow.
- [ ] Update `ci/scix_test_schema.sql` if the schema shape changes.
- [ ] Tests ship in the same commit.
- [ ] STOP at branch-ready. **Do not apply to prod.** The operator applies
      it in a window. (Applying is prod-DB-touching → §5.)

## 5. HALT-branch-ready — what stops at the branch

PROVISIONAL pending Stephanie (discovery Q5) on the exact boundary; the
conservative rule until then:

**Any change that (a) writes to the prod `scix` database, (b) executes DDL
against prod, (c) moves an ADR-pinned axis, (d) changes the corpus or a repo
pin, (e) adds NER labels, or (f) touches the sealed cold-text tier: implement
completely on a `bd/<id>-*` branch with tests in the same commits, write the
run plan, then HALT. Operator sign-off precedes any execution or merge.**

The incident that makes this non-negotiable — the **s7cy out-of-process
drop** (bead `s7cy`, OPEN, P1): ADR-015 (status: **Proposed**, explicitly
"artifacts only, NO prod exec") authorized dropping two INDUS _indexes_
(Stage 1, migration 071) after a soak gate ending **2026-07-11**. Instead, a
full `DROP TABLE paper_embeddings` (+ outbox) was executed against prod
out-of-process ~2026-06-29/30, with no NAS archive, without the companion
code cutover. Result: `daily_sync.sh` aborts at Step 5 every run since
2026-06-30, new papers (~1–3k/day) get no dense vectors, and the
committed embed path targets a table that no longer exists (§6). The lesson:
**"the ADR exists" is not authorization — the ADR's own status, stage, and
soak gates are the authorization,** and execution happens only in an
operator-run window.

### The current operator gate (dated note, 2026-07-07)

The rig's latest deep-audit (`.gc-reports/audit-2026-06-22.md` §3
BLOCKED_CHECK) is **RED — mechanically blocked on operator input**. The
entire dense-lane track (Qdrant INDUS flip canary `o9ib`, RAM pin `1z6s`,
throughput pilot `61j9`) and prod-touching epic work (`dbl.16/.17/.18`
ingests, `puln` storage reclamation) are parked on ONE operator action: a
scheduled maintenance window to free prod disk (98% full at audit time, swap
exhausted) and pin ~25 GB dense-lane RAM. Code for all of it is
branch-ready. **Do not attempt to route around this gate** by running any of
that work yourself; it is disk/RAM-gated, not effort-gated.
(PROVISIONAL framing per discovery Q1: this crisis is a dated operational
note, not the project's research campaign — that is the retrieval-quality
problem, see `scix-research-frontier`.)

## 6. Where truth lives — shallow git, bead-is-truth

**The git history is shallow and misleading. Do not do archaeology in git.**

Verified 2026-07-07: `git rev-list --count HEAD` = 706, but the earliest
reachable commit (`e7e958d`) is dated **2026-06-11** and the latest
(`452ab86`) 2026-06-30. CHANGELOG `v0.1.0` is dated **2026-04-20** and the
ADR/migration sequence long predates June — the repo was re-inited /
history-truncated around 2026-06-11. There are no `git revert` commits;
back-outs live in bead threads.

Where to look instead, in order:

| Question                     | Source                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------- |
| Why is X designed this way?  | `docs/ADR/` (006–016), then `docs/prd/`                                         |
| What went wrong before?      | ADR "context" sections, `.gc-reports/audit-*.md`, bead threads (`bd show <id>`) |
| What is in flight right now? | `bd list --status=open` (epics), working-tree vs HEAD diff                      |
| What was rejected/retired?   | Closed beads, ADR amendments, §7 below                                          |

**Committed HEAD vs working tree (dated note, 2026-07-07, PROVISIONAL per
discovery Q2):** the working tree carries the UNCOMMITTED s7cy remediation
(new `src/scix/qdrant_dense.py`, `migrations/072_indus_qdrant_synced.sql`,
`scripts/seed_indus_qdrant_synced.py`, modified `src/scix/embed.py` /
`scripts/embed.py` / `scripts/daily_sync.sh`, plus new tests). Committed
HEAD's `embed.py` and `scripts/embed_fast.py` still target the dropped
`paper_embeddings` table — **a newcomer reading committed code sees an
ingest pipeline that cannot run.** Rule: teach and build on committed
reality (`git show HEAD:src/scix/embed.py` when in doubt); treat the
uncommitted fix as proposed-not-landed until it lands through review.
Canonize nothing from an uncommitted tree. Details of the breakage and the
fix's own hazard (bibcode-only upsert wipes Qdrant payloads, the e4xv
pattern) → `scix-embedding-pipeline`.

## 7. Retired vs parked — do not resurrect, do not declare dead

| Thing                                                                                                                       | Status                                               | Evidence / rule                                                                                                                                                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Public docker + cloudflared deploy (`mcp.sjarmak.ai`)                                                                       | **Intentionally retired 2026-06-12** — not an outage | `CLAUDE.local.md` (operator note, gitignored): "intentionally retired — we don't want to open our network. Do not restart it." The MCP server runs as a local stdio process only. `deploy/` and the tunnel runbooks are historical. Do not "fix" its absence. (PROVISIONAL per discovery Q5.) |
| pgvectorscale/DiskANN dense serving                                                                                         | **Closed/settled** (bead `12rp`)                     | ADR-013. Returning dense to Postgres is possible only via PRD MH-12 + the §2 validation rules + a new ADR.                                                                                                                                                                                    |
| ADR-014 Qdrant sparse lexical lane                                                                                          | **Parked-on-disk**, not dead                         | Unblocks after storage relief. Check bead + branch state before re-landing.                                                                                                                                                                                                                   |
| scixmuse remote mirror                                                                                                      | **Parked remote target**                             | AGENTS.md: not "prod scix"; VPN-only, IP migrating.                                                                                                                                                                                                                                           |
| `bd/*` branches (~39)                                                                                                       | Mixed — check per branch                             | `bd show <id>` for the owning bead before reusing anything from one.                                                                                                                                                                                                                          |
| README "single PostgreSQL instance", CHANGELOG "pgvector HNSW dense", `scix-mcp` skill's OpenAI lane + trycloudflare config | **Stale docs, known**                                | Fix only under an explicit bead; never silently, and never treat them as authority — trust ADRs + code + beads.                                                                                                                                                                               |

## 8. Gas City machinery (INTERNAL-ORCHESTRATION — this section only)

This is the ONLY skill in the library that documents the fleet machinery
(per discovery Q4, PROVISIONAL: skills ship repo-local and repo-portable;
everything outside this section survives a clone with no `gc`/`~/.claude`).
Skip this section entirely if you are not running inside the ds-research
Gas City installation.

- The repo is the Gas City rig `scix-experiments` (bead prefix
  `scix_experiments-`), origin `sjarmak/scix-agent`. Work arrives as beads;
  a `scix-worker` pool executes them; a mayor agent dispatches; a PL runs
  weekly deep-audits into `.gc-reports/`.
- As a worker (`GC_AGENT` env set): execute autonomously, close with
  `bd close <id>`; if blocked, `gc mail send mayor "blocked on <reason>"`
  and stop (AGENTS.md).
- Never run `bd dolt start|stop|status` in this rig — it kills the live
  gc-managed Dolt server (fleet-wide rule).
- **Dated note (2026-07-07): direct dispatch only.** The city's default
  sling formula is `mol-focus-review`
  (`/home/ds/gas-city/.gc/sling-intercept.yaml`), and it is currently broken
  — open beads `fmso` (P0), `fv5k`, `w3ee` (P2), all titled
  `mol-focus-review`. Until those close, work is direct-dispatched by the
  mayor (`gc sling --no-formula` pattern / `gc-sling` wrapper) rather than
  routed through the formula. If you are asked to sling scix work, check
  `bd show scix_experiments-fmso` first; if still open, do not attach the
  formula.
- The `scix-batch` memory-isolation wrapper (`~/.local/bin/scix-batch`) is
  an operational requirement OF THIS INSTALLATION (co-hosted gascity
  supervisor + systemd-oomd), not of the codebase — full discipline in
  `scix-memory-and-batch-discipline`.
- External artifacts (pushes of data/results, PRs, issues, any comms) need
  per-action operator approval; routine rig code pushes of branch-ready
  worker code are pre-authorized (fleet rule, 2026-06-19).

## Provenance and maintenance

Authored 2026-07-07 against branch `bd/0yp5-external-copy-accuracy-audit`
(HEAD `452ab86` — NOTE: not `main`; the rig had this bead branch checked
out). All claims verified by source-reading only (no DB connections, no
script execution). Re-verify before trusting drift-prone facts:

```bash
git branch --show-current && git rev-parse --short HEAD   # provenance pin
grep -n "change only via ADR" AGENTS.md                    # §2 pinned-axes section still present
grep -n "VISIBLE_TOOL_CAP" src/scix/mcp_server.py          # cap still 15, import guard intact
grep -n "CONTRACT_VERSION" src/scix/mcp_contract.py        # contract version (was "1")
ls migrations/ | tail -3                                   # highest migration number (was 072, 072 uncommitted)
grep -c "INSERT INTO schema_migrations" migrations/*.sql | grep -v ":0" | wc -l   # self-recording migrations (was 5)
grep -n "Status" docs/ADR/015_offload_drop_paper_embeddings_indus.md | head -1    # ADR-015 still Proposed? soak was 2026-07-11
bd show scix_experiments-s7cy | head -5                    # live-fire bead still open?
bd show scix_experiments-fmso | head -3                    # mol-focus-review still broken? (direct-dispatch note expires when closed)
git status --porcelain | grep -c "qdrant_dense\|072_indus" # s7cy fix still uncommitted? (0 = landed; update §6)
ls .gc-reports/                                            # a newer audit supersedes the RED BLOCKED_CHECK note in §5
git log --reverse --format='%ad' --date=short | head -1    # history still starts 2026-06-11
```

Facts most likely to drift: the RED operator gate (§5, expires at the
maintenance window), the s7cy uncommitted-fix note (§6, expires when the fix
lands), the direct-dispatch rule (§8, expires when fmso closes), the ADR-015
soak date (2026-07-11), and the migration high-water mark. The
`schema_migrations` max=68 figure is reported, not re-verified (needs an
operator DB read).
