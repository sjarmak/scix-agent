# SciX Experiments

Python project running AI/ML experiments on the NASA ADS scientific literature corpus (~100GB across 6 years). Hybrid retrieval (INDUS dense + BM25 sparse, RRF fusion) over PostgreSQL 16 + pgvector. The corpus is recoverable from upstream ADS, so we accept "duplicate-to-NAS" as the only redundancy tier (no off-host backup).

Standing collaboration rules: `~/.claude/rules/common/agent-collaboration.md`. **Operator-local notes are in `CLAUDE.local.md`** (gitignored; current service URLs / endpoints / live state).

---

## Don't (each line names the failure mode it prevents)

**Database safety**

- Don't run `pytest tests/` without `SCIX_TEST_DSN` set → write/delete tests skip silently (or, before the guard existed, mutated production data). Default DSN points at the prod `scix` database.
- Don't hardcode `ADS_API_KEY` → secret in git. Use the env var; `.env` is gitignored.
- Don't commit `*.jsonl` / `*.jsonl.gz` / `*.jsonl.xz` → corpus data; they belong in `.gitignore`.

**Retrieval architecture (decisions, change only via ADR)**

- Don't add a paid-API embedding lane (e.g. `text-embedding-3-large`) → blocked by `feedback_no_paid_apis.md`; any "second dense lane" must be local-weight + ADR-approved.
- Don't use binary quantization for vector storage → >40% nDCG@10 loss on scientific retrieval. Use only as a first-pass filter; `halfvec` (float16) is the safe quantization.
- Don't serve dense ANN from pgvector/pgvectorscale → the INDUS dense lane serves from Qdrant (`scix_indus_v2_papers_s1`, `QDRANT_URL` gate in `vector_search()`) per ADR-013; pgvectorscale DiskANN cannot scan the expression index our dimensionless `embedding` column forces (56 h build lost 2026-06-11). New sections/chunks lanes target Qdrant collections, not pgvector.
- Don't trust a new index until one query has returned from it → ≤50k-row scratch build + forced-index-scan smoke test BEFORE any multi-hour build; benchmark DDL must be byte-identical to prod DDL (ADR-013 validation rules; the failure that bought them is bead `12rp`).
- Don't add an MCP tool past the 15-tool cap → agent tool-selection accuracy degrades (premortem-driven; consolidation governed by `docs/prd/prd_v1_tool_consolidation.md` + `docs/mcp_tool_audit_2026-04.md`).
- Don't change vector dimensionality off 768d in pgvector → block-size limits + TOAST overhead; 768d also matches INDUS native output.
- Don't run Qdrant or Postgres data dirs from NAS (`/mnt`) → NFS substrate is not safe for live-write workloads (see `docs/prd/qdrant_nas_migration.md`).

**Body-AI / closed-access**

- Don't run body-AI scripts (`scripts/run_ner_bodies.py`, `scripts/extract_citation_contexts.py`, `python -m scix.embeddings.section_pipeline`, `scripts/run_chunk_pass.py`) without OA/preprint gating → publisher TDM-clause risk (Wiley / Springer / etc.). They gate on `papers_is_oa_or_preprint(papers)` already; don't remove that. Abstract-only AI (INDUS title+abstract, GLiNER abstracts) is universally indexable.

**Memory isolation (this host also runs the gascity supervisor)**

- Don't run multi-GB or >1-minute scix scripts in the default shell cgroup → `user@1000.service` has `ManagedOOMMemoryPressure=kill` at 50%; oomd picks a casualty and frequently kills the gascity supervisor, taking down mayor and every worker session.

**Telemetry**

- Don't analyse `query_log` with `WHERE success=FALSE AND error_msg IS NOT NULL` → silently drops blocked-by-guard requests. Structured-error responses log `success=TRUE` because `_dispatch_tool` returns the error JSON without raising. Use `WHERE success=FALSE OR error_msg IS NOT NULL`.

**Remote target (scixmuse)**

- Don't confuse scixmuse with "prod scix" → prod scix is the local DB on this host (`scix` database). The 2xe pgvectorscale benchmark approved 2026-05-01 runs on local prod scix, not the remote mirror.
- Don't SSH to scixmuse without VPN, and don't hard-code its IP → IP is migrating early-to-mid May 2026; treat the alias in Steph's MacBook `~/.ssh/config` as canonical.

---

## Do (concrete point targets)

- **Heavy work:** `scix-batch <command>` — transient `systemd-run --scope` unit with `MemoryHigh=20G`, `MemoryMax=30G`, `ManagedOOMPreference=avoid`. Override per invocation: `scix-batch --mem-high 40G --mem-max 60G ...`. Cron jobs use `$SCIX_BATCH` with a PATH fallback.
- **Integration tests:** `export SCIX_TEST_DSN="dbname=scix_test"` before `pytest tests/`. The `scix_test` database has the full schema (all migrations applied), no data. Tests that write check `is_production_dsn()` and skip if pointed at prod.
- **Body-AI queries:** filter on `papers_is_oa_or_preprint(papers)` (migration 068, partial index `idx_papers_is_oa`); opt-in to closed via `--include-closed`.
- **Adding an MCP tool:** read `docs/prd/prd_v1_tool_consolidation.md` and `docs/mcp_tool_audit_2026-04.md` first; budget against the 15-tool cap. After an intentional change to the tool surface, regenerate the contract artifact with `python scripts/gen_mcp_contract.py` (else `tests/test_mcp_contract_conformance.py` fails); a breaking change bumps `CONTRACT_VERSION` in `scix.mcp_contract`.
- **Production scripts:** pass `--allow-prod`. The script self-checks `SYSTEMD_SCOPE` env (set automatically by `systemd-run --scope`) and refuses to run otherwise.
- **Finding current focus:** `bd list --status=open` then filter `issue_type == "epic"` — current open epics: `wqr`, `xz4`, `dbl`, `buu`, `xoas`.
- **As a Gas City worker** (`GC_AGENT` env set): execute autonomously, no plan-approval, close beads with `bd close <id>`; if blocked, `gc mail send mayor "blocked on <reason>"` and stop.
- **New code defaults to DS** (`/dev/nvme1n1p2`, 1.9 TB NVMe). Move to NAS (`/mnt`, 50 TB) only for: (a) doesn't fit, (b) export/snapshot intended as backup duplicate, (c) archival raw content.

---

## Codebase compasses

Summon by name when working in the area.

| Compass | Summon when |
| --- | --- |
| `compass-retrieval-stack` | retrieval, embeddings, vector index, RRF, ADR-pinned architecture |
| `compass-memory-isolation` | scix-batch, oomd, supervisor coexistence, heavy job sizing |
| `compass-db-safety` | DSN guards, migrations, production protection |
| `compass-mcp-tools` | MCP server, tool surface cap, tool audit, telemetry |
| `compass-scixmuse` | remote mirror target, VPN, migration plan |

Detailed playbooks in `docs/conventions/*.md`.

---

## Layout

```
src/scix/                  — package: mcp_server, search (RRF fusion), db, ingest, embed, graph_metrics, extract, session, sources/, jit/, eval/
scripts/                   — CLI tools (incl. scix-batch self-enforcement via SYSTEMD_SCOPE)
migrations/                — PostgreSQL schema migrations (068 introduced papers_is_oa_or_preprint)
tests/                     — pytest; integration tests need SCIX_TEST_DSN
docs/
  adr/, prd/, premortem/   — architecture decisions, product requirements, risk analyses
  paper_outline.md         — ADASS paper outline
  mcp_tool_audit_2026-04.md
  conventions/             — fetch-on-demand playbooks (this file's compass index points here)
ads_metadata_by_year_picard/  — raw JSONL data (gitignored, also duplicated to /mnt/scix_offload/)
CLAUDE.local.md            — operator-local notes (gitignored)
```

## Where current state lives (not here)

- **Active focus, in-flight work** → epic beads (see above) and `docs/`
- **Tool surface inventory** → `docs/mcp_tool_audit_2026-04.md`
- **Embedding model landscape** → `docs/prd/` and `docs/adr/`
- **Paper section pointers** → `docs/paper_outline.md`
- **Recently shipped** → `git log --oneline -50`

If something has a date stamp or describes "what we're doing right now", it belongs in an epic bead — not in this file.
