# scixmuse migration runbook

> Tracking bead: `scix_experiments-ffux` (scoping). Successor executor bead
> will be filed at the end of Phase 4 and gated on Steph's sign-off.
>
> **Background**: see `CLAUDE.md` § "scixmuse — the remote target machine".
> scixmuse is a separate physical machine accessed over SSH + VPN. Goal: mirror
> the full scix-mcp stack (code + Postgres + paper_embeddings + eval harness +
> runbooks) onto it so it can serve as a hot standby or parallel benchmark
> target.

## Phase 0 — Access verification

**Status: PATH CHOSEN (2026-05-02) — MacBook-Pro-as-bridge.** No corp VPN on
ds-5090 required. Pattern is validated by prior `/mnt/scix_offload/ads_metadata_by_year_picard/`
transfer (same topology, opposite direction).

### Why no VPN on ds-5090

scixmuse sits behind the CfA Cisco AnyConnect SAML VPN (`vpn1.cfa.harvard.edu`,
group `SAOVPN`, federated to SI Microsoft Entra IdP, MFA via Authenticator).
Initial plan was to install `openconnect-sso` on ds-5090. Three risks made
this unattractive:

1. **Conditional access**: SI's tenant likely requires "compliant device"
   (Intune/Jamf-enrolled). ds-5090 isn't enrolled; CA policy may reject it
   regardless of client quality.
2. **Headless SAML first-auth**: openconnect-sso uses Qt WebEngine; first
   auth needs X11 forwarding from a Mac.
3. **SAML session expiry**: Entra sessions are typically 8-24 h. Phase 2 is
   multi-day → would need a re-auth scheduler.

Steph's empirical observation supersedes the openconnect path: **while MBP
is on the corp VPN, parallel rsyncs MBP ⇄ scixmuse and MBP ⇄ ds-5090 work
fine**, even though SSH from MBP-on-VPN to ds-5090 does not. (Likely the
corp VPN's split-tunnel hijacks DNS / specific ports but leaves enough open
for rsync; root cause not investigated — empirical result is enough.)

### Topology

```
ds-5090  ⇄  MacBook Pro  ⇄  scixmuse
  (Tailscale         (corp VPN
  or LAN)             vpn1.cfa.harvard.edu / SAOVPN)
```

MBP runs in the middle. ds-5090 never sees the corp VPN.

### What we checked on ds-5090

| Check | Result |
|---|---|
| `~/.ssh/config` on ds-5090 | does not exist — alias canonically lives on MBP |
| Tailscale state (`tailscale0`) | up, IP `100.122.69.12` (peer name `ds-5090`) |
| Tailscale peers | `ds-5090`, `iphone-15-pro`, `stephanies-macbook-air` online; `stephanies-macbook-pro` offline 59d ago; `bens-macbook-air-1/2`, `instance-20251230-155636` offline. **scixmuse not on tailnet.** |
| DNS for `scixmuse` / variants | all fail (expected — corp-internal hostname) |

scixmuse on Tailscale is **not** an option per project context (corp-managed
machine, can't install third-party VPN clients).

### scixmuse access details

| Item | Value |
|---|---|
| SSH user | `scixmuse` |
| IP (as of 2026-05-02) | `131.142.194.21` |
| Subnet | `131.142.194.0/24` (CfA range) |
| IP-change window | early-to-mid May 2026 — re-resolve before bulk transfer |
| ssh command from MBP | `ssh scixmuse@131.142.194.21` |
| No alias in MBP `~/.ssh/config` | she types the IP each time today |
| Free disk on scixmuse | **877 GB available** (partition TBD via `df -h` — survey will resolve). Clears the Phase 3 disk-capacity gate (estimate <200 GB total transfer). |
| Pre-existing data on scixmuse | `ads_metadata_year_full.jsonl` files for **1940-2026** (per Steph 2026-05-02). **Wider** than ds-5090's `ads_metadata_by_year_picard/` (2021-2026 only). See "Scope question" below. |

### Inputs still required from Steph (Phase 2 only)

| # | Input | Source | Blocks |
|---|---|---|---|
| 1 | MBP free disk | `df -h ~` on MBP | Phase 2 staging plan |
| 2 | Last-used parallel rsync invocation | shell history / saved script | Phase 2 transport |

### Scope question — Mirror, Extension, or Asymmetric?

scixmuse already has `ads_metadata_year_full.jsonl` for 1940-2026 (wider than
ds-5090's 2021-2026 corpus). This forces a scope decision that didn't exist
in the original bead description:

| Scope | What it means | Cost |
|---|---|---|
| **Mirror** (2021-2026) | Push only 2021-2026 raw + derived from ds-5090. scixmuse's older jsonls untouched. | Smallest. Baseline plan. |
| **Extension** (1940-2026) | Pull 1940-2020 raw scixmuse→ds-5090, ingest into local Postgres, embed, derive. Then push everything to scixmuse. | High compute on ds-5090 (~5x more rows to embed). Unlocks deeper benchmark corpus. |
| **Asymmetric** | Migrate 2021-2026 stack only, ignore the 1940-2020 raw on scixmuse. | Cheapest, muddiest end state — scixmuse can't act as a true standby. |

Survey output answers two prerequisite questions before this decision:

1. Are scixmuse's `ads_metadata_year_full.jsonl` the same upstream dump as
   ds-5090's `picard` files, or a different ADS snapshot batch? (Schema/field
   drift matters for ingestion.)
2. Are the 2021-2026 entries on scixmuse current vs. ds-5090's? (Mtime + line
   count comparison.)

Decision deferred to Steph after Phase 1 lands. Default if she doesn't
choose: **Mirror** — it's the smallest scope and matches the bead's original
"hot standby" framing.

## Phase 1 — Survey (read-only) — PENDING MBP HOP

Run from MBP-on-VPN over SSH to scixmuse. Capture all output to one
`scixmuse_survey_<date>.tar.gz`, then rsync that file back to ds-5090
(VPN-on or VPN-off — empirical result says rsync MBP↔ds-5090 works either
way).

Survey captures:

- OS / kernel / distro (`uname -a`, `cat /etc/os-release`)
- CPU / RAM / disk (`lscpu`, `free -h`, `lsblk`, `df -h`)
- Postgres install + version + running daemons + listen_addresses
- pgvector / pgvectorscale availability in apt/yum
- Contents of `/home/<scix-user>/` and any `/mnt/<offload>/` paths
  (especially the existing ADS metadata jsonls — `find ... -maxdepth 3`)
- External IP (`curl ifconfig.me`) for whitelisting if needed
- Time sync state (`timedatectl` / `chronyc tracking`)
- Free disk on the partition that'll hold paper_embeddings (~100 GB needed)

### Survey script

Read-only. Writes a tarball to stdout. Leaves nothing on scixmuse.

Save this as `~/scixmuse_survey.sh` on the MBP:

```bash
#!/bin/bash
# scixmuse Phase 1 survey — read-only. Tarball to stdout.
set -uo pipefail

WORKDIR=$(mktemp -d /tmp/scixmuse_survey.XXXXXX)
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR"

run() {
  local name="$1"; shift
  echo "=== $name ===" >> survey.txt
  echo "\$ $*" >> survey.txt
  ("$@") >> survey.txt 2>&1 || echo "[exit $?]" >> survey.txt
  echo >> survey.txt
}

{
  echo "scixmuse Phase 1 survey — captured $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname) — user: $(whoami)"
} > survey.txt

# System
run uname           uname -a
run os-release      cat /etc/os-release
run lsb-release     lsb_release -a
run cpuinfo         lscpu
run mem             free -h
run disk            df -h
run blockdev        lsblk
run mounts          mount
run external-ip     curl -sS --max-time 5 ifconfig.me
run timesync        timedatectl
run uptime          uptime

# Postgres
run pg-which        bash -c "command -v psql || echo not-found"
run pg-version      bash -c "psql --version 2>/dev/null || echo not-installed"
run pg-listen       bash -c "ss -tlnp 2>/dev/null | grep -E ':5432|postgres' || echo no-pg-port"
run pg-conf-locate  bash -c "sudo -n find /etc -name postgresql.conf 2>/dev/null || find /etc/postgresql -name postgresql.conf 2>/dev/null || echo unknown"
run pg-services     bash -c "systemctl list-units --type=service --no-pager 2>/dev/null | grep -i postgres || echo none"

# pgvector / pgvectorscale availability via package manager
if command -v apt >/dev/null; then
  run apt-pgvector       apt-cache search pgvector
  run apt-pgvectorscale  apt-cache search pgvectorscale
elif command -v dnf >/dev/null; then
  run dnf-pgvector       dnf -q search pgvector
  run dnf-pgvectorscale  dnf -q search pgvectorscale
elif command -v yum >/dev/null; then
  run yum-pgvector       yum -q search pgvector
fi

# Home + offload
run home            ls -la /home/
run user-home       bash -c "ls -la \$HOME"
run mnt             bash -c "ls -la /mnt/ 2>/dev/null || echo no-/mnt"
run mnt-find-jsonl  bash -c "find /mnt -maxdepth 4 -name '*.jsonl*' 2>/dev/null | head -50"
run home-find-jsonl bash -c "find \$HOME -maxdepth 4 -name '*.jsonl*' 2>/dev/null | head -50"
run ads-paths       bash -c "find / -maxdepth 5 -iname 'ads*' 2>/dev/null | head -50"

tar czf - survey.txt
```

Run it from the MBP (with VPN connected):

```bash
chmod +x ~/scixmuse_survey.sh
ssh scixmuse@131.142.194.21 'bash -s' < ~/scixmuse_survey.sh \
  > ~/scixmuse_survey_$(date +%Y%m%d).tar.gz
```

Then ship the tarball to ds-5090 (Tailscale, no VPN needed):

```bash
rsync ~/scixmuse_survey_*.tar.gz ds-5090:/tmp/
```

ds-5090 will untar it into `docs/runbooks/scixmuse_survey/` and append the
key results to this runbook under "Phase 1 — captured output".

## Phase 1 — captured output (2026-05-02)

scixmuse confirmed as:

- **OS**: Ubuntu 24.04.2 LTS
- **Postgres**: not installed (clean install path)
- **Disk**: total ~1.77 TB on `/dev/nvme0n1p3` mounted at `/`. **893 GB used**, **877 GB free**.

**Pre-existing data on scixmuse `/home/scixmuse/`** (693 GB used in /home):

| Path | Size | Status |
|---|---:|---|
| `scix_data/` | 448 GB | Raw ADS jsonls 1940-2026. Not touched by our plan (we don't migrate raw ADS). |
| `scix_kg/` | 52 GB | **Do not touch** — pre-existing work, ownership TBD |
| `Embeddings/` | 50 GB | **Do not touch** — pre-existing work, ownership TBD |
| (other) | ~143 GB | Untracked — not blocking |

Outside `/home`: ~200 GB (Ubuntu system, `/var`, packages). Default Postgres
data dir at `/var/lib/postgresql/16/main/` lands here, alongside `/home/scixmuse/`
without conflict.

**Open question**: the original bead description claimed scixmuse had "no
scix-mcp code, no postgres database, no embeddings". `scix_kg/` and
`Embeddings/` contradict this — they're either (a) a different/older project
unrelated to ours, (b) someone else's work, or (c) Steph's earlier work she
didn't recall when the bead was written. Coexistence is fine; ownership flag
for Steph to clarify when convenient.

## Phase 2 — Per-component transfer plan

### Capacity reality

Local prod `scix` DB total: **1411 GB**. scixmuse: **877 GB free**. Full
mirror does not fit.

Top relations on prod (data + indexes split is the key insight):

| Relation | Total | Table | Indexes |
|---|---:|---:|---:|
| `papers_fulltext` | 493 GB | 4.5 GB | 489 GB |
| `papers` | 411 GB | 59 GB | 353 GB |
| `paper_embeddings` | 195 GB | 66 GB | 128 GB |
| `agent_document_context` | 58 GB | 54 GB | 5 GB |
| `citation_edges` | 45 GB | 19 GB | 26 GB |
| `document_entities` | 29 GB | 15 GB | 13 GB |
| `paper_metrics` | 18 GB | 7 GB | 11 GB |
| `_to_embed` | 16 GB | 16 GB | 0.5 GB |
| `entities` | 14 GB | 4.5 GB | 9.6 GB |
| (rest) | ~120 GB | ~10 GB | ~110 GB |

Aggregate: **~250 GB tables + ~1100 GB indexes**. The DB is mostly indexes.

### TOAST is the bulk, not indexes

Initial framing said "DB is mostly indexes" — wrong. Per-table breakdown:

| Table | Heap | TOAST | Indexes | Total |
|---|---:|---:|---:|---:|
| `papers_fulltext` | 4.5 GB | **455 GB** | 29 GB | 493 GB |
| `papers` | 59 GB | **292 GB** | 56 GB | 411 GB |
| `paper_embeddings` | 66 GB | 124 GB | 3 GB | 195 GB |

Most storage is TOAST (out-of-line big text/jsonb columns). Slim-the-indexes
alone saves only ~50 GB and doesn't fit the 534 GB shortfall. Real savings
require dropping TOAST-heavy tables.

### Path B — concrete plan (chosen 2026-05-02)

Verified against `src/scix/search.py`: BM25 sparse lane uses `papers.body`
(via `ix_papers_body_tsv`), not `papers_fulltext`. INDUS dense lane uses
`paper_embeddings` HNSW. Both retrieval lanes survive without
`papers_fulltext`.

| Action | Saves | Trade-off |
|---|---:|---|
| Drop `papers_fulltext` | 493 GB | `claim_search` MCP tool degrades; snippets fall back to body-extract instead of section-aware |
| Drop `agent_document_context` | 58 GB | Session/working-set state rebuilds per session |
| Drop `_to_embed` | 16 GB | Embedding work queue — scixmuse won't ingest, doesn't need it |
| Drop ~10 wide indexes on `papers` not on retrieval hot path | ~5 GB | Some facet/admin queries slower |
| **Total saved** | **~572 GB** | |
| **scixmuse final footprint** | **~840 GB** | 37 GB headroom on 877 GB |

**What stays (retrieval-critical):**
- `papers` (411 GB — body + tsvs preserved) → BM25 sparse lane ✅
- `paper_embeddings` (195 GB — HNSW intact) → INDUS dense lane ✅
- `papers_external_ids`, `paper_metrics`, `paper_uat_mappings` — ranking + filters
- `citation_edges` (45 GB), `document_entities` (29 GB), `entities` (14 GB) — graph + entity tools
- All small tables not enumerated

**What tools degrade on scixmuse:**
- `claim_search` (no `papers_fulltext.sections`) — fails or returns body-fallback
- Working-set / session tools — fresh state per session
- A few admin/facet queries (the dropped GIN indexes)

**Headroom alternatives** if 37 GB feels tight:
- Also drop `paper_metrics` (18 GB) if scixmuse doesn't rank-by-metrics
- Also drop `citation_edges` (45 GB) if scixmuse is purely retrieval (no graph traversal)

### Earlier paths — rejected

| Path | Why rejected |
|---|---|
| **A**: add disk | Steph chose B; revisitable if Path B headroom proves too tight in practice |
| **C**: drop a single major table without considering trade-offs | Subsumed into the surgical Path B plan above |
| **Extension** scope (1940-2026) | Capacity gap precludes — even current 2021-2026 prod doesn't fit |

### Impact on scope question

The earlier Mirror / Extension / Asymmetric fork narrows: **Extension is dead**
under current scixmuse capacity (can't fit 2021-2026 prod, let alone deeper
1940-2026). Real choices reduce to:
- **Mirror** (2021-2026, full prod data, with one of A/B/C above)
- **Asymmetric** (push only a thin slice — code + small tables, no
  embeddings) — useful only as a code/eval-harness mirror, not retrieval

### Per-component table — to fill once Path A/B/C is chosen

Filled after the path decision. Sizes confirmed against local prod via
`pg_size_pretty(pg_total_relation_size(...))`.

**Transport for every row**: dump to file on ds-5090 (or stream over
Tailscale to MBP staging), then `rsync` MBP→scixmuse over corp VPN. No
direct ds-5090↔scixmuse hop. MBP needs enough free disk to stage the
largest single artifact (largest single `pg_dump` table file); `rsync`
streams keep working-set bounded but at least one full table-dump must
fit on MBP at a time.

| Component | Method | Estimated size | Notes |
|---|---|---|---|
| scix-mcp code | rsync working tree (no public push) | ~50 MB | Trivial via MBP hop |
| postgres install on scixmuse | apt/yum, match local prod 16.13 | n/a | Done on scixmuse, no transfer |
| pgvector | package or build from source | n/a | Same |
| pgvectorscale | install only if `scix_experiments-2xe` says go | n/a | Coordinate with `2xe` |
| schema | `pg_dump --schema-only` → file → MBP → scixmuse | <100 MB | One-shot |
| paper_embeddings (32M × halfvec(768)) | `pg_dump -Fc -t paper_embeddings` → split → parallel rsync | ~70-80 GB raw + ~20 GB index | Largest item. Use `--checkpoint-segments` and `parallel` per the validated invocation Steph used last time |
| section_embeddings | dump+rsync | TBD | Coordinate with `wqr.9` / `zpm4` |
| Other tables (papers, papers_fulltext, citation_*) | `pg_dump -Fd --jobs=8` → directory → rsync | TBD | Per-table size from prod survey |
| eval harness + runbooks | rsync | <1 MB | |
| ADS metadata jsonls already on scixmuse | leave in place; document path | n/a | Refresh tracked separately |

### Babysitting cost

MBP is in the loop, so the bulk transfer is foreground work for Steph
(MBP plugged in, awake, on VPN, parallel rsync running). Estimate hours
for code+schema+small tables, ~1-2 days for paper_embeddings depending
on VPN throughput. Section_embeddings sizing TBD.

## Phase 3 — Execution gate — TBD

Will list a binary go/no-go checklist plus the migration's PASS/FAIL bar
(pre-committed: scix-mcp on scixmuse answers a 50-query eval against the
migrated DB within 1% nDCG@10 of prod).

## Phase 4 — Successor executor bead — TBD

Filed once Phases 0-3 are populated. Will be gated on
`gc.requires_human_approval=true` so mayor surfaces it to Steph before
slinging.
