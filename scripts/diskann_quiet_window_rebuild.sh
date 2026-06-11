#!/usr/bin/env bash
# diskann_quiet_window_rebuild.sh — quiet-window rebuild of the INDUS DiskANN
# index on the production `scix` DB, with the memory contention removed so the
# build doesn't thrash (see beads scix_experiments-kj37, scix_experiments-12rp).
#
# CANONICAL DDL (decided in scix_experiments-12rp; matches the benchmarked V1
# variant in scripts/build_streamingdiskann_variants.py and the halfvec
# rationale in CLAUDE.md):
#
#   CREATE INDEX idx_embed_diskann_indus ON paper_embeddings
#     USING diskann (embedding_hv halfvec_cosine_ops) WHERE model_name='indus';
#
# WHY this column/opclass and NOT the earlier (((embedding)::vector(768))
# vector_cosine_ops) form:
#   * CORRECTNESS — the cast form is an EXPRESSION index. pgvectorscale 0.9.0
#     DiskANN BUILDS it but cannot SCAN it: every kNN errors 'assertion failed:
#     attnum > 0' (expression keys have no attnum). The 2026-06-09..11 prod
#     build (56h, 21GB, 32.38M vectors, valid=t) was UNSCANNABLE and the planner
#     auto-selected it, crashing all INDUS dense queries (dropped 2026-06-11).
#     Reproduced at 50k rows. A fixed-dimension column scans fine — and
#     paper_embeddings.embedding_hv is exactly that: halfvec(768), fully
#     backfilled for INDUS (32.38M rows, finished 2026-04-29; migrations
#     053/054). So NO new column migration is needed — just index embedding_hv.
#   * STORAGE/SPEED — halfvec is 2 bytes/dim vs 4 for vector(768), so the build
#     chews ~half the bytes and the on-disk index is ~half the size (fits the
#     tight prod disk). halfvec is also CLAUDE.md's sanctioned quantization
#     (binary quant loses >40% nDCG@10; halfvec does not).
#   * VALIDATED OPCLASS — halfvec_cosine_ops is the opclass whose nDCG@10 was
#     measured against the C1 'within 1% of HNSW' gate (pgvectorscale_migration_decision.md).
#
# QUERY-PATH COUPLING: the planner only picks this index when the query orders
# by the BARE halfvec column (pe.embedding_hv <=> $::halfvec(768)). That is the
# SCIX_USE_HALFVEC=1 path in src/scix/search.py. The dense lane must be deployed
# with SCIX_USE_HALFVEC=1 or the index sits unused (see cmd_finish).
#
# Before any multi-hour build, `validate` runs a 50k-row scratch scannability
# gate (cmd_validate) that reproduces the attnum>0 check in seconds — REQUIRED
# by 12rp. (Standalone alternative still under consideration: skip pgvectorscale
# for the dense lane and move paper_embeddings to GPU-built Qdrant HNSW.)
#
# WHY: attempt 1 used maintenance_work_mem=8GB on a 62GB host that was already
# swap-bound (shared_buffers=16GB + gascity ~10GB + claude ~6GB) → 23% in 8.5h,
# ~48h ETA, and global swap pressure that risks oomd killing the gascity
# supervisor (user@1000.service has ManagedOOMMemoryPressure=kill @50%).
#
# This script frees the contention first, then builds with a large
# maintenance_work_mem, then restores everything.
#
# Phases are SEPARATE subcommands on purpose:
#   prep    — pause gascity, drop shared_buffers 16G→4G (PG restart), clear swap   [needs sudo]
#   build   — create the DiskANN index with maintenance_work_mem=$MWM             [no sudo; long]
#   restore — shared_buffers back, PG restart, resume gascity                      [needs sudo]
#   finish  — commit the server gate change + print redeploy/verify steps         [no sudo]
#   status  — show current memory / index / gascity state
#
# Typical run (operator, with sudo available):
#   ! bash scripts/diskann_quiet_window_rebuild.sh prep
#   ! bash scripts/diskann_quiet_window_rebuild.sh build      # walk away ~hours
#   ! bash scripts/diskann_quiet_window_rebuild.sh restore
#   ! bash scripts/diskann_quiet_window_rebuild.sh finish
#
# Knobs (env):
#   MWM=32GB            maintenance_work_mem for the build (default 32GB)
#   NUM_NEIGHBORS=      if set (e.g. 32), adds WITH(num_neighbors=N): smaller/faster, slight recall cost
#   SMALL_SHARED=4GB    shared_buffers during the build (default 4GB)
#   CONCURRENTLY=0      1 = CREATE INDEX CONCURRENTLY (online but slower + leaves INVALID idx on failure)

set -euo pipefail

DSN="${SCIX_DSN:-dbname=scix}"
REPO="/home/ds/projects/scix_experiments"
STATE="$REPO/logs/diskann_rebuild_state.env"
LOG="$REPO/logs/diskann_quiet_rebuild_$(date -u +%Y%m%dT%H%M%SZ).log"
IDX="idx_embed_diskann_indus"
# Canonical index target: the fixed-dim halfvec(768) column (migrations 053/054),
# NOT a (((embedding)::vector(768))) expression cast — see header for why.
HV_COL="embedding_hv"
PG_CLUSTER_VER=16
PG_CLUSTER_NAME=main
MWM="${MWM:-32GB}"
SMALL_SHARED="${SMALL_SHARED:-4GB}"
NUM_NEIGHBORS="${NUM_NEIGHBORS:-}"
CONCURRENTLY="${CONCURRENTLY:-0}"

mkdir -p "$REPO/logs"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }
q()   { psql "$DSN" -tA -c "$1"; }

assert_scix() {
  local db; db=$(q "SELECT current_database();")
  [ "$db" = "scix" ] || die "expected DB 'scix', got '$db' (DSN=$DSN)"
}

# build_ddl <index_name> <concurrently:0|1> <with_clause>
# Single source of truth for the canonical DiskANN DDL so build + build-coexist
# can't drift apart. Indexes the bare halfvec(768) column (scannable), NOT an
# expression cast (unscannable — see header).
build_ddl() {
  local name="$1" conc=""; [ "${2:-0}" = "1" ] && conc="CONCURRENTLY"
  echo "CREATE INDEX $conc $name ON paper_embeddings USING diskann ($HV_COL halfvec_cosine_ops)${3:-} WHERE model_name='indus';"
}

# cmd_preflight — refuse to build unless the canonical column exists, has the
# right type, and is fully backfilled (a partial build over a half-backfilled
# column silently under-indexes).
# Capability gate: pgvectorscale must expose a halfvec opclass for diskann.
# 0.9.0 ships diskann opclasses for `vector` only (vector_cosine_ops etc.) —
# NOT halfvec_cosine_ops — so the canonical DDL cannot build until the
# extension is upgraded. (The float32 vector(768) alternative needs a fixed-dim
# column + ~98GB rewrite that does not fit the prod disk; see header + 12rp.)
assert_diskann_halfvec_capable() {
  local has_hv; has_hv=$(q "SELECT count(*) FROM pg_opclass opc
                              JOIN pg_am am ON am.oid=opc.opcmethod
                             WHERE am.amname='diskann' AND opc.opcname='halfvec_cosine_ops';")
  [ "${has_hv:-0}" != "0" ] && return 0
  local ver; ver=$(q "SELECT extversion FROM pg_extension WHERE extname='vectorscale';")
  die "pgvectorscale ${ver:-?} has no 'halfvec_cosine_ops' opclass for diskann — the canonical halfvec DiskANN index CANNOT be built on this server.
    OPTIONS (owner/ADR decision, see bead scix_experiments-12rp + kj37):
      1. Upgrade pgvectorscale to a release that ships halfvec diskann opclasses, then re-run.
      2. Pivot the INDUS dense lane to Qdrant (GPU-built HNSW) per the paper_embeddings->Qdrant migration.
      3. Fallback: keep the validated HNSW-on-halfvec index (migration 054, idx_embed_hnsw_indus_hv).
    NOT viable: diskann on (((embedding)::vector(768))) — builds but is UNSCANNABLE (attnum>0); a fixed-dim vector(768) column would fit the opclass but not the disk (~98GB)."
}

cmd_preflight() {
  assert_scix
  assert_diskann_halfvec_capable
  local t; t=$(q "SELECT format_type(atttypid, atttypmod) FROM pg_attribute
                   WHERE attrelid='paper_embeddings'::regclass AND attname='$HV_COL';")
  [ "$t" = "halfvec(768)" ] || die "$HV_COL is '${t:-absent}', expected halfvec(768) — canonical DiskANN column missing (apply migrations 053/054 + backfill_halfvec.py)"
  local fin; fin=$(q "SELECT (finished_at IS NOT NULL)::text FROM halfvec_backfill_progress
                       WHERE model_name='indus' ORDER BY started_at DESC LIMIT 1;")
  [ "$fin" = "t" ] || die "INDUS halfvec backfill not marked finished — refusing to build a partial index (run scripts/backfill_halfvec.py --model indus first)"
  log "PREFLIGHT ok: $HV_COL=halfvec(768), INDUS backfill finished"
}

# cmd_validate — 50k-row scratch scannability gate. Reproduces the attnum>0
# expression-index failure in seconds, on the EXACT column+opclass the long
# build will use, BEFORE committing many hours. REQUIRED by scix_experiments-12rp.
cmd_validate() {
  assert_scix
  assert_diskann_halfvec_capable
  log "VALIDATE: 50k-row scratch DiskANN scannability gate on $HV_COL"
  local out rc=0
  out=$(psql "$DSN" -v ON_ERROR_STOP=1 -tA <<SQL 2>&1
CREATE TEMP TABLE _diskann_scratch (embedding_hv halfvec(768));
INSERT INTO _diskann_scratch (embedding_hv)
  SELECT $HV_COL FROM paper_embeddings
   WHERE model_name='indus' AND $HV_COL IS NOT NULL
   LIMIT 50000;
CREATE INDEX _diskann_scratch_idx ON _diskann_scratch
  USING diskann (embedding_hv halfvec_cosine_ops);
-- Session-level (NOT SET LOCAL): psql autocommits each statement, so SET LOCAL
-- would expire before the SELECT below and the planner could seqscan 50k rows
-- without ever touching the index — a false PASS. Force the index scan.
SET enable_seqscan = off;
-- The 'assertion failed: attnum > 0' fires HERE if the index is unscannable.
-- ON_ERROR_STOP makes that abort the whole gate.
SELECT count(*) FROM (
  SELECT 1 FROM _diskann_scratch
   ORDER BY embedding_hv <=> (SELECT embedding_hv FROM _diskann_scratch LIMIT 1)
   LIMIT 10
) s;
SELECT 'SCANNABLE_OK';
SQL
  ) || rc=$?
  if [ "$rc" -ne 0 ] || ! printf '%s' "$out" | grep -q 'SCANNABLE_OK'; then
    printf '%s\n' "$out" | tee -a "$LOG"
    die "scannability gate FAILED (rc=$rc) — do NOT start the multi-hour build; the canonical DDL is not scannable on this server"
  fi
  log "VALIDATE ok: scratch DiskANN index built and scanned cleanly (no attnum>0)"
}

pg_restart() {
  log "restarting postgres ($PG_CLUSTER_VER/$PG_CLUSTER_NAME) — needs sudo"
  sudo pg_ctlcluster "$PG_CLUSTER_VER" "$PG_CLUSTER_NAME" restart \
    || sudo systemctl restart "postgresql@${PG_CLUSTER_VER}-${PG_CLUSTER_NAME}" \
    || die "postgres restart failed — restart it manually and re-run"
  for _ in $(seq 1 30); do q "SELECT 1" >/dev/null 2>&1 && break; sleep 2; done
  q "SELECT 1" >/dev/null 2>&1 || die "postgres did not come back after restart"
  log "postgres back up; shared_buffers=$(q 'SHOW shared_buffers')"
}

cmd_status() {
  echo "=== memory ==="; free -h
  echo "=== swap ==="; swapon --show || true
  echo "=== shared_buffers / maintenance_work_mem ==="; q "SHOW shared_buffers" ; q "SHOW maintenance_work_mem"
  echo "=== gascity supervisor ==="; gc supervisor status 2>&1 | head -3 || true
  echo "=== $IDX ==="
  q "SELECT COALESCE((SELECT i.indisvalid::text FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid WHERE c.relname='$IDX'),'absent') AS idx_state,
            COALESCE((SELECT pg_size_pretty(pg_relation_size('$IDX')) WHERE EXISTS(SELECT 1 FROM pg_class WHERE relname='$IDX')),'-') AS size"
  echo "=== disk ==="; df -h / | awk 'NR==2{print "free="$4" used="$5}'
}

cmd_prep() {
  assert_scix
  log "PREP start (MWM target=$MWM, small shared_buffers=$SMALL_SHARED)"
  local free_disk; free_disk=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  [ "$free_disk" -ge 30 ] || die "only ${free_disk}G free on / — need >=30G for the ~22G index + WAL"

  # record original shared_buffers + that gascity was running, for restore
  local orig_sb; orig_sb=$(q "SHOW shared_buffers")
  echo "ORIG_SHARED_BUFFERS=$orig_sb" > "$STATE"
  echo "GASCITY_WAS_RUNNING=$(gc supervisor status >/dev/null 2>&1 && echo 1 || echo 0)" >> "$STATE"
  log "recorded state: orig shared_buffers=$orig_sb (-> $STATE)"

  log "pausing gascity (city sessions + supervisor) to free RAM and exit oomd blast radius"
  gc stop 2>&1 | tee -a "$LOG" || log "  (gc stop returned nonzero — continuing)"
  gc supervisor stop 2>&1 | tee -a "$LOG" || log "  (gc supervisor stop returned nonzero — continuing)"
  sleep 3

  log "lowering shared_buffers -> $SMALL_SHARED (frees ~12G for the build)"
  q "ALTER SYSTEM SET shared_buffers='$SMALL_SHARED';" >/dev/null
  pg_restart
  [ "$(q 'SHOW shared_buffers')" = "$SMALL_SHARED" ] || log "WARN: shared_buffers not $SMALL_SHARED — check manually"

  # clear stale swap ONLY if RAM can absorb it
  local avail_kb used_swap_kb
  avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  used_swap_kb=$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{print t-f}' /proc/meminfo)
  if [ "$avail_kb" -gt $((used_swap_kb + 4194304)) ]; then
    log "clearing swap (used=$((used_swap_kb/1048576))G, avail=$((avail_kb/1048576))G) — needs sudo"
    sudo swapoff -a && sudo swapon -a && log "swap cleared" || log "WARN: swap clear failed (non-fatal)"
  else
    log "SKIP swap clear: only $((avail_kb/1048576))G available < $((used_swap_kb/1048576))G swap-in-use + margin"
  fi
  log "PREP done. free now:"; free -h | tee -a "$LOG"
  log "next: bash $0 build"
}

cmd_build() {
  assert_scix
  cmd_preflight
  cmd_validate
  local sb; sb=$(q "SHOW shared_buffers")
  log "BUILD start (shared_buffers=$sb, maintenance_work_mem=$MWM, num_neighbors=${NUM_NEIGHBORS:-auto}, concurrently=$CONCURRENTLY)"
  [ "$sb" = "$SMALL_SHARED" ] || log "WARN: shared_buffers=$sb (expected $SMALL_SHARED) — did you run 'prep'? continuing anyway"

  # clean any leftover (e.g. INVALID index from a cancelled CONCURRENTLY build)
  q "DROP INDEX IF EXISTS $IDX;" >/dev/null

  local with_clause=""
  [ -n "$NUM_NEIGHBORS" ] && with_clause=" WITH (num_neighbors=$NUM_NEIGHBORS)"
  local ddl; ddl=$(build_ddl "$IDX" "$CONCURRENTLY" "$with_clause")
  log "DDL: $ddl"
  if [ "$CONCURRENTLY" != "1" ]; then
    log "NOTE: non-concurrent build takes a SHARE lock (blocks WRITES to paper_embeddings, reads OK). Avoid the 06:15 UTC daily_sync window."
  fi

  # scix-batch keeps OUR client in a bounded scope; the real work + memory is in
  # the postgres backend (maintenance_work_mem via PGOPTIONS).
  local SB; SB=$(command -v scix-batch || true)
  local t0=$SECONDS
  ${SB:+$SB --mem-high 8G --mem-max 12G} env PGOPTIONS="-c maintenance_work_mem=$MWM" \
    psql "$DSN" -v ON_ERROR_STOP=1 -c "$ddl" 2>&1 | tee -a "$LOG"
  local valid; valid=$(q "SELECT i.indisvalid FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid WHERE c.relname='$IDX';" || echo "")
  [ "$valid" = "t" ] || die "build finished but index not valid (got '$valid') — see $LOG"
  log "BUILD done in $(((SECONDS-t0)/60)) min; size=$(q "SELECT pg_size_pretty(pg_relation_size('$IDX'))")"
  log "next: bash $0 restore"
}

cmd_build_coexist() {
  # Build WITHOUT stopping gascity and WITHOUT a postgres restart.
  # Safety model: bound the build's memory (maintenance_work_mem) so it can't
  # exhaust RAM, + a watchdog that cancels the build if MemAvailable drops below
  # a floor or swap grows — so the build yields to gascity, never the reverse.
  assert_scix
  cmd_preflight
  cmd_validate
  local MWM_CO="${MWM_COEXIST:-16GB}"
  local NN="${NUM_NEIGHBORS:-32}"          # smaller graph: less build memory, smaller on disk, slight recall cost
  local PSI_LIMIT="${PSI_LIMIT:-35}"       # abort if user-slice 'full' mem-pressure avg10 > this %% (oomd kills @50)
  local GROWTH_GB="${SWAP_GROWTH_GB:-8}"   # abort if swap GROWS more than this from baseline (real thrash signal)
  local FLOOR_GB="${FLOOR_GB:-2}"          # hard backstop: abort if MemAvailable below this (near-OOM)
  local WATCH_INT="${WATCH_INT:-15}"
  local PSI_PATH="/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure"
  local MIN_FREE_GB="${MIN_FREE_GB:-28}"   # disk headroom: ~16G index (nn=32) + WAL/daily-sync/log churn

  # --- HARD disk pre-flight (the current blocker) ---
  local free_gb; free_gb=$(df -B1 --output=avail / | tail -1 | awk '{printf "%d",$1/1e9}')
  if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    die "ONLY ${free_gb}G FREE on / (need >=${MIN_FREE_GB}G). Building ~14-22G into a near-full prod disk risks
         filling it mid-build -> postgres can't write WAL -> crash/read-only. FREE DISK FIRST (move duplicate/
         archival to NAS per storage-tiering; this ties to the urgent paper_embeddings->Qdrant-NAS migration)."
  fi

  log "BUILD-COEXIST start: maintenance_work_mem=$MWM_CO, num_neighbors=$NN, free_disk=${free_gb}G"
  log "  gascity stays UP; watchdog floor MemAvailable=${FLOOR_GB}G, swap-growth abort=${GROWTH_GB}G"
  q "DROP INDEX IF EXISTS $IDX;" >/dev/null

  # --- memory watchdog: protects gascity by cancelling the build, not gc ---
  # Watchdog keys on the SAME signal oomd uses — the user-slice 'full' memory
  # pressure (oomd kills @50%; we abort @PSI_LIMIT, default 35, to preempt it).
  # Backstops: swap GROWTH (real thrash) and a low MemAvailable hard floor.
  # NOTE: absolute MemAvailable alone is NOT used as the primary trip — it false-
  # tripped attempt 2 (avail dipped to 4G while swap was SHRINKING = no real danger).
  local base_swap_kb; base_swap_kb=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
  ( while sleep "$WATCH_INT"; do
      local avail_kb used_swap_kb psi psi_x100
      avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
      used_swap_kb=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print t-f}' /proc/meminfo)
      psi=$(awk '/^full/{for(i=1;i<=NF;i++){if($i~/^avg10=/){sub("avg10=","",$i);print $i}}}' "$PSI_PATH" 2>/dev/null)
      psi_x100=$(awk -v v="${psi:-0}" 'BEGIN{printf "%d", v*100}')
      if [ "$psi_x100" -gt $((PSI_LIMIT*100)) ] \
         || [ $((used_swap_kb-base_swap_kb)) -gt $((GROWTH_GB*1048576)) ] \
         || [ "$avail_kb" -lt $((FLOOR_GB*1048576)) ]; then
        echo "[watchdog] TRIP psi_full_avg10=${psi:-?}% swapΔ=$(((used_swap_kb-base_swap_kb)/1048576))G avail=$((avail_kb/1048576))G — cancelling build to protect gascity" | tee -a "$LOG"
        psql "$DSN" -tA -c "SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE query LIKE 'CREATE INDEX%${IDX}%' AND pid<>pg_backend_pid();" >/dev/null 2>&1
        break
      fi
    done ) &
  local WPID=$!
  # also abort if the prod disk runs low DURING the build
  ( while sleep "$WATCH_INT"; do
      local fb; fb=$(df -B1 --output=avail / | tail -1 | awk '{printf "%d",$1/1e9}')
      if [ "$fb" -lt 6 ]; then
        echo "[diskdog] TRIP only ${fb}G free — cancelling build before disk hits 0" | tee -a "$LOG"
        psql "$DSN" -tA -c "SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE query LIKE 'CREATE INDEX%${IDX}%' AND pid<>pg_backend_pid();" >/dev/null 2>&1
        break
      fi
    done ) &
  local DPID=$!

  local ddl; ddl=$(build_ddl "$IDX" 0 " WITH (num_neighbors=$NN)")
  log "DDL: $ddl"
  log "NOTE: non-concurrent SHARE lock blocks WRITES to paper_embeddings (gascity doesn't write it; the 06:15 UTC daily_sync would wait). Set CONCURRENTLY=1 to avoid, at higher memory cost."
  local t0=$SECONDS rc=0
  env PGOPTIONS="-c maintenance_work_mem=$MWM_CO" psql "$DSN" -v ON_ERROR_STOP=1 -c "$ddl" 2>&1 | tee -a "$LOG" || rc=$?
  kill "$WPID" "$DPID" 2>/dev/null || true

  local valid; valid=$(q "SELECT i.indisvalid FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid WHERE c.relname='$IDX';" 2>/dev/null || echo "")
  if [ "$valid" = "t" ]; then
    log "BUILD-COEXIST done in $(((SECONDS-t0)/60)) min; size=$(q "SELECT pg_size_pretty(pg_relation_size('$IDX'))"). next: bash $0 finish"
  else
    q "DROP INDEX IF EXISTS $IDX;" >/dev/null 2>&1 || true
    die "build did not complete (rc=$rc, valid='$valid') — likely watchdog/disk abort or error; see $LOG. Index dropped."
  fi
}

cmd_restore() {
  assert_scix
  [ -f "$STATE" ] && . "$STATE" || log "WARN: no state file; restoring shared_buffers to 16GB default"
  local target="${ORIG_SHARED_BUFFERS:-16GB}"
  log "RESTORE: shared_buffers -> $target"
  q "ALTER SYSTEM SET shared_buffers='$target';" >/dev/null
  pg_restart
  log "resuming gascity supervisor + city"
  gc supervisor start 2>&1 | tee -a "$LOG" || log "  (supervisor start returned nonzero)"
  gc start 2>&1 | tee -a "$LOG" || log "  (gc start returned nonzero — start the city manually if needed)"
  log "RESTORE done."; cmd_status
  log "next: bash $0 finish"
}

cmd_finish() {
  cd "$REPO"
  log "FINISH: committing the MCP gate change (detects diskann OR hnsw)"
  if ! git diff --quiet -- src/scix/mcp_server.py tests/test_mcp_server.py; then
    git add src/scix/mcp_server.py tests/test_mcp_server.py
    git commit -m "feat(scix_experiments-kj37): dense-lane gate accepts DiskANN index, not just HNSW

_hnsw_index_exists now matches idx_embed_hnsw_<m> OR idx_embed_diskann_<m> via
_vector_index_names(); query path is index-agnostic (cosine on (embedding)::vector(768)).
Re-enables INDUS semantic search after the pgvectorscale StreamingDiskANN cutover.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 2>&1 | tee -a "$LOG"
  else
    log "  no uncommitted gate change found (already committed?)"
  fi
  cat <<EOF | tee -a "$LOG"

REDEPLOY + VERIFY (manual — depends on how the local MCP is launched):
  - REQUIRED: deploy the dense lane with SCIX_USE_HALFVEC=1. The new index is on
    the bare halfvec column (embedding_hv halfvec_cosine_ops); the planner only
    picks it when the query orders by pe.embedding_hv <=> \$::halfvec(768), which
    is the SCIX_USE_HALFVEC=1 path in src/scix/search.py. With it unset (default
    0) the query casts (embedding)::vector(768) and the DiskANN index sits unused.
  - If the MCP is stdio-spawned per Claude session: just start a NEW session; it imports the updated code.
  - If a persistent local MCP server runs: restart it.
  - Then verify the dense lane is live:
      * a scoped MCP search (e.g. arxiv_class=cs.AI, year_min=2025) should report vector_ms > 0
      * lit_review should return seeds (was returning 0 while lexical-only)
      * confirm the planner actually uses the index:
          SET enable_seqscan=off;
          EXPLAIN SELECT 1 FROM paper_embeddings WHERE model_name='indus'
            ORDER BY embedding_hv <=> (SELECT embedding_hv FROM paper_embeddings
              WHERE model_name='indus' AND embedding_hv IS NOT NULL LIMIT 1) LIMIT 10;
        -> plan must show 'Index Scan using idx_embed_diskann_indus'.
  - Optional eval: 50-query nDCG vs the recorded HNSW numbers; fill docs/prd/pgvectorscale_migration_decision.md
EOF
  log "FINISH done."
}

case "${1:-}" in
  prep)          cmd_prep ;;
  preflight)     cmd_preflight ;;
  validate)      cmd_validate ;;
  build)         cmd_build ;;
  build-coexist) cmd_build_coexist ;;
  restore)       cmd_restore ;;
  finish)        cmd_finish ;;
  status)        cmd_status ;;
  *) sed -n '2,40p' "$0"; echo; echo "usage: $0 {prep|preflight|validate|build|build-coexist|restore|finish|status}"; echo "  preflight:     assert embedding_hv halfvec(768) exists + INDUS backfill finished"; echo "  validate:      50k-row scratch scannability gate (run before any long build)"; echo "  build-coexist: build alongside gascity (bounded mem + watchdog, no gc stop, no PG restart) — needs ~28G free disk"; exit 1 ;;
esac
