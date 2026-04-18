# pgvectorscale Install + `scix_pgvs_pilot` Scratch DB Runbook

This runbook installs pgvectorscale 0.x on Ubuntu with PostgreSQL 16 and
bootstraps a dedicated scratch database (`scix_pgvs_pilot`) for evaluating
StreamingDiskANN against the production pgvector HNSW indexes. A human
operator runs every command below — nothing here executes automatically.

## 0. Why pgvectorscale, why now

From `CLAUDE.md`:

> For 30M+ vectors: pgvectorscale StreamingDiskANN — SSD-backed index,
> 471 QPS at 99% recall on 50M vectors, 28x lower p95 latency than Pinecone,
> 75% less cost.

The production SciX corpus has 32M+ INDUS embeddings. pgvector HNSW works
up to ~5M vectors comfortably; at 30M+ we expect memory pressure and long
build times. StreamingDiskANN uses the SSD as index backing store, which is
why we want to pilot it before committing to a full rebuild.

## 1. Licensing and dependencies

pgvectorscale is **Apache 2.0** licensed and does **NOT** require
TimescaleDB — vectorscale is a standalone PostgreSQL extension. The only
hard requirement is pgvector (>= 0.7.0). You may install it alongside
TimescaleDB, but TimescaleDB is not needed for vectorscale to load or for
`CREATE EXTENSION vectorscale` to succeed.

Orthogonality note: pgvectorscale and pg_search/BM25 are **orthogonal**.
You can have both extensions installed and loaded in the same database at
the same time — vectorscale provides a vector access method
(`diskann` / StreamingDiskANN), pg_search provides a lexical BM25 index
type. They do not share state, do not conflict, and are commonly combined
for hybrid retrieval. Our `migrations/004_per_model_hnsw_and_pg_search.sql`
already uses pg_search in the production DB; the pilot DB may load both.

## 2. Target environment

| Component            | Version                 | Notes                           |
| -------------------- | ----------------------- | ------------------------------- |
| Ubuntu               | 22.04 LTS / 24.04 LTS   | What the host runs              |
| PostgreSQL           | 16.x                    | Must be **PG16** exactly        |
| pgvector             | 0.8.2                   | Already installed on this host  |
| pgvectorscale        | 0.5.x (latest 0.x)      | Being installed by this runbook |

### PostgreSQL 16 specific notes

- pgvectorscale 0.x is built and packaged specifically per PG major
  version. Packages are split `postgresql-16-pgvectorscale`,
  `postgresql-17-pgvectorscale`, etc. Installing the wrong one will
  silently not load, because `CREATE EXTENSION` looks for shared objects
  under the current server's `pkglibdir`.
- Confirm the running server version before installing:
  ```bash
  pg_config --version          # should report: PostgreSQL 16.x
  psql -U postgres -c 'SHOW server_version;'
  ```
- Upstream release notes: <https://github.com/timescale/pgvectorscale/releases>
  — check this page for PG16 compatibility of the version you pick.
- Known gotcha (PG16 on Ubuntu): `shared_preload_libraries` is **not
  required** for vectorscale — unlike TimescaleDB. Do not add
  `vectorscale` to `shared_preload_libraries`; the extension loads on
  demand. Adding it will cause the postmaster to refuse to start.
- PG16 + pgvector 0.8.2 iterative-scan (`relaxed_order`) works correctly
  with vectorscale's `diskann` index type; no config change required.
- Local postgres on this machine only listens on loopback. See
  `CLAUDE.local.md` for the `pg_docker_proxy.py` workaround — the pilot DB
  lives on the same postgres instance, so the proxy remains required for
  containerized clients.

## 3. Install pgvectorscale — option A: Timescale apt repository (recommended)

This is the **preferred** path. It keeps the binary under `apt`, so
upgrades and rollbacks are one command.

```bash
# 1. Timescale's public signing key (apt needs this to trust the repo).
sudo apt-get install -y gnupg postgresql-common apt-transport-https lsb-release wget
sudo install -d /usr/share/postgresql-common/pgdg
sudo wget --quiet -O /usr/share/keyrings/timescale.keyring.asc \
    https://packagecloud.io/timescale/timescaledb/gpgkey

# 2. Add the Timescale repo for your Ubuntu codename.
echo "deb [signed-by=/usr/share/keyrings/timescale.keyring.asc] https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main" \
    | sudo tee /etc/apt/sources.list.d/timescaledb.list

# 3. Update and install the vectorscale package for PG16.
sudo apt-get update
sudo apt-get install -y postgresql-16-pgvectorscale

# 4. Sanity-check that the shared object landed in PG16's pkglibdir.
pg_config --pkglibdir     # e.g. /usr/lib/postgresql/16/lib
ls "$(pg_config --pkglibdir)" | grep -i vectorscale
#   → vectorscale-0.x.y.so
ls "$(pg_config --sharedir)"/extension | grep -i vectorscale
#   → vectorscale.control, vectorscale--0.x.y.sql

# 5. Restart is NOT required (no shared_preload_libraries change), but
#    reconnect any open psql session so it sees the new available extension.
```

If `apt-get install postgresql-16-pgvectorscale` reports
`Unable to locate package`, the Timescale repo is not yet publishing for
your Ubuntu codename — fall back to option B below.

## 4. Install pgvectorscale — option B: build from source

Use this when apt does not have a PG16 build for your Ubuntu codename or
when you need a specific commit. This is a supported path; pgvectorscale
is a Rust project built with pgrx.

```bash
# 1. System build deps (rust toolchain, PG16 server headers, libclang).
sudo apt-get install -y build-essential pkg-config libssl-dev clang \
    postgresql-server-dev-16 git curl

# 2. Install rustup + stable toolchain (skip if already present).
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 3. Install cargo-pgrx pinned to the version pgvectorscale expects.
#    Check the README at the tag you are building for the exact version.
cargo install --locked cargo-pgrx --version 0.12.9

# 4. Initialize pgrx against the system PG16.
cargo pgrx init --pg16 "$(which pg_config)"

# 5. Clone and build a released tag (do NOT build main — it may be
#    incompatible with pgvector 0.8.2).
git clone https://github.com/timescale/pgvectorscale
cd pgvectorscale/pgvectorscale
git checkout 0.5.1                          # or whichever 0.x tag is current
cargo pgrx install --release --pg-config "$(which pg_config)"

# 6. Verify the install the same way option A does.
ls "$(pg_config --pkglibdir)" | grep -i vectorscale
ls "$(pg_config --sharedir)"/extension | grep -i vectorscale
```

## 5. Create `scix_pgvs_pilot` and load the extensions

The pilot DB is deliberately a **separate database** on the same postgres
cluster as production `scix`. This gives us:

- Zero risk of rewriting production indexes or tables.
- Same host, same postgres binary, same tuning — so benchmark numbers are
  directly comparable to prod.
- Trivial teardown: `DROP DATABASE scix_pgvs_pilot;`.

```bash
# As the postgres superuser (adjust role/host as needed):
psql -U postgres -h localhost <<'SQL'
CREATE DATABASE scix_pgvs_pilot;
SQL

# Connect to the new DB and load the extensions.
psql -U postgres -h localhost -d scix_pgvs_pilot <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION vectorscale;
-- Optional, only if pg_search is already installed on this cluster:
-- CREATE EXTENSION IF NOT EXISTS pg_search;
SQL
```

## 6. Schema bootstrap — minimum migration set

The pilot only needs the tables required to host `paper_embeddings` with
the right shape and metadata columns. From `migrations/` we need exactly
these four SQL files, applied in order:

| # | Migration file                                | Why it is needed for the pilot                                                                                                                                      |
| - | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `001_initial_schema.sql`                      | Creates `papers` (the FK target) and the original `paper_embeddings` table with composite PK `(bibcode, model_name)` and `vector(768)`.                              |
| 2 | `003_search_infrastructure.sql`               | Adds the `input_type` and `source_hash` metadata columns to `paper_embeddings`. The pilot-loader script writes these; without them the COPY fails.                  |
| 3 | `004_per_model_hnsw_and_pg_search.sql`        | Creates the per-model partial HNSW index we will benchmark *against*. We intentionally keep this so the pilot has a like-for-like HNSW baseline next to DiskANN.     |
| 4 | `023_logged_embeddings.sql`                   | Converts `paper_embeddings` to LOGGED and widens the column from `vector(768)` to untyped `vector` so mixed-dimension pilots (e.g. 768d INDUS + 3072d OpenAI) work. |

All other migrations (entity graph, harvest runs, fusion MV, full-text
tables, etc.) are irrelevant to a vector-only pilot and are deliberately
skipped.

Apply them from the project root:

```bash
cd ~/projects/scix_experiments

export SCIX_PILOT_DSN="dbname=scix_pgvs_pilot host=localhost"

psql "$SCIX_PILOT_DSN" -v ON_ERROR_STOP=1 -f migrations/001_initial_schema.sql
psql "$SCIX_PILOT_DSN" -v ON_ERROR_STOP=1 -f migrations/003_search_infrastructure.sql
psql "$SCIX_PILOT_DSN" -v ON_ERROR_STOP=1 -f migrations/004_per_model_hnsw_and_pg_search.sql
psql "$SCIX_PILOT_DSN" -v ON_ERROR_STOP=1 -f migrations/023_logged_embeddings.sql
```

`-v ON_ERROR_STOP=1` is important — without it psql continues past failed
statements and you end up with a half-built schema.

Notes on expected warnings:
- `003` updates existing rows in `papers` to populate `tsv`; on an empty
  pilot DB this is a no-op.
- `004` emits
  `NOTICE: pg_search extension not available — skipping BM25 index creation`
  if pg_search is not installed. That is expected and safe for the pilot.

## 7. Create a pilot DiskANN index

Once the schema is in place and you have loaded a representative subset
of embeddings (e.g. 1M–5M rows — see the pilot-loader script in a separate
work unit), build the StreamingDiskANN index:

```sql
-- Connect: psql "$SCIX_PILOT_DSN"

-- Match the INDUS model (768d). Adjust m, ef_construction to taste.
CREATE INDEX idx_embed_diskann_indus
    ON paper_embeddings
    USING diskann (embedding vector_cosine_ops)
    WHERE model_name = 'indus';

-- Benchmark against the per-model HNSW baseline from migration 004.
```

## 8. Verification

Run all of the following and confirm the output. This is the
authoritative "did the install work?" check.

```bash
psql "$SCIX_PILOT_DSN" <<'SQL'
-- 1. Both extensions installed in the pilot DB.
SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('vector', 'vectorscale')
ORDER BY extname;

-- 2. Access methods: diskann should be present.
SELECT amname
FROM pg_am
WHERE amname IN ('hnsw', 'diskann', 'ivfflat')
ORDER BY amname;

-- 3. Required tables exist with expected columns.
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'paper_embeddings'
ORDER BY ordinal_position;

-- 4. input_type and source_hash columns were added by migration 003.
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'paper_embeddings'
  AND column_name IN ('input_type', 'source_hash');

-- 5. Confirm the table is LOGGED (migration 023).
SELECT relname, relpersistence
FROM pg_class
WHERE relname = 'paper_embeddings';
--   relpersistence should be 'p' (permanent / LOGGED), not 'u' (unlogged).
SQL
```

Expected:

- `pg_extension` returns two rows: `vector` (0.8.x) and `vectorscale`
  (0.5.x or whichever tag you installed).
- `pg_am` includes `diskann` (new), `hnsw` (from pgvector), and
  `ivfflat` (from pgvector).
- `paper_embeddings` has columns `bibcode`, `model_name`, `embedding`,
  `input_type`, `source_hash`.
- `relpersistence` is `p`.

## 9. Isolation guarantees

**Never** point pilot tooling at the production DSN. Concretely:

| Env var         | Intended DB         | Set by                                      |
| --------------- | ------------------- | ------------------------------------------- |
| `SCIX_DSN`      | `scix` (production) | Production env, `deploy/.env`               |
| `SCIX_PILOT_DSN`| `scix_pgvs_pilot`   | The pilot operator's shell only (see below) |

Set the pilot DSN in a scoped shell, and verify it resolves to the right
database before running anything destructive:

```bash
# Pick a shell just for pilot work.
export SCIX_PILOT_DSN="dbname=scix_pgvs_pilot host=localhost"

# Sanity check: this MUST print scix_pgvs_pilot. If it prints scix, stop.
psql "$SCIX_PILOT_DSN" -Atc 'SELECT current_database();'
#   → scix_pgvs_pilot

# Cross-check that SCIX_DSN (production) is NOT the same DB.
echo "Production DSN target:"
psql "$SCIX_DSN" -Atc 'SELECT current_database();'
#   → scix  (must NOT say scix_pgvs_pilot, and the pilot DSN must NOT say scix)
```

Guardrails:

- Production `SCIX_DSN` points at `scix`. It does **not** point at
  `scix_pgvs_pilot`, and nothing in `src/scix/` is configured to use
  `scix_pgvs_pilot` by default.
- All pilot scripts must read `SCIX_PILOT_DSN`, not `SCIX_DSN`. Code
  that mixes them is a P0 bug — revert and rewrite.
- The pilot DB has no `papers` data at bootstrap; you must load a
  subset explicitly. This means a misconfigured tool will fail fast with
  FK errors rather than corrupt production data.

## 10. Teardown

When the pilot is done:

```bash
# Disconnect all sessions, then drop.
psql -U postgres -h localhost -c 'DROP DATABASE IF EXISTS scix_pgvs_pilot;'

# Optional: remove the extension cluster-wide if you no longer want it.
# (The apt package stays installed; only the SQL objects go.)
# Do NOT run this if any other DB on the cluster depends on vectorscale.
```

To remove the package entirely (apt install path only):

```bash
sudo apt-get remove --purge postgresql-16-pgvectorscale
sudo apt-get autoremove
```

## 11. Summary of what this runbook did

- Installed pgvectorscale 0.x (Apache 2.0, no TimescaleDB dependency)
  for PostgreSQL 16 on this host, via apt or source build.
- Created an isolated `scix_pgvs_pilot` database on the same postgres
  cluster as production `scix`.
- Applied the minimum four migrations needed for `paper_embeddings`
  (`001`, `003`, `004`, `023`) and loaded both `vector` and `vectorscale`
  extensions.
- Established `SCIX_PILOT_DSN` as the only DSN pilot tooling uses, so
  production `scix` is never touched.
- Left the cluster ready to create StreamingDiskANN indexes and benchmark
  them against the existing pgvector HNSW baseline, side by side with
  pg_search/BM25 if desired.
