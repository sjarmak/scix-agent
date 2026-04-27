# Qdrant runbook

Operational notes for the `scix-qdrant` container. The PRD that drives
this deployment is `docs/prd/qdrant_nas_migration.md`; this file covers
the day-to-day lifecycle and safety policies.

## Contents

- [Binding policy (security)](#binding-policy-security)
- [Lifecycle](#lifecycle)
- [Storage layout](#storage-layout)
- [Smoke checks](#smoke-checks)

## Binding policy (security)

**Rule:** the `scix-qdrant` container must bind its REST (6333) and gRPC
(6334) ports to `127.0.0.1` only. The container must never be reachable
from the LAN or the public internet.

**Why:** Qdrant has no built-in authentication on the open-source
distribution unless `QDRANT__SERVICE__API_KEY` is set. The collection
`scix_papers_v1` carries 32M-paper-scale embeddings derived from
Postgres tables that are themselves on a single host without a
network-edge auth layer. A drive-by scan of the LAN range would let
anyone read or rewrite vectors, change collection schemas, or wipe
storage.

This isn't theoretical. On 2026-04-25 the running `scix-qdrant`
container was found bound to `0.0.0.0:6333,6334` (LAN-reachable), with
`http://192.168.1.67:6333/readyz` returning 200 from another host on the
network. The container had been started with a bare `docker run -p
6333:6333` — docker treats an empty `HostIp` as `0.0.0.0`. Bead
`scix_experiments-s1a` tracks the policy codification this runbook
section is part of.

### How the policy is enforced

Three layers, in order of authority:

1. **`deploy/qdrant-compose.yml`** — the only sanctioned way to bring
   up the container. Ports are written `127.0.0.1:6333:6333` and
   `127.0.0.1:6334:6334`, so `docker compose -f deploy/qdrant-compose.yml
   up -d` cannot accidentally produce a 0.0.0.0 binding.
2. **`scripts/preflight_qdrant_security.sh`** — a post-start check that
   inspects the running container and exits non-zero if any port
   binding has empty/0.0.0.0 `HostIp`. Wire it into deployment scripts,
   CI smoke jobs, and any periodic health probe.
3. **PRD MH-1** — the architectural acceptance criterion. Any change
   that loosens the binding requires updating the PRD first.

### How to verify in 5 seconds

```bash
scripts/preflight_qdrant_security.sh
```

Expected output: `preflight: 'scix-qdrant' port bindings are loopback-only ✓`.

If you want to inspect manually:

```bash
docker inspect scix-qdrant \
  --format '{{ range $p, $cs := .HostConfig.PortBindings }}{{ range $cs }}{{ $p }} {{ .HostIp }}:{{ .HostPort }}{{ "\n" }}{{ end }}{{ end }}'
```

Every line should start with `127.0.0.1` or `::1`. Empty values, `0.0.0.0`,
or any other host IP are all violations.

### What to do on a violation

1. Stop the container: `docker stop scix-qdrant`.
2. Re-create from the compose file: `docker compose -f deploy/qdrant-compose.yml up -d`.
3. Re-run the preflight script to confirm.
4. Open a bead capturing how the drift happened so the deployment path
   that produced the bare binding gets a fix.

Do not just `docker stop` and walk away — the Qdrant collection
metadata persists, but the running config is what was queryable
externally for however long the violation existed.

## Lifecycle

`deploy/qdrant-compose.yml` is the canonical spec. On hosts without the
`docker compose` plugin, `deploy/qdrant-run.sh` is a plain-`docker-run`
mirror — same image, ports, mounts, and restart policy. Pick whichever
matches your local toolchain (the rest of `deploy/` follows the same
compose-with-shell-fallback pattern).

```bash
# Start (compose)
docker compose -f deploy/qdrant-compose.yml up -d

# Start (no-compose fallback)
./deploy/qdrant-run.sh

# Both end with the same preflight check baked in (the shell script
# runs it automatically; with compose, run it yourself):
scripts/preflight_qdrant_security.sh

# Stop
docker compose -f deploy/qdrant-compose.yml down
# or:
./deploy/qdrant-run.sh stop

# Tail logs
docker logs --tail 100 -f scix-qdrant
# or:
./deploy/qdrant-run.sh logs

# Bounce after image upgrade
docker compose -f deploy/qdrant-compose.yml pull
docker compose -f deploy/qdrant-compose.yml up -d
scripts/preflight_qdrant_security.sh
```

The compose file and the shell script both set `restart:
unless-stopped`, so the container comes back across docker daemon
restarts and host reboots without manual intervention. The preflight
script auto-runs as the last step of `qdrant-run.sh`; with compose,
chain it explicitly.

## Storage layout

Per `docs/prd/qdrant_nas_migration.md` MH-1:

- **Live storage** (`/qdrant/storage` inside container): `.qdrant_storage/`
  in the project root, on local NVMe. NFS is unsafe for live writes —
  see memory `storage_tiering_policy.md`.
- **Snapshots** (`/qdrant/snapshots` inside container): `/mnt/qdrant_snapshots/`
  on NAS, when provisioned. The compose file has the mount line
  commented out until the NAS path exists; uncomment it once the NAS
  admin has set up the share.

## Smoke checks

```bash
# API responds
curl -sf http://127.0.0.1:6333/readyz

# Not LAN-reachable (this should fail with connection refused)
curl -sf "http://$(hostname -I | awk '{print $1}'):6333/readyz" && echo "VIOLATION"

# Collection inventory
curl -sf http://127.0.0.1:6333/collections | python3 -m json.tool

# scix_papers_v1 (the pilot) point count
curl -sf http://127.0.0.1:6333/collections/scix_papers_v1 | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print('points:', d['result']['points_count'])"
```
