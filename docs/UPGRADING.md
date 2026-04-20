# Upgrading the SciX MCP

The contract for how the ADS ops team pulls updates from this repo into
a running deployment. Read this alongside
[`CHANGELOG.md`](../CHANGELOG.md) and [`docs/DEPLOYMENT.md`](./DEPLOYMENT.md).

## TL;DR — routine upgrade

```bash
# 1. Fetch the new tag
cd /path/to/scix_experiments
git fetch --tags
git checkout v<X.Y.Z>

# 2. Skim the CHANGELOG between your current tag and the new one
git log --oneline v<current>..v<X.Y.Z>
less CHANGELOG.md   # find the target version's entry

# 3. Apply migrations. They are idempotent — re-running an already-applied
#    migration is a no-op. Simpler and safer than tracking "what's new":
for f in migrations/*.sql; do
  echo "=== $f ==="
  psql "$SCIX_DSN" -v ON_ERROR_STOP=1 -f "$f" || exit 1
done
#
# (If you need audit — which migration introduced a change — use
#  `git log -- migrations/NNN_*.sql` or the CHANGELOG entry.)

# 4. Rebuild the image at the new tag and push to your registry
docker build -f deploy/Dockerfile -t scix-mcp:v<X.Y.Z> .
docker tag scix-mcp:v<X.Y.Z> <your-registry>/scix-mcp:v<X.Y.Z>
docker push <your-registry>/scix-mcp:v<X.Y.Z>

# 5. Roll the deployment
#    k8s:
sed -i "s|image: scix-mcp:v.*|image: <your-registry>/scix-mcp:v<X.Y.Z>|" \
  deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl -n scix-mcp rollout status deploy/scix-mcp

#    compose:
docker compose -f deploy/compose/backoffice.yaml --env-file deploy/.env \
  up -d --force-recreate scix-mcp
```

A successful upgrade finishes with `curl -sf https://<host>/health`
returning 200.

## The contract

### 1. Migrations are append-only and sequentially numbered

- New schema changes go in a new file: `migrations/NNN_<description>.sql`
  where `NNN` is `max(existing) + 1`, zero-padded to 3 digits.
- **Never edit a migration after it has landed on `main` and been tagged.**
  If you need to change what a migration did, write a new one that
  fixes it. This is what lets any operator replay migrations in order
  on a fresh DB and arrive at a known state, no matter when they
  started.
- Migrations must be idempotent to the extent `psql` makes possible —
  use `CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`, guarded
  `DO $$ ... $$` blocks. An operator re-running an already-applied
  migration should never fail; the worst case is a no-op.

### 2. Tags are the handoff unit

- Every release is an annotated git tag: `v<MAJOR>.<MINOR>.<PATCH>`.
- The tag commit must contain a `CHANGELOG.md` entry describing what
  changed since the previous tag, in four sections at minimum:
  *Server*, *Retrieval / Schema*, *Deployment*, *Security*. Add more
  if useful.
- ADS pulls tags, not `main`. Work-in-progress on `main` never affects
  a deployment that has pinned to `v0.1.0` until the operator
  explicitly moves to a newer tag.
- Never force-push a tag. If a release is broken, cut a higher patch
  or minor and document the defect in its entry.

### 3. Semver

Given the MCP's two surfaces — the **MCP tool protocol** and the
**Postgres schema** — the version-bump rules are:

- **MAJOR** (`v1.0.0` → `v2.0.0`): a tool is renamed or removed, a tool
  argument becomes required, a tool's response shape changes in a
  breaking way, or a migration requires manual steps beyond `psql -f`
  (e.g. a downtime window, a data backfill longer than a few minutes).
- **MINOR** (`v0.1.0` → `v0.2.0`): new tool added, new optional
  argument, new migration that runs cleanly, new image-level capability
  (e.g. Kafka consumer support).
- **PATCH** (`v0.1.0` → `v0.1.1`): bug fix, performance improvement,
  documentation or deployment-manifest edit with no behavioural change.

Pre-1.0 this repo may still make breaking changes inside MINOR bumps,
but the CHANGELOG will call them out explicitly under a *Breaking*
subsection.

### 4. Breaking changes

Anything that would require an ADS operator to do more than pull,
apply migrations, and `rollout restart` is a **breaking change**. Each
breaking change must be listed in the CHANGELOG under its version's
*Breaking* section, with:

- Exactly what breaks (symptom the operator will see).
- The migration path (commands to run, config to change).
- A rollback procedure (how to get back to the previous tag if the
  new one misbehaves in production).

### 5. Rollback

Rolling back one version is a tag-checkout and an image pin — the
database is the hard part. Rules:

- **Code rollback** is always safe if no migrations have been applied
  from the newer tag. Roll the image tag back, `kubectl apply`.
- **Migration rollback** is **not** supported in-place. Migrations do
  not ship `DOWN` scripts. If a migration has run, plan forward — apply
  a new migration that undoes or supersedes the damage, then cut a
  patch release.
- This is deliberate: `DOWN` migrations tempt operators into
  destructive rollbacks under pressure. Forward-only with good review
  discipline is safer at production scale.

## What ADS needs to do the first time

For the v0.1.0 → first-production rollout specifically:

1. Fork or mirror this repo under `adsabs/` (so pushes from sjarmak
   flow through a PR review you control).
2. Stand up a dedicated Postgres 16 + pgvector 0.8.2 instance (see
   `docs/DEPLOYMENT.md` §Prerequisites for sizing).
3. Run every migration `001` … `054` in order on that empty DB.
4. Publish the image to your ECR/Honeycomb repo with the retag
   convention `tailor:scix-mcp-v0.1.0`.
5. Apply `deploy/k8s/` manifests, updating the `image:` field to your
   published tag and filling the Ingress host.
6. Backfill the corpus and generate embeddings (JSONL path in
   `docs/DEPLOYMENT.md` §2–§3). This takes real wall time — ingest is
   hours, embedding is 16h on a single RTX-class GPU or a week on CPU.

After that, every subsequent upgrade is the §TL;DR flow above.

## Getting help

- CHANGELOG entry unclear? File an issue on the shared repo.
- Migration hits an error? Don't re-run destructively — capture the
  `psql` error, the migration file number, and the current
  `schema_migrations` row count, and open an issue. Fix-forward
  migrations are easier to author from a clear failure trace than from
  a half-rolled-back DB.
