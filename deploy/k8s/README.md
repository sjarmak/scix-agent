# SciX MCP — Kubernetes deployment

Manifests for running the SciX MCP HTTP server on the ADS production
cluster (`v126.kube.adslabs.org`, k8s 1.26 on AWS), or any standard
Kubernetes cluster with access to a Postgres 16 + pgvector 0.8.2
database holding the SciX schema (see `migrations/` in the repo root).

## Files

| File | What it does |
|---|---|
| `namespace.yaml` | Creates the `scix-mcp` namespace |
| `configmap.yaml` | Non-secret env: port, pool sizes, per-tool timeouts, INDUS device |
| `secret.example.yaml` | **Template** — do NOT commit real values. Use `kubectl create secret` |
| `deployment.yaml` | Pod spec: unprivileged user, read-only root FS, probes, resources |
| `service.yaml` | ClusterIP on port 80 → container 8000 |
| `ingress.example.yaml` | Traefik ingress template (drop if internal-only) |

## Prerequisites

1. **Postgres with SciX schema** reachable from the cluster — either
   in-cluster (`postgres.scix-mcp.svc.cluster.local`) or managed (RDS).
   Apply the 54 migrations in `migrations/` in order; see
   `docs/DEPLOYMENT.md` for the bootstrap procedure.
2. **Container image** published to a registry the cluster can pull from.
   Build from repo root:
   ```bash
   docker build -f deploy/Dockerfile -t scix-mcp:v0.1.0 .
   # Re-tag for ADS ECR (adjust registry + repo to match your conventions):
   docker tag scix-mcp:v0.1.0 \
     084981688622.dkr.ecr.us-east-1.amazonaws.com/honeycomb:scix-mcp-v0.1.0
   docker push 084981688622.dkr.ecr.us-east-1.amazonaws.com/honeycomb:scix-mcp-v0.1.0
   # Update deployment.yaml `image:` field to match the pushed tag.
   ```

## Deploy

```bash
# 1. Namespace
kubectl apply -f namespace.yaml

# 2. Non-secret config
kubectl apply -f configmap.yaml

# 3. Secrets — create directly, do NOT apply secret.example.yaml with real values
kubectl -n scix-mcp create secret generic scix-mcp-secrets \
  --from-literal=MCP_AUTH_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  --from-literal=SCIX_DSN="host=<pg-host> dbname=scix user=scix_reader password=<pw>"

# 4. Deployment + service
kubectl apply -f deployment.yaml -f service.yaml

# 5. Ingress (skip if you only need in-cluster access)
#    Edit ingress.example.yaml host first, then:
kubectl apply -f ingress.example.yaml
```

Verify:
```bash
kubectl -n scix-mcp rollout status deploy/scix-mcp
kubectl -n scix-mcp port-forward svc/scix-mcp 8000:80
curl -sf http://127.0.0.1:8000/health
# {"status":"ok"}
```

## Rotating the bearer token

```bash
NEW_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
kubectl -n scix-mcp create secret generic scix-mcp-secrets \
  --from-literal=MCP_AUTH_TOKEN="$NEW_TOKEN" \
  --from-literal=SCIX_DSN="$(kubectl -n scix-mcp get secret scix-mcp-secrets \
       -o jsonpath='{.data.SCIX_DSN}' | base64 -d)" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n scix-mcp rollout restart deploy/scix-mcp
unset NEW_TOKEN
```

## INDUS embedding cache

`configmap.yaml` ships with `HF_HUB_OFFLINE=1` — the pod expects the
INDUS model (~1.5 GB, `nasa-impact/nasa-smd-ibm-st-v2`) already in the
cache mounted at `/home/scix/.cache/huggingface`.

Two options:

**A. Let the pod download on first start** (simplest; adds ~90s pod startup)
- In `configmap.yaml`: set `HF_HUB_OFFLINE: "0"` and `TRANSFORMERS_OFFLINE: "0"`
- Keep `hf-cache` volume as `emptyDir` (default in `deployment.yaml`)
- Pod must have egress to huggingface.co

**B. Pre-seed a PersistentVolumeClaim** (recommended for prod)
- Create a `scix-mcp-hf-cache` PVC (10 Gi on gp3 is plenty)
- Run a one-shot Job that mounts the PVC and runs
  `python -c "from transformers import AutoModel; AutoModel.from_pretrained('nasa-impact/nasa-smd-ibm-st-v2')"`
- Uncomment the `persistentVolumeClaim` block in `deployment.yaml`'s
  `hf-cache` volume and delete the `emptyDir` line

## GPU (optional)

INDUS runs on CPU by default. To move it to GPU:
1. In `configmap.yaml` change `SCIX_EMBED_DEVICE: "cuda"`
2. In `deployment.yaml` add to the container `resources.limits`:
   ```yaml
   nvidia.com/gpu: 1
   ```
3. Schedule onto a GPU node group (the cluster has DCGM-monitored GPU
   instances per `ads_devops_docs/source/services+software/services/dcgm.md`).

## Scaling

`replicas: 1` is fine for pilot traffic. For horizontal scale:
- Bump `replicas` — the MCP server is stateless, the pool is per-pod.
- Raise `SCIX_POOL_MAX` only if Postgres can handle the fan-out
  (`pool_max_per_pod × replicas` is the upper bound on concurrent DB
  connections).
- The app rate-limits per bearer token in-memory, so multiple replicas
  allow `N × rate` if the same token is used across all pods. Add a
  Redis-backed limiter later if that matters.

## Client config

Once the ingress is live at `https://mcp.<your-domain>/mcp/`:

```json
{
  "mcpServers": {
    "scix": {
      "url": "https://mcp.<your-domain>/mcp/",
      "headers": { "Authorization": "Bearer <MCP_AUTH_TOKEN>" }
    }
  }
}
```
