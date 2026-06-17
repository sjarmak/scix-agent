# Architecture diagram (LikeC4)

Architecture-as-code model of `scix` (the SciX Agent), rendered with
[LikeC4](https://likec4.dev). The model is the source of truth across
[`spec.c4`](spec.c4) (element kinds, tags, deployment node kinds),
[`model.c4`](model.c4) (the system), and [`views.c4`](views.c4) (structure,
walkthrough, and risk views), with the deployment model in
[`deployment.c4`](deployment.c4). The narrative companions are the repo-root
[`README.md`](../README.md) and [`AGENTS.md`](../AGENTS.md).

Every element `link`s to its source (`src/scix/…`, `migrations/`, `schema.sql`)
and, where one exists, to the relevant [`docs/ADR/`](../docs/ADR) or
[`docs/prd/`](../docs/prd) entry — so any box in the explorer is one click from
the code and the rationale behind it.

## Delivery state is tagged, not guessed

Every element carries a tag so **planned and research work renders distinctly
from what already serves on the 32.4M-paper corpus** (legend in `spec.c4`):

| Tag | Meaning | Render |
|---|---|---|
| `#built` | code path exists and is exercised on the production corpus | solid |
| `#evolving` | built, but the science/contract is still moving | solid |
| `#planned` | designed; not yet implemented (or v1 is a stub/heuristic) | **dashed, dimmed** |
| `#research` | speculative spike / `docs/prd/` research track | **dashed, indigo** |

Research / off-surface items in the model: the **graph-experiment spike**
(in-memory igraph + agent benchmark harness, not in the production MCP surface)
and **nanopub-style claim extraction**. Evolving-but-stubbed risk surfaces: the
**JIT local_ner** lane (CPU stub) and **query expansion** (in-memory index, no
production pgvector path yet).

## Views

**Structure** — the static map:

| View | Scope |
|---|---|
| `index` | system landscape — `scix` in context of ADS, OpenAlex, external corpus, vocabularies, inference models, and Qdrant |
| `scixSystem` | the `scix` system decomposed into containers over one Postgres + pgvector store |
| `ingestContainer` | ingest & sync — ADS harvest, COPY loader, source adapters, OA gate |
| `retrievalContainer` | retrieval stack — hybrid RRF, INDUS dense lane (Qdrant), BM25 lexical, embeddings, rerankers, eval |
| `graphContainer` | citation-graph engine — metrics, citation context/intent, claim_blame, grounding, negative results |
| `entityContainer` | entity & knowledge layer — extraction, M13 resolver, JIT lanes, linker, concept substrate, claims |
| `mcpContainer` | MCP server — the 15-tool surface, transport, contract, guards, session |
| `vizContainer` | visualization layer — viz API + trace stream, web dashboards |
| `research` | off-surface graph spike + research tracks, with built dependencies dimmed |
| `deployment` | where each piece runs — DB/GPU host, Qdrant, NAS tier, ADS AWS cluster target |

**Walkthrough flows** (dynamic / numbered-step views) — the narrative spine for
a design-review walkthrough:

| View | Flow |
|---|---|
| `ingestFlow` | corpus ingest from ADS + external sources (harvest → OA gate → COPY → embed → Qdrant) |
| `searchFlow` | one hybrid search request end-to-end (dense + lexical → RRF → optional rerank) |
| `claimBlameFlow` | claim provenance over the citation graph (reverse-reference walk weighted by intent) |
| `entityFlow` | multi-lane entity resolution (cache → local_ner canary → live_jit → static-core) |

**Risk lens:**

| View | Scope |
|---|---|
| `risks` | the `#risk`-flagged elements with each open question stated in-box (closed-access TDM gating, dense-lane-must-stay-on-Qdrant, sparse citation-context coverage, JIT local_ner stub, query-expansion stub, 15-tool cap) |

### Running the walkthrough

For a design review, present in this order: `index` → `scixSystem` (orient on
structure) → the four walkthrough flows in sequence (what actually happens) →
`deployment` (where it runs) → `risks` (what to probe) → `research` (what's
off-surface / next). In `npx likec4 start`, the dynamic views animate
step-by-step and each view's notes panel carries the gotchas (the OA-gate
publisher-clause caveat, the ADR-013 Qdrant constraint, the scratch-build
index-validation rule, the M13 lane fallbacks).

## Viewing & regenerating

```bash
# Interactive, hot-reloading explorer (recommended)
npx likec4 start architecture

# Re-export the static PNGs in exports/ (needs a one-time browser download:
#   npx playwright install chromium-headless-shell)
npx likec4 export png architecture -o architecture/exports

# Validate the model (strict — the source of truth for correctness)
npx likec4 validate architecture
```

### Viewing the interactive explorer over SSH (headless remote)

`likec4 start` serves a Vite dev server on `localhost:5173`. From a headless
remote (this DB/GPU host is headless), forward that port to your laptop and open
it locally — three options, easiest first:

1. **VS Code / Cursor Remote-SSH** — run `npx likec4 start architecture` in the
   integrated terminal; the editor auto-forwards 5173 and offers "Open in
   Browser". Nothing else to configure.
2. **SSH local port-forward** — on your laptop:
   ```bash
   ssh -N -L 5173:localhost:5173 user@remote   # leave running
   ```
   then on the remote `npx likec4 start architecture` and open
   <http://localhost:5173> locally. (Already in an SSH session? Add the tunnel
   without reconnecting: press `~C` then type `-L 5173:localhost:5173`.)
3. **Bind + reach directly** — `npx likec4 start architecture --listen 0.0.0.0`
   and browse to `http://<remote-ip>:5173` (only if that port is reachable /
   firewall-open; the tunnel in option 2 is safer).

No browser at all? Export the PNGs with `npx likec4 export png` — they need no
display; `scp` them down, or view inline if your terminal supports images.
