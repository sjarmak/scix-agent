# SciX Viz Demo — Operator & Presenter Guide

## Overview

The SciX viz demo exposes three complementary views of the 32M-paper NASA
ADS corpus, each built directly on top of the same hybrid retrieval +
graph-analytics stack that the MCP server uses. The goal of the talk is to
make it concrete that "agent-navigable scientific knowledge" is not a slide
deck — it is a running system that a person or an agent can point-click
through end-to-end.

The three views are layered: **V2** (temporal Sankey) shows how scientific
communities persist and split across decades; **V3** (UMAP browser) drops
to the paper level, rendering 100k INDUS embeddings in a 2-D layout you can
zoom and hover; **V4** (agent trace overlay) pipes live MCP tool calls
through a server-sent-event stream and highlights the papers an agent is
touching as it works through a prompt. Together they answer a single
question in three granularities: "what does the corpus look like, and what
happens when an agent navigates it?"

This document is for both the operator (how do I bring it up, what do I do
when the port is taken) and the presenter (what should I actually click on,
what flow tells the best story). Use it as a pre-flight checklist before a
demo.

## Quickstart

One command brings everything up:

```bash
make viz-demo
```

Behind the scenes this runs `scripts/viz/run.sh`, which:

1. Ensures `data/viz/sankey.json` exists (runs the builder in `--synthetic`
   mode if the default file is missing or `--synthetic` is passed).
2. Ensures `data/viz/umap.json` exists (same logic; 2000 synthetic points
   if the DB is unreachable).
3. Launches `uvicorn scix.viz.server:app --host 127.0.0.1 --port 8765`.

Then open:

- http://127.0.0.1:8765/viz/sankey.html — V2 temporal communities
- http://127.0.0.1:8765/viz/umap_browser.html — V3 embedding browser
- http://127.0.0.1:8765/viz/agent_trace.html — V4 agent trace overlay

If Postgres is up and you have real community + UMAP data, you can skip
the synthetic hint entirely — just `make viz-demo` and the runner will
reuse any existing `data/viz/*.json` instead of rebuilding.

To rebuild purely synthetic data without serving (useful for CI and
pre-flight checks):

```bash
make viz-demo-build
```

## V2 — Temporal community Sankey

**URL**: http://127.0.0.1:8765/viz/sankey.html

**What it shows**: each vertical column is a decade (1990, 2000, 2010,
2020, ...); each node is a semantic community (Leiden clustering over the
INDUS embedding graph, medium resolution); each link is the flow of a
community across adjacent decades, weighted by its paper count in the
later decade.

**Why it matters for the narrative**: communities are not static labels —
they merge, split, and turn over as fields evolve. The Sankey makes this
visible in one glance.

**Three talking-point flows to look for**:

1. **A persistent core.** Pick a high-flow community (e.g. the largest
   stable band in the centre of the diagram) and trace it across four or
   five decades — the community that *stays* is a sign of a mature
   subfield with consistent terminology.
2. **A late-arriving community.** Find a node that only appears in 2010
   or 2020 and carries significant weight — that is the birth of a new
   field (machine-learning methods in astrophysics is the canonical one
   once real data is loaded).
3. **A fading community.** Find a node in 1990 or 2000 whose outgoing
   link is noticeably thinner than its incoming one — that is a field
   whose vocabulary has been absorbed elsewhere.

With synthetic data the specific stories won't line up with reality, but
the structural hook ("persistent / new / fading") still reads clearly.

## V3 — UMAP embedding browser

**URL**: http://127.0.0.1:8765/viz/umap_browser.html

**What it shows**: a scatter plot of the INDUS 768-d paper embeddings
projected to 2-D with UMAP. Each dot is one paper, coloured by community
id. Clusters are regions of semantic similarity; the gaps between
clusters are the things two abstracts would have to bridge to be
retrieved together.

**Interactions**:

- **Zoom**: scroll / pinch to zoom in on any region — the layout stays
  responsive even at 100k points thanks to deck.gl's GPU-backed
  `ScatterplotLayer`.
- **Pan**: click-drag to move around.
- **Hover**: the tooltip shows the bibcode, the community id, and the
  short title via the `/viz/api/paper/{bibcode}` endpoint.
- **Click**: opens a side-panel with the abstract, authors, and a direct
  link to the ADS bibcode page.
- **Resolution toggle** (if backed by real data across all three Leiden
  resolutions): switch between coarse / medium / fine community colours
  to show the hierarchical structure.

**Presenter hint**: zoom into a dense cluster, hover three points, and
point out that their titles are clearly variations on the same topic.
Then zoom out and point at two adjacent clusters — the gap between them
is what the retriever has to cross, which is why hybrid (dense + sparse +
graph) beats any one signal alone.

## V4 — Agent trace overlay

**URL**: http://127.0.0.1:8765/viz/agent_trace.html

**What it shows**: a live SSE stream of MCP tool calls from the SciX MCP
server. Each tool invocation lights up the papers it touched on top of
the UMAP layout, so you can watch an agent "walk" across the embedding
space in real time. The right-hand rail lists each tool call with its
arguments, duration, and the bibcodes returned.

**How to drive it from another terminal**: open a second shell, point a
Claude client at the SciX MCP server, and issue prompts. Every
`mcp__scix__*` call the agent makes will appear in the trace overlay
within a few hundred milliseconds.

### Scenario 1 — Literature survey

- **Prompt template**:
  > Give me a literature survey of the last ten years of research on
  > exoplanet atmospheric retrieval. Focus on the major methodological
  > shifts and cite the highest-impact paper from each shift.
- **Expected tool sequence**:
  1. `search` with `"exoplanet atmospheric retrieval"` (hybrid dense +
     BM25), year filter `>= 2015`.
  2. `facet_counts` on the returned bibcodes, grouped by keyword /
     community — picks out methodological clusters.
  3. `temporal_evolution` on the dominant community to chart activity.
  4. `citation_graph` (PageRank top-10) per cluster to identify the
     flagship paper.
- **Observable behaviour**: after prompt submission, the UMAP view
  should light up a localized cluster (atmospheric retrieval lives
  fairly tightly in INDUS space), followed by jumps to neighbouring
  clusters as `citation_graph` resolves flagship papers that span the
  subfield.

### Scenario 2 — Related methods discovery

- **Prompt template**:
  > Here is a paper introducing a Gaussian-process-based noise model for
  > radial-velocity time series: {bibcode}. Find me three papers that use
  > *different* methodological approaches to solve the same problem, and
  > explain how each one differs.
- **Expected tool sequence**:
  1. `get_paper` on the anchor bibcode.
  2. `citation_similarity` to find co-cited / bibliographically-coupled
     papers (same problem, may share methods).
  3. `concept_search` for the topic keywords extracted from the
     abstract, but with a *negative* filter excluding the anchor's
     methodology terms.
  4. `read_paper` on the top 3 candidates to confirm methodological
     distinctness before returning.
- **Observable behaviour**: the overlay starts at the anchor bibcode's
  UMAP dot, then spreads to a ring of nearby points via
  `citation_similarity`, and finally jumps to a *different* UMAP region
  entirely when `concept_search` pulls the methodologically-distant
  candidates. The visual jump from the local ring to the remote region
  is the money shot.

### Scenario 3 — Entity disambiguation

- **Prompt template**:
  > I'm reading a paper that mentions "the Kepler mission" but I need to
  > be sure which one — the NASA space telescope or Johannes Kepler's
  > historical work. Ground the reference using the entity graph.
- **Expected tool sequence**:
  1. `entity` with surface form `"Kepler"`.
  2. `entity_context` on each candidate ID to see disambiguating context
     (mission launch year, instrument name, etc.).
  3. `graph_context` to see which neighbouring entities (instruments,
     principal investigators, transit detection methods) link back to
     each candidate.
  4. `search` to confirm usage patterns in the corpus.
- **Observable behaviour**: the trace rail shows the two candidate
  entity IDs side by side with their degree centrality and description;
  the UMAP overlay highlights the distinct paper clouds associated with
  each (one clustered around mission-operations and transit-detection
  communities, the other in a thin history-of-science cluster).

## Tips

- **Port conflict**: pass `--port 9876` to `scripts/viz/run.sh`, or set
  `PORT` on the Makefile call: `make viz-demo PORT=9876` (note: the
  bundled Makefile does not thread `PORT` through — fall back to
  `./scripts/viz/run.sh --port 9876` for custom ports).
- **Force a synthetic rebuild** (for a canned demo, ignore the
  production DB): `./scripts/viz/run.sh --synthetic`. Regenerates both
  JSON files with deterministic random data and then serves.
- **Reuse existing data** (skip the potentially slow UMAP projection
  step if the file is current): `./scripts/viz/run.sh --no-build`.
- **Headless build for CI**: `./scripts/viz/run.sh --build-only
  --synthetic` produces both JSON files and exits 0 without binding a
  port. This is what the pytest smoke test
  (`tests/test_viz_demo_runner.py`) drives.
- **Override the interpreter**: set `VENV_PY` (e.g. when running from a
  worktree that shares the main repo's `.venv`):
  `VENV_PY=/home/ds/projects/scix_experiments/.venv/bin/python
  ./scripts/viz/run.sh`.

## Known limitations

- The agent-trace overlay requires a running SciX MCP server that emits
  instrumentation events to the trace-stream SSE endpoint. In synthetic
  demo mode the overlay page renders but will sit idle until an MCP
  client drives traffic.
- `--synthetic` UMAP output uses 2000 random 768-d vectors — the
  resulting scatter is a fuzzy blob, not a structured embedding. For a
  narrative demo, generate UMAP data from the real DB.
- The Makefile delegates to `./scripts/viz/run.sh` without forwarding
  extra arguments. For ad-hoc flags (alt port, `--no-build`, etc.) call
  the script directly.
