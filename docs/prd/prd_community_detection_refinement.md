# PRD: Community Detection Refinement for SciX Citation Graph

## Problem Statement

Leiden community detection on the SciX citation graph produces 45,615 communities as a hard floor — matching the exact number of disconnected components in the connected subgraph. The calibration targets of 50/500/5,000 are structurally unreachable because Leiden (and any connectivity-respecting algorithm) cannot merge nodes across disconnected components. This makes the `explore_community` MCP tool ineffective: communities are mostly small disconnected clusters, not meaningful research areas.

The root cause is that the 2021-2026 corpus captures only 18% of citation edges (14.6M of 82.2M). The remaining 67.5M edges point to pre-2021 papers, creating a massively fragmented in-corpus topology. Additionally, the UAT relationship loading is broken (0 relationships loaded), so the concept hierarchy is non-functional.

## Goals & Non-Goals

### Goals

- Produce meaningful communities that align with astronomical domain structure
- Assign every paper (including 1.72M isolated nodes) to at least one community signal
- Make communities useful for agent navigation (interpretable labels, reasonable sizes)
- Fix the broken UAT hierarchy loading
- Preserve clean semantics: agents should know what each community signal _means_

### Non-Goals

- Single unified community system (parallel signals is the architecture)
- Real-time incremental updates (batch weekly is fine)
- Overlapping/soft community assignments (v2)

## Architecture: Parallel Community Signals

**Converged decision**: Keep multiple independent community signals rather than building one unified system. Each signal has clean semantics, updates independently, and agents choose the appropriate lens per query.

| Signal                                             | What it means                 | Coverage                      | Computation      |
| -------------------------------------------------- | ----------------------------- | ----------------------------- | ---------------- |
| **Citation community** (Leiden on giant component) | "Papers that cite each other" | ~65% (connected papers)       | igraph, ~15 min  |
| **Semantic community** (SPECTER2 k-means)          | "Papers with similar content" | ~95% (papers with embeddings) | sklearn, minutes |
| **Taxonomic community** (arXiv class + UAT)        | "Papers in the same field"    | ~90%+ (papers with metadata)  | Zero computation |
| **Union coverage**                                 | At least one signal           | **~99%+**                     |                  |

### Why not a unified graph?

The /converge debate resolved this tension decisively:

- Augmenting the citation graph with semantic edges **destroys signal clarity** — the resulting communities are neither purely structural nor purely topical
- The resolution parameter becomes uninterpretable on a hybrid graph (no ground truth for calibration)
- Each augmented edge introduces hyperparameters (similarity threshold, k) with no principled way to set them
- Independent signals update independently — no coupled pipeline where changing one invalidates others
- An agent asking "find related papers" can choose: related by citation lineage, by content similarity, or by field classification

### Schema additions

```sql
ALTER TABLE paper_metrics ADD COLUMN community_semantic_coarse INTEGER;
ALTER TABLE paper_metrics ADD COLUMN community_semantic_medium INTEGER;
ALTER TABLE paper_metrics ADD COLUMN community_semantic_fine INTEGER;
ALTER TABLE paper_metrics ADD COLUMN community_taxonomic TEXT;  -- arxiv_class or UAT top concept
```

## Implementation Plan

### Phase 1: Quick wins (1-2 days)

#### 1a. Giant component extraction for Leiden

- After `filter_isolated_nodes()`, extract the giant component via `graph.components()`
- Run Leiden only on the giant component (likely >90% of connected nodes)
- Assign small-component papers to nearest giant-component community by embedding distance
- Revised targets: coarse ~20, medium ~200, fine ~2K (domain-grounded)
- **Status**: RESOLVED. All positions agreed this is a strict improvement.

#### 1b. arXiv class as taxonomic community

- Populate `community_taxonomic` directly from `arxiv_class` field
- Zero computation, instant coverage for arXiv papers
- **Status**: RESOLVED. Highest feasibility (F=5) and impact (I=5) in brainstorm.

#### 1c. Fix UAT relationship loading

- Debug `load_relationships()` — 2,180 relationships parsed, 0 loaded
- Likely FK constraint or COPY format issue
- Once fixed, enables hierarchical concept queries via `concept_search`
- **Status**: RESOLVED. Production bug discovered during research.

### Phase 2: Semantic communities (3-5 days)

#### 2a. SPECTER2 embedding clustering

- Run k-means (or mini-batch k-means) on ~4.76M SPECTER2 vectors
- Target k=20 (coarse), k=200 (medium), k=2000 (fine)
- Store in `community_semantic_coarse/medium/fine` columns
- Generate labels via top TF-IDF keywords per cluster
- Covers 95% of papers including most isolates
- **Status**: RESOLVED. Highest coverage single signal.

#### 2b. Update MCP tools

- Modify `explore_community` to accept a `signal` parameter: `citation`, `semantic`, `taxonomic`
- Modify `get_paper_metrics` to return all community signals
- Default to `semantic` (highest coverage) when no signal specified

### Phase 3: Refinements (future)

#### 3a. Bibliographic coupling (should-have)

- Build coupling edges from shared out-of-corpus references (67.5M skipped edges)
- Use MinHash for scalable fingerprint comparison
- Add as weighted edges to citation graph before Leiden
- **Would improve citation communities** without mixing signals

#### 3b. UAT concept co-occurrence graph (should-have)

- 2,122-node concept graph, edges weighted by paper co-occurrence
- Run Leiden on this tiny graph, assign papers via keyword mapping
- Provides a domain-expert-curated alternative to data-driven clustering

#### 3c. Consensus clustering (nice-to-have)

- Run multiple independent clusterings, build consensus matrix
- Store as a 4th column: `community_consensus`
- Most robust but most complex

## Design Considerations

**Resolved tension — unified vs. parallel**: The parallel-signals position won on signal clarity, coverage, and maintenance. Key argument: "Citation cluster means papers that cite each other. Semantic cluster means similar content. Taxonomic group means same field. An agent knows what it's querying." Graph augmentation muddles these distinctions.

**Resolved tension — community count targets**: The original 50/500/5K targets were arbitrary. Revised to 20/200/2K based on domain knowledge: 11 UAT top concepts, 22 Infomap-study astrophysics clusters, ~70 UAT level-1 concepts, ~2,122 total UAT concepts. These align with how astronomers structure their field.

**Resolved — giant component extraction**: Unanimous agreement across all positions. The 45K floor is a topology fact, not an algorithm problem. Giant component extraction makes calibration effective.

## Open Questions (Reduced)

1. ~~What is the giant component size?~~ **TO VERIFY**: Run `graph.components()` — expected >90% of connected nodes
2. ~~Why does `load_relationships()` insert 0 rows?~~ **TO FIX**: Phase 1c
3. What fraction of isolated papers have arxiv_class? (Determines taxonomic coverage)
4. Optimal k for SPECTER2 k-means — use silhouette score or domain expert validation?

## Research Provenance

**Diverge (3 agents)**: Algorithms agent proved 45K = component count. Hierarchy agent grounded targets in domain knowledge and found UAT bug. Hybrid agent proposed concept co-occurrence graph and parallel semantic columns.

**Brainstorm (30 ideas)**: Top quick wins: arXiv class (#2, 12/15), reference fingerprint hashing (#4, 12/15). Most creative: agent-driven lazy discovery (#6), PQ fractal communities (#27), random walk with embedding bridges (#28).

**Converge (3 debaters)**: Parallel-signals position won the architecture debate. All positions agreed on giant component extraction and arXiv class as immediate wins. Graph augmentation deferred to Phase 3 refinement (improves citation signal without mixing with semantic signal).
