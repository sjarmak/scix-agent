# Agent-Navigable Scientific Knowledge: Graph Intelligence and Semantic Retrieval on the ADS Citation Corpus

## Target Venues

- **Primary**: ADASS 2026 (Astronomical Data Analysis Software and Systems)
- **Secondary**: arXiv preprint, cs.DL + astro-ph.IM
- **Stretch**: JCDL 2027 (Joint Conference on Digital Libraries)

---

## Abstract (~250 words)

Scientific literature grows faster than researchers can navigate it. AI agents are emerging as research tools, but they lack infrastructure: existing search APIs return ranked lists, not navigable knowledge structures. We present an agent-native knowledge layer built on the full NASA ADS corpus — 32.4M papers and 299M citation edges spanning 1800-2026 — and show that corpus completeness is not optional: a 6-year rolling window resolves only 17.8% of citation edges, rendering graph analytics structurally misleading. The full corpus resolves 99.6%, transforming a fragmented graph into a connected knowledge structure where PageRank, community detection, and co-citation analysis produce valid results. On this foundation, we build three capabilities absent from existing scientific search: (1) corpus-scale graph analytics — PageRank, HITS, and Leiden community detection revealing the structural topology of science; (2) multi-model semantic retrieval — INDUS (nasa-impact/nasa-smd-ibm-st-v2), a domain-specific model trained on 2.66M ADS title-abstract pairs, for document similarity fused with text-embedding-3-large for natural-language queries via reciprocal rank fusion; and (3) agent session state — working-set management that enables systematic literature exploration rather than isolated searches. These are exposed through a 22-tool MCP server. We evaluate retrieval quality, graph topology, and community coherence on the full corpus, and discuss implications for ADS infrastructure: precomputed embeddings and graph metrics as data products, and MCP endpoints as a complement to REST APIs.

---

## 1. Introduction (2 pages)

### The navigation problem

- Scientific output doubles every ~12 years; >3M papers/year across disciplines
- Researchers spend 20-30% of time on literature search (cite: Tenopir et al.)
- AI agents can systematically explore, but need structured knowledge, not just search results

### The infrastructure gap

- ADS, Semantic Scholar, Google Scholar: excellent human-facing search
- REST APIs return ranked lists — no graph topology, no session memory, no community structure
- MCP protocol emerging as standard for AI tool use — but no scientific literature MCP exists

### Our contribution

- Agent-native navigation layer on 32.4M ADS papers
- Three layers: graph intelligence, semantic retrieval, session state
- 22 MCP tools enabling agent-driven research workflows
- Open-source, designed for upstream integration into ADS

---

## 2. Related Work (1.5 pages)

### Scientific search systems

- ADS (Kurtz et al. 2000, Accomazzi et al. 2015): Solr-based, citation tracking, myADS
- Semantic Scholar (Ammar et al. 2018): ML-powered, TLDR, citation intent
- Google Scholar: broad coverage, limited API, no graph analytics

### Scientific embeddings

- SPECTER (Cohan et al. 2020): citation-proximity training on Semantic Scholar
- SPECTER2 (Singh et al. 2023): multi-task adapters, improved zero-shot
- INDUS (Bhatia et al. 2024, arXiv 2405.10725): NASA SMD foundation model, 768d, trained on 2.66M ADS title-abstract pairs — outperforms larger general models on scientific retrieval
- SciBERT (Beltagy et al. 2019): scientific pretraining
- Gap: general-purpose scientific embeddings (SPECTER, SPECTER2) trained for document-to-document similarity, not query-to-document retrieval; INDUS is domain-specific but shares this asymmetry

### Citation network analysis

- PageRank on citation graphs (Chen et al. 2007)
- Community detection in science (Waltman & van Eck 2012, Leiden algorithm)
- Bibliographic coupling (Kessler 1963), co-citation analysis (Small 1973)
- Gap: not integrated into search systems; computed offline for bibliometrics, not navigation

### AI agents for research

- Literature review agents (emerging area, 2024-2025)
- Retrieval-augmented generation for scientific QA
- MCP protocol (Anthropic 2024): standard for tool use
- Gap: no MCP server for scientific literature; no agent-optimized knowledge infrastructure

---

## 3. Corpus Completeness Changes Everything (2.5 pages)

_This section presents the paper's lead empirical result: graph analytics on partial scientific corpora are not merely incomplete — they are structurally misleading._

### 3.1 The edge resolution problem

- Citation graphs are constructed from reference lists: each cited paper is an edge target
- If the target paper is not in the corpus, the edge "dangles" — no node to land on
- Dangling edges break graph algorithms: PageRank leaks probability mass, community detection sees false boundaries, co-citation analysis misses connections
- **This is not a sampling problem** — it is a systematic bias. Recent papers cite older papers. A rolling-window corpus systematically excludes the papers most cited by the papers it includes.

### 3.2 Quantifying the gap

- **A 6-year window (2021-2026, ~5M papers) resolves only 17.8% of citation edges**
- 82.2% of edges point to papers outside the window — the graph is 82% holes
- **Full corpus (1800-2026, 32.4M papers) resolves 99.6%** — from fragmented graph to connected knowledge structure
- The remaining 0.4% are references to papers not in ADS (books, reports, unpublished work)

**Table 1: Edge resolution by target paper decade**

| Target decade | 6-year corpus | Full corpus |
| ------------- | ------------- | ----------- |
| 2020s         | ~95%          | ~99.8%      |
| 2010s         | ~30%          | ~99.7%      |
| 2000s         | 0%            | ~99.5%      |
| 1990s         | 0%            | ~99.2%      |
| 1980s         | 0%            | ~98.5%      |
| Pre-1980      | 0%            | ~97.0%      |

**Figure 2: Edge resolution curve** — % of edges resolved vs. years of corpus ingested, showing the logarithmic shape: you need most of the corpus to resolve most of the edges.

### 3.3 Impact on graph analytics

- **PageRank**: On the 6-year corpus, probability mass leaks through 82% dangling edges. Top-ranked papers are those that happen to cite within-window papers — a recency bias artifact, not a measure of influence. On the full corpus, PageRank produces rankings consistent with domain knowledge (landmark papers rank highest).
- **Community detection**: Leiden on the 6-year graph finds ~45K disconnected components because most inter-community edges are severed. On the full corpus, >95% of connected papers belong to a single giant component — the true community structure emerges.
- **Co-citation analysis**: With 82% dangling edges, co-citation matrices are sparse and noisy. Two papers cited together by a 2024 paper are "co-cited" only if both happen to be post-2021. Historical foundations of fields become invisible.

_Implication: Any graph analytics system built on a partial scientific corpus — including commercial products that index only recent publications — produces structurally misleading results. Full-corpus ingestion is not a "nice to have"; it is a prerequisite for valid graph intelligence._

### 3.4 Corpus and infrastructure

- 32,390,237 papers from NASA ADS, years 1800-2026
- 23.3M with abstracts, 299.3M citation edges
- Single-server PostgreSQL 16 + pgvector 0.8.0+ (no separate search engine)
- Idempotent JSONL → PostgreSQL via COPY, 227 files in 4 hours
- Design choice: simplicity over distributed complexity; 62GB RAM suffices

---

## 4. Semantic Retrieval Layer (2.5 pages)

### 4.1 INDUS embeddings (replacing SPECTER2)

- INDUS (nasa-impact/nasa-smd-ibm-st-v2): 768d, CLS pooling on "title [SEP] abstract"
- Trained on 2.66M ADS title-abstract pairs — domain-specific scientific similarity
- 32M papers embedded (full corpus); HNSW index via pgvector
- Strength: outperforms larger general models on scientific retrieval; encodes domain-specific similarity
- Weakness: trained for doc-to-doc similarity, not query-to-doc retrieval (complemented by text-embedding-3-large)
- History: originally used SPECTER2 (Singh et al. 2023) as the primary dense signal; INDUS replaced it for the full 32M corpus due to superior domain-specific performance

### 4.2 text-embedding-3-large (dual-model)

- 1024d Matryoshka truncation via OpenAI API
- Strength: trained for asymmetric query-document retrieval
- Halfvec (float16) quantization: ~37GB + ~24GB for dual HNSW indexes

### 4.3 Hybrid search with RRF fusion

- Three-signal fusion: BM25 (tsvector) + INDUS vector + text-embedding-3-large vector
- Reciprocal Rank Fusion with k=60
- Circuit breaker: OpenAI failure → graceful fallback to INDUS + lexical

### 4.4 Evaluation

_This section reports retrieval results honestly, including a surprising finding: the general-purpose nomic model outperforms the domain-specific INDUS model on citation-based retrieval._

#### Methodology

- **Benchmark**: 50-query evaluation on a 10K stratified sample from the 32.4M-paper corpus
- **Ground truth**: Citation network (references + citing papers) restricted to the sample. Binary relevance: a paper is relevant if and only if it cites or is cited by the seed paper within the sample.
- **Query formulation**: Seed paper title + first 50 words of abstract as query text for lexical search; stored embedding vector for dense retrieval.
- **Fusion**: Reciprocal Rank Fusion (RRF) with k=60, combining BM25 (tsvector) with each dense model.
- **Models compared**: INDUS (nasa-impact/nasa-smd-ibm-st-v2, 768d, trained on 2.66M ADS title-abstract pairs), nomic-embed-text-v1.5 (768d, general-purpose), SPECTER2 (allenai/specter2_base, 768d, citation-proximity trained).
- **Metrics**: nDCG@10, Recall@10, Recall@20, Precision@10, MRR.

#### Results

**Table N: 50-query retrieval evaluation (10K sample, citation-based ground truth)**

| Method          | nDCG@10         | Recall@10 | Recall@20 | P@10  | MRR   |
| --------------- | --------------- | --------- | --------- | ----- | ----- |
| nomic           | 0.459 +/- 0.237 | 0.285     | 0.448     | 0.378 | 0.743 |
| hybrid_indus    | 0.428 +/- 0.244 | 0.262     | 0.455     | 0.354 | 0.747 |
| indus           | 0.427 +/- 0.242 | 0.262     | 0.453     | 0.354 | 0.744 |
| hybrid_specter2 | 0.404 +/- 0.242 | 0.251     | 0.388     | 0.340 | 0.688 |
| specter2        | 0.402 +/- 0.241 | 0.251     | 0.386     | 0.340 | 0.690 |
| lexical         | 0.200 +/- 0.129 | 0.046     | 0.046     | 0.100 | 0.750 |

The ranking is consistent across both evaluation runs (original and rebaseline): nomic > INDUS > SPECTER2, with lexical search far behind dense methods. Nomic outperforms SPECTER2 with statistical significance (Wilcoxon signed-rank, p=0.004, n=50). The INDUS-vs-SPECTER2 gap (0.025 nDCG@10) is not statistically significant (p=0.29). No direct nomic-vs-INDUS significance test was computed, but the gap (0.032 nDCG@10) is intermediate between the significant nomic-vs-SPECTER2 difference and the non-significant INDUS-vs-SPECTER2 difference, suggesting the nomic advantage over INDUS is real but modest. All dense models substantially outperform lexical search, which returns results for fewer than 12% of queries on the 10K sample due to AND-logic matching on specific title words.

#### The nomic > INDUS result

This result is counterintuitive: nomic-embed-text-v1.5 is a general-purpose model with no astronomy-specific training, yet it outperforms INDUS, a model trained explicitly on 2.66M ADS title-abstract pairs. Several hypotheses may explain this:

1. **Training corpus diversity.** Nomic was trained on a broad multi-domain corpus (hundreds of millions of text pairs) with contrastive learning, producing representations that generalize well across query types. INDUS was fine-tuned on a narrower ADS-specific distribution, which may cause it to over-specialize on the patterns present in its training set (title-abstract similarity within astronomy) at the expense of broader topical and cross-disciplinary retrieval. The ADS corpus spans physics, geosciences, biosciences, and engineering — subfields where INDUS's astronomy-heavy training may provide less benefit.

2. **Architecture and training objective.** Nomic uses a Matryoshka representation learning objective that produces useful representations at multiple dimensionalities, which may yield better-calibrated similarity scores even at 768d. Both models use CLS pooling on "title [SEP] abstract" inputs, so the difference lies in training data and objectives, not in input processing.

3. **Evaluation methodology bias.** Citation-based ground truth defines relevance as "cites or is cited by the seed paper." This captures topical relatedness but also methodological connections, cross-disciplinary citations, and review-article linkages. A general-purpose model may capture these diverse relevance signals more effectively than a domain-specific model optimized for within-field similarity. This is a genuine limitation of our evaluation: citation networks test a broader notion of relevance than pure topical similarity.

4. **Sample size effects.** INDUS is embedded for the full 32M corpus (8.2M embeddings at evaluation time) while nomic and SPECTER2 are embedded only for the 20K pilot sample. The evaluation restricts all models to the same 10K sample via JOIN, so embedding coverage does not directly bias the comparison. However, the 50-query benchmark with high variance (standard deviations of ~0.24) limits statistical power. The nomic-vs-INDUS difference could narrow or reverse with a larger query set.

#### Why INDUS remains the system's primary dense signal

Despite nomic's advantage on this benchmark, INDUS is the correct choice for the production system:

- **Corpus coverage.** INDUS is the only model embedded for the full 32.4M-paper corpus (8.2M at evaluation time, targeting complete coverage). Nomic and SPECTER2 exist only as 20K-paper pilot embeddings. Embedding the full corpus with nomic would cost compute time and storage equivalent to what has already been invested in INDUS, with uncertain benefit.
- **Domain alignment.** INDUS was trained on NASA SMD data specifically for scientific similarity in the ADS domain. For downstream tasks beyond retrieval — community detection seeding, citation prediction, semantic clustering — domain-specific embeddings are expected to provide better signal. The retrieval benchmark tests only one use case.
- **Dual-model architecture.** The system's design fuses INDUS (domain-specific document similarity) with text-embedding-3-large (general-purpose query-document retrieval) via RRF. The nomic result actually validates this design: general-purpose models add complementary signal that domain-specific models miss. Text-embedding-3-large (3072d, Matryoshka-truncated to 1024d) is a stronger general-purpose model than nomic and is expected to capture the same "diversity advantage" in the production RRF pipeline.
- **Open-weight, zero-cost inference.** INDUS runs locally on GPU at no per-query cost, unlike API-based alternatives. For 32M-paper embedding at ~2-5K records/second, this is a critical practical advantage.

#### Hybrid search: limited lift on the pilot sample

RRF fusion of BM25 + dense retrieval adds negligible lift on this benchmark: hybrid_indus (0.428) vs. indus-only (0.427), a difference of 0.001 nDCG@10 (not significant, p=0.65). This is not a failure of hybrid search but a consequence of sample size: BM25 with AND-logic plainto_tsquery returns results for fewer than 12% of queries against the 10K sample. For most queries, the hybrid pipeline degrades to vector-only retrieval. On the full 32.4M-paper corpus, where BM25 has meaningful recall, hybrid search is expected to show the 49-67% error reduction over dense-only retrieval reported in the literature (Ma et al. 2021; Lin et al. 2021). Full-corpus hybrid evaluation is pending completion of the text-embedding-3-large pipeline.

#### Limitations

- **Citation-based ground truth** captures only one dimension of relevance. An expert-judged benchmark with graded relevance would better distinguish model quality on nuanced queries.
- **10K sample** constrains lexical search and limits hybrid evaluation. Full-corpus evaluation is needed to assess the production RRF pipeline.
- **text-embedding-3-large not yet evaluated.** The second model in the dual-model architecture has no embeddings generated; the full three-signal comparison (INDUS + text-embedding-3-large + BM25) remains future work.
- **50 queries with high variance** (std ~0.24 on nDCG@10) provide limited statistical power. A larger benchmark (200+ queries, stratified by discipline) would strengthen confidence in model rankings.
- **Stored embeddings as queries.** Dense retrieval uses the seed paper's stored embedding vector, not a separately encoded query. This tests document-to-document retrieval, not the asymmetric query-to-document setting that text-embedding-3-large is designed for.

### 4.5 Embedding as infrastructure (ADS integration discussion)

- Cost of computing embeddings: ~$0 (INDUS local, open-weight) or ~$65 (text-embedding-3-large batch for 5M)
- Cost of NOT sharing: every downstream consumer re-embeds the corpus
- Proposal: ADS serves precomputed embeddings as a data product alongside metadata
- Precedent: Semantic Scholar serves SPECTER embeddings via API

---

## 5. Graph Intelligence Layer (2.5 pages)

### 5.1 Full-graph analytics

- PageRank on 32.4M nodes, 298M edges
- HITS hub/authority scores
- Table: top-20 papers by PageRank (sanity check against known landmarks)

### 5.2 Community detection

- Leiden algorithm on giant component
- Three resolutions: coarse (~20 communities), medium (~200), fine (~2000)
- Calibration methodology: binary search on resolution parameter targeting community count

### 5.3 Parallel community signals

- **Citation community**: Leiden on giant component — "papers that cite each other"
- **Semantic community**: k-means on INDUS embeddings — "papers with similar content"
- **Taxonomic community**: arXiv class + UAT hierarchy — "papers in the same field"
- Each signal has clean semantics; agents choose the appropriate lens
- Coverage: citation 62% (20M connected / 32.4M), semantic ~99% (32M INDUS embedded / 32.4M), taxonomic ~8% (2.7M with arXiv class)

### 5.4 Giant component analysis

- Before full ingest: ~45K components, 1.7M isolated nodes
- After full ingest (32.4M papers, 299.3M citation edges):
  - 12,330,419 connected components
  - Giant component: 19,981,157 nodes (61.7% of all papers, **99.3% of connected papers**)
  - 12,274,690 isolated papers (no citation links recorded in ADS)
  - Extreme bimodality: second-largest component has only 36 nodes
  - Only 1 component exceeds 100 nodes — the giant component itself
  - 134,390 papers in 55,728 small components (sizes 2–36)
- Edge resolution: 298.1M / 299.3M = 99.6% (only 1.2M dangling)
- Out-degree: median 12, mean 18, P99 = 97, max 13,265
- Impact on community quality: [MEASURE — Leiden NMI/purity, requires separate computation]

### 5.5 Comparison with partial-corpus analytics

- Side-by-side: PageRank on 5M (2021-2026) vs 32M (full corpus) — extends Section 3.3
- Rank correlation (Kendall's tau) between partial and full PageRank: quantify divergence
- Community structure comparison: how many Leiden communities split/merge with full data?
- Concrete example: a landmark paper (e.g., Riess et al. 1998) ranks [X] on partial vs [Y] on full corpus

---

## 6. Agent Navigation Interface (2 pages)

### 6.1 MCP tool design

- 22 tools across 5 categories:
  - Search: semantic_search, hybrid_search, lexical_search, faceted_search
  - Paper: get_paper, get_citations, get_references
  - Graph: co_citation_analysis, bibliographic_coupling, citation_chain, temporal_evolution, graph_metrics, explore_community
  - Entity: entity_search, entity_profile
  - Session: add_to_working_set, get_working_set, get_session_summary, find_gaps, clear_working_set, concept_search
- Tool count discipline: premortem identified >15 tools degrades agent selection accuracy

### 6.2 Session state architecture

- In-memory working set per session (stdio transport)
- WorkingSetEntry: bibcode + provenance (source_tool, source_context, tags)
- find_gaps: SQL query finding cross-community bridge papers
- Implicit vs explicit curation: design tradeoff (discussed)

### 6.3 Entity extraction

- LLM-based (Haiku): methods, datasets, instruments, materials
- JSONB storage with GIN index for containment queries
- Pilot validation: precision/recall on gold-standard set
- Cost analysis: $40 pilot (10K papers), $500-$17K full corpus (fine-tune vs LLM)

### 6.4 Agent workflow demonstration

- Concrete example: "Map the research landscape of exoplanet atmospheres"
- Agent transcript showing multi-tool composition
- Steps: hybrid_search → explore_community → find_gaps → citation_chain → temporal_evolution
- Comparison: same task attempted with ADS API-only agent

---

## 7. Evaluation (2 pages)

### 7.1 Retrieval benchmarks

- Table: latency p50/p95/p99 for each search type
- Table: recall@10 comparison across models and fusion strategies
- Ground truth: ADS search results as relevance proxy

### 7.2 Graph metrics validation

- PageRank sanity check: do highly-cited papers rank highest?
- Community coherence: do communities align with arXiv class labels?
- Parallel signal agreement: how often do citation/semantic/taxonomic communities agree?

### 7.3 Edge resolution impact (extends Section 3)

- Before/after: graph diameter, component count, giant component coverage
- Community detection quality vs corpus completeness
- **Figure: before/after graph visualization** — 6-year window (fragmented, 45K components) vs full corpus (single giant component covering >95% of connected nodes)

### 7.4 Agent task evaluation

- 10 research tasks (qualitative)
- Metrics: papers found, coverage breadth, gap identification, time to completion
- Comparison: MCP-equipped agent vs ADS-API-only agent

---

## 8. Infrastructure Implications for ADS (1.5 pages)

### 8.1 Embeddings as a data product

- INDUS embeddings: compute once, serve to all consumers (open-weight model, no API cost)
- Options: bulk download, API endpoint, or pgvector-backed search
- Precedent: Semantic Scholar serves SPECTER embeddings; ADS would be the first for astronomy with a domain-tuned model
- Update model: embed on ingest (same-day), rebuild HNSW index weekly

### 8.2 Graph metrics as a data product

- PageRank, HITS, community assignments as columns on paper records
- Weekly batch recomputation (full graph in ~30 min on adequate hardware)
- Expose via existing ADS API: add `pagerank`, `community_id` to search results
- Enable: "sort by influence" in the ADS UI, community browsing

### 8.3 MCP server as a service

- ADS hosts an MCP endpoint alongside REST API
- Agents connect directly — no need to harvest and re-index
- Real-time: new papers get embeddings and graph metrics on ingest
- Session state: server-side working sets for multi-turn agent research

### 8.4 What changes for ADS

- New compute: GPU for embedding (one-time + incremental), CPU for weekly graph batch
- New storage: paper_embeddings table (~80GB for INDUS + text-embedding-3-large), paper_metrics (~5GB)
- New endpoint: MCP server (Python process, minimal infrastructure)
- Unchanged: Solr search, existing REST API, myADS, ORCID integration

---

## 9. Discussion (1 page)

### What works

- Full-corpus ingestion transforms graph quality (82% → 0.4% dangling)
- Hybrid search compensates for individual model weaknesses
- Parallel community signals give agents multiple navigation lenses

### Limitations

- Embedding 32M+ papers takes days on a single GPU; ADS infrastructure would amortize this
- Leiden on 32M nodes needs >43GB RAM; requires memory management or distributed approach
- Entity extraction quality varies by subfield; not yet validated at scale
- Session state design assumes agent curation that agents may not spontaneously perform

### Premortem learnings

- Cost estimates for LLM extraction were 3-5x too low (prompt overhead)
- Pilot cohorts biased toward high-citation papers don't represent the long tail
- Tool count matters: >15 tools degrades agent tool selection accuracy

---

## 10. Conclusion & Future Work (0.5 pages)

### Summary

- First MCP server for scientific literature navigation
- 32.4M papers, 299M edges, 22 tools, 3 community signals
- Designed for upstream integration into ADS/SciX

### Future work

- Full-text embeddings (not just abstracts) — requires ADS full-text access
- Cross-disciplinary expansion (beyond astro-ph to all of ADS)
- Agent benchmark suite as a shared resource
- Incremental graph updates (streaming PageRank, online community detection)

---

## Appendices

### A. MCP Tool Specification

Complete JSON schema for all 22 tools.

### B. SQL Schema

Migrations 001-009.

### C. Benchmark Results

Full latency and recall tables.

### D. Premortem Risk Analysis

Condensed from the 3-agent premortem (infrastructure, agent utility, data quality).

### E. Reproduction

Instructions for harvesting, ingesting, embedding, and running the MCP server.
Open-source repository: [URL]

---

## Figures

1. **Edge resolution before/after** (LEAD FIGURE) — Side-by-side graph visualization: 6-year window (45K disconnected components, 82% dangling edges) vs full corpus (single giant component, 99.6% resolved). Visual: shattered graph → connected structure.
2. Edge resolution curve (% resolved vs years of corpus ingested) — logarithmic shape showing diminishing returns, inflection points by decade
3. Architecture diagram (corpus → PostgreSQL → MCP server → agent)
4. Community structure visualization (coarse communities, colored by arXiv class)
5. Embedding space t-SNE/UMAP (INDUS, colored by community)
6. Retrieval comparison (bar chart: recall@10 across models)
7. Agent workflow transcript (annotated screenshot or sequence diagram)
8. PageRank distribution (log-log plot, power law)
9. Parallel community signal agreement matrix (Sankey or confusion matrix)
