# PRD: Full-Text Body Applications for Agent-Navigable Scientific Knowledge

## Problem Statement

The SciX project has ~6M full-text paper bodies (median ~65K chars) now stored in a dedicated `body` column, but no applications consume them. The original PRD focused on title+abstract retrieval and citation graph navigation. Full text unlocks capabilities impossible with abstracts alone: understanding WHY papers cite each other, extracting methodology details, surfacing negative results, and enabling deep reading by AI agents. However, only 19% of 32M papers have full text, creating coverage bias that must be managed.

This PRD defines a phased approach that uses full text primarily to **enrich the existing citation graph and agent tools** rather than building a separate full-text retrieval system, maximizing value while respecting the coverage constraint.

## Goals & Non-Goals

### Goals

- Enrich 299M citation edges with semantic context and intent labels
- Enable agents to deeply read papers (section-level access, within-paper search)
- Extract structured entities from full text using local GPU (near-zero marginal cost)
- Quantify and surface coverage bias so agents/users understand limitations

### Non-Goals

- Building a standalone full-text search engine (BM25 on body is a byproduct, not the goal)
- Chunk-level vector embeddings for all 6M papers (defer until section-level BM25 proves insufficient)
- Full knowledge graph with reasoning (entity extraction yes, complex inference no)
- Processing papers without body text differently than today (graceful degradation, not separate pipelines)

## Requirements

### Must-Have

- **Citation context extraction pipeline**
  - Extract ~250-word context windows around inline citation markers in body text
  - Resolve numbered `[N]` markers to target bibcodes via the `reference[]` array
  - Store in `citation_contexts` table: (source_bibcode, target_bibcode, context_text, char_offset)
  - Acceptance: `SELECT count(*) FROM citation_contexts` returns > 50M rows after processing 6M papers; spot-check 100 random contexts against source papers shows >95% correct bibcode resolution

- **Citation intent classification**
  - Classify each extracted context as: background, method, result_comparison (3-class)
  - Use SciBERT/SPECTER2 fine-tuned on SciCite (11K labeled examples) or LLM classification
  - Add `intent` column to `citation_contexts`
  - Acceptance: Macro-F1 > 0.80 on held-out test set of 500 manually labeled astronomy citation contexts

- **Section-aware text parsing utility**
  - Regex-based section splitter for astronomy papers (Introduction, Methods/Observations, Results, Discussion, Conclusions)
  - Returns list of (section_name, start_char, end_char, text) tuples
  - Acceptance: On a sample of 200 random body texts, correctly identifies section boundaries for >80% of papers that have standard section headers

- **Coverage bias analysis**
  - Compare subfield distribution (arxiv_class), year distribution, citation count distribution, and journal distribution between full-text and abstract-only papers
  - Store results as a reference document and surface in agent tool responses
  - Acceptance: Analysis report exists at `docs/full_text_coverage_analysis.md` with charts showing distribution comparisons

### Should-Have

- **`read_paper_section` MCP tool**
  - Parameters: bibcode, section (intro|methods|results|discussion|conclusions|full)
  - Returns section text with `has_body: true/false` flag, falls back to abstract
  - Pagination support for long sections (char_offset, limit)
  - Acceptance: Tool returns correct section content for 10 test bibcodes with known body text; returns abstract with `has_body: false` for papers without body

- **`search_within_paper` MCP tool**
  - Parameters: bibcode, query string
  - Returns matching passages with context (200 chars before/after) using `ts_headline`
  - Acceptance: Searching for "spectral energy distribution" in a paper known to contain that phrase returns the relevant passage with context

- **`get_citation_context` MCP tool**
  - Parameters: source_bibcode, target_bibcode
  - Returns the context text and intent label for why source cites target
  - Acceptance: Returns correct context for 10 known citation pairs; returns null gracefully for pairs without context

- **Astronomical NER pipeline (local GPU)**
  - Fine-tune astroBERT or GLiNER on WIESP2022-NER dataset (from ADS team, on HuggingFace)
  - Extract entities from 6M full-text papers: celestial objects, instruments, surveys, methods
  - Store in normalized `kg_entities` table with canonical names and external IDs (SIMBAD/NED cross-refs)
  - Acceptance: Process 6M papers in <14 days on RTX 5090; entity extraction F1 > 0.75 on WIESP2022-NER test set

- **Cross-encoder reranking with full text**
  - Implement reranker in existing hybrid search pipeline (already has `reranker` parameter stub)
  - For papers with body text, use most relevant section + abstract as reranking input
  - Acceptance: nDCG@10 improves by >5% on a benchmark of 50 astronomy queries compared to current RRF-only pipeline

### Nice-to-Have

- **Negative results index**
  - Identify null/negative results in body text using section role + hedging language patterns
  - Store as extraction type in `extractions` table
  - Acceptance: On 100 randomly sampled papers with body text, recall >60% of negative results identified by human annotator

- **Claim contradiction detector**
  - Extract quantitative claims with uncertainties (e.g., "H0 = 73.4 +/- 1.1 km/s/Mpc")
  - Flag statistically inconsistent measurements across papers in the same subfield
  - Acceptance: Correctly identifies the Hubble tension papers as contradictory; processes 1000 cosmology papers and surfaces >10 genuine measurement disagreements

- **Body text as embedding training signal**
  - Use abstract-body pairs as contrastive learning data to fine-tune SPECTER2
  - Acceptance: Fine-tuned model shows >3% improvement on SciRepEval astronomy subset over base SPECTER2

- **Object-centric knowledge aggregator**
  - For named astronomical objects, aggregate all mentions across corpus into unified profiles
  - Acceptance: Profile for "M87" includes properties from >50 papers spanning multiple measurement types

- **Implicit knowledge gap detector**
  - Extract "not well understood", "further work needed" patterns from body text
  - Aggregate into a per-subfield map of acknowledged unknowns
  - Acceptance: For 5 major astronomy subfields, produces a ranked list of top-10 open questions supported by >5 papers each

## Design Considerations

### Coverage Bias vs. Value

19% coverage skews toward arXiv-deposited, recent, high-impact papers. This is a feature for agent-facing tools (these are the papers agents query most) but a bug for corpus-wide analysis (selection bias). Resolution: use full text to enrich the universal citation graph (100% coverage) rather than building full-text-only features.

### Graph Enrichment vs. Separate Retrieval

The citation graph already encodes what introduction/related-work sections contain in prose form. Full text should primarily annotate graph edges with semantics (citation intent, context), not duplicate the graph as a text retrieval index. Section-level access and within-paper search are complementary capabilities, not replacements for graph navigation.

### Cost Tiers for Extraction

- **Tier 1 (free)**: Local GPU NER with GLiNER/astroBERT — all 6M papers
- **Tier 2 (cheap)**: Local Qwen3-8B for relation extraction — top 500K papers by PageRank
- **Tier 3 (targeted)**: Claude Batch API for claim/finding extraction — top 50K papers
  Total estimated cost: <$5K for all tiers.

### Chunking: Start Simple

Recursive 512-token splitting beat semantic chunking in 2026 benchmarks (69% vs 54% accuracy). Start with section-level BM25 for within-paper search. Add chunk embeddings only if measured retrieval quality is insufficient.

## Open Questions

- What is the actual quality distribution of body text (clean text vs. LaTeX artifacts vs. OCR noise)?
- What fraction of papers use `[N]` numbered citations vs. author-year format?
- How well do astronomy papers follow standard IMRaD section structure?
- What are licensing constraints on processed full text from publisher sources (vs. arXiv)?
- Should the knowledge graph support temporal versioning of claims?

## Research Provenance

### Divergent Research (5 independent agents)

1. **RAG & Retrieval**: Section-aware chunking, cross-encoder reranking, two-phase retrieval. Key finding: chunking strategy matters as much as embedding model choice.
2. **Entity Extraction**: GLiNER efficiency, WIESP2022-NER dataset, 3-tier cost model. Key finding: ADS team's own NER dataset available on HuggingFace.
3. **Citation Context**: Regex extraction, 250-word windows, SciCite intent classification. Key finding: ADS reference[] arrays make citation resolution trivial.
4. **Agent Capabilities**: Section reading tools, cross-paper comparison, evidence chains. Key finding: existing schema is 90% ready, ~200 lines of Python needed.
5. **Contrarian**: Coverage bias, compute costs, graph-first alternative. Key finding: 19% is not a random sample, and the citation graph may already encode what full-text intros contain.

### Brainstorm (30 ideas, top 7 by score)

| Score | Idea                                   | Why it matters                                     |
| ----- | -------------------------------------- | -------------------------------------------------- |
| 14/15 | Negative results index                 | Surfaces systematically hidden information         |
| 14/15 | Implicit knowledge gap detector        | Maps the frontier of acknowledged unknowns         |
| 13/15 | Citation intent fingerprinting         | WHY papers cite, not just THAT                     |
| 13/15 | Assumption dependency chains           | Traces which results break under which assumptions |
| 13/15 | Semantic section role classifier       | Universal paper skeleton by rhetorical role        |
| 13/15 | Temporal knowledge decay detector      | Finds claims that may be superseded                |
| 13/15 | Body text as embedding training signal | Abstract-body pairs for contrastive learning       |

### Key Convergence

All 5 research agents agreed: **use full text to enrich the citation graph** (citation context + intent) rather than building standalone full-text search. This aligns graph-first investment (100% coverage) with full-text capabilities (19% coverage).

### Key Divergence

The contrarian agent argues graph intelligence should be prioritized over full-text processing. The agent-tools agent argues full text corrects abstract-level bias. Both are right — the phased approach in this PRD resolves the tension by using full text as a graph enrichment tool first.
