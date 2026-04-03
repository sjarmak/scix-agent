# PRD: Knowledge Enrichment for SciX Agent-Navigable Layer

## Problem Statement

The SciX corpus (5M papers, 2021-2026) has a functional retrieval layer: semantic search (SPECTER2), keyword search, citation graph traversal, community detection, and UAT concept search across 15 MCP tools. However, the corpus lacks structured entity extraction (methods, datasets, instruments) and the single-model embedding architecture has a fundamental gap: SPECTER2 was trained for document-to-document similarity (citation proximity), not for the asymmetric query-to-document retrieval that agents actually perform. Additionally, agents have no session memory -- every tool call is stateless, so agents cannot build and reason over a working set of papers across a research session.

This PRD addresses bead 9jf.4 (Phase 3: Knowledge enrichment) in the parent epic 9jf (Agent-Navigable Knowledge Layer). It unifies three research perspectives: NLP/entity extraction architecture, embedding model evaluation, and agent session state.

## Tensions Identified and Resolved

### Tension 1: Pilot scope overlap (entity extraction vs embedding both want 10K papers)

Both entity extraction and embedding evaluation target the same 10K high-cited pilot set. This is a feature, not a conflict -- using the same papers enables cross-validation (entities extracted from a paper should correlate with its embedding neighborhood). **Decision**: Use a single 10K pilot cohort selected by citation count, shared across both workstreams.

### Tension 2: Schema storage for multi-dimensional embeddings

The current `paper_embeddings.embedding` column is typed `vector(768)` (SPECTER2 fixed dimension). The embedding perspective recommends `text-embedding-3-large` at 1024d, which cannot fit. Options: (a) ALTER to untyped `vector`, (b) separate table, (c) use the existing composite PK `(bibcode, model_name)` with an untyped column. **Decision**: ALTER the column to untyped `vector` (no dimension constraint). The composite PK already separates models. This avoids table proliferation and keeps `hybrid_search` modifications minimal. The HNSW indexes are already per-model (migration 004), so dimension mismatch between models is not an indexing problem.

### Tension 3: Memory budget for dual HNSW indexes

Two full-precision HNSW indexes (768d SPECTER2 + 1024d OpenAI) would consume ~43.7GB of 62.4GB RAM. **Decision**: Use halfvec (float16) quantization for both indexes. pgvector 0.8.0+ (already deployed via migration 005) supports halfvec natively. Combined memory: ~21.9GB (35% RAM), leaving comfortable headroom. Quantization has negligible recall impact (<1% at typical HNSW parameters).

### Tension 4: Entity storage -- single row vs multi-row per paper

The NLP perspective proposes one row per paper with a JSONB payload containing all entity types. The existing `extractions` table uses multi-row design (one row per `extraction_type`). **Decision**: Use the existing multi-row design with `extraction_type = 'entities'` and `extraction_version = 'haiku-v1'`. This aligns with the existing schema, supports idempotent re-extraction, and the JSONB payload still contains all entity types in a single document. Add the missing unique constraint for idempotent upserts.

### Tension 5: Session state storage -- PostgreSQL vs in-memory

The session state perspective correctly identifies that in-memory Python dicts are sufficient for the single-process stdio MCP transport. **Decision**: In-memory with `session_id` keying for future HTTP transport. No database tables, no external dependencies. The `find_gaps` tool performs a server-side SQL query using the in-memory working set as input, bridging the two worlds cleanly.

### Tension 6: Phasing -- parallel vs sequential workstreams

Entity extraction requires Anthropic API calls, embedding evaluation requires OpenAI API calls, and session state is pure Python. All three are independent. **Decision**: Phase 1 (entity extraction pilot) and Phase 2 (embedding pilot) can run in parallel since they use different APIs and touch different tables. Phase 3 (session state) has zero database dependencies and can be implemented at any time. Phase 4 (scale-up) depends on pilot validation results from Phases 1-2.

## Goals & Non-Goals

### Goals

- Extract structured entities (methods, datasets, instruments, materials) from papers via LLM
- Add a second embedding model (text-embedding-3-large) optimized for query-to-document retrieval
- Give agents session memory via an in-memory working set with curation tools
- Validate all three capabilities on a 10K pilot before committing to full-corpus scale
- Keep total pilot cost under $50

### Non-Goals

- Fine-tuning a custom extraction model (Phase 4 decision, contingent on pilot results)
- Replacing SPECTER2 (complementary models, not competing)
- Persistent session state across server restarts (ephemeral is correct for stdio transport)
- Real-time extraction (batch pipeline is sufficient)
- Entity resolution / knowledge graph construction (future phase)

## Architecture Overview

```
                    Agent (Claude)
                         |
                    MCP Protocol
                         |
              +----------+----------+
              |    MCP Server       |
              |  (scix.mcp_server)  |
              |                     |
              |  Session State      |  <-- NEW: in-memory working set
              |  (Python dicts)     |
              +----+-------+-------+
                   |       |
          +--------+       +--------+
          |                         |
   PostgreSQL + pgvector      Anthropic/OpenAI
          |                    Batch APIs
   +------+------+                  |
   |      |      |           +------+------+
papers  paper_  extrac-     Entity    Embedding
        embed-  tions       Extraction  Pipeline
        dings               (Haiku)   (text-emb-3)
```

**Data flow for entity extraction**: Papers (title + abstract) --> Anthropic Messages Batches API (Haiku) --> structured JSONB --> `extractions` table --> `entity_search` / `entity_profile` MCP tools.

**Data flow for dual-model search**: Agent query --> embed with SPECTER2 + embed with OpenAI API --> two parallel vector searches --> RRF fusion --> ranked results. The existing `hybrid_search()` and `rrf_fuse()` functions extend naturally.

**Data flow for session state**: Agent calls `add_to_working_set` --> Python dict stores entry with metadata --> `find_gaps` reads working set bibcodes, runs SQL to find cross-community bridge papers --> returns gap-filling suggestions.

## Phase 1: Entity Extraction Pilot (10K papers)

### 1a. Schema preparation

Add unique constraint and GIN index to existing `extractions` table:

```sql
-- migration 009_knowledge_enrichment.sql
ALTER TABLE extractions
    ADD CONSTRAINT uq_extractions_bibcode_type_version
    UNIQUE (bibcode, extraction_type, extraction_version);

CREATE INDEX IF NOT EXISTS idx_extractions_payload_gin
    ON extractions USING GIN (payload jsonb_path_ops);
```

### 1b. Pilot cohort selection

Select 10K papers by citation count, ensuring subfield diversity:

```sql
SELECT bibcode, title, abstract
FROM papers
WHERE abstract IS NOT NULL
  AND length(abstract) > 100
ORDER BY citation_count DESC NULLS LAST
LIMIT 10000;
```

Verify distribution across arxiv_class to ensure representation of at least 4 astronomy subfields (astro-ph.GA, astro-ph.SR, astro-ph.CO, astro-ph.EP).

### 1c. Extraction pipeline

- **Model**: Claude Haiku via Anthropic Messages Batches API (50% discount)
- **Prompt**: Few-shot with 3-5 examples across subfields, using tool-use/structured output mode
- **Entity types**: methods, datasets, instruments, materials
- **Output schema per paper**:
  ```json
  {
    "methods": [
      {
        "name": "MCMC",
        "canonical": "Markov chain Monte Carlo",
        "confidence": 0.95,
        "span": "..."
      }
    ],
    "datasets": [
      {
        "name": "SDSS DR17",
        "canonical": "Sloan Digital Sky Survey Data Release 17",
        "confidence": 0.92,
        "span": "..."
      }
    ],
    "instruments": [
      {
        "name": "JWST NIRCam",
        "canonical": "James Webb Space Telescope Near Infrared Camera",
        "confidence": 0.98,
        "span": "..."
      }
    ],
    "materials": [
      {
        "name": "silicate dust",
        "canonical": "silicate dust grains",
        "confidence": 0.88,
        "span": "..."
      }
    ]
  }
  ```
- **Batch size**: 10K papers per batch request
- **Resumability**: Follow existing `IngestLog` pattern -- track progress by bibcode batch
- **Estimated cost**: $17-34 (10K papers, ~500 tokens input + ~200 tokens output each)

### 1d. Validation (4-tier)

1. **Structural validation**: Automated check that every response parses as valid JSON matching the schema, all required fields present, confidence in [0,1]
2. **Gold-standard eval**: Manually annotate 200 papers (50 per subfield), compute precision/recall/F1 per entity type. Target: P>0.8, R>0.7
3. **Cross-validation**: Compare extracted entities against `papers.keywords`, UAT concept assignments, and community labels. Entity mentions should correlate with domain metadata
4. **Temporal consistency**: Same instrument/dataset mentioned across years should get the same canonical name

## Phase 2: Embedding Model Evaluation (10K papers)

### 2a. Schema preparation (same migration 009)

```sql
-- Remove fixed dimension constraint to support multiple embedding sizes
ALTER TABLE paper_embeddings
    ALTER COLUMN embedding TYPE vector USING embedding::vector;
```

### 2b. Pilot embedding

- **Model**: OpenAI `text-embedding-3-large` at 1024 dimensions (Matryoshka truncation)
- **API**: OpenAI Batch API (50% discount)
- **Input**: Same 10K pilot cohort (title + abstract concatenated)
- **Storage**: Insert into `paper_embeddings` with `model_name = 'text-embedding-3-large'`
- **Estimated cost**: $0.13 (10K papers at $0.13/1M tokens)

### 2c. HNSW index for pilot

```sql
CREATE INDEX IF NOT EXISTS idx_embed_hnsw_openai ON paper_embeddings
    USING hnsw ((embedding::halfvec(1024)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200)
    WHERE model_name = 'text-embedding-3-large';
```

Note: Uses halfvec quantization from the start to validate the production memory profile.

### 2d. Evaluation benchmark

Build a 30-50 query test set with known-relevant papers:

| Query type        | Example                                                 | Expected signal       |
| ----------------- | ------------------------------------------------------- | --------------------- |
| Topical           | "dark energy constraints from CMB"                      | OpenAI should excel   |
| Method-specific   | "Bayesian hierarchical modeling in exoplanet detection" | OpenAI should excel   |
| Citation-proximal | "Papers similar to [specific bibcode]"                  | SPECTER2 should excel |
| Cross-domain      | "Machine learning in stellar spectroscopy"              | Fusion should excel   |

Metrics: Recall@10, NDCG@10, MRR for each model individually and RRF fusion. The fusion must beat the better single model on at least 60% of queries to justify the memory cost.

### 2e. Dual-model search integration

Extend `hybrid_search()` in `search.py` to perform two vector searches and fuse:

```python
# Pseudocode -- ~10 lines added to existing hybrid_search
specter_results = vector_search(conn, specter_embedding, model_name="specter2", ...)
openai_results = vector_search(conn, openai_embedding, model_name="text-embedding-3-large", ...)
fused = rrf_fuse([specter_results, openai_results, lexical_results], k=RRF_K)
```

The OpenAI embedding is obtained via API call (cached) rather than local model inference.

## Phase 3: Agent Session State (Working Set Tools)

### 3a. Session state module

New file: `src/scix/session.py`

```python
@dataclass
class WorkingSetEntry:
    bibcode: str
    added_at: str          # ISO timestamp
    source_tool: str       # which MCP tool surfaced this paper
    source_context: str    # query or bibcode that led here
    relevance_hint: str    # agent's note on why this is relevant
    tags: list[str]        # agent-assigned tags

class SessionState:
    """In-memory session state keyed by session_id."""
    working_set: dict[str, WorkingSetEntry]   # bibcode -> entry
    seen_papers: dict[str, str]               # bibcode -> first_seen_at (auto-tracked)
```

Key design decisions:

- `session_id` defaults to `"_default"` (single-process stdio), ready for HTTP transport
- `working_set` is explicitly curated (only `add_to_working_set` adds entries)
- `seen_papers` is auto-tracked (all tool results annotated with `in_working_set: true/false`)
- Maximum working set size: 1000 entries (soft limit with warning)

### 3b. MCP tools (5 new tools)

| Tool                  | Description                                                                     | Complexity |
| --------------------- | ------------------------------------------------------------------------------- | ---------- |
| `add_to_working_set`  | Add one or more bibcodes with metadata                                          | Low        |
| `get_working_set`     | Return current working set with optional tag filter                             | Low        |
| `get_session_summary` | Statistics: count, community distribution, temporal spread, tag counts          | Low        |
| `find_gaps`           | SQL query finding papers that bridge between communities the agent has explored | Medium     |
| `clear_working_set`   | Reset working set (with optional tag-scoped clearing)                           | Low        |

### 3c. `find_gaps` implementation

This is the highest-value tool. Given the working set's community assignments, it finds papers that:

1. Belong to a community NOT yet in the working set
2. Have citation edges TO papers in communities the agent HAS explored
3. Rank by PageRank (authority in the bridging community)

```sql
-- Pseudocode: find bridge papers between explored communities
WITH explored AS (
    SELECT DISTINCT community_id_coarse FROM paper_metrics
    WHERE bibcode = ANY(%s)  -- working set bibcodes
),
candidates AS (
    SELECT pm.bibcode, pm.community_id_coarse, pm.pagerank
    FROM paper_metrics pm
    JOIN citation_edges ce ON ce.source_bibcode = pm.bibcode
    WHERE ce.target_bibcode = ANY(%s)  -- working set bibcodes
      AND pm.community_id_coarse NOT IN (SELECT community_id_coarse FROM explored)
)
SELECT bibcode, community_id_coarse, pagerank
FROM candidates
ORDER BY pagerank DESC
LIMIT %s;
```

### 3d. Tool result annotation

All existing search tools that return papers will annotate results with:

```json
{ "bibcode": "...", "title": "...", "in_working_set": true }
```

This requires a lightweight check against the in-memory working set dict in `_dispatch_tool`. No database queries added.

## Phase 4: Scale-Up (contingent on pilot validation)

### 4a. Entity extraction to 5M papers

**Decision point**: Based on gold-standard eval results from Phase 1d:

- If P>0.85, R>0.75 and agents actively use entities: fine-tune ModernBERT/DeBERTa on Haiku's silver labels. Cost: ~$500 compute for fine-tuning, inference is local.
- If P>0.85, R>0.75 but extraction quality varies by subfield: continue with Haiku at full scale. Cost: ~$8,500 via Batches API.
- If P<0.80: iterate on prompt engineering, expand few-shot examples, re-run pilot.

### 4b. Full embedding batch (5M papers)

- **Cost**: ~$65 via OpenAI Batch API (5M papers)
- **Timeline**: ~24 hours batch processing
- **Index build**: HNSW on halfvec(1024) -- estimated build time ~4 hours, memory ~14.6GB

### 4c. HNSW index migration to halfvec

At scale, rebuild SPECTER2 index with halfvec too:

```sql
-- Drop old full-precision index
DROP INDEX IF EXISTS idx_embed_hnsw;

-- Rebuild both as halfvec
CREATE INDEX idx_embed_hnsw_specter2 ON paper_embeddings
    USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200)
    WHERE model_name = 'specter2';

CREATE INDEX idx_embed_hnsw_openai ON paper_embeddings
    USING hnsw ((embedding::halfvec(1024)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200)
    WHERE model_name = 'text-embedding-3-large';
```

Combined memory at full scale: ~21.9GB (35% of 62.4GB RAM).

## Schema Changes (Consolidated Migration 009)

```sql
-- 009_knowledge_enrichment.sql
-- Entity extraction support + multi-dimensional embedding support

BEGIN;

-- 1. Unique constraint for idempotent entity extraction upserts
ALTER TABLE extractions
    ADD CONSTRAINT uq_extractions_bibcode_type_version
    UNIQUE (bibcode, extraction_type, extraction_version);

-- 2. GIN index for JSONB entity queries (e.g., find papers mentioning "JWST")
CREATE INDEX IF NOT EXISTS idx_extractions_payload_gin
    ON extractions USING GIN (payload jsonb_path_ops);

-- 3. Remove fixed dimension constraint on embeddings
--    Allows storing 768d (SPECTER2) and 1024d (OpenAI) in same table
--    Composite PK (bibcode, model_name) already separates models
ALTER TABLE paper_embeddings
    ALTER COLUMN embedding TYPE vector USING embedding::vector;

-- 4. Halfvec HNSW index for OpenAI embeddings (pilot, 10K papers)
--    Uses float16 quantization to halve memory footprint
CREATE INDEX IF NOT EXISTS idx_embed_hnsw_openai ON paper_embeddings
    USING hnsw ((embedding::halfvec(1024)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200)
    WHERE model_name = 'text-embedding-3-large';

COMMIT;
```

## MCP Tool Additions (7 new tools)

### Entity tools (2)

| Tool             | Description                              | Input                                                                      |
| ---------------- | ---------------------------------------- | -------------------------------------------------------------------------- |
| `entity_search`  | Find papers mentioning a specific entity | `entity_type` (method/dataset/instrument/material), `entity_name`, `limit` |
| `entity_profile` | Get all extracted entities for a paper   | `bibcode`                                                                  |

`entity_search` uses the GIN index with `jsonb_path_ops`:

```sql
SELECT e.bibcode, e.payload, p.title, p.first_author, p.year
FROM extractions e
JOIN papers p ON p.bibcode = e.bibcode
WHERE e.extraction_type = 'entities'
  AND e.payload @> %s  -- e.g., '{"instruments": [{"canonical": "JWST NIRCam"}]}'
ORDER BY p.citation_count DESC
LIMIT %s;
```

### Session tools (5)

| Tool                  | Description                                     | Input                                                                 |
| --------------------- | ----------------------------------------------- | --------------------------------------------------------------------- |
| `add_to_working_set`  | Add papers to working set with metadata         | `bibcodes`, `source_tool`, `source_context`, `relevance_hint`, `tags` |
| `get_working_set`     | Return current working set                      | `tag_filter` (optional)                                               |
| `get_session_summary` | Session statistics and coverage analysis        | (none)                                                                |
| `find_gaps`           | Find bridge papers between explored communities | `limit`, `resolution`                                                 |
| `clear_working_set`   | Clear working set (optionally by tag)           | `tag` (optional)                                                      |

Total tool count after implementation: 22 (15 existing + 7 new).

## Validation & Benchmarks

### Entity extraction quality

| Metric                      | Target   | Measurement                                        |
| --------------------------- | -------- | -------------------------------------------------- |
| Structural validity         | 100%     | Automated JSON schema validation                   |
| Precision (per entity type) | >0.80    | 200-paper gold set, 4 subfields                    |
| Recall (per entity type)    | >0.70    | 200-paper gold set, 4 subfields                    |
| Canonical consistency       | >0.90    | Same entity gets same canonical name across papers |
| Keyword correlation         | Positive | Extracted entities overlap with paper keywords     |

### Embedding quality

| Metric                        | Target                  | Measurement                    |
| ----------------------------- | ----------------------- | ------------------------------ |
| Recall@10 (OpenAI, query-doc) | >0.70                   | 30-50 query test set           |
| NDCG@10 (fusion)              | > max(SPECTER2, OpenAI) | Same test set                  |
| Fusion win rate               | >60% of queries         | Fusion beats best single model |
| Memory (dual halfvec HNSW)    | <25GB                   | pg_relation_size()             |

### Session state

| Metric                  | Target      | Measurement                                                          |
| ----------------------- | ----------- | -------------------------------------------------------------------- |
| Working set add latency | <1ms        | In-memory dict, no DB                                                |
| find_gaps latency       | <500ms      | SQL with working set of 100 papers                                   |
| Agent task completion   | Qualitative | Agent can build, review, and use a working set in a research session |

## Risk Analysis (Premortem-Annotated)

> This section was generated by a 3-agent premortem analysis (infrastructure, agent utility, data quality) that projected forward 6 months and identified realistic failure modes. Risks are grouped by severity tier.

### CRITICAL Risks

| #   | Risk                                                                                                                                                                                                                                                      | Likelihood | Mitigation                                                                                                                                                                                                                                                                            |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1  | **ALTER COLUMN TYPE on 5M-row `paper_embeddings` acquires ACCESS EXCLUSIVE lock for 4-6 hours**, blocking all search queries. Existing HNSW indexes must be rebuilt after the type change.                                                                | HIGH       | Never ALTER in-place. Create `paper_embeddings_v2` with untyped `vector`, backfill in batches, rename atomically. Build new HNSW indexes before the rename so the swap is instant.                                                                                                    |
| C2  | **Agents never use `entity_search`** because entity types (methods/datasets/instruments/materials) don't match how agents decompose queries. Agents use `hybrid_search` with full natural-language queries and never decompose into typed entity lookups. | HIGH       | Run a 2-week observational study of real agent transcripts before building extraction. Log all queries to `hybrid_search`. If agents don't decompose queries, fold entity data into existing tools (e.g., `include_entities` param on `hybrid_search`) instead of creating new tools. |
| C3  | **7 new MCP tools (22 total) overwhelm agent tool selection.** Claude's tool selection accuracy degrades with >15 tools. Agents make suboptimal choices, calling `entity_search` when `hybrid_search` suffices.                                           | HIGH       | Strict tool budget: max 12-15 tools. Fold entity data into existing tools (`hybrid_search`, `get_paper`). Collapse 5 session tools into 2 (`get_session_context`, `find_related_unexplored`). Net new tools: 0-2, not 7.                                                              |
| C4  | **HNSW halfvec index build OOM-kills PostgreSQL at 5M scale.** Build-time memory (~28GB for both indexes) + shared_buffers (~15.6GB) exceeds 62.4GB RAM.                                                                                                  | HIGH       | Build indexes sequentially. Set `maintenance_work_mem = 4GB`. Use `CREATE INDEX CONCURRENTLY`. Test on 1M vectors first.                                                                                                                                                              |
| C5  | **LLM extraction hallucinates entities** — 8% of instrument extractions reference instruments not used in the paper (e.g., JWST attributed to pre-2022 papers).                                                                                           | HIGH       | Validate instrument mentions against a known-instrument registry (NASA mission list, ESO database). Flag extractions where instrument launch date > paper publication date.                                                                                                           |
| C6  | **Fine-tuned model amplifies Haiku's systematic errors.** ModernBERT trained on silver labels learns to hallucinate the same instruments Haiku did, at higher rates. 0.91 F1 against teacher outputs is meaningless.                                      | MEDIUM     | Require fine-tuned model passes the same gold-standard eval as Haiku (human annotations, not teacher outputs). Budget $2K for continued Haiku extraction as fallback.                                                                                                                 |

### HIGH Risks

| #   | Risk                                                                                                                                                                                                          | Likelihood | Mitigation                                                                                                                                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H1  | **Cost estimate 3-5x too low.** Actual input per request ~2,550 tokens (not 500) due to few-shot examples, system prompt, and tool-use overhead. Pilot: ~$87, not $17-34. Full-scale LLM: ~$43.5K, not $8.5K. | HIGH       | Run 100-paper micro-pilot to measure actual token consumption. Use prompt caching (60% input cost reduction). Reduce few-shot to 2 examples.                                                                                                 |
| H2  | **Working set tools go unused.** Agents don't maintain research strategies across turns. `add_to_working_set` requires deliberate curation that agents don't spontaneously perform.                           | HIGH       | Switch to implicit auto-population (last N papers retrieved). Reduce to 2 tools: `get_session_context` + `find_related_unexplored`. Remove metadata fields requiring agent judgment.                                                         |
| H3  | **Dual-model search doubles query latency.** OpenAI API call adds 200-800ms per query. P50 latency: 120ms → 380ms. API errors cause total search failure.                                                     | HIGH       | Cache query embeddings. Return SPECTER2 results immediately, merge OpenAI results asynchronously. Circuit breaker fallback to SPECTER2-only when OpenAI error rate >5%.                                                                      |
| H4  | **No sub-batch checkpointing.** Pipeline crash at row 8,742 of 10K rolls back all writes; API cost already spent.                                                                                             | HIGH       | Write batch API results to local JSONL first (durable checkpoint). Load to PostgreSQL in 500-row chunks with COMMIT between. Use `ON CONFLICT DO UPDATE` for idempotent upserts.                                                             |
| H5  | **Untyped `vector` column silently accepts wrong-dimension embeddings.** 512d vectors inserted where 1024d expected; pgvector zero-pads, producing garbage search results.                                    | MEDIUM     | Add CHECK constraint: `CHECK (CASE WHEN model_name = 'specter2' THEN array_length(embedding, 1) = 768 WHEN model_name = 'text-embedding-3-large' THEN array_length(embedding, 1) = 1024 END)`. Also validate in pipeline code before insert. |
| H6  | **Gold-standard eval too small (200 papers) with wide confidence intervals.** P=0.83 has 95% CI of [0.76, 0.90]. True precision may be below threshold.                                                       | HIGH       | Increase to 500 papers. Stratify by abstract length quartile, doctype, and year. Include 20 adversarial examples (short abstracts, non-English, erratum). Report CIs and require lower bound > threshold.                                    |
| H7  | **Pilot cohort bias (high-cited = easy papers).** Extraction precision P=0.87 on pilot drops to P=0.71 on papers with citation_count < 10 (60% of corpus).                                                    | MEDIUM     | Stratify pilot: 5K high-cited, 3K medium, 2K low-cited. Report metrics by citation quartile.                                                                                                                                                 |
| H8  | **Entity search returns irrelevant results due to canonical name mismatch.** Agent queries "JWST" but canonical is "James Webb Space Telescope". GIN index requires exact containment.                        | MEDIUM     | Don't expose raw entity search. Instead, add extracted entity names to tsvector for lexical search. Entities become a search-quality improvement, not a separate tool.                                                                       |
| H9  | **Temporal drift on 2025-2026 papers.** Novel instruments (Rubin/LSST, DESI Y1) absent from or inconsistent in Haiku's training data.                                                                         | MEDIUM     | Stratify pilot by year. Validate on 2025-2026 holdout. Maintain known-instrument registry for post-extraction validation.                                                                                                                    |

### MEDIUM Risks

| #   | Risk                                                                                                                                                                                       | Likelihood | Mitigation                                                                                                                                                                                    |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M1  | **halfvec recall degradation worse than expected under filtered search.** In dense clusters (astro-ph.CO), recall@10 drops from 0.89 (float32) to 0.76 (halfvec) — 13-point loss, not <1%. | MEDIUM     | Benchmark on actual corpus distribution. Use float32 for SPECTER2 (smaller index), halfvec only for OpenAI. Increase `ef_search` to 100-200 for filtered queries.                             |
| M2  | **GIN index on 5M JSONB rows bloats to 8GB**, thrashing shared_buffers.                                                                                                                    | MEDIUM     | Consider normalized `entity_mentions(bibcode, entity_type, canonical_name, confidence)` table with B-tree indexes (~2GB) instead of GIN on JSONB.                                             |
| M3  | **OpenAI Batch API 24-hour expiry silently cancels unprocessed papers.** 5M-paper batch may not complete in time.                                                                          | MEDIUM     | Split into 50K-paper sub-batches (each completes in ~15 min). Always download partial results from expired batches.                                                                           |
| M4  | **find_gaps returns noise.** Bridge papers are structurally interesting but topically irrelevant — mostly survey articles and methods papers.                                              | HIGH       | Replace community-bridge heuristic with semantic approach: find papers whose embeddings are far from working set centroid but have high citation overlap. Or delegate gap-finding to the LLM. |
| M5  | **Cross-validation (entity-community alignment) is coincidental.** High-cited papers use the same instruments, creating false correlation.                                                 | HIGH       | Define null model (shuffled entity assignments). Use PMI instead of raw correlation. Treat cross-validation as exploratory, not confirmatory.                                                 |
| M6  | **Dual-model adds complexity without measurable agent task improvement.** Marginal Recall@10 gain of 0.02-0.04 doesn't change which papers agents find.                                    | MEDIUM     | Add decision gate: "Does fusion change the top-5 result set?" If >80% overlap with SPECTER2+lexical alone, don't scale the second model. Also measure end-to-end latency, not just recall.    |

### Cross-Cutting Recommendations from Premortem

1. **Add Phase 3.5 "infrastructure stress test" at 500K papers** (10% of corpus, ~$850) before committing to 5M. Catches non-linear scaling issues in memory, I/O, index build, and GIN bloat.
2. **Adopt "API result → local file → database" pattern.** Never couple batch API responses directly to database transactions. Write to JSONL checkpoint first.
3. **Observe before building.** Run 2 weeks of agent transcript analysis to discover actual query patterns before designing entity types and session tools.
4. **Run 100-paper micro-pilot** before 10K pilot to validate cost estimates and prompt token budgets.
5. **Increase gold-standard eval to 500 papers**, stratified by year, citation quartile, doctype, and abstract length. Require CI lower bounds exceed thresholds.

## Cost Summary

> **Premortem correction (Risk H1)**: Original extraction cost estimates used ~500 tokens/request input. Actual input with few-shot examples + system prompt + tool-use overhead is ~2,550 tokens/request. Costs below reflect the corrected estimates. Prompt caching can reduce input costs by ~60%.

| Item                                          | Phase       | Original Est. | Corrected Est. (premortem) | With Prompt Caching |
| --------------------------------------------- | ----------- | ------------- | -------------------------- | ------------------- |
| Micro-pilot (100 papers, cost validation)     | 0 (NEW)     | --            | $1-2                       | $0.50-1             |
| Entity extraction pilot (10K, Haiku batches)  | 1           | $17-34        | $70-90                     | $30-40              |
| Embedding pilot (10K, OpenAI batch)           | 2           | $0.13         | $0.13                      | --                  |
| Gold-standard annotation (500 papers, manual) | 1 (was 200) | ~8 hours      | ~20 hours labor            | --                  |
| Query test set creation (50 queries)          | 2           | ~4 hours      | ~4 hours labor             | --                  |
| Infrastructure stress test (500K papers)      | 3.5 (NEW)   | --            | ~$850                      | ~$350               |
| **Pilot total**                               | **0-2**     | **~$35**      | **~$92 + 24 hours**        | **~$42 + 24 hours** |
| Entity extraction full (5M, Haiku batches)    | 4           | $8,500        | $43,500                    | $17,400             |
| Entity extraction full (5M, fine-tuned model) | 4           | ~$500         | ~$500                      | --                  |
| Embedding full (5M, OpenAI batch)             | 4           | $65           | $65                        | --                  |
| **Scale-up total (LLM path)**                 | **4**       | **~$8,565**   | **~$43,565**               | **~$17,465**        |
| **Scale-up total (fine-tune path)**           | **4**       | **~$565**     | **~$565**                  | **~$565**           |

## Research Provenance

**Perspective 1 (NLP/Entity Extraction)**: Recommended hybrid LLM-then-fine-tune approach, identified missing unique constraint and GIN index, designed batch pipeline with Batches API, proposed 4-tier validation.

**Perspective 2 (Embedding Model Evaluation)**: Identified SPECTER2's query-document gap, recommended text-embedding-3-large at 1024d with Matryoshka, proposed dual-model RRF fusion (not routing), flagged memory constraint and halfvec solution.

**Perspective 3 (Agent Session State)**: Designed in-memory working set with explicit curation, identified `find_gaps` as highest-value tool, scoped sessions to process lifetime with future HTTP readiness.

**Convergence decisions**: Shared 10K pilot cohort, untyped vector column (not separate table), multi-row extractions (not single-row), in-memory sessions (not PostgreSQL), parallel phasing for independent workstreams, halfvec from day one.
