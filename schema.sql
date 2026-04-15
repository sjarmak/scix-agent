-- schema.sql
-- Consolidated schema for scix database (PostgreSQL 16 + pgvector 0.8.2)
-- Generated from migrations 001-043 on 2026-04-14

CREATE EXTENSION IF NOT EXISTS vector;

-- 001_initial_schema.sql
-- 001_initial_schema.sql
-- SciX Knowledge Infrastructure: PostgreSQL + pgvector schema
-- Stores ~5M NASA ADS metadata records with citation graph and vector embeddings

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- Core paper metadata (one row per ADS record)
CREATE TABLE IF NOT EXISTS papers (
    bibcode TEXT PRIMARY KEY,
    title TEXT,
    abstract TEXT,
    year SMALLINT,
    doctype TEXT,
    pub TEXT,
    pub_raw TEXT,
    volume TEXT,
    issue TEXT,
    page TEXT[],
    authors TEXT[],
    first_author TEXT,
    affiliations TEXT[],
    keywords TEXT[],
    arxiv_class TEXT[],
    database TEXT[],
    doi TEXT[],
    identifier TEXT[],
    alternate_bibcode TEXT[],
    bibstem TEXT[],
    bibgroup TEXT[],
    orcid_pub TEXT[],
    orcid_user TEXT[],
    property TEXT[],
    copyright TEXT,
    lang TEXT,
    pubdate TEXT,
    entry_date TEXT,
    indexstamp TEXT,
    citation_count INTEGER,
    read_count INTEGER,
    reference_count INTEGER,
    raw JSONB
);

-- Citation graph edges (source cites target)
-- No FK constraints: citations frequently reference papers outside the 2021-2026
-- corpus or papers not yet ingested. Integrity is enforced at the application level.
CREATE TABLE IF NOT EXISTS citation_edges (
    source_bibcode TEXT NOT NULL,
    target_bibcode TEXT NOT NULL,
    PRIMARY KEY (source_bibcode, target_bibcode)
);
CREATE INDEX IF NOT EXISTS idx_cite_target ON citation_edges(target_bibcode);

-- Separate embeddings table (allows re-indexing without disrupting metadata)
-- Composite PK (bibcode, model_name) supports multiple embedding models per paper.
CREATE TABLE IF NOT EXISTS paper_embeddings (
    bibcode TEXT NOT NULL REFERENCES papers(bibcode),
    model_name TEXT NOT NULL,
    embedding vector(768),
    PRIMARY KEY (bibcode, model_name)
);
CREATE INDEX IF NOT EXISTS idx_embed_hnsw ON paper_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Flexible entity/relationship storage (JSONB, versioned)
CREATE TABLE IF NOT EXISTS extractions (
    id SERIAL PRIMARY KEY,
    bibcode TEXT NOT NULL REFERENCES papers(bibcode),
    extraction_type TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_extractions_bibcode ON extractions(bibcode);
CREATE INDEX IF NOT EXISTS idx_extractions_type ON extractions(extraction_type);

-- GIN indexes for faceted filtering on array fields
CREATE INDEX IF NOT EXISTS idx_papers_authors ON papers USING GIN (authors);
CREATE INDEX IF NOT EXISTS idx_papers_keywords ON papers USING GIN (keywords);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers USING GIN (arxiv_class);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers USING GIN (doi);

-- B-tree indexes for common filters
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers (year);
CREATE INDEX IF NOT EXISTS idx_papers_doctype ON papers (doctype);

COMMIT;

-- 002_ingest_log.sql
-- 002_ingest_log.sql
-- Tracks ingestion progress per file for resumability.

BEGIN;

CREATE TABLE IF NOT EXISTS ingest_log (
    filename TEXT PRIMARY KEY,
    records_loaded INTEGER NOT NULL DEFAULT 0,
    errors_skipped INTEGER NOT NULL DEFAULT 0,
    edges_loaded INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'in_progress',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

COMMIT;

-- 003_search_infrastructure.sql
-- 003_search_infrastructure.sql
-- Search infrastructure: custom text search config for scientific text,
-- tsvector column with GIN index, and embedding metadata columns.

BEGIN;

-- ---------------------------------------------------------------------------
-- Custom text search configuration for scientific text
-- ---------------------------------------------------------------------------
-- The built-in 'english' config strips hyphens (breaking "X-ray", "gamma-ray"),
-- drops numeric tokens (losing wavelengths like "21cm", redshifts like "z=0.5"),
-- and uses generic stop words. This config handles scientific text better.

-- Simple dictionary that preserves tokens as-is (no stemming, no stop words).
-- Used for hyphenated terms and numeric tokens.
DO $$ BEGIN
    CREATE TEXT SEARCH DICTIONARY simple_nostem (
        TEMPLATE = pg_catalog.simple,
        STOPWORDS = english  -- still filter common stop words
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Scientific text search config based on english but with tweaks.
DO $$ BEGIN
    CREATE TEXT SEARCH CONFIGURATION scix_english (COPY = english);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Map hyphenated words and numeric types through the simple dictionary
-- so "X-ray" stays as "x-ray" rather than being split into "x" and "ray".
ALTER TEXT SEARCH CONFIGURATION scix_english
    ALTER MAPPING FOR hword, hword_part, hword_asciipart
    WITH simple_nostem;

ALTER TEXT SEARCH CONFIGURATION scix_english
    ALTER MAPPING FOR int, uint, float
    WITH simple_nostem;

-- ---------------------------------------------------------------------------
-- tsvector column on papers (weighted: title=A, abstract=B, keywords=C)
-- ---------------------------------------------------------------------------
-- Using our custom scix_english config instead of built-in english.
-- GENERATED ALWAYS columns cannot reference custom configs in all PG versions,
-- so we use a trigger-maintained column instead.

ALTER TABLE papers ADD COLUMN IF NOT EXISTS tsv tsvector;

-- Populate existing rows
UPDATE papers SET tsv =
    setweight(to_tsvector('scix_english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('scix_english', coalesce(abstract, '')), 'B') ||
    setweight(to_tsvector('scix_english', coalesce(array_to_string(keywords, ' '), '')), 'C')
WHERE tsv IS NULL;

-- Trigger to keep tsv updated on INSERT/UPDATE
CREATE OR REPLACE FUNCTION papers_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv :=
        setweight(to_tsvector('scix_english', coalesce(NEW.title, '')), 'A') ||
        setweight(to_tsvector('scix_english', coalesce(NEW.abstract, '')), 'B') ||
        setweight(to_tsvector('scix_english', coalesce(array_to_string(NEW.keywords, ' '), '')), 'C');
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_papers_tsv ON papers;
CREATE TRIGGER trig_papers_tsv
    BEFORE INSERT OR UPDATE OF title, abstract, keywords ON papers
    FOR EACH ROW EXECUTE FUNCTION papers_tsv_trigger();

-- GIN index for fast tsvector queries
CREATE INDEX IF NOT EXISTS idx_papers_tsv ON papers USING GIN (tsv);

-- ---------------------------------------------------------------------------
-- Embedding metadata columns
-- ---------------------------------------------------------------------------
ALTER TABLE paper_embeddings ADD COLUMN IF NOT EXISTS input_type TEXT NOT NULL DEFAULT 'title_abstract';
ALTER TABLE paper_embeddings ADD COLUMN IF NOT EXISTS source_hash TEXT;

-- Partial HNSW index per model_name for blue-green transitions
-- (The existing idx_embed_hnsw covers all models; add per-model indexes as needed.)

-- ---------------------------------------------------------------------------
-- First-author normalized index for author search
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_papers_first_author ON papers (first_author);

-- Index for year range queries (already exists from 001, but ensure it's there)
-- CREATE INDEX IF NOT EXISTS idx_papers_year ON papers (year);

COMMIT;

-- 004_per_model_hnsw_and_pg_search.sql
-- 004_per_model_hnsw_and_pg_search.sql
-- Per-model partial HNSW indexes for blue-green embedding model transitions,
-- and optional pg_search BM25 index for search quality evaluation.

BEGIN;

-- ---------------------------------------------------------------------------
-- Per-model partial HNSW indexes
-- ---------------------------------------------------------------------------
-- The global idx_embed_hnsw (from 001) covers all models but the planner cannot
-- use it efficiently when filtering by model_name because the WHERE clause is
-- applied after the ANN scan. Partial indexes let pgvector restrict the scan
-- to a single model's embeddings, improving recall and latency.
--
-- During a blue-green model transition (e.g. specter2 -> specter3), both
-- partial indexes coexist. Once the old model is retired, drop its index.

CREATE INDEX IF NOT EXISTS idx_embed_hnsw_specter2
    ON paper_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200)
    WHERE model_name = 'specter2';

-- Placeholder for future models (uncomment when ready):
-- CREATE INDEX IF NOT EXISTS idx_embed_hnsw_specter3
--     ON paper_embeddings USING hnsw (embedding vector_cosine_ops)
--     WITH (m = 16, ef_construction = 200)
--     WHERE model_name = 'specter3';

-- ---------------------------------------------------------------------------
-- pg_search BM25 index (optional — requires ParadeDB pg_search extension)
-- ---------------------------------------------------------------------------
-- pg_search provides a true Okapi BM25 implementation via a custom index type.
-- This is superior to ts_rank_cd for ranking because BM25 accounts for
-- document length normalization and term saturation.
--
-- If pg_search is not installed, this section is a no-op.

DO $$ BEGIN
    -- Check if pg_search extension is available
    IF EXISTS (
        SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search'
    ) THEN
        CREATE EXTENSION IF NOT EXISTS pg_search;

        -- Create BM25 index on papers for title + abstract + keywords
        -- Only if the index does not already exist
        IF NOT EXISTS (
            SELECT 1 FROM pg_indexes WHERE indexname = 'idx_papers_bm25'
        ) THEN
            EXECUTE $idx$
                CREATE INDEX idx_papers_bm25
                ON papers USING bm25 (bibcode, title, abstract, keywords)
                WITH (
                    key_field = 'bibcode',
                    text_fields = '{
                        "title":   {"tokenizer": {"type": "default"}, "boost": 2.0},
                        "abstract": {"tokenizer": {"type": "default"}},
                        "keywords": {"tokenizer": {"type": "default"}, "boost": 1.5}
                    }'
                );
            $idx$;
            RAISE NOTICE 'pg_search: created BM25 index idx_papers_bm25';
        END IF;
    ELSE
        RAISE NOTICE 'pg_search extension not available — skipping BM25 index creation';
    END IF;
END $$;

COMMIT;

-- 005_pgvector_upgrade.sql
-- 005_pgvector_upgrade.sql
-- Upgrade pgvector extension and enable iterative index scans.
--
-- pgvector 0.8.0+ adds iterative index scans which solve the post-filtering
-- problem: with selective filters (e.g., "astrophysics papers from 2023"),
-- the old approach could return fewer than k results because it scans a fixed
-- number of index entries then post-filters. Iterative scans automatically
-- expand the search until enough results pass the filter.
--
-- This migration is safe to run on any pgvector version:
-- - ALTER EXTENSION ... UPDATE upgrades to the latest installed version
-- - The SET commands are session-level and only take effect in 0.8.0+
--
-- Also enables in 0.8.0+:
-- - halfvec (float16) quantization — halves embedding storage
-- - Binary quantization for further compression
-- - Improved HNSW build performance

BEGIN;

-- Upgrade pgvector to the latest version available on the server.
-- If already at the latest version, this is a no-op.
ALTER EXTENSION vector UPDATE;

-- Note: iterative scan settings are session-level (SET/SET LOCAL), not
-- persisted in the schema. Application code in search.py configures
-- hnsw.iterative_scan per-transaction when pgvector >= 0.8.0 is detected.
--
-- Available modes:
--   off           — default pre-0.8.0 behavior (post-filtering only)
--   relaxed_order — iterative scan, may return results slightly out of order
--                   (best for filtered search where recall > exact ordering)
--   strict_order  — iterative scan with exact distance ordering
--                   (slower but guarantees order; use for unfiltered top-k)

COMMIT;

-- 006_graph_metrics.sql
-- 006_graph_metrics.sql
-- Precomputed graph metrics: PageRank, HITS, Leiden communities

BEGIN;

CREATE TABLE IF NOT EXISTS paper_metrics (
    bibcode TEXT PRIMARY KEY REFERENCES papers(bibcode),
    pagerank DOUBLE PRECISION,
    hub_score DOUBLE PRECISION,
    authority_score DOUBLE PRECISION,
    community_id_coarse INTEGER,
    community_id_medium INTEGER,
    community_id_fine INTEGER,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pm_community_coarse ON paper_metrics(community_id_coarse);
CREATE INDEX IF NOT EXISTS idx_pm_community_medium ON paper_metrics(community_id_medium);
CREATE INDEX IF NOT EXISTS idx_pm_community_fine ON paper_metrics(community_id_fine);
CREATE INDEX IF NOT EXISTS idx_pm_pagerank ON paper_metrics(pagerank DESC);

CREATE TABLE IF NOT EXISTS communities (
    community_id INTEGER NOT NULL,
    resolution TEXT NOT NULL CHECK (resolution IN ('coarse', 'medium', 'fine')),
    label TEXT,
    paper_count INTEGER NOT NULL DEFAULT 0,
    top_keywords TEXT[] NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (community_id, resolution)
);

COMMIT;

-- 007_uat_concepts.sql
-- 007_uat_concepts.sql
-- Unified Astronomy Thesaurus (UAT) concept hierarchy

BEGIN;

CREATE TABLE IF NOT EXISTS uat_concepts (
    concept_id TEXT PRIMARY KEY,
    preferred_label TEXT NOT NULL,
    alternate_labels TEXT[] NOT NULL DEFAULT '{}',
    definition TEXT,
    level INTEGER
);

CREATE INDEX IF NOT EXISTS idx_uat_preferred_label ON uat_concepts(preferred_label);
CREATE INDEX IF NOT EXISTS idx_uat_alternate_labels ON uat_concepts USING GIN (alternate_labels);

CREATE TABLE IF NOT EXISTS uat_relationships (
    parent_id TEXT NOT NULL REFERENCES uat_concepts(concept_id),
    child_id TEXT NOT NULL REFERENCES uat_concepts(concept_id),
    PRIMARY KEY (parent_id, child_id)
);

CREATE INDEX IF NOT EXISTS idx_uat_rel_child ON uat_relationships(child_id);

CREATE TABLE IF NOT EXISTS paper_uat_mappings (
    bibcode TEXT NOT NULL REFERENCES papers(bibcode),
    concept_id TEXT NOT NULL REFERENCES uat_concepts(concept_id),
    match_type TEXT NOT NULL CHECK (match_type IN ('exact', 'fuzzy', 'parent')),
    PRIMARY KEY (bibcode, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_pum_concept ON paper_uat_mappings(concept_id);
CREATE INDEX IF NOT EXISTS idx_pum_match_type ON paper_uat_mappings(match_type);

COMMIT;

-- 008_community_taxonomic.sql
-- 008_community_taxonomic.sql
-- Add taxonomic community column populated from papers.arxiv_class

BEGIN;

ALTER TABLE paper_metrics ADD COLUMN IF NOT EXISTS community_taxonomic TEXT;

CREATE INDEX IF NOT EXISTS idx_pm_community_taxonomic
    ON paper_metrics(community_taxonomic);

COMMIT;

-- 009_knowledge_enrichment.sql
-- 009_knowledge_enrichment.sql
-- Knowledge enrichment schema changes:
--   1. Unique constraint on extractions to prevent duplicate entries
--   2. GIN index on extractions.payload for JSONB path queries
--   3. Untyped vector column on paper_embeddings to support multiple dimensions
--   4. CHECK constraint enforcing embedding dimension per model_name
--   5. halfvec HNSW index placeholder for text-embedding-3-large (1024-dim)

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Unique constraint on extractions
-- ---------------------------------------------------------------------------
-- Prevents duplicate (bibcode, extraction_type, extraction_version) tuples.
-- The serial id remains as PK for FK references; this constraint enforces
-- logical uniqueness.

ALTER TABLE extractions
    ADD CONSTRAINT uq_extractions_bibcode_type_version
    UNIQUE (bibcode, extraction_type, extraction_version);

-- ---------------------------------------------------------------------------
-- 2. GIN index on extractions.payload with jsonb_path_ops
-- ---------------------------------------------------------------------------
-- jsonb_path_ops is smaller and faster than the default GIN operator class
-- when queries only use the @> (containment) operator, which is the common
-- case for structured extraction payloads.

CREATE INDEX idx_extractions_payload_gin
    ON extractions USING GIN (payload jsonb_path_ops);

-- ---------------------------------------------------------------------------
-- 3. Untyped vector column on paper_embeddings
-- ---------------------------------------------------------------------------
-- The original schema fixed embedding to vector(768) (specter2 dimensions).
-- To support models with different dimensions (e.g. text-embedding-3-large
-- at 1024), we remove the fixed dimension so the column accepts any size.

ALTER TABLE paper_embeddings
    ALTER COLUMN embedding TYPE vector;

-- ---------------------------------------------------------------------------
-- 4. CHECK constraint: embedding dimension must match model_name
-- ---------------------------------------------------------------------------
-- Ensures data integrity after removing the fixed dimension: each model's
-- embeddings must have the correct number of dimensions.

ALTER TABLE paper_embeddings
    ADD CONSTRAINT chk_embedding_dim CHECK (
        (model_name = 'specter2' AND vector_dims(embedding) = 768)
        OR (model_name = 'text-embedding-3-large' AND vector_dims(embedding) = 1024)
    );

-- ---------------------------------------------------------------------------
-- 5. halfvec HNSW index for text-embedding-3-large
-- ---------------------------------------------------------------------------
-- Uses halfvec (float16) quantization from pgvector 0.8.0+ to halve storage
-- for the 1024-dim OpenAI embeddings. Partial index scoped to model_name.

CREATE INDEX idx_embed_hnsw_openai
    ON paper_embeddings USING hnsw ((embedding::halfvec(1024)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200)
    WHERE model_name = 'text-embedding-3-large';

COMMIT;

-- 010_body_column.sql
-- Add dedicated body column for full-text paper content.
-- Previously stored inside the raw JSONB blob; now a first-class column
-- for full-text search and RAG pipelines.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS body TEXT;

-- 011_citation_contexts.sql
-- 011_citation_contexts.sql
-- Citation context snippets: the surrounding text where a citation appears
-- Enables citation intent classification and context-aware retrieval

BEGIN;

-- Citation contexts (one row per in-text citation mention)
-- No FK constraints: citations frequently reference papers outside the corpus
-- (same rationale as citation_edges in 001_initial_schema.sql)
CREATE TABLE IF NOT EXISTS citation_contexts (
    id SERIAL PRIMARY KEY,
    source_bibcode TEXT NOT NULL,
    target_bibcode TEXT NOT NULL,
    context_text TEXT NOT NULL,
    char_offset INTEGER,
    section_name TEXT,
    intent TEXT
);

CREATE INDEX IF NOT EXISTS idx_citctx_source_target
    ON citation_contexts (source_bibcode, target_bibcode);
CREATE INDEX IF NOT EXISTS idx_citctx_target
    ON citation_contexts (target_bibcode);

COMMIT;

-- 012_full_field_coverage.sql
-- 012_full_field_coverage.sql
-- Add dedicated columns for all ADS API fields harvested by scripts/harvest_full.py.
-- Previously these fields were either dropped or buried in the raw JSONB blob.
-- Promotes them to first-class columns for indexing, filtering, and agent queries.
--
-- All columns are nullable — most papers won't have all fields populated.
-- ALTER TABLE ... ADD COLUMN IF NOT EXISTS is catalog-only for nullable columns
-- (no table rewrite), so this migration is safe on large tables.

BEGIN;

-- Text fields
ALTER TABLE papers ADD COLUMN IF NOT EXISTS ack TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS date TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS eid TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS entdate TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS first_author_norm TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS page_range TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS pubnote TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS series TEXT;

-- Array fields (TEXT[])
ALTER TABLE papers ADD COLUMN IF NOT EXISTS aff_id TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS alternate_title TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS author_norm TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS caption TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS comment TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS data TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS esources TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS facility TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS grant_facet TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS grant_agencies TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS grant_id TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS isbn TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS issn TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS keyword_norm TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS keyword_schema TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS links_data TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS nedid TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS nedtype TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS orcid_other TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS simbid TEXT[];
ALTER TABLE papers ADD COLUMN IF NOT EXISTS vizier TEXT[];

-- Integer fields
ALTER TABLE papers ADD COLUMN IF NOT EXISTS author_count INTEGER;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS page_count INTEGER;

-- Real (float) fields — normalized scores from ADS
ALTER TABLE papers ADD COLUMN IF NOT EXISTS citation_count_norm REAL;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS cite_read_boost REAL;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS classic_factor REAL;

-- Indexes for high-value queryable fields
-- GIN indexes on arrays frequently used for faceted search
CREATE INDEX IF NOT EXISTS idx_papers_data ON papers USING GIN (data);
CREATE INDEX IF NOT EXISTS idx_papers_facility ON papers USING GIN (facility);
CREATE INDEX IF NOT EXISTS idx_papers_esources ON papers USING GIN (esources);
CREATE INDEX IF NOT EXISTS idx_papers_nedid ON papers USING GIN (nedid);
CREATE INDEX IF NOT EXISTS idx_papers_simbid ON papers USING GIN (simbid);
CREATE INDEX IF NOT EXISTS idx_papers_keyword_norm ON papers USING GIN (keyword_norm);
CREATE INDEX IF NOT EXISTS idx_papers_author_norm ON papers USING GIN (author_norm);

-- B-tree index on author_count for range queries
CREATE INDEX IF NOT EXISTS idx_papers_author_count ON papers (author_count);

COMMIT;

-- 013_entity_dictionary.sql
-- 013_entity_dictionary.sql
-- Entity dictionary for canonical names, aliases, and metadata

BEGIN;

CREATE TABLE IF NOT EXISTS entity_dictionary (
    id SERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    source TEXT NOT NULL,
    external_id TEXT,
    aliases TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    UNIQUE (canonical_name, entity_type, source)
);

CREATE INDEX IF NOT EXISTS idx_entity_dict_type ON entity_dictionary(entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_dict_aliases ON entity_dictionary USING GIN (aliases);

COMMIT;

-- 014_discipline_and_indexes.sql
-- 014_discipline_and_indexes.sql
-- Add discipline column to entity_dictionary with backfill and functional index

BEGIN;

-- 1. Add nullable discipline column
ALTER TABLE entity_dictionary ADD COLUMN IF NOT EXISTS discipline TEXT;

-- 2. Btree index on discipline
CREATE INDEX IF NOT EXISTS idx_entity_dict_discipline
    ON entity_dictionary (discipline);

-- 3. Functional index for case-insensitive canonical_name lookups
CREATE INDEX IF NOT EXISTS idx_entity_dict_canonical_lower
    ON entity_dictionary (lower(canonical_name));

-- 4. Backfill discipline='astrophysics' for all known astronomy sources
UPDATE entity_dictionary
   SET discipline = 'astrophysics'
 WHERE source IN ('ascl', 'aas', 'physh', 'pwc', 'astromlab', 'vizier', 'ads_data')
   AND discipline IS NULL;

COMMIT;

-- 015_staging_schema.sql
-- 015_staging_schema.sql
-- Staging schema for extraction pipeline writes.
-- Isolates write-heavy extraction workloads from read-heavy MCP queries
-- by directing pipeline writes to staging.extractions, then batch-promoting
-- canonical results to public.extractions via staging.promote_extractions().

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Create staging schema
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS staging;

-- ---------------------------------------------------------------------------
-- 2. Create staging.extractions mirroring public.extractions structure
-- ---------------------------------------------------------------------------
-- No FK to papers: staging data may reference bibcodes not yet in public.

CREATE TABLE IF NOT EXISTS staging.extractions (
    id SERIAL PRIMARY KEY,
    bibcode TEXT NOT NULL,
    extraction_type TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_staging_extractions_bibcode_type_version
        UNIQUE (bibcode, extraction_type, extraction_version)
);

CREATE INDEX IF NOT EXISTS idx_staging_extractions_bibcode
    ON staging.extractions(bibcode);

CREATE INDEX IF NOT EXISTS idx_staging_extractions_type
    ON staging.extractions(extraction_type);

-- ---------------------------------------------------------------------------
-- 3. promote_extractions() — batch upsert from staging to public
-- ---------------------------------------------------------------------------
-- Uses the unique constraint (bibcode, extraction_type, extraction_version)
-- on public.extractions for ON CONFLICT upsert. After promotion, staging
-- is truncated.

CREATE OR REPLACE FUNCTION staging.promote_extractions()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    promoted_count INTEGER;
BEGIN
    -- Upsert from staging into public
    WITH upserted AS (
        INSERT INTO public.extractions (bibcode, extraction_type, extraction_version, payload, created_at)
        SELECT bibcode, extraction_type, extraction_version, payload, created_at
        FROM staging.extractions
        ON CONFLICT (bibcode, extraction_type, extraction_version)
        DO UPDATE SET
            payload = EXCLUDED.payload,
            created_at = EXCLUDED.created_at
        RETURNING 1
    )
    SELECT count(*) INTO promoted_count FROM upserted;

    -- Clear staging after successful promotion
    TRUNCATE staging.extractions;

    RETURN promoted_count;
END;
$$;

COMMIT;

-- 016_query_log.sql
-- 016_query_log.sql
-- Query logging for MCP tool calls
-- Note: This duplicates 013_query_log.sql with IF NOT EXISTS guards for safety.

BEGIN;

CREATE TABLE IF NOT EXISTS query_log (
    id SERIAL PRIMARY KEY,
    tool_name TEXT NOT NULL,
    params_json JSONB,
    latency_ms REAL,
    success BOOLEAN NOT NULL,
    error_msg TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_log_tool_name ON query_log(tool_name);
CREATE INDEX IF NOT EXISTS idx_query_log_created_at ON query_log(created_at);

COMMIT;

-- 017_entity_provenance.sql
-- Migration 017: Add provenance metadata to extractions
--
-- Tracks the origin and confidence of each extraction record:
--   source           — where the extraction came from (llm, metadata, ner, etc.)
--   confidence_tier  — high / medium / low
--   extraction_model — which model produced it (nullable for non-LLM sources)

BEGIN;

ALTER TABLE extractions
    ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'llm',
    ADD COLUMN IF NOT EXISTS confidence_tier TEXT NOT NULL DEFAULT 'medium',
    ADD COLUMN IF NOT EXISTS extraction_model TEXT;

-- Constrain to known values
ALTER TABLE extractions
    ADD CONSTRAINT chk_extractions_source
        CHECK (source IN ('metadata', 'ner', 'llm', 'openalex', 'citation_propagation'));

ALTER TABLE extractions
    ADD CONSTRAINT chk_extractions_confidence_tier
        CHECK (confidence_tier IN ('high', 'medium', 'low'));

-- Index for filtering by provenance
CREATE INDEX IF NOT EXISTS idx_extractions_source ON extractions(source);
CREATE INDEX IF NOT EXISTS idx_extractions_confidence_tier ON extractions(confidence_tier);

COMMIT;

-- 018_openalex_linking.sql
-- 018_openalex_linking.sql
-- Add OpenAlex work ID and topic annotations to papers table.

BEGIN;

ALTER TABLE papers ADD COLUMN IF NOT EXISTS openalex_id TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS openalex_topics JSONB;

CREATE INDEX IF NOT EXISTS idx_papers_openalex_id ON papers(openalex_id);

COMMIT;

-- 019_schema_migrations.sql
-- 019_schema_migrations.sql
-- Schema migration tracking table

BEGIN;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    filename TEXT NOT NULL
);

COMMIT;

-- 020_harvest_runs.sql
-- 020_harvest_runs.sql
-- Harvest run tracking for external data source ingestion

BEGIN;

CREATE TABLE IF NOT EXISTS harvest_runs (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    records_fetched INT NOT NULL DEFAULT 0,
    records_upserted INT NOT NULL DEFAULT 0,
    cursor_state JSONB,
    error_message TEXT,
    config JSONB,
    counts JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_harvest_runs_source ON harvest_runs(source);

COMMIT;

-- 021_entity_graph.sql
-- 021_entity_graph.sql
-- Entity graph tables, compatibility view, and seed migration from entity_dictionary.
-- Normalizes entity_dictionary into a proper graph schema with separate tables
-- for identifiers, aliases, and relationships.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. entities — canonical entity records
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    discipline TEXT,
    source TEXT NOT NULL,
    harvest_run_id INT REFERENCES harvest_runs(id),
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (canonical_name, entity_type, source)
);

CREATE INDEX IF NOT EXISTS idx_entities_entity_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_discipline ON entities(discipline);
CREATE INDEX IF NOT EXISTS idx_entities_canonical_lower ON entities(lower(canonical_name));
CREATE INDEX IF NOT EXISTS idx_entities_properties ON entities USING GIN (properties jsonb_path_ops);

-- ---------------------------------------------------------------------------
-- 2. entity_identifiers — external IDs (e.g. Wikidata QID, DOI)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_identifiers (
    entity_id INT REFERENCES entities(id) ON DELETE CASCADE,
    id_scheme TEXT NOT NULL,
    external_id TEXT NOT NULL,
    is_primary BOOLEAN DEFAULT false,
    PRIMARY KEY (id_scheme, external_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_identifiers_entity_id ON entity_identifiers(entity_id);

-- ---------------------------------------------------------------------------
-- 3. entity_aliases — alternate names for entities
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id INT REFERENCES entities(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    alias_source TEXT,
    PRIMARY KEY (entity_id, alias)
);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_lower ON entity_aliases(lower(alias));

-- ---------------------------------------------------------------------------
-- 4. entity_relationships — entity-to-entity links
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_relationships (
    id SERIAL PRIMARY KEY,
    subject_entity_id INT REFERENCES entities(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_entity_id INT REFERENCES entities(id) ON DELETE CASCADE,
    source TEXT,
    harvest_run_id INT REFERENCES harvest_runs(id),
    confidence REAL DEFAULT 1.0,
    UNIQUE (subject_entity_id, predicate, object_entity_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_relationships_object ON entity_relationships(object_entity_id);

-- ---------------------------------------------------------------------------
-- 5. document_entities — bridge: bibcode <-> entity
-- ---------------------------------------------------------------------------
-- No FK on bibcode (matches citation_edges pattern — papers may not be ingested yet)

CREATE TABLE IF NOT EXISTS document_entities (
    bibcode TEXT NOT NULL,
    entity_id INT REFERENCES entities(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL,
    confidence REAL,
    match_method TEXT,
    evidence JSONB,
    harvest_run_id INT REFERENCES harvest_runs(id),
    PRIMARY KEY (bibcode, entity_id, link_type)
);

-- ---------------------------------------------------------------------------
-- 6. datasets — external dataset records
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    discipline TEXT,
    source TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    description TEXT,
    temporal_start DATE,
    temporal_end DATE,
    properties JSONB DEFAULT '{}',
    harvest_run_id INT REFERENCES harvest_runs(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source, canonical_id)
);

-- ---------------------------------------------------------------------------
-- 7. dataset_entities — bridge: dataset <-> entity
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dataset_entities (
    dataset_id INT REFERENCES datasets(id) ON DELETE CASCADE,
    entity_id INT REFERENCES entities(id) ON DELETE CASCADE,
    relationship TEXT NOT NULL,
    PRIMARY KEY (dataset_id, entity_id, relationship)
);

-- ---------------------------------------------------------------------------
-- 8. document_datasets — bridge: bibcode <-> dataset
-- ---------------------------------------------------------------------------
-- No FK on bibcode

CREATE TABLE IF NOT EXISTS document_datasets (
    bibcode TEXT NOT NULL,
    dataset_id INT REFERENCES datasets(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL,
    confidence REAL,
    match_method TEXT,
    harvest_run_id INT REFERENCES harvest_runs(id),
    PRIMARY KEY (bibcode, dataset_id, link_type)
);

-- ---------------------------------------------------------------------------
-- 9. entity_dictionary_compat — backward-compatible view
-- ---------------------------------------------------------------------------
-- Exposes the same column shape as entity_dictionary so that existing code
-- (e.g. src/scix/dictionary.py) continues to work via SELECT queries.

CREATE OR REPLACE VIEW entity_dictionary_compat AS
SELECT
    e.id,
    e.canonical_name,
    e.entity_type,
    e.source,
    ei.external_id,
    COALESCE(
        (SELECT array_agg(ea.alias) FROM entity_aliases ea WHERE ea.entity_id = e.id),
        '{}'::TEXT[]
    ) AS aliases,
    e.properties AS metadata
FROM entities e
LEFT JOIN entity_identifiers ei
    ON ei.entity_id = e.id AND ei.is_primary = true;

-- ---------------------------------------------------------------------------
-- 10. Seed migration — copy data from entity_dictionary into new tables
-- ---------------------------------------------------------------------------

-- 10a. Copy entities
INSERT INTO entities (canonical_name, entity_type, source, discipline, properties)
SELECT canonical_name, entity_type, source, discipline, metadata
FROM entity_dictionary
ON CONFLICT DO NOTHING;

-- 10b. Copy aliases (unnest entity_dictionary.aliases[])
INSERT INTO entity_aliases (entity_id, alias, alias_source)
SELECT e.id, unnested.alias, 'entity_dictionary'
FROM entity_dictionary ed
CROSS JOIN LATERAL unnest(ed.aliases) AS unnested(alias)
JOIN entities e
    ON e.canonical_name = ed.canonical_name
    AND e.entity_type = ed.entity_type
    AND e.source = ed.source
ON CONFLICT DO NOTHING;

-- 10c. Copy external_ids where not null
INSERT INTO entity_identifiers (entity_id, id_scheme, external_id, is_primary)
SELECT e.id, ed.source, ed.external_id, true
FROM entity_dictionary ed
JOIN entities e
    ON e.canonical_name = ed.canonical_name
    AND e.entity_type = ed.entity_type
    AND e.source = ed.source
WHERE ed.external_id IS NOT NULL
ON CONFLICT DO NOTHING;

COMMIT;

-- 022_staging_entities.sql
-- 022_staging_entities.sql
-- Staging tables for entity graph: entities, entity_identifiers, entity_aliases.
-- Follows the staging pattern from 015_staging_schema.sql.
-- Pipeline writes land in staging.*, then staging.promote_entities() batch-upserts
-- into public.* and truncates staging.

BEGIN;

-- Public entity tables (entities, entity_identifiers, entity_aliases) are
-- created by migration 021_entity_graph.sql. This migration only adds
-- their staging counterparts and the promote function.

-- ---------------------------------------------------------------------------
-- 1. Staging entity tables (no FK enforcement)
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.entities (
    id SERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    discipline TEXT,
    source TEXT NOT NULL,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (canonical_name, entity_type, source)
);

CREATE TABLE IF NOT EXISTS staging.entity_identifiers (
    entity_id INT NOT NULL,
    id_scheme TEXT NOT NULL,
    external_id TEXT NOT NULL,
    is_primary BOOLEAN DEFAULT false,
    PRIMARY KEY (id_scheme, external_id)
);

CREATE TABLE IF NOT EXISTS staging.entity_aliases (
    entity_id INT NOT NULL,
    alias TEXT NOT NULL,
    alias_source TEXT,
    PRIMARY KEY (entity_id, alias)
);

-- ---------------------------------------------------------------------------
-- 3. promote_entities() — atomic batch upsert from staging to public
-- ---------------------------------------------------------------------------
-- Promotes all 3 tables in a single call:
--   1. Upsert entities (ON CONFLICT updates properties + updated_at)
--   2. Upsert identifiers (remaps staging entity_id -> public entity_id
--      via canonical_name + entity_type + source natural key)
--   3. Upsert aliases (same remapping, ON CONFLICT DO NOTHING)
--   4. Truncate all staging tables
-- Returns the number of promoted entities.

CREATE OR REPLACE FUNCTION staging.promote_entities()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    promoted_count INTEGER;
BEGIN
    -- 1. Upsert entities
    WITH upserted AS (
        INSERT INTO public.entities
            (canonical_name, entity_type, discipline, source, properties, created_at, updated_at)
        SELECT canonical_name, entity_type, discipline, source, properties, created_at, updated_at
        FROM staging.entities
        ON CONFLICT (canonical_name, entity_type, source)
        DO UPDATE SET
            properties = EXCLUDED.properties,
            updated_at = NOW()
        RETURNING 1
    )
    SELECT count(*) INTO promoted_count FROM upserted;

    -- 2. Upsert identifiers (remap entity_id through natural key)
    INSERT INTO public.entity_identifiers (entity_id, id_scheme, external_id, is_primary)
    SELECT pe.id, si.id_scheme, si.external_id, si.is_primary
    FROM staging.entity_identifiers si
    JOIN staging.entities se ON se.id = si.entity_id
    JOIN public.entities pe ON pe.canonical_name = se.canonical_name
                            AND pe.entity_type = se.entity_type
                            AND pe.source = se.source
    ON CONFLICT (id_scheme, external_id)
    DO UPDATE SET
        entity_id = EXCLUDED.entity_id,
        is_primary = EXCLUDED.is_primary;

    -- 3. Upsert aliases (remap entity_id through natural key)
    INSERT INTO public.entity_aliases (entity_id, alias, alias_source)
    SELECT pe.id, sa.alias, sa.alias_source
    FROM staging.entity_aliases sa
    JOIN staging.entities se ON se.id = sa.entity_id
    JOIN public.entities pe ON pe.canonical_name = se.canonical_name
                            AND pe.entity_type = se.entity_type
                            AND pe.source = se.source
    ON CONFLICT (entity_id, alias)
    DO NOTHING;

    -- 4. Clear staging tables
    TRUNCATE staging.entity_aliases;
    TRUNCATE staging.entity_identifiers;
    TRUNCATE staging.entities;

    RETURN promoted_count;
END;
$$;

COMMIT;

-- 023_logged_embeddings.sql
-- 023_logged_embeddings.sql
-- Convert paper_embeddings from UNLOGGED to LOGGED for crash safety,
-- and widen the vector column to support multiple embedding dimensions.
--
-- Context: paper_embeddings was UNLOGGED for faster bulk writes during
-- initial embedding. A PostgreSQL crash wiped all 32M embeddings because
-- UNLOGGED tables are truncated on recovery. This migration prevents that.
--
-- Also changes embedding column from vector(768) to vector (untyped),
-- allowing different models to store different-dimensional embeddings
-- in the same table (e.g. 768 for SPECTER2/INDUS, 1024 for GTE).

BEGIN;

-- 1. Convert to LOGGED so WAL protects the data.
--    This is a metadata-only change when the table is empty (current state).
--    On a populated table it rewrites all pages to WAL — plan for downtime.
ALTER TABLE paper_embeddings SET LOGGED;

-- 2. Widen embedding column: vector(768) -> vector (any dimension).
--    The model_name column already partitions by model, so mixed dimensions
--    are safe. Per-model partial HNSW indexes (migration 004) enforce
--    dimensional consistency at the index level.
ALTER TABLE paper_embeddings ALTER COLUMN embedding TYPE vector;

-- 3. Drop the old global HNSW index from migration 001 if it exists.
--    We use per-model partial indexes (migration 004) instead.
DROP INDEX IF EXISTS idx_embed_hnsw;
DROP INDEX IF EXISTS idx_embeddings_hnsw;

-- 4. Also convert _to_embed if it exists (scratch table for embedding pipeline).
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM pg_class WHERE relname = '_to_embed' AND relpersistence = 'u') THEN
        ALTER TABLE _to_embed SET LOGGED;
        RAISE NOTICE 'Converted _to_embed to LOGGED';
    END IF;
END $$;

COMMIT;

-- 024_agent_context_views.sql
-- 024_agent_context_views.sql
-- Materialized views for agent context: document, entity, and dataset.
-- Pre-compute JOINs so MCP tools can serve single-row lookups in <1ms.
-- Benchmarked at scale (10M document_entities): all REFRESH <30s.
-- See .claude/prd-build-artifacts/matview-benchmark.md for details.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. agent_document_context — one row per paper with aggregated entities
-- ---------------------------------------------------------------------------

CREATE MATERIALIZED VIEW IF NOT EXISTS agent_document_context AS
SELECT
    p.bibcode,
    p.title,
    p.abstract,
    p.year,
    p.citation_count,
    p.reference_count,
    COALESCE(
        jsonb_agg(
            DISTINCT jsonb_build_object(
                'entity_id', e.id,
                'name', e.canonical_name,
                'type', e.entity_type,
                'link_type', de.link_type,
                'confidence', de.confidence
            )
        ) FILTER (WHERE e.id IS NOT NULL),
        '[]'::jsonb
    ) AS linked_entities
FROM papers p
LEFT JOIN document_entities de ON de.bibcode = p.bibcode
LEFT JOIN entities e ON e.id = de.entity_id
GROUP BY p.bibcode, p.title, p.abstract, p.year, p.citation_count, p.reference_count;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_doc_ctx_bibcode
    ON agent_document_context (bibcode);

-- ---------------------------------------------------------------------------
-- 2. agent_entity_context — one row per entity with identifiers, aliases,
--    relationships, and citing paper count
-- ---------------------------------------------------------------------------

CREATE MATERIALIZED VIEW IF NOT EXISTS agent_entity_context AS
SELECT
    e.id AS entity_id,
    e.canonical_name,
    e.entity_type,
    e.discipline,
    e.source,
    COALESCE(
        jsonb_agg(DISTINCT jsonb_build_object('scheme', ei.id_scheme, 'id', ei.external_id))
            FILTER (WHERE ei.external_id IS NOT NULL),
        '[]'::jsonb
    ) AS identifiers,
    COALESCE(
        array_agg(DISTINCT ea.alias) FILTER (WHERE ea.alias IS NOT NULL),
        ARRAY[]::text[]
    ) AS aliases,
    COALESCE(
        jsonb_agg(DISTINCT jsonb_build_object(
            'predicate', er.predicate,
            'object_id', er.object_entity_id,
            'confidence', er.confidence
        )) FILTER (WHERE er.id IS NOT NULL),
        '[]'::jsonb
    ) AS relationships,
    COALESCE(cnt.doc_count, 0) AS citing_paper_count
FROM entities e
LEFT JOIN entity_identifiers ei ON ei.entity_id = e.id
LEFT JOIN entity_aliases ea ON ea.entity_id = e.id
LEFT JOIN entity_relationships er ON er.subject_entity_id = e.id
LEFT JOIN LATERAL (
    SELECT count(*) AS doc_count
    FROM document_entities de
    WHERE de.entity_id = e.id
) cnt ON true
GROUP BY e.id, e.canonical_name, e.entity_type, e.discipline, e.source, cnt.doc_count;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_entity_ctx_id
    ON agent_entity_context (entity_id);

-- ---------------------------------------------------------------------------
-- 3. agent_dataset_context — one row per dataset with linked entities and
--    citing papers
-- ---------------------------------------------------------------------------

CREATE MATERIALIZED VIEW IF NOT EXISTS agent_dataset_context AS
SELECT
    d.id AS dataset_id,
    d.name AS dataset_name,
    d.source,
    d.discipline,
    d.description,
    COALESCE(
        jsonb_agg(DISTINCT jsonb_build_object(
            'entity_id', e.id,
            'name', e.canonical_name,
            'type', e.entity_type,
            'relationship', dse.relationship
        )) FILTER (WHERE e.id IS NOT NULL),
        '[]'::jsonb
    ) AS linked_entities,
    COALESCE(
        jsonb_agg(DISTINCT jsonb_build_object(
            'bibcode', p.bibcode,
            'title', p.title,
            'link_type', dd.link_type
        )) FILTER (WHERE p.bibcode IS NOT NULL),
        '[]'::jsonb
    ) AS citing_papers
FROM datasets d
LEFT JOIN dataset_entities dse ON dse.dataset_id = d.id
LEFT JOIN entities e ON e.id = dse.entity_id
LEFT JOIN document_datasets dd ON dd.dataset_id = d.id
LEFT JOIN papers p ON p.bibcode = dd.bibcode
GROUP BY d.id, d.name, d.source, d.discipline, d.description;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_dataset_ctx_id
    ON agent_dataset_context (dataset_id);

COMMIT;

-- 044_converge_entity_dictionary.sql
-- 044_converge_entity_dictionary.sql
-- Migrate entity_dictionary entries into the entities + entity_aliases +
-- entity_identifiers graph so the linking pipeline can resolve against all
-- ontology sources (ASCL, PhySH, PwC, VizieR, AAS) — not just GCMD/SPASE.

BEGIN;

-- Step 1: Insert entities from dictionary, skipping any that already exist.
INSERT INTO entities (canonical_name, entity_type, source, discipline, properties)
SELECT ed.canonical_name, ed.entity_type, ed.source, ed.discipline, ed.metadata
FROM entity_dictionary ed
ON CONFLICT (canonical_name, entity_type, source) DO NOTHING;

-- Step 2: Expand aliases array into entity_aliases rows.
-- Only process dictionary entries that have aliases.
INSERT INTO entity_aliases (entity_id, alias, alias_source)
SELECT e.id, unnest(ed.aliases), ed.source
FROM entity_dictionary ed
JOIN entities e ON e.canonical_name = ed.canonical_name
    AND e.entity_type = ed.entity_type
    AND e.source = ed.source
WHERE ed.aliases IS NOT NULL AND cardinality(ed.aliases) > 0
ON CONFLICT DO NOTHING;

-- Step 3: Insert external identifiers where they exist.
INSERT INTO entity_identifiers (entity_id, id_scheme, external_id, is_primary)
SELECT e.id, ed.source, ed.external_id, true
FROM entity_dictionary ed
JOIN entities e ON e.canonical_name = ed.canonical_name
    AND e.entity_type = ed.entity_type
    AND e.source = ed.source
WHERE ed.external_id IS NOT NULL AND ed.external_id <> ''
ON CONFLICT (id_scheme, external_id) DO NOTHING;

COMMIT;

-- 025_entity_audit_log.sql
-- 025_entity_audit_log.sql
-- Audit log for entity merge and split operations.
-- Tracks when resolution decisions change (entity A merged with B, or split).

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. entity_merge_log — records when two entities are merged
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_merge_log (
    id SERIAL PRIMARY KEY,
    old_entity_id INT NOT NULL,
    new_entity_id INT NOT NULL REFERENCES entities(id),
    reason TEXT,
    merged_by TEXT,
    merged_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entity_merge_log_old ON entity_merge_log(old_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_merge_log_new ON entity_merge_log(new_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_merge_log_at ON entity_merge_log(merged_at);

-- ---------------------------------------------------------------------------
-- 2. entity_split_log — records when an entity is split into children
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_split_log (
    id SERIAL PRIMARY KEY,
    parent_entity_id INT NOT NULL,
    child_entity_ids INT[] NOT NULL,
    reason TEXT,
    split_by TEXT,
    split_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entity_split_log_parent ON entity_split_log(parent_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_split_log_at ON entity_split_log(split_at);

COMMIT;

-- 026_spdf_spase_crosswalk.sql
-- 026_spdf_spase_crosswalk.sql
-- Explicit mapping between CDAWeb (SPDF) dataset IDs and SPASE ResourceIDs.
-- e.g. AC_H2_MFI -> spase://NASA/NumericalData/ACE/MAG/L2/PT16S

BEGIN;

CREATE TABLE IF NOT EXISTS spdf_spase_crosswalk (
    id SERIAL PRIMARY KEY,
    spdf_id TEXT NOT NULL,
    spase_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'spdf_harvest',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (spdf_id, spase_id)
);

CREATE INDEX IF NOT EXISTS idx_crosswalk_spdf ON spdf_spase_crosswalk(spdf_id);
CREATE INDEX IF NOT EXISTS idx_crosswalk_spase ON spdf_spase_crosswalk(spase_id);

-- Populate from existing datasets harvested by SPDF
-- The canonical_id column holds the CDAWeb dataset ID, and
-- properties->>'spase_resource_id' holds the SPASE ResourceID.
INSERT INTO spdf_spase_crosswalk (spdf_id, spase_id, source)
SELECT
    canonical_id,
    properties->>'spase_resource_id',
    'spdf_harvest_seed'
FROM datasets
WHERE source = 'spdf'
  AND properties->>'spase_resource_id' IS NOT NULL
  AND properties->>'spase_resource_id' != ''
ON CONFLICT DO NOTHING;

COMMIT;

-- 027_per_model_hnsw_rebuild.sql
-- 027_per_model_hnsw_rebuild.sql
-- Create per-model partial HNSW indexes for all active embedding models.
--
-- pgvector 0.8.2 supports parallel HNSW builds (max_parallel_maintenance_workers=7)
-- and iterative scan (relaxed_order) for filtered queries.
--
-- Index parameters:
--   m=16           — graph degree, good balance of recall vs build time
--   ef_construction=64 — lower than 200 for 32M rows (acceptable recall,
--                         significantly faster build). Matches CLAUDE.md guidance.
--
-- The indus model (32M rows, 768d) is the critical index. specter2 and nomic
-- (20K rows each) are small but included for completeness.
--
-- NOTE: The indus index build on 32M rows will take significant time even with
-- parallel workers. Run during low-traffic window.

BEGIN;

-- Drop stale indexes from earlier migrations if they somehow survived
DROP INDEX IF EXISTS idx_embed_hnsw_specter2;
DROP INDEX IF EXISTS idx_embed_hnsw_openai;

-- Per-model HNSW indexes with cosine distance.
-- The embedding column is untyped vector (migration 023), so we must cast
-- to vector(768) in the index expression to pin the dimensionality.

-- indus: 32M rows, 768d — primary retrieval model
CREATE INDEX IF NOT EXISTS idx_embed_hnsw_indus
    ON paper_embeddings USING hnsw ((embedding::vector(768)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE model_name = 'indus';

-- specter2: 20K rows, 768d — citation-proximity similarity
CREATE INDEX IF NOT EXISTS idx_embed_hnsw_specter2
    ON paper_embeddings USING hnsw ((embedding::vector(768)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE model_name = 'specter2';

-- nomic: 20K rows, 768d
CREATE INDEX IF NOT EXISTS idx_embed_hnsw_nomic
    ON paper_embeddings USING hnsw ((embedding::vector(768)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE model_name = 'nomic';

COMMIT;

-- 028_entity_schema_hardening.sql
-- 028_entity_schema_hardening.sql
-- M1: Harden the entity graph for tiered linking.
--   - New ENUMs for ambiguity_class and link_policy on entities.
--   - Add tier / tier_version to document_entities.
--   - Replace document_entities PK with (bibcode, entity_id, link_type, tier)
--     so the same (bibcode, entity, link_type) can exist at multiple tiers.
--
-- Idempotent: safe to re-run.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. ENUM: entity_ambiguity_class
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    CREATE TYPE entity_ambiguity_class AS ENUM (
        'unique',
        'domain_safe',
        'homograph',
        'banned'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;

-- ---------------------------------------------------------------------------
-- 2. ENUM: entity_link_policy
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    CREATE TYPE entity_link_policy AS ENUM (
        'open',
        'context_required',
        'llm_only',
        'banned'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END
$$;

-- ---------------------------------------------------------------------------
-- 3. entities: add ambiguity_class and link_policy columns (nullable — legacy
--    rows predate the classification pass)
-- ---------------------------------------------------------------------------

ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS ambiguity_class entity_ambiguity_class;

ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS link_policy entity_link_policy;

-- ---------------------------------------------------------------------------
-- 4. document_entities: add tier / tier_version
--    tier semantics (u04 will document in detail):
--       0 = legacy / default
--       1 = high-precision (exact ID match, unique canonical)
--       2 = medium (alias + context)
--       3 = low / LLM-adjudicated
-- ---------------------------------------------------------------------------

ALTER TABLE document_entities
    ADD COLUMN IF NOT EXISTS tier SMALLINT NOT NULL DEFAULT 0;

ALTER TABLE document_entities
    ADD COLUMN IF NOT EXISTS tier_version INT NOT NULL DEFAULT 1;

-- ---------------------------------------------------------------------------
-- 5. Replace primary key with one that includes tier
--    (DROP + ADD inside DO-block for idempotency: if the PK already matches
--    the new shape, leave it alone)
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    current_pk_cols text;
BEGIN
    SELECT string_agg(a.attname, ',' ORDER BY array_position(c.conkey, a.attnum))
      INTO current_pk_cols
      FROM pg_constraint c
      JOIN pg_attribute a
        ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
     WHERE c.conrelid = 'public.document_entities'::regclass
       AND c.contype  = 'p';

    IF current_pk_cols IS DISTINCT FROM 'bibcode,entity_id,link_type,tier' THEN
        IF current_pk_cols IS NOT NULL THEN
            EXECUTE 'ALTER TABLE document_entities DROP CONSTRAINT document_entities_pkey';
        END IF;
        EXECUTE 'ALTER TABLE document_entities
                 ADD CONSTRAINT document_entities_pkey
                 PRIMARY KEY (bibcode, entity_id, link_type, tier)';
    END IF;
END
$$;

-- Helpful index for tier-scoped deletes/queries
CREATE INDEX IF NOT EXISTS idx_document_entities_tier
    ON document_entities(tier);

COMMIT;

-- 029_ontology_version_pinning.sql
-- 029_ontology_version_pinning.sql
-- S4: Pin each entity to the ontology version it was harvested from, and
-- allow a newer entity row to supersede an older one via self-reference.
--
-- Note on type: entities.id is INTEGER (SERIAL). The PRD spec text said
-- "supersedes_id BIGINT" but the foreign-key column MUST match the referent.
-- We use INTEGER here; if the id sequence is migrated to bigint later, a
-- follow-up migration will widen supersedes_id in lockstep.
--
-- Idempotent: safe to re-run.

BEGIN;

ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS source_version TEXT;

ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS supersedes_id INTEGER;

-- Self-FK guarded by DO block for idempotency
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint
         WHERE conname = 'entities_supersedes_id_fkey'
           AND conrelid = 'public.entities'::regclass
    ) THEN
        ALTER TABLE entities
            ADD CONSTRAINT entities_supersedes_id_fkey
            FOREIGN KEY (supersedes_id)
            REFERENCES entities(id)
            ON DELETE SET NULL;
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_entities_supersedes_id
    ON entities(supersedes_id)
    WHERE supersedes_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_entities_source_version
    ON entities(source, source_version)
    WHERE source_version IS NOT NULL;

COMMIT;

-- 030_staging_and_promote_harvest.sql
-- 030_staging_and_promote_harvest.sql
-- D3: Public-schema staging tables for entity-graph harvests and a
-- promote_harvest(run_id) skeleton function.
--
-- Distinct from the `staging.*` schema created in 022_staging_entities.sql:
-- those are used by the existing entity pipeline; these are for the new
-- harvest-run-based promote flow (u04 will fill in the function body).
--
-- Idempotent: safe to re-run.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. entities_staging — mirrors public.entities + staging_run_id
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entities_staging (
    id               BIGSERIAL PRIMARY KEY,
    staging_run_id   BIGINT  NOT NULL,
    canonical_name   TEXT    NOT NULL,
    entity_type      TEXT    NOT NULL,
    discipline       TEXT,
    source           TEXT    NOT NULL,
    source_version   TEXT,
    ambiguity_class  TEXT,
    link_policy      TEXT,
    properties       JSONB   DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_entities_staging_run
    ON entities_staging(staging_run_id);
CREATE INDEX IF NOT EXISTS idx_entities_staging_natural_key
    ON entities_staging(canonical_name, entity_type, source);

-- ---------------------------------------------------------------------------
-- 2. entity_aliases_staging
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_aliases_staging (
    id              BIGSERIAL PRIMARY KEY,
    staging_run_id  BIGINT NOT NULL,
    staging_entity_id BIGINT,  -- local FK into entities_staging.id (unenforced for COPY speed)
    canonical_name  TEXT,      -- natural-key fallback when staging_entity_id is null
    entity_type     TEXT,
    source          TEXT,
    alias           TEXT NOT NULL,
    alias_source    TEXT
);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_staging_run
    ON entity_aliases_staging(staging_run_id);

-- ---------------------------------------------------------------------------
-- 3. entity_identifiers_staging
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_identifiers_staging (
    id              BIGSERIAL PRIMARY KEY,
    staging_run_id  BIGINT NOT NULL,
    staging_entity_id BIGINT,
    canonical_name  TEXT,
    entity_type     TEXT,
    source          TEXT,
    id_scheme       TEXT NOT NULL,
    external_id     TEXT NOT NULL,
    is_primary      BOOLEAN DEFAULT false
);

CREATE INDEX IF NOT EXISTS idx_entity_identifiers_staging_run
    ON entity_identifiers_staging(staging_run_id);

-- ---------------------------------------------------------------------------
-- 4. promote_harvest(run_id BIGINT) — stub
--
-- u04 will replace the body with an atomic batch upsert from *_staging into
-- the public entity tables, with tier assignment based on match quality, and
-- clean-up of the staging rows on success. For now this stub exists purely so
-- u01 can guarantee the function signature is present; it returns 0 and does
-- NOT raise, so downstream scaffolding can call it safely.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION promote_harvest(run_id BIGINT)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    staged_count INTEGER;
BEGIN
    SELECT count(*) INTO staged_count
      FROM entities_staging
     WHERE staging_run_id = run_id;

    -- Stub: u04 will implement the full upsert + tier assignment here.
    -- For now just return the number of staged rows so callers can assert
    -- the function exists and is wired to the staging table.
    RETURN COALESCE(staged_count, 0);
END
$$;

COMMENT ON FUNCTION promote_harvest(BIGINT) IS
    'STUB — u01. u04 will implement atomic promote of *_staging rows for '
    'staging_run_id into public entity tables with tier assignment.';

COMMIT;

-- 031_query_log.sql
-- 031_query_log.sql
-- M3.5.0: Extend the existing query_log table (created in 016_query_log.sql)
-- with the columns the new MCP instrumentation pass needs.
--
-- We ALTER the existing table rather than recreate it so historical rows and
-- the existing id sequence stay intact. The acceptance criterion for this
-- migration only requires that the new columns exist.
--
-- Idempotent: safe to re-run.

BEGIN;

-- New instrumentation columns (tool/query/result_count/session_id/is_test).
-- `ts` is an alias-style timestamp that defaults to now() so freshly inserted
-- rows are tagged without needing every call site to set it explicitly.
ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS ts TIMESTAMPTZ NOT NULL DEFAULT now();

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS tool TEXT;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS query TEXT;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS result_count INT;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS session_id TEXT;

ALTER TABLE query_log
    ADD COLUMN IF NOT EXISTS is_test BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_query_log_ts          ON query_log(ts);
CREATE INDEX IF NOT EXISTS idx_query_log_tool        ON query_log(tool);
CREATE INDEX IF NOT EXISTS idx_query_log_session_id  ON query_log(session_id);

COMMIT;

-- 032_core_promotion_log.sql
-- 032_core_promotion_log.sql
-- M3.5.2: Curated entity core lifecycle tables.
--
-- Creates:
--   * curated_entity_core — the actual core membership list (10K cap)
--   * core_promotion_log  — append-only event log of promote/demote events
--
-- Idempotent: safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS curated_entity_core (
    entity_id      INT PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    query_hits_14d INT NOT NULL DEFAULT 0,
    promoted_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_curated_entity_core_hits
    ON curated_entity_core(query_hits_14d);

CREATE TABLE IF NOT EXISTS core_promotion_log (
    id             SERIAL PRIMARY KEY,
    entity_id      INT NOT NULL,
    action         TEXT NOT NULL CHECK (action IN ('promote', 'demote')),
    query_hits_14d INT,
    reason         TEXT,
    ts             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_core_promotion_log_entity
    ON core_promotion_log(entity_id);

CREATE INDEX IF NOT EXISTS idx_core_promotion_log_ts
    ON core_promotion_log(ts);

COMMIT;

-- 033_fusion_mv.sql
-- 033_fusion_mv.sql
-- M8: Fused confidence materialized view for document_entities.
--
--   - tier_weight(SMALLINT) IMMUTABLE SQL function mapping tier -> calibration weight.
--   - tier_weight_calibration_log table records each weight version.
--   - document_entities_canonical materialized view computes fused confidence via
--     noisy-OR:  fused = 1 - exp(sum(ln(1 - c_t * w_t)))
--   - fusion_mv_state table records dirty bit and last_refresh_at for the
--     rate-limited refresh loop.
--
-- Idempotent: safe to re-run.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. tier_weight SQL function (IMMUTABLE LEAKPROOF PARALLEL SAFE)
--
--    Placeholder calibration weights — will be replaced by calibration runs
--    that write to tier_weight_calibration_log. The function body is
--    recreated by each calibration migration.
--
--    Tier semantics (u04 / M1):
--      1 = exact id match (high precision)
--      2 = alias + context
--      3 = deprecated keyword tier
--      4 = LLM-adjudicated fallback
--      5 = JIT path
--      else (including 0 / legacy) -> 0.50 default
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION tier_weight(tier SMALLINT)
RETURNS DOUBLE PRECISION
LANGUAGE sql
IMMUTABLE
LEAKPROOF
PARALLEL SAFE
AS $$
    SELECT CASE tier
        WHEN 1::SMALLINT THEN 0.98::float8
        WHEN 2::SMALLINT THEN 0.85::float8
        WHEN 3::SMALLINT THEN 0.92::float8
        WHEN 4::SMALLINT THEN 0.50::float8
        WHEN 5::SMALLINT THEN 0.88::float8
        ELSE 0.50::float8
    END
$$;

-- ---------------------------------------------------------------------------
-- 2. tier_weight_calibration_log — one row per calibration version
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tier_weight_calibration_log (
    id          SERIAL PRIMARY KEY,
    version     TEXT NOT NULL UNIQUE,
    weights     JSONB NOT NULL,
    notes       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO tier_weight_calibration_log (version, weights, notes)
VALUES (
    'placeholder_2026-04-12',
    '{"1": 0.98, "2": 0.85, "3": 0.92, "4": 0.50, "5": 0.88, "default": 0.50}'::jsonb,
    'Initial placeholder weights. Tier 3 is deprecated; tier 5 is the JIT path. Replace via a calibration run.'
)
ON CONFLICT (version) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 3. fusion_mv_state — dirty bit + last refresh timestamp for the refresh job
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fusion_mv_state (
    id              INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    dirty           BOOLEAN NOT NULL DEFAULT true,
    last_refresh_at TIMESTAMPTZ
);

INSERT INTO fusion_mv_state (id, dirty, last_refresh_at)
VALUES (1, true, NULL)
ON CONFLICT (id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 4. document_entities_canonical materialized view
--
--    Noisy-OR fusion of per-tier confidences:
--        fused = 1 - exp(sum(ln(1 - LEAST(0.9999, c_t * w_t))))
--
--    The LEAST(0.9999, ...) clamp prevents ln(0) when a (confidence, weight)
--    pair happens to land on 1.0. For typical weights (<= 0.98) and
--    confidences (<= 1.0), the clamp is never binding, so the closed-form
--    equivalence holds within floating-point tolerance.
--
--    Drop + recreate (idempotent). The UNIQUE index below is what makes
--    REFRESH MATERIALIZED VIEW CONCURRENTLY legal.
-- ---------------------------------------------------------------------------

DROP MATERIALIZED VIEW IF EXISTS document_entities_canonical CASCADE;

CREATE MATERIALIZED VIEW document_entities_canonical AS
SELECT
    de.bibcode,
    de.entity_id,
    1 - exp(
        sum(
            ln(
                1 - LEAST(
                    0.9999::float8,
                    GREATEST(
                        0.0::float8,
                        de.confidence::float8 * tier_weight(de.tier)
                    )
                )
            )
        )
    ) AS fused_confidence,
    count(*)                           AS link_count,
    array_agg(DISTINCT de.tier ORDER BY de.tier) AS contributing_tiers,
    max(de.tier_version)               AS max_tier_version,
    max(de.harvest_run_id)             AS latest_harvest_run_id
FROM document_entities de
WHERE de.confidence IS NOT NULL
GROUP BY de.bibcode, de.entity_id;

-- UNIQUE index required for REFRESH CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS idx_dec_bibcode_entity
    ON document_entities_canonical (bibcode, entity_id);

-- Entity-scoped top-k lookups (hot path for resolver reads)
CREATE INDEX IF NOT EXISTS idx_dec_entity_fused
    ON document_entities_canonical (entity_id, fused_confidence DESC);

-- Bibcode lookups (document-centric reads)
CREATE INDEX IF NOT EXISTS idx_dec_bibcode
    ON document_entities_canonical (bibcode);

COMMIT;

-- 034_jit_cache.sql
-- 034_jit_cache.sql
-- M11b: Persisted JIT cache (partitioned by expires_at).
--
-- Creates:
--   * document_entities_jit_cache — partitioned cache table with tier=5,
--     candidate_set_hash / model_version / expires_at columns.
--   * document_entities_jit_cache_default — catch-all DEFAULT partition so
--     INSERTs succeed before the daily partition-creation cron runs.
--   * Supporting index on (bibcode, candidate_set_hash, model_version).
--
-- Idempotent: safe to re-run.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Partitioned parent table
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS document_entities_jit_cache (
    bibcode             TEXT        NOT NULL,
    entity_id           INT         NOT NULL,
    link_type           TEXT        NOT NULL,
    confidence          REAL,
    match_method        TEXT,
    evidence            JSONB,
    harvest_run_id      INT,
    tier                SMALLINT    NOT NULL DEFAULT 5,
    tier_version        INT         NOT NULL DEFAULT 1,
    candidate_set_hash  TEXT        NOT NULL,
    model_version       TEXT        NOT NULL,
    expires_at          TIMESTAMPTZ NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT document_entities_jit_cache_tier_check CHECK (tier = 5),
    PRIMARY KEY (
        bibcode,
        entity_id,
        link_type,
        candidate_set_hash,
        model_version,
        expires_at
    )
) PARTITION BY RANGE (expires_at);

-- ---------------------------------------------------------------------------
-- 2. DEFAULT partition — catches any row whose expires_at does not fall into
--    a dated partition. The daily cron (scripts/jit_cache_cleanup.py) will
--    create forward-dated range partitions and eventually detach/drop the
--    default — but for tests and bootstrap the DEFAULT is sufficient so
--    INSERTs always succeed.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS document_entities_jit_cache_default
    PARTITION OF document_entities_jit_cache DEFAULT;

-- ---------------------------------------------------------------------------
-- 3. Lookup index — the hot path is
--    (bibcode, candidate_set_hash, model_version) equality.
--    Defined on the parent so all partitions inherit.
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_document_entities_jit_cache_lookup
    ON document_entities_jit_cache (bibcode, candidate_set_hash, model_version);

CREATE INDEX IF NOT EXISTS idx_document_entities_jit_cache_expires
    ON document_entities_jit_cache (expires_at);

COMMIT;

-- 035_entity_link_audits.sql
-- 035_entity_link_audits.sql
-- M9: Human + LLM-judge audit table for entity linking.
--
-- Stores per-(tier, bibcode, entity_id, annotator) labels from either
-- human annotators or an LLM-judge. Labels are one of
-- ('correct','incorrect','ambiguous').
--
-- Primary key is (tier, bibcode, entity_id, annotator) so multiple
-- annotators can label the same link and disagreement is first-class.
--
-- Idempotent: safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS entity_link_audits (
    tier       SMALLINT NOT NULL,
    bibcode    TEXT     NOT NULL,
    entity_id  BIGINT   NOT NULL,
    annotator  TEXT     NOT NULL,
    label      TEXT     NOT NULL CHECK (label IN ('correct','incorrect','ambiguous')),
    note       TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tier, bibcode, entity_id, annotator)
);

-- Secondary indexes for analytics reads
CREATE INDEX IF NOT EXISTS idx_entity_link_audits_tier_label
    ON entity_link_audits (tier, label);

CREATE INDEX IF NOT EXISTS idx_entity_link_audits_annotator
    ON entity_link_audits (annotator);

COMMIT;

-- 036_link_runs_watermark.sql
-- 036_link_runs_watermark.sql
-- M10 / u13: Watermark table for the incremental entity-linking pipeline,
-- plus a tiny alerts sink used by the circuit-breaker and
-- watermark-staleness checks.
--
-- link_runs is append-only: every incremental run inserts one row with
-- the max entry_date it processed, the row count, and a status so that
-- a tripped circuit breaker (graceful degradation) can still advance the
-- watermark while recording that it did so with zero link writes.
--
-- alerts is a generic sink the breaker and other M10 health checks write
-- into. Operational paging reads rows with severity='page' and marks
-- acked_at when handled.
--
-- Idempotent: safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS link_runs (
    run_id         BIGSERIAL   PRIMARY KEY,
    max_entry_date TIMESTAMPTZ,
    timestamp      TIMESTAMPTZ NOT NULL DEFAULT now(),
    rows_linked    INTEGER     NOT NULL DEFAULT 0,
    status         TEXT        NOT NULL DEFAULT 'ok'
        CHECK (status IN ('ok', 'tripped', 'failed')),
    trip_count     INTEGER     NOT NULL DEFAULT 0,
    note           TEXT
);

CREATE INDEX IF NOT EXISTS idx_link_runs_timestamp_desc
    ON link_runs (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_link_runs_max_entry_date_desc
    ON link_runs (max_entry_date DESC NULLS LAST);

CREATE TABLE IF NOT EXISTS alerts (
    id         BIGSERIAL   PRIMARY KEY,
    severity   TEXT        NOT NULL
        CHECK (severity IN ('info', 'warn', 'page')),
    source     TEXT        NOT NULL,
    message    TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    acked_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_alerts_unacked_severity
    ON alerts (severity, created_at DESC)
    WHERE acked_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_alerts_source
    ON alerts (source);

COMMIT;

-- 037_citation_consistency_and_disputes.sql
-- Migration 037: Citation consistency proxy column + entity link disputes.
--
-- Part of PRD §S1 (citation consistency precision proxy) and §S5 (researcher
-- feedback loop). See u14-should-haves work unit and
-- .claude/prd-build-artifacts/plan-u14-should-haves.md.
--
-- S1: document_entities.citation_consistency stores the fraction of outbound
--     citations from `bibcode` that also link to `entity_id` at the same
--     link_type. Null means "not computed yet".
--
-- S5: entity_link_disputes is a lightweight append-only table where
--     researchers (or human curators) can flag suspected incorrect links.
--     Not joined into the hot path — consumed by offline audit jobs.

BEGIN;

-- -----------------------------------------------------------------------------
-- 1. document_entities.citation_consistency
-- -----------------------------------------------------------------------------
ALTER TABLE document_entities
    ADD COLUMN IF NOT EXISTS citation_consistency REAL;

COMMENT ON COLUMN document_entities.citation_consistency IS
    'Fraction of outbound citations from bibcode that also link to entity_id '
    '(precision proxy, 0..1). NULL = not yet computed. See PRD §S1.';

-- -----------------------------------------------------------------------------
-- 2. entity_link_disputes
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entity_link_disputes (
    id          BIGSERIAL PRIMARY KEY,
    bibcode     TEXT,
    entity_id   BIGINT,
    reason      TEXT,
    reported_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    tier        SMALLINT
);

CREATE INDEX IF NOT EXISTS idx_entity_link_disputes_bibcode
    ON entity_link_disputes(bibcode);

CREATE INDEX IF NOT EXISTS idx_entity_link_disputes_entity
    ON entity_link_disputes(entity_id);

CREATE INDEX IF NOT EXISTS idx_entity_link_disputes_reported_at
    ON entity_link_disputes(reported_at);

COMMENT ON TABLE entity_link_disputes IS
    'Append-only researcher feedback on suspected incorrect document->entity '
    'links. Consumed by offline audit jobs, not the hot path. See PRD §S5.';

COMMIT;

-- 038_papers_external_ids.sql
-- Migration 038: papers_external_ids crosswalk table.
--
-- Part of PRD Build 5 (scix_experiments-wqr) W1 (wqr.2). First of the
-- external-source ingestion pipeline: a single crosswalk table that joins
-- ADS bibcodes to identifiers from OpenAlex / arXiv / Semantic Scholar /
-- PubMed / PMC, plus has_* boolean flags tracking which structured full-text
-- sources have been harvested.
--
-- SAFETY: this table MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory for
-- the 32M-embedding loss that prompted this rule. The block at the bottom of
-- this file asserts relpersistence='p' and raises if not.

BEGIN;

CREATE TABLE IF NOT EXISTS papers_external_ids (
    bibcode              TEXT PRIMARY KEY REFERENCES papers(bibcode),
    doi                  TEXT,
    arxiv_id             TEXT,
    openalex_id          TEXT,
    s2_corpus_id         BIGINT,
    s2_paper_id          TEXT,
    pmcid                TEXT,
    pmid                 BIGINT,
    has_ads_body         BOOLEAN NOT NULL DEFAULT false,
    has_arxiv_source     BOOLEAN NOT NULL DEFAULT false,
    has_ar5iv_html       BOOLEAN NOT NULL DEFAULT false,
    has_s2orc_body       BOOLEAN NOT NULL DEFAULT false,
    openalex_has_pdf_url BOOLEAN NOT NULL DEFAULT false,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE papers_external_ids IS
    'Crosswalk from ADS bibcode to external identifiers (OpenAlex / arXiv / '
    'Semantic Scholar / PubMed). Populated incrementally by PRD Build 5 work '
    'units W1-W6. has_* flags track which structured full-text sources have '
    'been ingested for each paper.';

-- B-tree lookup indexes on the join keys. Not unique — a single external id
-- can in rare cases point to multiple bibcodes (arxiv preprints with multiple
-- ADS records, DOI collisions between reprints, etc.).
CREATE INDEX IF NOT EXISTS idx_papers_external_ids_doi
    ON papers_external_ids (doi);
CREATE INDEX IF NOT EXISTS idx_papers_external_ids_arxiv_id
    ON papers_external_ids (arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_external_ids_openalex_id
    ON papers_external_ids (openalex_id);
CREATE INDEX IF NOT EXISTS idx_papers_external_ids_s2_corpus_id
    ON papers_external_ids (s2_corpus_id);

-- updated_at trigger: the DEFAULT now() only fires on INSERT. The column is
-- meaningful only if it also advances on UPDATE.
CREATE OR REPLACE FUNCTION papers_external_ids_touch() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_papers_external_ids_updated_at ON papers_external_ids;
CREATE TRIGGER trig_papers_external_ids_updated_at
    BEFORE UPDATE ON papers_external_ids
    FOR EACH ROW EXECUTE FUNCTION papers_external_ids_touch();

-- Safety assertion: refuse to leave the migration with an UNLOGGED table.
DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'papers_external_ids' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'papers_external_ids did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'papers_external_ids must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;

-- 039_papers_ads_body.sql
-- Migration 039: papers_ads_body — full text from the ADS `body` field.
--
-- Part of PRD Build 5 (scix_experiments-wqr) W1 (wqr.2). Dedicated table for
-- the ADS-harvested full text of each paper. Although papers.body already
-- exists (migration 010), this dedicated table:
--   - Gives the body its own primary key so the 17.8M-row bulk load can drop
--     indexes cleanly without touching the hot `papers` table.
--   - Stores a body-only GENERATED STORED tsvector, enabling BM25-style
--     queries on full text independent of the title/abstract tsv on papers
--     (migration 003), which is trigger-maintained and doesn't cover body.
--   - Isolates the 17.8M-row write volume from papers, so reindexing the body
--     tsv does not block queries against papers.
--
-- SAFETY: this table MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory. The
-- block at the bottom of this file asserts relpersistence='p'.
--
-- Note on text search config: the tsv column uses the built-in 'english'
-- config, not scix_english (migration 003), because GENERATED STORED columns
-- require an immutable configuration reference that round-trips cleanly
-- through pg_dump. scix_english is reachable by name but upgrading this
-- column to use it is a follow-up (separate bead, not this one). The
-- tradeoff is that hyphenated tokens like "X-ray" get split on hyphen; for
-- full-text recall on body text this is acceptable.

BEGIN;

CREATE TABLE IF NOT EXISTS papers_ads_body (
    bibcode      TEXT PRIMARY KEY REFERENCES papers(bibcode),
    body_text    TEXT NOT NULL,
    body_length  INT  NOT NULL,
    harvested_at TIMESTAMPTZ NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', body_text)) STORED
);

COMMENT ON TABLE papers_ads_body IS
    'Full text from the ADS `body` field, harvested ~55% of ADS papers. '
    'Populated by scripts/ingest_ads_body.py. body-only tsvector enables '
    'full-text search independent of the title/abstract tsv on papers.';

COMMENT ON COLUMN papers_ads_body.tsv IS
    'Generated stored tsvector using the built-in `english` config. See '
    'migration 039 preamble for the tradeoff vs scix_english.';

CREATE INDEX IF NOT EXISTS idx_papers_ads_body_tsv
    ON papers_ads_body USING GIN (tsv);

-- Safety assertion: refuse to leave the migration with an UNLOGGED table.
DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'papers_ads_body' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'papers_ads_body did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'papers_ads_body must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;

-- 040_openalex_tables.sql
-- Migration 040: OpenAlex S3 snapshot tables — papers_openalex + works_references.
--
-- Part of PRD Build 5 (scix_experiments-wqr) W2 (wqr.3). OpenAlex is the
-- graph backbone: ~260M works, CC0 licensed, no auth. These two tables hold
-- the pruned Work records and their citation edges independently of the ADS
-- `papers` table, enabling graph analytics across the full OpenAlex corpus
-- without requiring a bibcode match.
--
-- The loader (src/scix/sources/openalex.py) populates these via DuckDB-over-
-- Parquet staging and binary COPY, then joins on DOI/arXiv ID to update
-- papers_external_ids.openalex_id for matched papers.
--
-- SAFETY: both tables MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory for
-- the 32M-embedding loss that prompted this rule. The block at the bottom of
-- this file asserts relpersistence='p' for both tables.

BEGIN;

-- ---------------------------------------------------------------------------
-- papers_openalex — pruned OpenAlex Work records
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS papers_openalex (
    openalex_id          TEXT PRIMARY KEY,
    doi                  TEXT,
    title                TEXT,
    publication_year     SMALLINT,
    abstract             TEXT,
    topics               JSONB,
    open_access          JSONB,
    best_oa_location     JSONB,
    cited_by_count       INT,
    referenced_works_count INT,
    type                 TEXT,
    updated_date         DATE,
    created_date         DATE
);

COMMENT ON TABLE papers_openalex IS
    'Pruned OpenAlex Work records from the S3 snapshot. Populated by '
    'src/scix/sources/openalex.py. CC0 licensed, ~260M works. The '
    'abstract field is reconstructed from abstract_inverted_index.';

-- Lookup indexes for join operations
CREATE INDEX IF NOT EXISTS idx_papers_openalex_doi
    ON papers_openalex (doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_openalex_year
    ON papers_openalex (publication_year);
CREATE INDEX IF NOT EXISTS idx_papers_openalex_updated
    ON papers_openalex (updated_date);

-- ---------------------------------------------------------------------------
-- works_references — citation edges between OpenAlex works
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS works_references (
    source_openalex_id      TEXT NOT NULL,
    referenced_openalex_id  TEXT NOT NULL,
    PRIMARY KEY (source_openalex_id, referenced_openalex_id)
);

COMMENT ON TABLE works_references IS
    'Citation edges between OpenAlex works. source_openalex_id cites '
    'referenced_openalex_id. No FK to papers_openalex because referenced '
    'works may be outside the ingested corpus (xpac expansion).';

-- Reverse-lookup index for cited-by queries
CREATE INDEX IF NOT EXISTS idx_works_references_target
    ON works_references (referenced_openalex_id);

-- ---------------------------------------------------------------------------
-- Safety assertions: both tables MUST be LOGGED (relpersistence='p')
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    rp CHAR(1);
BEGIN
    -- Check papers_openalex
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'papers_openalex' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'papers_openalex did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'papers_openalex must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;

    -- Check works_references
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'works_references' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'works_references did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'works_references must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;

-- 041_papers_fulltext.sql
-- Migration 041: papers_fulltext — structured full-text from multiple sources.
--
-- Part of PRD Build 5 (scix_experiments-wqr) W4 (wqr.5). Single destination
-- table for structured full-text parsed from ar5iv HTML, arXiv LaTeX, S2ORC,
-- ADS body, Docling, or abstract-only fallback. Each source parser writes
-- sections (with heading/level/text/offset), inline citations, figures, tables,
-- and equations as JSONB arrays.
--
-- The `source` column indicates provenance and determines licensing treatment:
-- ar5iv/arxiv_local are LaTeX-derived (ADR-006 internal-use-only), s2orc has
-- its own license, ads_body is ADS-licensed, abstract is always open.
--
-- SAFETY: this table MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory for
-- the 32M-embedding loss that prompted this rule. The block at the bottom of
-- this file asserts relpersistence='p' and raises if not.

BEGIN;

CREATE TABLE IF NOT EXISTS papers_fulltext (
    bibcode        TEXT PRIMARY KEY REFERENCES papers(bibcode),
    source         TEXT NOT NULL,
    sections       JSONB NOT NULL,
    inline_cites   JSONB NOT NULL,
    figures        JSONB NOT NULL DEFAULT '[]',
    tables         JSONB NOT NULL DEFAULT '[]',
    equations      JSONB NOT NULL DEFAULT '[]',
    parser_version TEXT NOT NULL,
    parsed_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE papers_fulltext IS
    'Structured full-text from multiple sources (ar5iv, arxiv_local, s2orc, '
    'ads_body, docling, abstract). Each row holds parsed sections, inline '
    'citations, figures, tables, and equations as JSONB arrays. Source column '
    'determines licensing treatment per ADR-006.';

COMMENT ON COLUMN papers_fulltext.source IS
    'Provenance tag: ar5iv | arxiv_local | s2orc | ads_body | docling | abstract. '
    'ar5iv/arxiv_local are LaTeX-derived (ADR-006 internal-use-only).';

COMMENT ON COLUMN papers_fulltext.sections IS
    'Array of {heading, level, text, offset} objects representing the document '
    'structure. Level 1 = top-level section, 2 = subsection, etc.';

COMMENT ON COLUMN papers_fulltext.inline_cites IS
    'Array of {offset, bib_ref, target_bibcode_or_null} objects representing '
    'inline citation references found during parsing.';

CREATE INDEX IF NOT EXISTS idx_papers_fulltext_source
    ON papers_fulltext (source);

-- Safety assertion: refuse to leave the migration with an UNLOGGED table.
DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'papers_fulltext' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'papers_fulltext did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'papers_fulltext must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;

-- 042_s2_datasets.sql
-- Migration 042: Semantic Scholar Datasets tables.
--
-- Part of PRD Build 5 (scix_experiments-wqr) W5 (wqr.6). Creates tables for
-- ingesting data from the Semantic Scholar Open Data Platform:
--   - papers_s2orc_raw: raw S2ORC records (parsed papers with body_text)
--   - papers_s2ag: pruned S2AG metadata (225M papers)
--   - s2_citations: S2AG citation edges with intent + influence flags
--   - ALTER citation_edges to add edge_attrs JSONB column
--
-- SAFETY: all tables MUST be LOGGED. An UNLOGGED table is truncated on crash
-- recovery — see migration 023 and the feedback_unlogged_tables memory for
-- the 32M-embedding loss that prompted this rule. The block at the bottom of
-- this file asserts relpersistence='p' for each new table.

BEGIN;

-- ---------------------------------------------------------------------------
-- papers_s2orc_raw — raw S2ORC parsed papers
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS papers_s2orc_raw (
    s2_corpus_id     BIGINT PRIMARY KEY,
    external_ids     JSONB,
    content          JSONB NOT NULL,
    source_release   TEXT,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE papers_s2orc_raw IS
    'Raw S2ORC records from the Semantic Scholar Datasets API. Keyed by '
    's2_corpus_id. content column holds body_text[], cite_spans[], bib_entries[]. '
    'Populated by src/scix/sources/s2_datasets.py.';

CREATE INDEX IF NOT EXISTS idx_s2orc_raw_release
    ON papers_s2orc_raw (source_release);

-- ---------------------------------------------------------------------------
-- papers_s2ag — pruned Semantic Scholar Academic Graph metadata
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS papers_s2ag (
    s2_corpus_id     BIGINT PRIMARY KEY,
    external_ids     JSONB,
    title            TEXT,
    authors          JSONB,
    year             SMALLINT,
    venue            TEXT,
    citation_count   INTEGER,
    reference_count  INTEGER,
    influential_citation_count INTEGER,
    is_open_access   BOOLEAN,
    fields_of_study  JSONB,
    publication_types TEXT[],
    publication_date TEXT,
    journal          JSONB,
    source_release   TEXT,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE papers_s2ag IS
    'Pruned S2AG metadata from the Semantic Scholar Datasets API. 225M papers. '
    'Pruned to essential fields; raw data can be re-fetched from S2 snapshots. '
    'Populated by src/scix/sources/s2_datasets.py.';

CREATE INDEX IF NOT EXISTS idx_s2ag_year
    ON papers_s2ag (year);
CREATE INDEX IF NOT EXISTS idx_s2ag_release
    ON papers_s2ag (source_release);

-- ---------------------------------------------------------------------------
-- s2_citations — S2AG citation edges with intent and influence
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS s2_citations (
    citing_corpus_id BIGINT NOT NULL,
    cited_corpus_id  BIGINT NOT NULL,
    intents          TEXT[],
    is_influential   BOOLEAN NOT NULL DEFAULT false,
    source_release   TEXT,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (citing_corpus_id, cited_corpus_id)
);

COMMENT ON TABLE s2_citations IS
    'S2AG citation edges with intent labels (methodology, background, '
    'result_comparison) and influence flags. 2.8B edges in full dataset. '
    'Populated by src/scix/sources/s2_datasets.py.';

CREATE INDEX IF NOT EXISTS idx_s2_citations_cited
    ON s2_citations (cited_corpus_id);
CREATE INDEX IF NOT EXISTS idx_s2_citations_influential
    ON s2_citations (is_influential) WHERE is_influential = true;

-- ---------------------------------------------------------------------------
-- ALTER citation_edges: add edge_attrs JSONB for S2AG intent/influence merge
-- ---------------------------------------------------------------------------

ALTER TABLE citation_edges
    ADD COLUMN IF NOT EXISTS edge_attrs JSONB;

COMMENT ON COLUMN citation_edges.edge_attrs IS
    'JSONB attributes merged from external sources. Currently holds '
    's2_intents (TEXT[]) and s2_is_influential (BOOLEAN) from S2AG citation data. '
    'Populated by the citation intent merger in src/scix/sources/s2_datasets.py.';

-- ---------------------------------------------------------------------------
-- Safety assertions: all new tables must be LOGGED (relpersistence='p')
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    tbl TEXT;
    rp  CHAR(1);
BEGIN
    FOREACH tbl IN ARRAY ARRAY['papers_s2orc_raw', 'papers_s2ag', 's2_citations']
    LOOP
        SELECT relpersistence INTO rp
        FROM pg_class
        WHERE relname = tbl AND relnamespace = 'public'::regnamespace;
        IF rp IS NULL THEN
            RAISE EXCEPTION '% did not get created', tbl;
        END IF;
        IF rp <> 'p' THEN
            RAISE EXCEPTION '% must be LOGGED (relpersistence=''p''), got %', tbl, rp;
        END IF;
    END LOOP;
END
$$;

COMMIT;

-- 043_consolidate_promote_harvest.sql
-- 043_consolidate_promote_harvest.sql
-- Consolidate promote_harvest: drop the migration-030 stub and install the
-- full v2 body as the canonical promote_harvest function.
--
-- Background: migration 030 shipped a promote_harvest(BIGINT) RETURNS INTEGER
-- stub.  The u04 work unit implemented the real logic as promote_harvest_v2()
-- (installed lazily by Python at runtime) to avoid colliding with the stub.
-- This migration makes promote_harvest the single canonical function and
-- removes promote_harvest_v2.
--
-- Idempotent: uses CREATE OR REPLACE and DROP IF EXISTS.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Drop the old stub (BIGINT -> INTEGER signature)
-- ---------------------------------------------------------------------------

DROP FUNCTION IF EXISTS promote_harvest(BIGINT);

-- ---------------------------------------------------------------------------
-- 2. Drop promote_harvest_v2 if it was previously installed by Python
-- ---------------------------------------------------------------------------

DROP FUNCTION IF EXISTS promote_harvest_v2(BIGINT, JSONB, NUMERIC, NUMERIC, INTEGER);

-- ---------------------------------------------------------------------------
-- 3. Install the canonical promote_harvest with full signature
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION promote_harvest(
    run_id            BIGINT,
    floors            JSONB   DEFAULT '{}'::jsonb,
    canonical_max     NUMERIC DEFAULT 0.02,
    alias_max         NUMERIC DEFAULT 0.05,
    orphan_threshold  INTEGER DEFAULT 1000
)
RETURNS JSONB
LANGUAGE plpgsql
AS $fn$
DECLARE
    result             JSONB := '{}'::jsonb;
    staging_total      BIGINT := 0;
    alias_staging_tot  BIGINT := 0;
    prod_entity_total  BIGINT := 0;
    prod_alias_total   BIGINT := 0;
    canonical_shrink   NUMERIC := 0;
    alias_shrink       NUMERIC := 0;
    floor_violations   JSONB := '[]'::jsonb;
    orphan_violations  JSONB := '[]'::jsonb;
    schema_errors      JSONB := '[]'::jsonb;
    per_source_json    JSONB := '{}'::jsonb;
    lock_acquired      BOOLEAN := FALSE;
    src_rec            RECORD;
    orphan_rec         RECORD;
    schema_rec         RECORD;
    n_promoted_ent     INTEGER := 0;
    n_promoted_ali     INTEGER := 0;
    n_promoted_ids     INTEGER := 0;
BEGIN
    -- 1. Advisory lock ------------------------------------------------------
    lock_acquired := pg_try_advisory_lock(hashtext('entities_promotion'));
    IF NOT lock_acquired THEN
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'lock_unavailable',
            'diff', '{}'::jsonb
        );
    END IF;

    -- 2. Schema compatibility check ----------------------------------------
    FOR schema_rec IN
        SELECT column_name
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'entities_staging'
           AND column_name NOT IN ('id', 'staging_run_id', 'created_at')
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = 'entities'
               AND column_name = schema_rec.column_name
        ) THEN
            schema_errors := schema_errors || to_jsonb(schema_rec.column_name);
        END IF;
    END LOOP;

    IF jsonb_array_length(schema_errors) > 0 THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'schema_mismatch',
            'diff', jsonb_build_object('schema_errors', schema_errors)
        );
    END IF;

    -- 3. Counts -------------------------------------------------------------
    SELECT COUNT(*) INTO staging_total
      FROM entities_staging WHERE staging_run_id = run_id;
    SELECT COUNT(*) INTO alias_staging_tot
      FROM entity_aliases_staging WHERE staging_run_id = run_id;

    WITH run_sources AS (
        SELECT DISTINCT source FROM entities_staging WHERE staging_run_id = run_id
    )
    SELECT COUNT(*) INTO prod_entity_total
      FROM entities e
      JOIN run_sources rs ON rs.source = e.source;

    WITH run_sources AS (
        SELECT DISTINCT source FROM entities_staging WHERE staging_run_id = run_id
    )
    SELECT COUNT(*) INTO prod_alias_total
      FROM entity_aliases ea
      JOIN entities e ON ea.entity_id = e.id
      JOIN run_sources rs ON rs.source = e.source;

    IF prod_entity_total > 0 THEN
        canonical_shrink := (prod_entity_total - staging_total)::NUMERIC
                            / prod_entity_total;
    END IF;
    IF prod_alias_total > 0 THEN
        alias_shrink := (prod_alias_total - alias_staging_tot)::NUMERIC
                        / prod_alias_total;
    END IF;

    FOR src_rec IN
        SELECT source, COUNT(*) AS n
          FROM entities_staging
         WHERE staging_run_id = run_id
         GROUP BY source
    LOOP
        per_source_json := per_source_json
            || jsonb_build_object(src_rec.source, src_rec.n);

        IF floors ? src_rec.source THEN
            IF src_rec.n < (floors ->> src_rec.source)::BIGINT THEN
                floor_violations := floor_violations
                    || jsonb_build_object(
                        'source', src_rec.source,
                        'observed', src_rec.n,
                        'floor', (floors ->> src_rec.source)::BIGINT
                    );
            END IF;
        END IF;
    END LOOP;

    -- 4. Orphan check -------------------------------------------------------
    FOR orphan_rec IN
        WITH run_sources AS (
            SELECT DISTINCT source FROM entities_staging WHERE staging_run_id = run_id
        ),
        heavy AS (
            SELECT e.id, e.canonical_name, e.entity_type, e.source,
                   COUNT(de.*) AS doc_count
              FROM entities e
              JOIN run_sources rs ON rs.source = e.source
              JOIN document_entities de ON de.entity_id = e.id
             GROUP BY e.id, e.canonical_name, e.entity_type, e.source
            HAVING COUNT(de.*) >= orphan_threshold
        )
        SELECT h.*
          FROM heavy h
         WHERE NOT EXISTS (
            SELECT 1 FROM entities_staging s
             WHERE s.staging_run_id = run_id
               AND s.canonical_name = h.canonical_name
               AND s.entity_type    = h.entity_type
               AND s.source         = h.source
         )
    LOOP
        orphan_violations := orphan_violations
            || jsonb_build_object(
                'id', orphan_rec.id,
                'canonical_name', orphan_rec.canonical_name,
                'entity_type', orphan_rec.entity_type,
                'source', orphan_rec.source,
                'doc_count', orphan_rec.doc_count
            );
    END LOOP;

    -- 5. Build the diff object ----------------------------------------------
    result := jsonb_build_object(
        'staging_entity_count', staging_total,
        'staging_alias_count', alias_staging_tot,
        'prod_entity_count_for_sources', prod_entity_total,
        'prod_alias_count_for_sources', prod_alias_total,
        'canonical_shrinkage', canonical_shrink,
        'alias_shrinkage', alias_shrink,
        'per_source_counts', per_source_json,
        'floor_violations', floor_violations,
        'orphan_violations', orphan_violations
    );

    -- 6. Gate decisions -----------------------------------------------------
    IF canonical_shrink > canonical_max THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'canonical_shrinkage',
            'diff', result
        );
    END IF;

    IF alias_shrink > alias_max THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'alias_shrinkage',
            'diff', result
        );
    END IF;

    IF jsonb_array_length(floor_violations) > 0 THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'floor_violation',
            'diff', result
        );
    END IF;

    IF jsonb_array_length(orphan_violations) > 0 THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
        UPDATE harvest_runs SET status = 'rejected_by_diff'
         WHERE id = run_id;
        RETURN jsonb_build_object(
            'accepted', false,
            'reason', 'orphan_violation',
            'diff', result
        );
    END IF;

    -- 7. Atomic upserts ----------------------------------------------------
    WITH ins AS (
        INSERT INTO entities (
            canonical_name, entity_type, discipline, source, source_version,
            ambiguity_class, link_policy, properties, harvest_run_id
        )
        SELECT
            s.canonical_name,
            s.entity_type,
            s.discipline,
            s.source,
            s.source_version,
            s.ambiguity_class::entity_ambiguity_class,
            s.link_policy::entity_link_policy,
            COALESCE(s.properties, '{}'::jsonb),
            run_id
          FROM entities_staging s
         WHERE s.staging_run_id = run_id
        ON CONFLICT (canonical_name, entity_type, source) DO UPDATE
            SET discipline     = COALESCE(EXCLUDED.discipline, entities.discipline),
                source_version = COALESCE(EXCLUDED.source_version, entities.source_version),
                ambiguity_class = COALESCE(EXCLUDED.ambiguity_class, entities.ambiguity_class),
                link_policy    = COALESCE(EXCLUDED.link_policy, entities.link_policy),
                properties     = entities.properties || EXCLUDED.properties,
                harvest_run_id = EXCLUDED.harvest_run_id,
                updated_at     = now()
        RETURNING 1
    )
    SELECT COUNT(*) INTO n_promoted_ent FROM ins;

    WITH resolved AS (
        SELECT DISTINCT e.id AS entity_id, sa.alias, sa.alias_source
          FROM entity_aliases_staging sa
          JOIN entities e
            ON e.canonical_name = sa.canonical_name
           AND e.entity_type    = sa.entity_type
           AND e.source         = sa.source
         WHERE sa.staging_run_id = run_id
           AND sa.alias IS NOT NULL
    ),
    ins AS (
        INSERT INTO entity_aliases (entity_id, alias, alias_source)
        SELECT entity_id, alias, alias_source FROM resolved
        ON CONFLICT (entity_id, alias) DO UPDATE
            SET alias_source = COALESCE(EXCLUDED.alias_source, entity_aliases.alias_source)
        RETURNING 1
    )
    SELECT COUNT(*) INTO n_promoted_ali FROM ins;

    WITH resolved AS (
        SELECT DISTINCT e.id AS entity_id, si.id_scheme, si.external_id,
               COALESCE(si.is_primary, false) AS is_primary
          FROM entity_identifiers_staging si
          JOIN entities e
            ON e.canonical_name = si.canonical_name
           AND e.entity_type    = si.entity_type
           AND e.source         = si.source
         WHERE si.staging_run_id = run_id
           AND si.id_scheme IS NOT NULL
           AND si.external_id IS NOT NULL
    ),
    ins AS (
        INSERT INTO entity_identifiers (entity_id, id_scheme, external_id, is_primary)
        SELECT entity_id, id_scheme, external_id, is_primary FROM resolved
        ON CONFLICT (id_scheme, external_id) DO UPDATE
            SET entity_id  = EXCLUDED.entity_id,
                is_primary = EXCLUDED.is_primary
        RETURNING 1
    )
    SELECT COUNT(*) INTO n_promoted_ids FROM ins;

    result := result || jsonb_build_object(
        'promoted_entities', n_promoted_ent,
        'promoted_aliases', n_promoted_ali,
        'promoted_identifiers', n_promoted_ids
    );

    UPDATE harvest_runs SET status = 'promoted', finished_at = now()
     WHERE id = run_id;

    PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
    RETURN jsonb_build_object(
        'accepted', true,
        'reason', NULL,
        'diff', result
    );
EXCEPTION WHEN OTHERS THEN
    IF lock_acquired THEN
        PERFORM pg_advisory_unlock(hashtext('entities_promotion'));
    END IF;
    RAISE;
END
$fn$;

COMMENT ON FUNCTION promote_harvest(BIGINT, JSONB, NUMERIC, NUMERIC, INTEGER) IS
    'Atomic promote of *_staging rows into public entity tables with '
    'shadow-diff gating (shrinkage, floor, orphan checks). '
    'Consolidates migration 030 stub + promote_harvest_v2 into single function.';

COMMIT;

