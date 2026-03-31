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
