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
