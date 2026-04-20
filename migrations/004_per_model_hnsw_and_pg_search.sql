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
