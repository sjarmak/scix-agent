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
