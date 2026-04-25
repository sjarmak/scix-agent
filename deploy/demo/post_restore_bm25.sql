-- Regenerate papers.tsv after a subset restore.
--
-- Why: restore_subset.sh sets session_replication_role=replica to defer
-- FK checks during bulk insert. That also disables the papers_tsv_trigger
-- defined in migration 003, so every restored row lands with tsv=NULL.
-- Without this file, the BM25 leg of hybrid search returns zero hits.
--
-- Uses the same scix_english config + weighting as the trigger
-- (title=A, abstract=B, keywords=C).

SET maintenance_work_mem = '512MB';

UPDATE papers
SET tsv =
    setweight(to_tsvector('scix_english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('scix_english', coalesce(abstract, '')), 'B') ||
    setweight(to_tsvector('scix_english', coalesce(array_to_string(keywords, ' '), '')), 'C')
WHERE tsv IS NULL;

-- GIN index was already created by migration 003; rebuilding stats helps
-- the planner pick it for workshop queries.
ANALYZE papers;

-- Sanity count the caller can eyeball.
SELECT count(*) AS papers_with_tsv FROM papers WHERE tsv IS NOT NULL;
