-- 055_agent_entity_context_rewrite.sql
-- Rewrite agent_entity_context materialized view to eliminate the per-entity
-- LATERAL count(*) on document_entities. The original definition (see
-- migrations/024_agent_context_views.sql lines 46-83) joined a LATERAL
-- subquery per entity, which forces a separate index scan over
-- document_entities for each row in entities — O(entities * log(rows)).
--
-- Replacement: pre-aggregate document_entities once with a CTE, then LEFT
-- JOIN on the pre-aggregated doc_count. The outer GROUP BY still produces
-- one row per entity. Column set, order, and types are identical to the
-- original — only the plan shape changes.
--
-- Preserves: idx_agent_entity_ctx_id UNIQUE (entity_id).

BEGIN;

DROP MATERIALIZED VIEW IF EXISTS agent_entity_context CASCADE;

CREATE MATERIALIZED VIEW IF NOT EXISTS agent_entity_context AS
WITH de_counts AS (
    SELECT entity_id, count(*) AS doc_count
    FROM document_entities
    GROUP BY entity_id
)
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
    COALESCE(dc.doc_count, 0) AS citing_paper_count
FROM entities e
LEFT JOIN entity_identifiers ei ON ei.entity_id = e.id
LEFT JOIN entity_aliases ea ON ea.entity_id = e.id
LEFT JOIN entity_relationships er ON er.subject_entity_id = e.id
LEFT JOIN de_counts dc ON dc.entity_id = e.id
GROUP BY e.id, e.canonical_name, e.entity_type, e.discipline, e.source, dc.doc_count;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_entity_ctx_id
    ON agent_entity_context (entity_id);

ANALYZE agent_entity_context;

COMMIT;
