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
