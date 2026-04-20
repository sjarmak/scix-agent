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
