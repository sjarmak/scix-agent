-- 056_concepts_vocabularies.sql
-- Generic concept-vocabulary substrate (cross-discipline expansion, dbl.1).
--
-- Purpose: hold open, permissively-licensed taxonomies (OpenAlex Topics,
-- ACM CCS, MSC, PhySH, GCMD, plus future MeSH/ChEBI/etc.) under one
-- schema instead of a per-vocabulary table per source. UAT keeps its own
-- legacy tables (uat_concepts/uat_relationships/paper_uat_mappings) for
-- backwards compatibility — concept_search may union UAT into this view
-- in a follow-up bead.
--
-- Tables:
--   vocabularies          one row per source vocabulary (license + provenance)
--   concepts              composite PK (vocabulary, concept_id); preferred
--                         label, alt labels, definition, level, properties
--   concept_relationships parent_id → child_id within one vocabulary,
--                         relationship type (broader / narrower / related)
--
-- Idempotent: every DDL uses IF NOT EXISTS. Safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS vocabularies (
    vocabulary      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    license         TEXT NOT NULL,
    license_url     TEXT,
    homepage_url    TEXT,
    source_url      TEXT NOT NULL,
    version         TEXT,
    record_count    INTEGER NOT NULL DEFAULT 0,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    properties      JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS concepts (
    vocabulary        TEXT NOT NULL REFERENCES vocabularies(vocabulary) ON DELETE CASCADE,
    concept_id        TEXT NOT NULL,
    preferred_label   TEXT NOT NULL,
    alternate_labels  TEXT[] NOT NULL DEFAULT '{}'::text[],
    definition        TEXT,
    external_uri      TEXT,
    level             INTEGER,
    properties        JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (vocabulary, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_concepts_vocab
    ON concepts (vocabulary);
CREATE INDEX IF NOT EXISTS idx_concepts_label_lower
    ON concepts (vocabulary, lower(preferred_label));
CREATE INDEX IF NOT EXISTS idx_concepts_alt_labels
    ON concepts USING GIN (alternate_labels);
CREATE INDEX IF NOT EXISTS idx_concepts_external_uri
    ON concepts (external_uri) WHERE external_uri IS NOT NULL;

CREATE TABLE IF NOT EXISTS concept_relationships (
    vocabulary    TEXT NOT NULL,
    parent_id     TEXT NOT NULL,
    child_id      TEXT NOT NULL,
    relationship  TEXT NOT NULL DEFAULT 'broader'
                  CHECK (relationship IN ('broader', 'narrower', 'related')),
    PRIMARY KEY (vocabulary, parent_id, child_id, relationship),
    FOREIGN KEY (vocabulary, parent_id)
        REFERENCES concepts (vocabulary, concept_id) ON DELETE CASCADE,
    FOREIGN KEY (vocabulary, child_id)
        REFERENCES concepts (vocabulary, concept_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_concept_rel_child
    ON concept_relationships (vocabulary, child_id);
CREATE INDEX IF NOT EXISTS idx_concept_rel_parent
    ON concept_relationships (vocabulary, parent_id);

INSERT INTO schema_migrations (version, filename)
    VALUES (56, '056_concepts_vocabularies.sql')
    ON CONFLICT (version) DO NOTHING;

COMMIT;
