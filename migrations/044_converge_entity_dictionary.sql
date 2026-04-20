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
