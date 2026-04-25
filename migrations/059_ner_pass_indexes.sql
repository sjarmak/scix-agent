-- 059_ner_pass_indexes.sql
-- Indexes that accelerate the dbl.3 GLiNER NER pass and downstream sweeps.
--
-- The pipeline (src/scix/extract/ner_pass.py) writes:
--   - entities  with source='gliner'
--   - document_entities  with match_method='gliner', tier=4
--
-- New match_method values are not constrained at the schema level — the
-- column has no CHECK constraint. The values 'gliner' and (later)
-- 'scibert_routed' are documented here as the canonical NER methods.
-- Tier 4 follows the existing tier ordering (0=lexical, 4=ML/NER).
--
-- Idempotent: every CREATE INDEX uses IF NOT EXISTS.

BEGIN;

-- Hot path during NER ingest: every novel mention does a SELECT id FROM
-- entities WHERE lower(canonical_name)=$1 AND entity_type=$2 AND source='gliner'
-- before deciding whether to INSERT. The partial index keeps the working
-- set small (only gliner-sourced rows) and avoids scanning the full
-- entities table.
CREATE INDEX IF NOT EXISTS idx_entities_gliner_lookup
    ON entities (lower(canonical_name), entity_type)
    WHERE source = 'gliner';

-- Sweep queries (alias merge, precision eval, discipline coverage report)
-- filter document_entities by match_method='gliner'. The partial index
-- is small relative to the 28 M+ row table.
CREATE INDEX IF NOT EXISTS idx_document_entities_gliner
    ON document_entities (bibcode)
    WHERE match_method = 'gliner';

INSERT INTO schema_migrations (version, filename)
    VALUES (59, '059_ner_pass_indexes.sql')
    ON CONFLICT (version) DO NOTHING;

COMMIT;
