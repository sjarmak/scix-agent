-- Migration 063: section_entities — section-grain entity links
--
-- Bead: scix_experiments-67e
-- Pairs with the wqr.9 section-embeddings work (migration 061): once papers
-- have section-grain text and embeddings, entity linking can run per-section
-- so retrieval can distinguish "paper uses JWST" from "paper cites JWST in
-- the introduction."
--
-- WHY A NEW TABLE INSTEAD OF EXTENDING document_entities
-- ------------------------------------------------------
-- document_entities holds 122M paper-grain rows today and the PK
-- (bibcode, entity_id, link_type, tier) is shaped for "this paper has this
-- entity." Adding section_index would either:
--   1. Change PK semantics so a paper appears once per section per entity —
--      breaks every existing query that joins on (bibcode, entity_id) and
--      assumes paper-grain cardinality, OR
--   2. Add NULL section_index for paper-grain rows AND non-NULL rows for
--      section-grain — confusing and forces partial indexes everywhere.
-- Section-grain links are a parallel signal, so they get their own table.
-- Paper-grain retrieval still uses document_entities; section-grain
-- retrieval uses section_entities; the MCP entity tool joins both.
--
-- ROW SHAPE
-- ---------
-- One row per (bibcode, section_index, entity_id, link_type, tier). A
-- single section may match the same entity through multiple surface forms;
-- those collapse on PK so we only keep the earliest mention. Multi-mention
-- evidence (additional spans, surface forms) lives in evidence JSONB.
--
-- section_index is the 0-based position inside papers_fulltext.sections —
-- same convention as section_embeddings (migration 061). The linker
-- denormalizes section_heading and a 5-role canonical section_role
-- (background/method/result/conclusion/other) computed via
-- scix.section_role.classify_section_role. Storing the role on the row
-- avoids a join on every retrieval call and pays for itself in index size.
--
-- INDEXES
-- -------
-- * PK already supports prefix scans by bibcode and (bibcode, section_index).
-- * idx_section_entities_entity_id: reverse lookups entity_id -> sections.
--   Essential for the MCP `entity` action='papers' with section breakdown.
-- * idx_section_entities_role: partial GIN-less btree on section_role for
--   role-filtered retrieval (e.g. "papers that USE JWST in their methods").
--   Partial because section_role can be NULL when the role classifier
--   declined to match.
-- * idx_section_entities_bibcode_entity: covers (bibcode, entity_id) joins
--   for the paper -> sections drill-down.
--
-- SAFETY: section_entities MUST be LOGGED. An UNLOGGED table is truncated
-- on crash recovery — see migration 023 / feedback_unlogged_tables for the
-- 32M-embedding loss that prompted this rule.
--
-- Idempotent: every DDL uses IF NOT EXISTS so this file is safe to re-run.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. section_entities table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS section_entities (
    bibcode         TEXT NOT NULL,
    section_index   INT  NOT NULL,
    entity_id       INTEGER NOT NULL,
    link_type       TEXT NOT NULL,
    tier            SMALLINT NOT NULL DEFAULT 2,
    tier_version    INTEGER NOT NULL DEFAULT 1,
    confidence      REAL,
    match_method    TEXT,
    section_heading TEXT,
    section_role    TEXT,
    evidence        JSONB,
    harvest_run_id  INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (bibcode, section_index, entity_id, link_type, tier),
    CONSTRAINT section_entities_bibcode_fkey
        FOREIGN KEY (bibcode) REFERENCES papers(bibcode),
    CONSTRAINT section_entities_entity_id_fkey
        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
    CONSTRAINT section_entities_harvest_run_id_fkey
        FOREIGN KEY (harvest_run_id) REFERENCES harvest_runs(id)
);

COMMENT ON TABLE section_entities IS
    'Section-grain entity links. One row per (bibcode, section_index, '
    'entity_id, link_type, tier). Pairs with section_embeddings (migration '
    '061) and parallels paper-grain document_entities. Bead '
    'scix_experiments-67e.';

COMMENT ON COLUMN section_entities.section_index IS
    '0-based position inside papers_fulltext.sections. Same convention as '
    'section_embeddings.section_index.';

COMMENT ON COLUMN section_entities.section_role IS
    'Canonical 5-role classification from scix.section_role: background, '
    'method, result, conclusion, other. NULL if the classifier declined.';

COMMENT ON COLUMN section_entities.evidence IS
    'JSONB. Mirrors document_entities.evidence shape: matched_surface, '
    'start, end, ambiguity_class, is_alias, plus optional additional_spans '
    'when the same entity matched multiple surface forms in this section.';

-- ---------------------------------------------------------------------------
-- 2. Reverse-lookup index: entity_id -> sections
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_section_entities_entity_id
    ON section_entities (entity_id);

-- ---------------------------------------------------------------------------
-- 3. Section-role filter index (partial)
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_section_entities_role
    ON section_entities (section_role)
    WHERE section_role IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 4. Composite index for paper -> sections drill-down
-- ---------------------------------------------------------------------------
-- The PK supports (bibcode) and (bibcode, section_index) prefix scans, but
-- the most common drill-down join is "give me the sections in paper X that
-- mention entity Y" — i.e. (bibcode, entity_id). Adding this index makes
-- that O(matching rows) instead of full PK scan + filter.
CREATE INDEX IF NOT EXISTS idx_section_entities_bibcode_entity
    ON section_entities (bibcode, entity_id);

-- ---------------------------------------------------------------------------
-- 5. Safety assertion: section_entities must be LOGGED
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'section_entities'
      AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'section_entities did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION
            'section_entities must be LOGGED (relpersistence=''p''), got %',
            rp;
    END IF;
END
$$;

COMMIT;
