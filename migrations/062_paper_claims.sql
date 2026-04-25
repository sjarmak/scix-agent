-- Migration 062: paper_claims — nanopub-inspired claim provenance.
--
-- Part of PRD Build "nanopub-claim-extraction". One row per scientific
-- claim extracted from a paper's full text. Each row carries:
--
--   * A provenance contract:
--       (bibcode, section_index, paragraph_index, char_span_start, char_span_end)
--     uniquely identifies the source span. Indices and char spans index into
--     papers_fulltext.sections[i].text per migration 041 — section_index is
--     the position in the JSONB sections array; paragraph_index is the
--     paragraph offset within that section's text; char_span_{start,end}
--     are character offsets within that section's text.
--
--   * The claim itself (claim_text) plus a structured (subject, predicate,
--     object) decomposition where the extractor was confident enough.
--
--   * A claim_type tag (factual / methodological / comparative / speculative
--     / cited_from_other) — CHECK-constrained so unknown labels can't slip in.
--
--   * Extraction provenance (extraction_model, extraction_prompt_version,
--     extracted_at, confidence) so we can audit and re-run with newer
--     models without losing history.
--
--   * Optional links into the entity graph (linked_entity_subject_id,
--     linked_entity_object_id). These are bigint REFERENCES-less by design:
--     entities.id is currently SERIAL (int4) in migration 021, but we want
--     headroom for the table to migrate to bigint without a follow-up
--     ALTER, and we don't want a hard FK that blocks claim insertion when
--     the linker hasn't run yet (matches the document_entities pattern of
--     not FK-constraining bibcode).
--
-- SAFETY: this table MUST be LOGGED. An UNLOGGED table is truncated on
-- crash recovery — see migration 023 and the feedback_unlogged_tables
-- memory for the 32M-embedding loss that prompted this rule. The block
-- at the bottom of this file asserts relpersistence='p' and raises if not.
--
-- Idempotent: CREATE TABLE IF NOT EXISTS + CREATE INDEX IF NOT EXISTS
-- everywhere, so a second `psql -f` succeeds.

BEGIN;

CREATE TABLE IF NOT EXISTS paper_claims (
    claim_id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    bibcode                   text NOT NULL REFERENCES papers(bibcode),
    section_index             int NOT NULL,
    paragraph_index           int NOT NULL,
    char_span_start           int NOT NULL,
    char_span_end             int NOT NULL,
    claim_text                text NOT NULL,
    claim_type                text NOT NULL,
    subject                   text,
    predicate                 text,
    object                    text,
    confidence                real,
    extraction_model          text NOT NULL,
    extraction_prompt_version text NOT NULL,
    extracted_at              timestamptz NOT NULL DEFAULT now(),
    linked_entity_subject_id  bigint,
    linked_entity_object_id   bigint,
    CONSTRAINT paper_claims_claim_type_check
        CHECK (claim_type IN (
            'factual',
            'methodological',
            'comparative',
            'speculative',
            'cited_from_other'
        ))
);

COMMENT ON TABLE paper_claims IS
    'Nanopub-inspired claim provenance. One row per scientific claim '
    'extracted from a paper. Provenance contract: '
    '(bibcode, section_index, paragraph_index, char_span_start, char_span_end) '
    'uniquely identifies the source span into papers_fulltext.sections[i].text.';

COMMENT ON COLUMN paper_claims.claim_id IS
    'UUID primary key. Stable across re-extraction because new extractions '
    'should INSERT new rows (with a new claim_id and new extraction_model / '
    'extraction_prompt_version), not UPDATE existing ones.';

COMMENT ON COLUMN paper_claims.bibcode IS
    'Source paper bibcode. FK to papers(bibcode).';

COMMENT ON COLUMN paper_claims.section_index IS
    'Index into papers_fulltext.sections[] (zero-based). Identifies which '
    'section this claim was extracted from.';

COMMENT ON COLUMN paper_claims.paragraph_index IS
    'Paragraph offset within the section text (zero-based, paragraph-split '
    'is whatever the extractor uses; extractor must be deterministic so '
    'the contract holds).';

COMMENT ON COLUMN paper_claims.char_span_start IS
    'Inclusive character offset of the claim within '
    'papers_fulltext.sections[section_index].text.';

COMMENT ON COLUMN paper_claims.char_span_end IS
    'Exclusive character offset of the claim within '
    'papers_fulltext.sections[section_index].text.';

COMMENT ON COLUMN paper_claims.claim_text IS
    'Verbatim text of the claim as extracted from the paper.';

COMMENT ON COLUMN paper_claims.claim_type IS
    'One of: factual | methodological | comparative | speculative | '
    'cited_from_other. CHECK-constrained so unknown labels cannot slip in.';

COMMENT ON COLUMN paper_claims.subject IS
    'Optional structured-claim subject (free text). Set when the extractor '
    'produced a subject-predicate-object decomposition.';

COMMENT ON COLUMN paper_claims.predicate IS
    'Optional structured-claim predicate (free text).';

COMMENT ON COLUMN paper_claims.object IS
    'Optional structured-claim object (free text).';

COMMENT ON COLUMN paper_claims.confidence IS
    'Optional extractor confidence in [0, 1]. Semantics defined by '
    'extraction_model + extraction_prompt_version.';

COMMENT ON COLUMN paper_claims.extraction_model IS
    'Model name + version that produced this claim '
    '(e.g. "claude-opus-4-7", "gpt-5.4-mini-2026-03-15").';

COMMENT ON COLUMN paper_claims.extraction_prompt_version IS
    'Prompt template version (e.g. "v1", "v2.1"). Combined with '
    'extraction_model uniquely identifies the extraction recipe.';

COMMENT ON COLUMN paper_claims.extracted_at IS
    'Timestamp of extraction.';

COMMENT ON COLUMN paper_claims.linked_entity_subject_id IS
    'Optional link from claim subject into the entity graph. '
    'bigint, not REFERENCES-constrained: entities.id is currently SERIAL '
    '(int4) but we leave headroom, and we do not want claim INSERTs to '
    'fail when the entity-linking pass has not yet run.';

COMMENT ON COLUMN paper_claims.linked_entity_object_id IS
    'Optional link from claim object into the entity graph. See '
    'linked_entity_subject_id for type/FK rationale.';

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- Lookup-by-paper, ordered by section. Every "show me claims from paper X"
-- and every section-scoped extraction backfill walks this index.
CREATE INDEX IF NOT EXISTS ix_paper_claims_bibcode_section
    ON paper_claims (bibcode, section_index);

-- Reverse lookups from the entity graph: "what claims mention entity E
-- as the subject / object?". Two btree indexes (one per side) so the
-- planner can use either in isolation.
CREATE INDEX IF NOT EXISTS ix_paper_claims_linked_entity_subject_id
    ON paper_claims (linked_entity_subject_id);

CREATE INDEX IF NOT EXISTS ix_paper_claims_linked_entity_object_id
    ON paper_claims (linked_entity_object_id);

-- Filter by claim type ("show me only methodological claims from this corpus").
-- Low-cardinality (5 distinct values) but useful as a partial-scan key when
-- combined with bibcode or model filters.
CREATE INDEX IF NOT EXISTS ix_paper_claims_claim_type
    ON paper_claims (claim_type);

-- Full-text search over claim_text. GIN over to_tsvector('english', ...)
-- so MCP tools can do natural-language claim search without scanning
-- the whole table.
CREATE INDEX IF NOT EXISTS ix_paper_claims_claim_text_tsv
    ON paper_claims USING GIN (to_tsvector('english', claim_text));

-- ---------------------------------------------------------------------------
-- Safety assertion: refuse to leave the migration with an UNLOGGED table.
-- Mirrors migration 041's pattern.
-- ---------------------------------------------------------------------------

DO $$
DECLARE
    rp CHAR(1);
BEGIN
    SELECT relpersistence INTO rp
    FROM pg_class
    WHERE relname = 'paper_claims' AND relnamespace = 'public'::regnamespace;
    IF rp IS NULL THEN
        RAISE EXCEPTION 'paper_claims did not get created';
    END IF;
    IF rp <> 'p' THEN
        RAISE EXCEPTION 'paper_claims must be LOGGED (relpersistence=''p''), got %', rp;
    END IF;
END
$$;

COMMIT;
