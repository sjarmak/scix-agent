-- 057_v_claim_edges.sql
-- MH-2 of SciX Deep Search v1 PRD (docs/prd/scix_deep_search_v1.md):
-- Materialized derived view `v_claim_edges` over citation_contexts JOIN
-- citation_edges JOIN papers. Each row exposes one in-text citation mention
-- with both endpoints' publication years materialized for downstream
-- chronology-based ranking (claim_blame, find_replications).
--
-- Acceptance for the host work unit (v-claim-edges-migration):
--   * View columns: source_bibcode, target_bibcode, context_snippet (≤1000
--     chars of context_text), intent, section_name, source_year, target_year,
--     char_offset.
--   * Unique index on (source_bibcode, target_bibcode, char_offset) — REQUIRED
--     for REFRESH MATERIALIZED VIEW CONCURRENTLY.
--   * B-tree indexes on (source_bibcode, intent) and (target_bibcode, intent)
--     to keep claim_blame / find_replications single-bibcode lookups in the
--     <100 ms p95 budget against a primed cache.
--   * Idempotent: DROP IF EXISTS + CREATE pattern. Safe to re-apply.
--
-- Notes on the citation_edges INNER JOIN: citation_contexts is upstream of
-- citation_edges in the ingest pipeline, but a context row without a matching
-- edge represents a context the resolver could not promote to the canonical
-- citation graph (e.g. ambiguous bibcode, target outside corpus). We require
-- the edge so v_claim_edges only surfaces citations that have a structurally
-- validated counterpart — claim lineage walks should never traverse
-- unresolved contexts.
--
-- The papers join is INNER on both endpoints because source_year /
-- target_year are required columns for chronological ranking; rows where
-- either paper is outside the corpus are dropped. This is consistent with
-- the rest of the Deep Search retrieval stack which only ranks within-corpus
-- evidence.
--
-- citation_contexts contains ~0.11% exact-duplicate rows (898 of 823k as of
-- 2026-04-25, almost entirely from re-ingest leftovers — same source, target,
-- char_offset, context_text). The natural key (source_bibcode,
-- target_bibcode, char_offset) is therefore not unique in the source table.
-- We dedupe at view-construction time with DISTINCT ON so the unique index
-- below can be enforced without a destructive cleanup of citation_contexts.
-- ORDER BY prefers a non-NULL intent (post-MH-1 backfill) so the survivor row
-- carries classifier output if any duplicate has it; ctid is the final
-- deterministic tiebreaker.

BEGIN;

DROP MATERIALIZED VIEW IF EXISTS v_claim_edges CASCADE;

CREATE MATERIALIZED VIEW v_claim_edges AS
SELECT DISTINCT ON (cc.source_bibcode, cc.target_bibcode, cc.char_offset)
    cc.source_bibcode,
    cc.target_bibcode,
    substring(cc.context_text FROM 1 FOR 1000) AS context_snippet,
    cc.intent,
    cc.section_name,
    sp.year AS source_year,
    tp.year AS target_year,
    cc.char_offset
FROM citation_contexts cc
JOIN citation_edges ce
    ON ce.source_bibcode = cc.source_bibcode
   AND ce.target_bibcode = cc.target_bibcode
JOIN papers sp ON sp.bibcode = cc.source_bibcode
JOIN papers tp ON tp.bibcode = cc.target_bibcode
ORDER BY
    cc.source_bibcode,
    cc.target_bibcode,
    cc.char_offset,
    cc.intent NULLS LAST,
    cc.ctid;

-- Unique index required by REFRESH MATERIALIZED VIEW CONCURRENTLY.
-- (source_bibcode, target_bibcode, char_offset) is the natural key: one row
-- per in-text citation mention. char_offset disambiguates multiple mentions
-- of the same target within the same source. CONCURRENT refresh requires a
-- unique index over plain column references (no expressions), so we lean on
-- Postgres 15+ NULLS NOT DISTINCT to make NULL char_offsets compare equal —
-- otherwise two un-offset mentions of the same (source, target) would be
-- considered distinct by the btree and the CONCURRENT refresh would still
-- succeed but uniqueness would not be enforced semantically.
CREATE UNIQUE INDEX idx_v_claim_edges_pk
    ON v_claim_edges (source_bibcode, target_bibcode, char_offset)
    NULLS NOT DISTINCT;

-- claim_blame: walk reverse references from a source_bibcode, optionally
-- filtered/ranked by intent.
CREATE INDEX idx_v_claim_edges_source_intent
    ON v_claim_edges (source_bibcode, intent);

-- find_replications: forward citations to target_bibcode, optionally
-- filtered/ranked by intent.
CREATE INDEX idx_v_claim_edges_target_intent
    ON v_claim_edges (target_bibcode, intent);

ANALYZE v_claim_edges;

COMMIT;
