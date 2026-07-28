-- Migration 072: INDUS dense-lane "synced" watermark (bead s7cy)
--
-- Replaces the dropped `paper_embeddings` table as the source of truth for
-- "which papers already have an INDUS vector in Qdrant". The daily embed
-- pipeline (scix.embed) anti-joins papers against this table to find the
-- unembedded set, and inserts a row per bibcode after a successful Qdrant
-- upsert. Tiny (bibcode + timestamp), unlike the ~195 GB table it replaces.
--
-- Seed once from the live Qdrant collection via
-- scripts/seed_indus_qdrant_synced.py so the pipeline only embeds the post-drop
-- gap, not the entire 32 M corpus already served from Qdrant.
--
-- Idempotent: safe to re-run. No migration auto-runner exists in this repo
-- (migrations are applied by hand), so this does not replay 069-071.

-- No FK to papers(bibcode) on purpose: the seed derives from the Qdrant
-- collection, which can briefly diverge from `papers` (bibcode canonicalization,
-- migration 046), and a FK violation would abort the bulk seed. A stray
-- watermark row is harmless — worst case one paper is treated as already-synced;
-- the daily anti-join self-corrects once the bibcode reconciles.
CREATE TABLE IF NOT EXISTS indus_qdrant_synced (
    bibcode   TEXT PRIMARY KEY,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE indus_qdrant_synced IS
    'Watermark: bibcodes whose INDUS vector is present in Qdrant '
    'scix_indus_v2_papers_s1. Replaces paper_embeddings for unembedded-detection '
    '(bead s7cy / ADR-015). Seeded from Qdrant, maintained by scix.embed.';
