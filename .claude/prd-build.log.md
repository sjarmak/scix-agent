# PRD Build Log: Entity Enrichment Strategy

## 2026-04-12

- **00:00** — Decomposition complete — 14 units across 5 layers (integration branch: prd-build/entity-enrichment)
  - Layer 0 (no deps): u01-schema-migrations (medium), u02-collision-bug-fix (small)
  - Layer 1: u03-resolve-entities-service (large), u04-staging-promotion-hardening (medium), u05-ambiguity-classification (medium), u06-tier1-keyword-match (medium)
  - Layer 2: u07-query-log-and-curated-core (medium), u08-fusion-mv (medium)
  - Layer 3: u09-tier2-aho-corasick (large), u10-jit-lane (large), u11-eval-harness (medium)
  - Layer 4: u12-m4-three-way-eval (large), u13-incremental-sync (medium), u14-should-haves (medium)
  - Key constraint: M13 resolve_entities() (u03) is the single write/read contract for document_entities/\_canonical/\_jit_cache; u09/u10/u13 route through it
