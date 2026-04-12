# Research — u09 Tier 2 Aho-Corasick + Adeft

## Dependencies

- `pyahocorasick` 2.3.0 — installed into `.venv` via `uv pip install`.
  Pickleable `ahocorasick.Automaton()`. Verified round-trip.
- `scikit-learn` 1.8.0 — already present (`sklearn.feature_extraction.text.TfidfVectorizer`,
  `sklearn.linear_model.LogisticRegression`). No install needed.

## M13 `resolve_entities()` API (src/scix/resolve_entities.py)

- Public `resolve_entities(bibcode, EntityResolveContext) -> EntityLinkSet`.
- `EntityResolveContext(candidate_set=frozenset[int], mode="auto"|..., ttl_max, budget_remaining, model_version)`.
- Current u03 state: **mock, in-memory only**. All four lanes (`static`,
  `jit_cache_hit`, `live_jit`, `local_ner`) are in-module dicts. None of
  them write to `document_entities`.
- AST lint (`scripts/ast_lint_resolver.py`) bans `INSERT/UPDATE/DELETE
document_entities` and `FROM document_entities_canonical` in any file
  under `src/` except `resolve_entities.py`. **Scripts under `scripts/`
  are NOT scanned.** `link_tier1.py` already uses the `# noqa:
resolver-lint` marker on its SQL string _and_ on the `INSERT INTO
document_entities (...)` line — that is belt+braces.
- Consequence for u09: the `scripts/link_tier2.py` inserter is exempt
  from the lint (it's outside `src/`) but I still annotate the SQL with
  `# noqa: resolver-lint` for parity with u06.

## u05 ambiguity classification (src/scix/ambiguity.py)

- `entities.ambiguity_class ∈ {banned, homograph, domain_safe, unique}`.
- Filter for Tier 2: `ambiguity_class IN ('unique', 'domain_safe', 'homograph')`.
- Homograph → requires disambiguation alias co-presence.

## u07 curated core (`curated_entity_core`)

- Schema: `(entity_id PK, query_hits_14d, promoted_at)`.
- Tier 2 reads `curated_entity_core JOIN entities` to pick ~10K entities
  for the automaton (not all 90K).

## Tier 1 pattern (scripts/link_tier1.py)

- Single SQL pass, `ON CONFLICT (bibcode, entity_id, link_type, tier)
DO NOTHING`.
- `tier = 2` rows use the same PK shape; clash only inside their own
  tier bucket (u04).

## Schema: document_entities

- `(bibcode, entity_id, link_type, confidence, match_method, evidence,
harvest_run_id, tier, tier_version)`.
- PK: `(bibcode, entity_id, link_type, tier)` — so Tier 2 rows with
  `link_type='abstract_match'` will never collide with Tier 1's
  `keyword_match`.

## entities columns used

- `id, canonical_name, source, ambiguity_class, link_policy`.
- Aliases live in `entity_aliases(entity_id, alias, alias_source)`.

## Plan shape

1. Build automaton: query `curated_entity_core JOIN entities` filtered
   on `ambiguity_class`, plus all `entity_aliases` for those entities.
   For each (canonical or alias) surface form, add `(entity_id,
surface, ambiguity_class, is_long_form_alias)` payload.
2. For homographs, compute a per-entity set of "long-form aliases"
   (alias strings ≥ 6 tokens _or_ ≥ 10 chars — conservative) used as
   disambiguators.
3. Linker: iterate automaton over abstract, emit candidate for every
   hit. Post-process: for homograph candidates, only keep ones where
   at least one long-form alias of the same entity is co-present in
   the abstract.
4. Parallel map over (bibcode, abstract) tuples via multiprocessing
   Pool, fork start method. Per-worker: rebuild automaton from pickled
   blob.
5. Per-entity linkage cap (25,000) enforced in the writer: track
   counts in a `dict[int, int]`; once an entity exceeds cap, skip
   inserts _and_ `UPDATE entities SET link_policy='llm_only' WHERE id=?`.
6. Confidence placeholder = 0.85 (TODO link to M9/u11).
