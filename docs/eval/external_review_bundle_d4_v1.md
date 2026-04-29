# D4 entity-enrichment eval — external reviewer bundle

**Bead**: scix_experiments-xz4.1.16 (Parallel Validation Track 2: D4 eval set
external review)
**Source PRD**: `docs/prd/prd_entity_enrichment_rollout.md` §D4
**Source gold sets**: `data/eval/entity_value_props/*.yaml`
**Bundle assembled**: 2026-04-29
**Scope**: 60 queries across 6 value props (10 per prop)
**Status**: ready for distribution; no review feedback received yet

---

## What you're being asked to review

We built a retrieval system over a 32M-paper scientific literature corpus
(NASA ADS metadata + arXiv mirror across cs/stat/physics/q-bio/...). The
system layers an entity-extraction graph on top of standard hybrid
retrieval (lexical BM25 + INDUS dense embeddings fused via RRF).

We want to evaluate **the entity-graph value props** — the parts of the
system where having entities should give measurably better results than
hybrid alone. We've handwritten 10 queries per value prop with explicit
expectations for what "good" looks like. Each prop has a 0–3 rubric.

We have an LLM judge running these queries today. Your job is **not** to
judge query results — you're reviewing the **eval set itself**:

1. **Are these the right tests?** Does each value prop have queries that
   genuinely stress the named capability, or are they too easy / too
   contrived?
2. **Are there obvious gaps?** What capabilities should we be testing
   that we're not?
3. **Add up to 50 of your own queries** (across any props) — same shape
   as the existing entries.

You do **not** need to run the queries. We'll re-run the LLM judge on
the combined (existing + your additions) set and call that "M4 v2".

### Reviewer panel (target)

- ≥ 1 NASA ADS librarian / digital-library specialist
- ≥ 1 scientist outside astronomy (cs / bio / chem / physics-non-astro)
  to stress cross-discipline coverage

### Turnaround

No deadline. The in-house eval run gates production decisions today;
external review feeds the next quarterly re-run.

### How to return feedback

Two options:

1. **Markdown reply** — annotate this doc inline with comments on
   specific queries; add new queries in the same shape as existing ones,
   inside the relevant `## Prop:` section.
2. **YAML diff** — clone the repo (or just the
   `data/eval/entity_value_props/` directory), edit the YAMLs directly,
   send a patch / diff back.

If neither is practical, free-form prose listing query suggestions plus
"this prop is weak / strong because ..." comments is fine.

---

## How the queries are scored

For each query, the judge sees:

1. The query text and its declared value prop.
2. The retrieval results (top-10 papers).
3. An expectation note hand-written by us ("should retrieve papers that
   ...").
4. A 0–3 rubric tailored to the prop.

The judge is a Claude Code subagent invoked via OAuth (no paid API). It
returns a score and a 1–2 sentence rationale per query.

---

## Prop 1 — Alias expansion

**What's tested**: a query using one surface form ("HST", "JWST")
returns papers that only mention the canonical form ("Hubble Space
Telescope", "James Webb Space Telescope"), not the alias.

**Rubric**:

- 0 — no alias expansion; results only mention the alias as written.
- 1 — partial; one or two canonical-form papers.
- 2 — mostly works; mix of alias and canonical, plausibly ranked.
- 3 — works correctly; canonical-form-only papers appear high.

### Queries

| id | query | alias | canonical |
|---|---|---|---|
| alias-001 | HST observations of M31 | HST | Hubble Space Telescope |
| alias-002 | JWST exoplanet atmospheres | JWST | James Webb Space Telescope |
| alias-003 | LIGO gravitational waves | LIGO | Laser Interferometer Gravitational-Wave Observatory |
| alias-004 | ALMA protoplanetary disks | ALMA | Atacama Large Millimeter Array |
| alias-005 | TESS exoplanet survey | TESS | Transiting Exoplanet Survey Satellite |
| alias-006 | Chandra X-ray observations | Chandra | Chandra X-ray Observatory |
| alias-007 | SDSS galaxy survey | SDSS | Sloan Digital Sky Survey |
| alias-008 | LSST cadence strategy | LSST | Legacy Survey of Space and Time |
| alias-009 | DESI cosmology measurements | DESI | Dark Energy Spectroscopic Instrument |
| alias-010 | XMM-Newton observations | XMM-Newton | X-ray Multi-Mirror Mission |

> Source: `data/eval/entity_value_props/alias_expansion.yaml`

---

## Prop 2 — Ontology expansion

**What's tested**: a generic query ("exoplanet") retrieves papers that
mention specific instances (HD 209458b, TRAPPIST-1b) but not the
generic term itself.

**Rubric**:

- 0 — no expansion; only generic-term papers.
- 1 — partial; one or two specific-instance papers.
- 2 — mostly works; mix of generic and specific instances.
- 3 — works correctly; specific-instance papers ranked high.

> Source: `data/eval/entity_value_props/ontology_expansion.yaml` (10 queries — see file for full list and per-query expectation notes)

---

## Prop 3 — Disambiguation

**What's tested**: when a term has multiple entity senses (Hubble the
person vs. mission; Psyche the asteroid vs. spacecraft), the retrieval
routes to the intended sense based on query context.

**Rubric**:

- 0 — wrong sense dominates.
- 1 — mixed senses, no clear preference.
- 2 — mostly works; majority correct sense.
- 3 — near-pure filtering to intended sense.

> Source: `data/eval/entity_value_props/disambiguation.yaml`

---

## Prop 4 — Type filter

**What's tested**: agent constrains a query with `entity_type =
instrument` (or `mission`, `dataset`, ...) and gets clean type-matched
results.

**Rubric**:

- 0 — type filter ignored; results span types.
- 1 — partial filtering; minority off-type leakage.
- 2 — mostly works; >70% on-type.
- 3 — clean filtering; ≥90% on-type.

> Source: `data/eval/entity_value_props/type_filter.yaml`

---

## Prop 5 — Specific entity

**What's tested**: given a resolved `entity_id=X`, return only papers
that actually mention entity X (not soft / surface-form matches).

**Rubric**:

- 0 — surface-form collisions dominate (e.g., wrong-sense "Hubble").
- 1 — partial; some collisions present.
- 2 — mostly works; minor collisions.
- 3 — clean entity_id match.

> Source: `data/eval/entity_value_props/specific_entity.yaml`

---

## Prop 6 — Community expansion

**What's tested**: given an entity in a graph community (e.g., HST),
retrieval surfaces sibling-community papers (WFC3, STIS, ACS, COS,
STScI) even when those papers don't mention the seed entity by name.

**Rubric**:

- 0 — no expansion; only seed-entity papers.
- 1 — partial; a few sibling papers.
- 2 — mostly works; sibling papers ranked plausibly.
- 3 — sibling papers surface as intended.

> Source: `data/eval/entity_value_props/community_expansion.yaml`

> **Implementation note for reviewers**: the community-expansion lane
> is currently being rewritten (bead xz4.1.38) from paper-Leiden
> communities to entity co-occurrence. The 10 queries here are
> appropriate for either implementation; the rubric is unchanged.

---

## Adding your own queries

Use this shape (YAML) — paste into the relevant
`data/eval/entity_value_props/<prop>.yaml` `queries:` list:

```yaml
- id: <prop>-011        # next free index
  query: "<your free-text query>"
  # plus per-prop fields:
  # alias_expansion:    alias, canonical
  # ontology_expansion: generic_term, specific_instances [list]
  # disambiguation:     intended_sense, off_target_sense
  # type_filter:        entity_type
  # specific_entity:    entity_name (resolved by harness)
  # community_expansion: seed_entity, community_label, expected_siblings [list]
  expectation: >
    <one-line description of what 'good' looks like>
  tags: [<freeform>]
```

If a value prop is **missing a category of test you'd expect**, flag
it inline in the corresponding section above.

---

## What we'd especially like opinions on

1. **Cross-discipline coverage.** All 60 queries today are astronomy /
   astrophysics. We need stress queries from cs / bio / chem / physics
   reviewers. Same value props should hold; the entities just look
   different (PyTorch, BERT, AlphaFold, CRISPR, ...).
2. **Disambiguation depth.** We have 10 disambiguation queries built
   around 5 well-known ambiguities. Are there ambiguities specific to
   your subfield that we should add?
3. **"Mostly works" floor.** The rubric's 2 score is loosely defined
   ("mostly works"). Is the 0/1/2/3 bucketing useful, or should we
   collapse to 0/1/2 / "fail/partial/pass"?
4. **What's missing.** Which value props *are not* in this eval that
   you'd want a retrieval system over scientific literature to
   demonstrate?

---

## Provenance and contact

- Project: SciX Experiments (NASA ADS / SMD-discipline literature
  retrieval research project).
- Repo: `~/projects/scix_experiments` (private).
- Bead: scix_experiments-xz4.1.16.
- Per-query metadata, full YAML source: `data/eval/entity_value_props/`.
- Eval harness: `scripts/eval_entity_value_props.py`.
- Rollout PRD: `docs/prd/prd_entity_enrichment_rollout.md`.

Reach Stephanie Jarmak (`stephanie.jarmak1@gmail.com`) with feedback or
questions.
