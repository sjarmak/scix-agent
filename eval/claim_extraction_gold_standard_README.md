# Claim Extraction Gold Standard

`claim_extraction_gold_standard.jsonl` is a hand-curated evaluation set for the
nanopub / claim-extraction pipeline. Each line is one paragraph annotated with
the claims a downstream extractor should recover from it.

The gold standard is intentionally small (12-20 entries). It is meant for
schema-and-shape regression checks and quick sanity evals during development —
not for statistical benchmarking.

## File format

JSON Lines (one JSON object per line, UTF-8). Every line MUST contain the
following keys.

| Field             | Type                       | Description                                                                                                  |
| ----------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `bibcode`         | string                     | ADS-style bibcode identifier for the source paper. **Bibcodes prefixed `GOLD` are placeholder** — they are not real ADS records and exist only as stable identifiers within this gold set. |
| `section_index`   | integer                    | Zero-based index of the section the paragraph belongs to in the source document.                              |
| `paragraph_index` | integer                    | Zero-based index of the paragraph within that section.                                                        |
| `paragraph_text`  | string                     | Full text of the paragraph as it would be presented to the extractor. Realistic discipline-flavoured prose, 2-5 sentences.       |
| `expected_claims` | array of claim objects     | Non-empty list of claims a correct extractor should produce from the paragraph.                                |
| `discipline`      | string enum                | One of `astrophysics`, `planetary_science`, `earth_science`.                                                   |

### Claim object schema

Every element of `expected_claims` MUST contain:

| Field             | Type    | Description                                                                                                                      |
| ----------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `claim_text`      | string  | A clean, atomic, human-readable statement of the claim. Need not appear verbatim in the paragraph — the curator may paraphrase. |
| `claim_type`      | string enum | One of `factual`, `methodological`, `comparative`, `speculative`, `cited_from_other`. See semantics below.                  |
| `subject`         | string  | Subject of the (subject, predicate, object) decomposition.                                                                       |
| `predicate`       | string  | Predicate connecting subject to object. Free-text; not a constrained vocabulary.                                                 |
| `object`          | string  | Object of the relation.                                                                                                          |
| `char_span_start` | integer | Inclusive start offset (in characters) into `paragraph_text` of the **anchor span** that supports the claim.                     |
| `char_span_end`   | integer | Exclusive end offset (in characters) into `paragraph_text`.                                                                       |

`paragraph_text[char_span_start:char_span_end]` MUST be a non-empty substring
of `paragraph_text`. The substring is the **anchor**: the contiguous region of
the paragraph that grounds the claim. It is **not** required to equal
`claim_text`. The anchor may be longer or shorter than the claim text and
will typically be a fragment of the original sentence.

## Discipline labels

Three top-level discipline buckets are used. They mirror the SciX coverage
areas relevant to this project.

| Label               | Coverage                                                                              |
| ------------------- | ------------------------------------------------------------------------------------- |
| `astrophysics`      | Stars, galaxies, cosmology, gravitational waves, transients, instrumentation thereof. |
| `planetary_science` | Solar system bodies, planetary atmospheres, surfaces, mission data, small bodies.     |
| `earth_science`     | Atmosphere, oceans, cryosphere, biosphere, remote sensing, climate.                   |

The set is balanced with at least 4 entries per discipline.

## Curation guidelines

### What is an atomic claim?

An atomic claim asserts **one** verifiable proposition. If a sentence makes
multiple assertions ("we used X to compute Y, and the result is Z"), split it
into separate claims (one methodological, one factual). Do not bundle multiple
findings into a single claim.

A claim should be self-contained enough that a downstream nanopub triple has a
plausible (subject, predicate, object) decomposition.

### Anchor spans

The anchor span is a substring of `paragraph_text`. Its purpose is to indicate
which part of the paragraph supports the claim — it is the evidence pointer.

- The anchor MUST be a non-empty substring of `paragraph_text`.
- Anchors do not need to equal `claim_text`; the claim is curator-cleaned, the
  anchor is raw paragraph prose.
- Prefer the **smallest fragment that fully supports the claim**. Whole-sentence
  anchors are acceptable when the supporting evidence is distributed over the
  full sentence.
- Anchors for different claims in the same paragraph may overlap.

### Claim type semantics

| Type                | Definition                                                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `factual`           | A measurement, observation, or fact stated as the paper's own contribution. Includes the paper's own headline numerical results.        |
| `methodological`    | A procedural / method statement: "we used X to compute Y", "the algorithm proceeds in three steps", "fitting performed with package Z". |
| `comparative`       | A comparison with other work, baseline, or alternative method: "our X outperforms Y by N%", "this is consistent / inconsistent with…".  |
| `speculative`       | A hypothesis, projection, future-work statement, or claim hedged with words like *could*, *may*, *will*, *plan to*, *expect*.           |
| `cited_from_other`  | A claim **attributed** to another paper, e.g. "Smith et al. (2020) reported X". The claim is not the current paper's own contribution.  |

When a sentence does both — e.g. a comparison that cites prior work for the
baseline value — annotate the dominant intent of the sentence and, if the two
roles are clearly separable, emit two claims.

### Choosing bibcodes

- Real ADS bibcodes are allowed when they accurately attribute the source.
- For invented or paraphrased prose, use placeholder bibcodes prefixed `GOLD`
  (e.g. `GOLD2024ApJ...001A...01`) so they are obviously not real ADS records.

## Validation

`tests/test_gold_standard_format.py` enforces the schema, enums, per-discipline
balance, and span integrity for every entry. Run:

```bash
pytest tests/test_gold_standard_format.py
```

before merging changes to the gold standard.

## Extending the set

To add an entry, append a JSON line with the schema above. The helper script
`eval/build_gold_standard.py` constructs entries from anchor substrings and
computes character offsets automatically — recommended over computing offsets
by hand. Re-run the helper after editing it:

```bash
python3 eval/build_gold_standard.py
pytest tests/test_gold_standard_format.py
```
