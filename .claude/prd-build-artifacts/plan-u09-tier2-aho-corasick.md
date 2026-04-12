# Plan — u09 Tier 2 Aho-Corasick + Adeft

## Files

1. `src/scix/aho_corasick.py` — pure library. `EntityRow` dataclass,
   `build_automaton(rows)`, `link_abstract(abstract, automaton,
disambiguator=None)`. Pickleable. No DB, no logging side effects.
   Exports `LinkCandidate` dataclass.
2. `src/scix/adeft_disambig.py` — pure library. `AdeftClassifier`
   dataclass wraps sklearn Pipeline. `train_classifier(acronym,
examples)`, `load_classifier(acronym, path)`, `save_classifier(clf,
path)`.
3. `scripts/link_tier2.py` — end-to-end CLI. Reads
   `curated_entity_core`, builds automaton, streams papers + abstracts,
   runs `link_abstract` in a worker pool, enforces 25K cap, writes
   tier=2 rows with `# noqa: resolver-lint`.
4. `tests/test_tier2.py` — seeds scix_test fixture; asserts HST alone
   → no link; HST + "Hubble Space Telescope" → link emitted; cap
   behavior; end-to-end CLI run → ≥1 tier=2 row written.
5. `tests/test_adeft.py` — trains classifiers for HST/JET/AI on
   synthetic data, asserts ≥90% held-out accuracy for each.

## Acceptance plumbing

- AC1: `build_automaton`, `link_abstract`, `AhocorasickAutomaton`
  type alias, `EntityRow`, `LinkCandidate` exports.
- AC2: homograph-with-disambiguator test case in `test_tier2.py`.
- AC3: char_wb (2-5) TF-IDF + LogisticRegression Pipeline; synthetic
  training corpus hand-written (50 positive per long-form per
  acronym).
- AC4: cap enforcement + `link_policy='llm_only'` update path;
  integration test seeds an entity with 2 bibcodes and sets
  `--max-per-entity 1` to exercise the cap.
- AC5: `pytest tests/test_tier2.py tests/test_adeft.py -v` green.
- AC6: `scripts/link_tier2.py` is outside `src/` (AST lint does not
  scan it) — unaffected. Still mark SQL with noqa comment for
  consistency.

## Risks

- Synthetic Adeft corpora can be too easy; keep balanced + add
  shuffled negatives so ≥90% is meaningful.
- `pyahocorasick` expects bytes on some platforms; I'll pass
  lower-cased str and add to automaton via `add_word(str)`.
- Parallel fork: sklearn + ahocorasick must pickle — both do.
