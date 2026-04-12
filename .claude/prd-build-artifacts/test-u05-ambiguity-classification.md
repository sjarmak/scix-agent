# Test Results — u05 Ambiguity Classification

Command:

```
SCIX_TEST_DSN=dbname=scix_test .venv/bin/python -m pytest tests/test_ambiguity.py -v
```

Result: **16 passed, 0 failed** (0.19s).

## Coverage

### Unit tests (`is_banned_name`)

- empty string -> banned (too short)
- 1 char -> banned
- 2 chars (UV, Hi) -> banned
- 3 chars non-common (xyz) -> not banned
- common English word (the, hubble) -> banned via Zipf >= 3.0
- rare long name (GALFA-HI) -> not banned

### Unit tests (`classify`)

- 2-char name (UV) -> banned
- common word (the) -> banned
- banned alias propagates -> banned
- HST + collision -> homograph
- banned precedes homograph (the + collision) -> banned
- GALFA-HI unique single-source -> domain_safe
- 3-char non-banned XYZ -> unique
- long unique multi-source -> unique
- 3-char multi-source -> unique

### Integration test (`test_classify_all_populates_four_buckets_on_fixture`)

Runs against `scix_test`, seeds 9 entities, runs the classifier end-to-end:

| canonical    | source   | expected    |
| ------------ | -------- | ----------- |
| the          | uat      | banned      |
| UV           | uat      | banned      |
| hubble       | uat      | banned      |
| HST          | ads_aas  | homograph   |
| HST          | wikidata | homograph   |
| GALFA-HI     | uat      | domain_safe |
| CHANDRA-XRAY | uat      | domain_safe |
| XYZ          | uat      | unique      |
| QZX          | uat      | unique      |

All 9 seeded entities received their expected classification. The post-run
`GROUP BY ambiguity_class` query returned all four buckets with non-null counts
(3 banned, 2 homograph, 2 domain_safe, 2 unique).

The audit report was written to `build-artifacts/ambiguity_audit.md` and
verified to contain one `## <class>` section per bucket.

## Fixture note

The initial fixture used `"Hubble Space Telescope"` as an alias on one of the
HST rows, which unexpectedly matched the banned rule (`zipf_frequency` of the
full phrase is 3.03, over the 3.0 threshold). The rule is working as
specified — a banned alias correctly propagates to banned — but the fixture
intent was to land HST in `homograph`, so the alias was swapped to `HST-ACS`
(zipf 2.52), which still exercises the alias-collision pathway without
tripping the banned rule. This was a fixture bug, not a classifier bug.

## Acceptance criteria check

1. Script runs against scix_test, populates ambiguity_class, emits progress -- yes, verified via integration test invocation of `classify_all()` plus stdout-logging in the main path.
2. `src/scix/ambiguity.py` exposes pure `classify(...)` -- yes, no DB/IO imports.
3. Unit tests cover all four classifications + edge cases (2-char, 'the', HST collision, GALFA-HI) -- yes.
4. `GROUP BY ambiguity_class` returns 4 non-null buckets -- yes.
5. `build-artifacts/ambiguity_audit.md` with up to 50 per class -- yes.
6. `pytest tests/test_ambiguity.py -v` -- 16/16 passed.
