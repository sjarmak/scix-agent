# Research — ner-wiesp-eval

## Existing eval patterns
- `scripts/eval_extraction_quality.py`, `scripts/eval_retrieval.py`, `scripts/eval_retrieval_50q.py` — all use argparse, `sys.path.insert(0, "src")`, frozen dataclasses, logging, and JSON/markdown output.
- `src/scix/eval/` contains audit.py, lane_delta.py, llm_judge.py, metrics.py, wilson.py — pure metric utilities.
- `src/scix/jit/local_ner.py` is a stub of the local NER pathway; the real one will wrap a checkpoint like `adsabs/nasa-smd-ibm-v0.1_NER_DEAL`.

## Dependencies
- `pyproject.toml` declares optional `embed = ["transformers>=4.36", "torch>=2.1"]`.
- `seqeval` is NOT yet declared anywhere in pyproject.toml or installed.
- `datasets` is installed at 4.8.4 (sufficient for loading `adsabs/WIESP2022-NER`).
- `transformers` is NOT installed in the default interpreter — must be optional at script import time.

## Entity type conventions
- `extract.py` uses types: `instruments`, `datasets`, `methods`, `observables`, `materials`, `software`.
- WIESP2022-NER uses BIO-tagged astronomy entity types (e.g., Instrument, CelestialObject, Mission, Software, Dataset). We'll treat the dataset's own label set as the source of truth rather than force-map to extract.py taxonomy.

## Model info
- `adsabs/nasa-smd-ibm-v0.1_NER_DEAL` on HuggingFace. Must pin a commit SHA. Latest commit on `main` is `87ce76dbc8c3b1e3f2bbe2c64fee5d25bc03c03d` (as of 2026-04 per project CLAUDE context; we hardcode a pinned SHA string — see plan).

## Test patterns
- Existing `tests/test_embed.py` uses frozen dataclasses, no pytest marks, imports from scix module. `tests/` has `pythonpath = ["src", "tests"]` in pyproject.
- Monkeypatch fixture available via pytest.

## seqeval
- Planned: add `seqeval>=1.2` to a new optional `ner_eval` extra so we don't affect the default install path. Script imports it lazily and falls back to a pure-Python BIO F1 for offline/test use.
