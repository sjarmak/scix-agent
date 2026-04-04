# Plan: Citation Intent Classifier

## Files

1. `src/scix/citation_intent.py` — Core module
2. `scripts/train_citation_intent.py` — Training script
3. `tests/test_citation_intent.py` — Tests

## Implementation

### src/scix/citation_intent.py

- `VALID_INTENTS = frozenset({"background", "method", "result_comparison"})`
- `SCICITE_LABEL_MAP = {0: "background", 1: "method", 2: "result_comparison"}`
- `IntentClassifier` Protocol with `classify_intent(context_text: str) -> str` and `classify_batch(texts: list[str]) -> list[str]`
- `SciBertClassifier` class wrapping `transformers.pipeline("text-classification", model=model_path)`
  - Maps SciCite labels to our 3 classes
  - `classify_batch()` uses pipeline batching
- `LLMClassifier` class using Anthropic Messages API
  - System prompt constraining output to one of 3 labels
  - `classify_batch()` iterates sequentially
- `update_intents(conn, classifier, batch_size)` — reads citation_contexts WHERE intent IS NULL, classifies, updates

### scripts/train_citation_intent.py

- Download SciCite from HuggingFace
- Map labels
- Fine-tune SciBERT with Trainer
- Save model

### tests/test_citation_intent.py

- Mock transformers pipeline for SciBertClassifier
- Mock anthropic client for LLMClassifier
- Test label validation
- Test batch processing
- Test update_intents with mock DB
