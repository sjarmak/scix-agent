"""Citation intent classification pipeline.

Classifies citation contexts into three intent classes — background, method,
result_comparison — using either a local SciBERT model or an Anthropic LLM
fallback.  Updates the intent column in the citation_contexts table.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import psycopg

from scix.db import get_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_INTENTS: frozenset[str] = frozenset({"background", "method", "result_comparison"})

# SciCite dataset uses integer labels: 0=background, 1=method, 2=result
# We map "result" -> "result_comparison" for our schema.
SCICITE_LABEL_MAP: dict[int, str] = {
    0: "background",
    1: "method",
    2: "result_comparison",
}

# Reverse map from SciCite string labels emitted by HuggingFace pipeline
SCICITE_STRING_LABEL_MAP: dict[str, str] = {
    "background": "background",
    "method": "method",
    "result": "result_comparison",
    # Also handle LABEL_N format from generic classifiers
    "LABEL_0": "background",
    "LABEL_1": "method",
    "LABEL_2": "result_comparison",
}

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IntentClassifier(Protocol):
    """Protocol for citation intent classifiers."""

    def classify_intent(self, context_text: str) -> str:
        """Classify a single citation context.

        Returns one of: 'background', 'method', 'result_comparison'.
        """
        ...

    def classify_batch(self, texts: list[str]) -> list[str]:
        """Classify multiple citation contexts.

        Returns a list of intent labels, one per input text.
        """
        ...


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_intent(label: str) -> str:
    """Validate and return an intent label, raising ValueError if invalid."""
    if label not in VALID_INTENTS:
        raise ValueError(f"Invalid intent label {label!r}; expected one of {sorted(VALID_INTENTS)}")
    return label


def _map_scicite_label(raw_label: str) -> str:
    """Map a raw SciCite or LABEL_N string to a valid intent."""
    mapped = SCICITE_STRING_LABEL_MAP.get(raw_label)
    if mapped is None:
        raise ValueError(
            f"Unknown SciCite label {raw_label!r}; "
            f"expected one of {sorted(SCICITE_STRING_LABEL_MAP)}"
        )
    return mapped


# ---------------------------------------------------------------------------
# SciBERT classifier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SciBertClassifier:
    """Citation intent classifier using a fine-tuned SciBERT model.

    Wraps a ``transformers.pipeline("text-classification", ...)`` for local
    inference.  The model should be fine-tuned on the SciCite dataset.

    Parameters
    ----------
    model_path : str
        Path to the saved model directory (or HuggingFace model ID).
    batch_size : int
        Batch size for pipeline inference.
    device : int
        Device ordinal (-1 for CPU, 0+ for GPU).
    """

    model_path: str
    batch_size: int = 32
    device: int = -1

    def _get_pipeline(self) -> Any:
        """Lazily build the transformers pipeline."""
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError(
                "transformers is required for SciBertClassifier. "
                "Install with: pip install transformers torch"
            )
        return hf_pipeline(
            "text-classification",
            model=self.model_path,
            device=self.device,
            truncation=True,
        )

    def classify_intent(self, context_text: str) -> str:
        """Classify a single citation context using the SciBERT model."""
        pipe = self._get_pipeline()
        result = pipe(context_text)[0]
        raw_label: str = result["label"]
        return _validate_intent(_map_scicite_label(raw_label))

    def classify_batch(self, texts: list[str]) -> list[str]:
        """Classify a batch of citation contexts using batched inference."""
        if not texts:
            return []
        pipe = self._get_pipeline()
        results = pipe(texts, batch_size=self.batch_size)
        labels: list[str] = []
        for result in results:
            raw_label: str = result["label"]
            labels.append(_validate_intent(_map_scicite_label(raw_label)))
        return labels


# ---------------------------------------------------------------------------
# LLM classifier (Anthropic fallback)
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
You are a citation intent classifier for scientific papers. Given a citation \
context (a passage surrounding a citation marker), classify the intent of the \
citation into exactly one of these three categories:

- background: The citation provides general context, motivation, or prior \
work that sets up the current study. It is not directly used in the method \
or compared against.
- method: The citation describes a method, tool, dataset, or technique that \
the current paper adopts, extends, or builds upon.
- result_comparison: The citation is used to compare results, validate \
findings, or contrast with the current paper's outcomes.

Respond with ONLY the label — one of: background, method, result_comparison
No explanation, no punctuation, no extra text."""


@dataclass(frozen=True)
class LLMClassifier:
    """Citation intent classifier using the Anthropic Messages API.

    Parameters
    ----------
    model : str
        Anthropic model ID.
    max_tokens : int
        Maximum tokens in the classification response.
    """

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 16

    def _get_client(self) -> Any:
        """Build an Anthropic client from environment."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic is required for LLMClassifier. " "Install with: pip install anthropic"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it before using LLMClassifier."
            )
        return anthropic.Anthropic(api_key=api_key)

    def _classify_single(self, client: Any, context_text: str) -> str:
        """Classify one context using the Anthropic Messages API."""
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context_text}],
        )
        raw_label = response.content[0].text.strip().lower()
        return _validate_intent(raw_label)

    def classify_intent(self, context_text: str) -> str:
        """Classify a single citation context via the Anthropic API."""
        client = self._get_client()
        return self._classify_single(client, context_text)

    def classify_batch(self, texts: list[str]) -> list[str]:
        """Classify multiple citation contexts sequentially via the API."""
        if not texts:
            return []
        client = self._get_client()
        labels: list[str] = []
        for text in texts:
            label = self._classify_single(client, text)
            labels.append(label)
        return labels


# ---------------------------------------------------------------------------
# Database update pipeline
# ---------------------------------------------------------------------------


def update_intents(
    conn: psycopg.Connection,
    classifier: IntentClassifier,
    batch_size: int = 256,
    limit: int | None = None,
) -> int:
    """Read citation contexts with NULL intent, classify, and update.

    Parameters
    ----------
    conn : psycopg.Connection
        Database connection (should NOT be in autocommit mode).
    classifier : IntentClassifier
        Any object satisfying the IntentClassifier protocol.
    batch_size : int
        Number of rows to classify and update per round-trip.
    limit : int | None
        Maximum total rows to process (None = all).

    Returns
    -------
    int
        Total number of rows updated.
    """
    total_updated = 0

    while True:
        if limit is not None:
            remaining = limit - total_updated
            if remaining <= 0:
                break
            fetch_size = min(batch_size, remaining)
        else:
            fetch_size = batch_size

        # Fetch a batch of unclassified contexts
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_bibcode, target_bibcode, char_offset, context_text
                FROM citation_contexts
                WHERE intent IS NULL
                LIMIT %s
                """,
                (fetch_size,),
            )
            rows = cur.fetchall()

        if not rows:
            break

        texts = [row[3] for row in rows]
        intents = classifier.classify_batch(texts)

        # Update each row
        with conn.cursor() as cur:
            for row, intent in zip(rows, intents):
                source_bibcode, target_bibcode, char_offset, _ = row
                cur.execute(
                    """
                    UPDATE citation_contexts
                    SET intent = %s
                    WHERE source_bibcode = %s
                      AND target_bibcode = %s
                      AND char_offset = %s
                    """,
                    (intent, source_bibcode, target_bibcode, char_offset),
                )

        conn.commit()
        total_updated += len(rows)
        logger.info("Updated %d intents (total: %d)", len(rows), total_updated)

    logger.info("Intent update complete: %d rows", total_updated)
    return total_updated
