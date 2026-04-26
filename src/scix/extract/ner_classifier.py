"""INDUS-based post-classifier for GLiNER NER mentions (dbl.3 quality lift).

Two-stage NER per the dbl.3 eval-driven design:

  1. GLiNER (existing) finds candidate spans + assigns predicted_type.
     Span detection is high-recall but typing is noisy (~58% modern, 54%
     pre-1990 aggregate precision per the 2026-04-25 evals).
  2. This module re-types the span via INDUS embedding similarity to
     hand-curated anchor centroids per type. Where the classifier
     disagrees with GLiNER, the mention is flagged
     ``agreement=false`` and downstream tools default-filter it out.

Anchors live in ``data/ner_anchors/seed_v{N}.json``: ~150 hand-curated
canonical examples per type, each with a usage sentence so the
embedding captures how the entity is referred to in real prose.
The per-type centroid is the mean of those anchor embeddings under
INDUS (``nasa-impact/nasa-smd-ibm-st-v2``).

This module is pure logic — no DB writes. The companion
``ner_classify_pass.py`` drives the post-pass over
``document_entities``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ANCHORS_PATH = Path(__file__).resolve().parents[3] / "data" / "ner_anchors" / "seed_v1.json"
DEFAULT_INDUS_MODEL = "indus"  # resolves via scix.embed.load_model

#: Sentence-boundary regex used to extract the sentence containing a
#: mention from the surrounding paragraph. Matches a sentence-final
#: punctuation followed by whitespace and a capital letter, similar to
#: NLTK's punkt rule but without the dependency.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\d])")


def extract_sentence(text: str, mention: str) -> str:
    """Return the sentence in ``text`` that contains ``mention`` (case-insensitive).

    Falls back to ``text[:300]`` if the mention isn't found, so a missing
    span never produces an empty embedding input.
    """
    if not text or not mention:
        return text[:300] if text else ""
    lower = text.lower()
    pos = lower.find(mention.lower())
    if pos < 0:
        return text[:300]
    sentences = _SENTENCE_BOUNDARY_RE.split(text)
    cumulative = 0
    for s in sentences:
        end = cumulative + len(s)
        if cumulative <= pos < end:
            return s.strip()
        cumulative = end + 1  # +1 for the split whitespace
    return text[:300]


@dataclass(frozen=True)
class ClassificationResult:
    """Output of :meth:`NerClassifier.classify`."""

    predicted_type: str  # GLiNER's call
    classifier_type: str  # this module's call
    classifier_score: float  # cosine sim to the winning centroid
    agreement: bool


class NerClassifier:
    """Cosine-similarity classifier over per-type INDUS anchor centroids.

    Lazy-loads INDUS on first ``classify()`` call so unit tests can inject
    a stub embedder via the ``embedder`` constructor argument.
    """

    def __init__(
        self,
        anchors_path: Path = DEFAULT_ANCHORS_PATH,
        embedder: Any = None,  # injectable for tests; expects .embed_batch(texts) -> list[list[float]]
        device: str = "cuda",
    ) -> None:
        self.anchors_path = anchors_path
        self.device = device
        self._embedder = embedder
        self._anchor_payload: dict[str, Any] | None = None
        self._centroids: dict[str, list[float]] | None = None
        self._known_types: tuple[str, ...] = ()

    # ----- loading & embedding ---------------------------------------------

    def _load_anchor_payload(self) -> dict[str, Any]:
        if self._anchor_payload is not None:
            return self._anchor_payload
        with self.anchors_path.open() as fh:
            payload = json.load(fh)
        self._anchor_payload = payload
        self._known_types = tuple(payload["anchors"].keys())
        return payload

    def _load_embedder(self) -> Any:
        """Return an object with ``.embed_batch(list[str]) -> list[list[float]]``."""
        if self._embedder is not None:
            return self._embedder

        # Wrap the existing scix.embed primitives so the caller doesn't need
        # to know about model/tokenizer/pooling. We import locally because
        # the embed module pulls in torch + transformers, which we don't
        # want at module-import time.
        from scix.embed import embed_batch, load_model

        class _IndusEmbedder:
            def __init__(self, device: str) -> None:
                self.model, self.tokenizer = load_model("indus", device=device)

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                if not texts:
                    return []
                return embed_batch(self.model, self.tokenizer, texts, batch_size=64, pooling="mean")

        self._embedder = _IndusEmbedder(self.device)
        return self._embedder

    def _ensure_centroids(self) -> dict[str, list[float]]:
        """Compute per-type centroids from anchor (text, context) pairs."""
        if self._centroids is not None:
            return self._centroids

        payload = self._load_anchor_payload()
        embedder = self._load_embedder()

        # Flatten so we make a single embed_batch call across all anchors.
        flat_texts: list[str] = []
        index: list[tuple[str, int]] = []  # (type, position-within-type)
        for etype, anchors in payload["anchors"].items():
            for i, a in enumerate(anchors):
                # Same shape we'll use at classify time: "MENTION | CONTEXT".
                flat_texts.append(f"{a['text']} | {a['context']}")
                index.append((etype, i))

        logger.info("ner_classifier: embedding %d anchors", len(flat_texts))
        embeddings = embedder.embed_batch(flat_texts)

        per_type: dict[str, list[list[float]]] = {t: [] for t in self._known_types}
        for (etype, _), emb in zip(index, embeddings, strict=True):
            per_type[etype].append(emb)

        self._centroids = {
            etype: _centroid(vectors) for etype, vectors in per_type.items() if vectors
        }
        logger.info("ner_classifier: built centroids for types %s", sorted(self._centroids.keys()))
        return self._centroids

    # ----- classification ---------------------------------------------------

    def classify(
        self,
        mention: str,
        context: str,
        predicted_type: str,
    ) -> ClassificationResult:
        """Classify a single mention.

        ``mention`` is the surface text GLiNER found. ``context`` is a
        short snippet (ideally the sentence containing the mention).
        ``predicted_type`` is GLiNER's call — used to compute
        ``agreement`` and never affects the cosine ranking itself.
        """
        results = self.classify_batch([(mention, context, predicted_type)])
        return results[0]

    def classify_batch(
        self,
        items: list[tuple[str, str, str]],
    ) -> list[ClassificationResult]:
        """Vectorized classify over a list of ``(mention, context, predicted_type)``.

        Issues one ``embed_batch`` call for the whole list, so this is
        much faster than calling :meth:`classify` per mention.
        """
        if not items:
            return []
        centroids = self._ensure_centroids()
        embedder = self._load_embedder()

        texts = [_compose_input(m, c) for m, c, _ in items]
        embeddings = embedder.embed_batch(texts)

        type_names = list(centroids.keys())
        type_vecs = [centroids[t] for t in type_names]

        out: list[ClassificationResult] = []
        for emb, (_, _, predicted) in zip(embeddings, items, strict=True):
            sims = [_cosine(emb, tv) for tv in type_vecs]
            best_idx = max(range(len(sims)), key=sims.__getitem__)
            best_type = type_names[best_idx]
            out.append(
                ClassificationResult(
                    predicted_type=predicted,
                    classifier_type=best_type,
                    classifier_score=float(sims[best_idx]),
                    agreement=(best_type == predicted),
                )
            )
        return out


# ---------------------------------------------------------------------------
# Pure-math helpers
# ---------------------------------------------------------------------------


def _compose_input(mention: str, context: str) -> str:
    """Build the embedding input string. Mirrors the anchor format exactly
    so anchor centroids and live mention embeddings share the same input
    distribution."""
    mention = (mention or "").strip()
    context = (context or "").strip()
    if not mention:
        return context[:300]
    if not context:
        return mention
    return f"{mention} | {context}"


def _centroid(vectors: list[list[float]]) -> list[float]:
    """Element-wise mean of a list of equal-length vectors."""
    if not vectors:
        raise ValueError("cannot centroid empty list")
    n = len(vectors[0])
    sums = [0.0] * n
    for v in vectors:
        if len(v) != n:
            raise ValueError(f"vector length mismatch: expected {n}, got {len(v)}")
        for i, x in enumerate(v):
            sums[i] += x
    inv = 1.0 / len(vectors)
    return [s * inv for s in sums]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 if either vector has zero norm."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
