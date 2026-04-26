"""INDUS embedder: thin wrapper around nasa-impact/nasa-smd-ibm-st-v2.

Encodes batches of chunk texts to 768-dim vectors using mean pooling
over non-padding tokens. Mirrors the ``GlinerExtractor`` pattern from
``scix.extract.ner_pass``: lazy ``_load`` of model + tokenizer so that
constructing an :class:`INDUSEmbedder` (e.g. in a test) does not pay
the torch / transformers import cost. Both ``model`` and ``tokenizer``
are injectable via ``__init__`` for tests.

The tokenizer is exposed via the :attr:`INDUSEmbedder.tokenizer`
property so that the chunker (which needs to count tokens against the
same vocab) can share a single tokenizer instance with the embedder.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: HuggingFace model id. Pinned so re-runs produce reproducible vectors.
DEFAULT_MODEL_NAME = "nasa-impact/nasa-smd-ibm-st-v2"

#: Hidden-state dimension exposed by INDUS. Used downstream as the
#: Qdrant collection's ``vector_size`` and as a sanity check on every
#: returned embedding.
DEFAULT_VECTOR_SIZE = 768

#: GPU inference batch size. 64 matches the PRD's ``--inference-batch
#: 64`` default and keeps an RTX 5090 (32 GB VRAM) busy on 512-token
#: windows without spilling.
DEFAULT_INFERENCE_BATCH = 64

#: Tokenizer truncation cap. INDUS shares BERT's 512-token positional
#: embedding ceiling; chunk windows above this are silently truncated
#: by the tokenizer.
DEFAULT_MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class INDUSEmbedder:
    """Batch-encodes chunk texts to 768-dim vectors via mean pooling.

    Lazy-loads model + tokenizer on first call so that callers (CLI,
    tests) can construct the embedder without paying the ~30s
    transformers import + model download. Tests substitute either by
    passing ``model=`` and/or ``tokenizer=`` explicitly.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cuda",
        inference_batch: int = DEFAULT_INFERENCE_BATCH,
        model: Any = None,  # injectable for tests
        tokenizer: Any = None,  # injectable for tests; also shared with chunker
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.inference_batch = inference_batch
        self.max_length = max_length
        self._model = model  # may be None; loaded lazily
        self._tokenizer = tokenizer  # may be None; loaded lazily

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer  # local import — heavy

        logger.info("embedder: loading tokenizer for %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        import torch  # local import — heavy
        from transformers import AutoModel  # local import — heavy

        logger.info("embedder: loading %s on %s", self.model_name, self.device)
        m = AutoModel.from_pretrained(self.model_name)
        m = m.to(self.device)
        m.eval()
        # Reference torch so the import is exercised under the type
        # checker; suppresses an unused-import warning while keeping
        # the import inside the lazy-load boundary.
        _ = torch
        self._model = m
        return m

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        """Return the loaded tokenizer (lazy-loading on first access).

        Exposed so that the chunker can share a single tokenizer
        instance with the embedder — chunk boundaries computed from
        one vocab must be encoded by the same vocab.
        """
        return self._load_tokenizer()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode ``texts`` to one 768-dim vector per input, in order.

        Empty input list returns an empty output list and does not
        touch the model. Pooling is mean over non-padding tokens
        (matches :func:`scix.embed.embed_batch` with ``pooling="mean"``).
        """
        if not texts:
            return []

        import torch  # local import — heavy

        tokenizer = self._load_tokenizer()
        model = self._load_model()

        out: list[list[float]] = []
        for start in range(0, len(texts), self.inference_batch):
            batch = texts[start : start + self.inference_batch]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Move tensors to the model's device. ``next(model.parameters())``
            # works for both real torch.nn.Module and the test stub which
            # yields a single zero tensor with a ``.device`` attribute.
            try:
                target_device = next(model.parameters()).device
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
            except (StopIteration, AttributeError):
                # Stub model with no parameters; leave inputs on CPU.
                pass

            with torch.no_grad():
                outputs = model(**inputs)

            # Mean pooling over non-padding tokens.
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (B, T, 1)
            masked = outputs.last_hidden_state * attention_mask
            summed = masked.sum(dim=1)  # (B, D)
            counts = attention_mask.sum(dim=1).clamp(min=1)  # (B, 1)
            embeddings = (summed / counts).cpu().tolist()
            out.extend(embeddings)

        return out
