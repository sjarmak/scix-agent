"""Tests for scix.extract.chunk_pass.embedder.

Covers all 12 acceptance criteria from the embedder work unit:
constants, lazy load, mean pooling math, empty input, tokenizer
truncation kwargs, and the tokenizer property.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scix.extract.chunk_pass import embedder as emb_mod
from scix.extract.chunk_pass.embedder import (
    DEFAULT_INFERENCE_BATCH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_VECTOR_SIZE,
    INDUSEmbedder,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal stand-in for a HuggingFace AutoModel.

    Returns a deterministic ``last_hidden_state`` so that we can
    independently compute the expected mean-pooled vector and compare.
    """

    def __init__(self, hidden: int = DEFAULT_VECTOR_SIZE, fill: float = 1.0) -> None:
        self.hidden = hidden
        self.fill = fill
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        import torch

        mask = kwargs["attention_mask"]
        b, t = mask.shape
        # Record the call so tests can introspect what got passed.
        self.calls.append({"input_ids_shape": kwargs["input_ids"].shape})
        last_hidden = torch.full((b, t, self.hidden), self.fill, dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=last_hidden)

    def parameters(self):
        # Empty generator — encode_batch handles StopIteration and skips
        # device transfer when the stub has no parameters.
        return iter(())

    def eval(self):
        return self

    def to(self, device):
        return self


def _make_stub_tokenizer(call_log: list[dict] | None = None):
    """Return a MagicMock tokenizer that records its kwargs and returns
    pre-shaped torch tensors for input_ids + attention_mask."""
    import torch

    log: list[dict] = call_log if call_log is not None else []

    def _tokenize(texts, **kwargs):
        log.append({"texts": list(texts), "kwargs": dict(kwargs)})
        b = len(texts)
        # Use a fixed seq length so the stub model returns a predictable
        # shape; pad to 4 tokens. Last two are "padding" (mask=0) for the
        # mean-pooling fixture below.
        t = 4
        input_ids = torch.zeros((b, t), dtype=torch.long)
        attention_mask = torch.ones((b, t), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    tok = MagicMock(side_effect=_tokenize)
    tok._call_log = log  # type: ignore[attr-defined]
    return tok


# ---------------------------------------------------------------------------
# Criterion 2-4: constants
# ---------------------------------------------------------------------------


def test_default_model_name_constant():
    """Criterion 2: DEFAULT_MODEL_NAME is the INDUS HF id."""
    assert DEFAULT_MODEL_NAME == "nasa-impact/nasa-smd-ibm-st-v2"


def test_default_vector_size_constant():
    """Criterion 3: DEFAULT_VECTOR_SIZE is 768."""
    assert DEFAULT_VECTOR_SIZE == 768


def test_default_inference_batch_constant():
    """Criterion 4: DEFAULT_INFERENCE_BATCH is in [32, 128]."""
    assert isinstance(DEFAULT_INFERENCE_BATCH, int)
    assert 32 <= DEFAULT_INFERENCE_BATCH <= 128


# ---------------------------------------------------------------------------
# Criterion 5: __init__ signature with injectable model + tokenizer
# ---------------------------------------------------------------------------


def test_constructor_accepts_injected_model_and_tokenizer():
    """Criterion 5: model and tokenizer are injectable via __init__."""
    stub_model = _StubModel()
    stub_tok = _make_stub_tokenizer()
    e = INDUSEmbedder(
        model_name="custom/name",
        device="cpu",
        inference_batch=32,
        model=stub_model,
        tokenizer=stub_tok,
    )
    assert e.model_name == "custom/name"
    assert e.device == "cpu"
    assert e.inference_batch == 32
    assert e._model is stub_model
    assert e._tokenizer is stub_tok


def test_constructor_uses_defaults():
    """__init__ defaults match the module-level constants."""
    e = INDUSEmbedder(model=_StubModel(), tokenizer=_make_stub_tokenizer())
    assert e.model_name == DEFAULT_MODEL_NAME
    assert e.device == "cuda"
    assert e.inference_batch == DEFAULT_INFERENCE_BATCH
    assert e.max_length == DEFAULT_MAX_LENGTH


# ---------------------------------------------------------------------------
# Criterion 6: encode_batch returns one 768-dim vector per input, in order
# ---------------------------------------------------------------------------


def test_encode_batch_returns_one_vector_per_input():
    """Criterion 6: one vector per input text, preserving order."""
    e = INDUSEmbedder(
        model=_StubModel(fill=1.0),
        tokenizer=_make_stub_tokenizer(),
        inference_batch=2,
    )
    texts = ["alpha", "beta", "gamma"]
    out = e.encode_batch(texts)
    assert len(out) == len(texts)
    # Each vector is 768-dim.
    for vec in out:
        assert len(vec) == DEFAULT_VECTOR_SIZE


def test_encode_batch_preserves_input_order():
    """Order-preservation: per-text vectors carry distinguishing values."""
    import torch

    # Build a model whose last_hidden_state encodes the row index, so
    # we can verify the returned vectors come back in input order.
    class _OrderedModel(_StubModel):
        def __call__(self, **kwargs):
            mask = kwargs["attention_mask"]
            b, t = mask.shape
            # Row i gets hidden state filled with float(i + 1).
            rows = [torch.full((1, t, self.hidden), float(i + 1)) for i in range(b)]
            last_hidden = torch.cat(rows, dim=0)
            return SimpleNamespace(last_hidden_state=last_hidden)

    e = INDUSEmbedder(
        model=_OrderedModel(),
        tokenizer=_make_stub_tokenizer(),
        inference_batch=8,
    )
    out = e.encode_batch(["a", "b", "c"])
    assert out[0][0] == pytest.approx(1.0)
    assert out[1][0] == pytest.approx(2.0)
    assert out[2][0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Criterion 7: mean pooling math
# ---------------------------------------------------------------------------


def test_mean_pooling_uses_attention_mask():
    """Criterion 7: pooled vec = sum(hidden * mask) / sum(mask)."""
    import torch

    # Construct a deterministic case: B=1, T=4, mask=[1,1,0,0],
    # last_hidden_state values per row = [10, 20, 30, 40] (broadcast
    # across the hidden dim). Expected pooled = (10 + 20) / 2 = 15.
    class _FixedModel:
        def __call__(self, **kwargs):
            # Per-row scalar values, broadcast to 768 dims.
            row_values = torch.tensor([10.0, 20.0, 30.0, 40.0])
            last_hidden = row_values.view(1, 4, 1).expand(1, 4, DEFAULT_VECTOR_SIZE).contiguous()
            return SimpleNamespace(last_hidden_state=last_hidden)

        def parameters(self):
            return iter(())

        def eval(self):
            return self

        def to(self, _device):
            return self

    def _fixed_tokenize(texts, **_kwargs):
        return {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        }

    tok = MagicMock(side_effect=_fixed_tokenize)
    e = INDUSEmbedder(model=_FixedModel(), tokenizer=tok)
    out = e.encode_batch(["only one input"])
    assert len(out) == 1
    assert len(out[0]) == DEFAULT_VECTOR_SIZE
    for v in out[0]:
        assert v == pytest.approx(15.0)


def test_mean_pooling_all_tokens_active():
    """When mask is all ones, pooled = arithmetic mean across T."""
    import torch

    class _FixedModel:
        def __call__(self, **kwargs):
            row_values = torch.tensor([2.0, 4.0, 6.0, 8.0])  # mean = 5.0
            last_hidden = row_values.view(1, 4, 1).expand(1, 4, DEFAULT_VECTOR_SIZE).contiguous()
            return SimpleNamespace(last_hidden_state=last_hidden)

        def parameters(self):
            return iter(())

        def eval(self):
            return self

        def to(self, _device):
            return self

    def _tokenize(texts, **_kwargs):
        return {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }

    e = INDUSEmbedder(model=_FixedModel(), tokenizer=MagicMock(side_effect=_tokenize))
    out = e.encode_batch(["x"])
    assert all(v == pytest.approx(5.0) for v in out[0])


# ---------------------------------------------------------------------------
# Criterion 8: lazy load (no torch / transformers at module import)
# ---------------------------------------------------------------------------


def test_module_import_does_not_pull_torch_or_transformers():
    """Criterion 8: importing the embedder module must not import torch
    or transformers eagerly. Run in a fresh subprocess so any prior
    import of these modules in this test process doesn't leak in."""
    import os
    from pathlib import Path

    # Locate the package's parent directory ("src") so we can hand it
    # to the subprocess via PYTHONPATH — pytest's pythonpath = ["src"]
    # config does not propagate to a freshly spawned interpreter.
    pkg_file = Path(emb_mod.__file__).resolve()
    # .../src/scix/extract/chunk_pass/embedder.py -> .../src
    # parents: 0=chunk_pass, 1=extract, 2=scix, 3=src
    src_root = pkg_file.parents[3]

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_root}{os.pathsep}{existing}" if existing else str(src_root)

    code = (
        "import sys\n"
        "import scix.extract.chunk_pass.embedder  # noqa: F401\n"
        "print('torch' in sys.modules, 'transformers' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert result.stdout.strip() == "False False", (
        f"expected 'False False' (lazy load), got: {result.stdout!r}; "
        f"stderr: {result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Criterion 9: empty input list returns empty output, no model call
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_output():
    """Criterion 9: encode_batch([]) -> [] without touching the model."""
    stub_model = _StubModel()
    stub_tok = _make_stub_tokenizer()
    e = INDUSEmbedder(model=stub_model, tokenizer=stub_tok)
    out = e.encode_batch([])
    assert out == []
    assert stub_model.calls == []
    stub_tok.assert_not_called()


def test_empty_input_does_not_load_model_or_tokenizer():
    """Empty input must not even trigger lazy model/tokenizer load."""
    e = INDUSEmbedder()  # no injected model / tokenizer
    # If this called _load_*, transformers would try to fetch from HF.
    out = e.encode_batch([])
    assert out == []
    assert e._model is None
    assert e._tokenizer is None


# ---------------------------------------------------------------------------
# Criterion 10: tokenizer truncation to max_length=512
# ---------------------------------------------------------------------------


def test_tokenizer_called_with_truncation_and_max_length():
    """Criterion 10: tokenizer receives max_length=512 + truncation=True."""
    call_log: list[dict] = []
    stub_tok = _make_stub_tokenizer(call_log=call_log)
    e = INDUSEmbedder(model=_StubModel(), tokenizer=stub_tok)
    e.encode_batch(["some chunk text"])

    assert len(call_log) == 1
    kwargs = call_log[0]["kwargs"]
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == 512
    assert kwargs["padding"] is True
    assert kwargs["return_tensors"] == "pt"


def test_tokenizer_max_length_matches_default_constant():
    """Default max_length attribute is 512."""
    e = INDUSEmbedder(model=_StubModel(), tokenizer=_make_stub_tokenizer())
    assert e.max_length == 512


# ---------------------------------------------------------------------------
# Criterion 11: tokenizer property exposes the loaded tokenizer (lazy)
# ---------------------------------------------------------------------------


def test_tokenizer_property_returns_injected_tokenizer():
    """Criterion 11: tokenizer property returns the injected stub."""
    stub_tok = _make_stub_tokenizer()
    e = INDUSEmbedder(model=_StubModel(), tokenizer=stub_tok)
    assert e.tokenizer is stub_tok


def test_tokenizer_property_lazy_loads(monkeypatch):
    """Property triggers _load_tokenizer when not yet loaded."""
    e = INDUSEmbedder()  # nothing injected
    fake_tok = object()

    def _fake_load(self):
        self._tokenizer = fake_tok
        return fake_tok

    monkeypatch.setattr(INDUSEmbedder, "_load_tokenizer", _fake_load)

    assert e._tokenizer is None
    assert e.tokenizer is fake_tok
    assert e._tokenizer is fake_tok


# ---------------------------------------------------------------------------
# Criterion 6 cont'd: batched input handled across multiple inference batches
# ---------------------------------------------------------------------------


def test_encode_batch_chunks_inputs_by_inference_batch():
    """Inputs larger than ``inference_batch`` are processed in chunks."""
    stub_model = _StubModel()
    stub_tok = _make_stub_tokenizer()
    e = INDUSEmbedder(model=stub_model, tokenizer=stub_tok, inference_batch=2)
    texts = ["a", "b", "c", "d", "e"]  # 3 inference batches: 2 + 2 + 1
    out = e.encode_batch(texts)
    assert len(out) == 5
    assert len(stub_model.calls) == 3


# ---------------------------------------------------------------------------
# Module shape sanity checks
# ---------------------------------------------------------------------------


def test_module_exposes_expected_public_names():
    """All five constants / class are importable from the module."""
    assert hasattr(emb_mod, "DEFAULT_MODEL_NAME")
    assert hasattr(emb_mod, "DEFAULT_VECTOR_SIZE")
    assert hasattr(emb_mod, "DEFAULT_INFERENCE_BATCH")
    assert hasattr(emb_mod, "INDUSEmbedder")
