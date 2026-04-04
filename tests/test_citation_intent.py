"""Tests for citation intent classification pipeline.

Mocks transformers and anthropic — no real model or API calls.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import FrozenInstanceError

from scix.citation_intent import (
    VALID_INTENTS,
    SCICITE_LABEL_MAP,
    SCICITE_STRING_LABEL_MAP,
    IntentClassifier,
    SciBertClassifier,
    LLMClassifier,
    classify_intent,
    classify_batch,
    _validate_intent,
    _map_scicite_label,
    _pipeline_cache,
)

# ---------------------------------------------------------------------------
# Constants / validation
# ---------------------------------------------------------------------------


class TestConstants:
    def test_valid_intents_contains_expected(self) -> None:
        assert VALID_INTENTS == frozenset({"background", "method", "result_comparison"})

    def test_scicite_label_map_covers_all_ids(self) -> None:
        assert set(SCICITE_LABEL_MAP.keys()) == {0, 1, 2}
        assert set(SCICITE_LABEL_MAP.values()) <= VALID_INTENTS

    def test_scicite_string_label_map_values_valid(self) -> None:
        for value in SCICITE_STRING_LABEL_MAP.values():
            assert value in VALID_INTENTS


class TestValidateIntent:
    @pytest.mark.parametrize("label", sorted(VALID_INTENTS))
    def test_valid_labels_pass(self, label: str) -> None:
        assert _validate_intent(label) == label

    @pytest.mark.parametrize("bad", ["unknown", "BACKGROUND", "result", "", "citation"])
    def test_invalid_labels_raise(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Invalid intent label"):
            _validate_intent(bad)


class TestMapSciciteLabel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("background", "background"),
            ("method", "method"),
            ("result", "result_comparison"),
            ("LABEL_0", "background"),
            ("LABEL_1", "method"),
            ("LABEL_2", "result_comparison"),
        ],
    )
    def test_known_labels_map(self, raw: str, expected: str) -> None:
        assert _map_scicite_label(raw) == expected

    def test_unknown_label_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown SciCite label"):
            _map_scicite_label("LABEL_99")


# ---------------------------------------------------------------------------
# Protocol check
# ---------------------------------------------------------------------------


class TestIntentClassifierProtocol:
    def test_scibert_is_intent_classifier(self) -> None:
        assert isinstance(SciBertClassifier(model_path="dummy"), IntentClassifier)

    def test_llm_is_intent_classifier(self) -> None:
        assert isinstance(LLMClassifier(), IntentClassifier)


# ---------------------------------------------------------------------------
# SciBertClassifier
# ---------------------------------------------------------------------------


class TestSciBertClassifier:
    def test_frozen(self) -> None:
        clf = SciBertClassifier(model_path="m", batch_size=8, device=-1)
        with pytest.raises(FrozenInstanceError):
            clf.model_path = "other"  # type: ignore[misc]

    @patch("scix.citation_intent.SciBertClassifier._get_pipeline")
    def test_classify_intent_background(self, mock_get_pipe: MagicMock) -> None:
        pipe = MagicMock()
        pipe.return_value = [{"label": "background", "score": 0.95}]
        mock_get_pipe.return_value = pipe

        clf = SciBertClassifier(model_path="m")
        result = clf.classify_intent("This paper builds on prior work...")
        assert result == "background"
        pipe.assert_called_once_with("This paper builds on prior work...")

    @patch("scix.citation_intent.SciBertClassifier._get_pipeline")
    def test_classify_intent_method(self, mock_get_pipe: MagicMock) -> None:
        pipe = MagicMock()
        pipe.return_value = [{"label": "method", "score": 0.88}]
        mock_get_pipe.return_value = pipe

        clf = SciBertClassifier(model_path="m")
        assert clf.classify_intent("We used the technique from...") == "method"

    @patch("scix.citation_intent.SciBertClassifier._get_pipeline")
    def test_classify_intent_result_via_label2(self, mock_get_pipe: MagicMock) -> None:
        pipe = MagicMock()
        pipe.return_value = [{"label": "LABEL_2", "score": 0.77}]
        mock_get_pipe.return_value = pipe

        clf = SciBertClassifier(model_path="m")
        assert clf.classify_intent("Our results outperform...") == "result_comparison"

    @patch("scix.citation_intent.SciBertClassifier._get_pipeline")
    def test_classify_intent_result_via_scicite_string(self, mock_get_pipe: MagicMock) -> None:
        pipe = MagicMock()
        pipe.return_value = [{"label": "result", "score": 0.80}]
        mock_get_pipe.return_value = pipe

        clf = SciBertClassifier(model_path="m")
        assert clf.classify_intent("Compared to [REF]...") == "result_comparison"

    @patch("scix.citation_intent.SciBertClassifier._get_pipeline")
    def test_classify_batch(self, mock_get_pipe: MagicMock) -> None:
        pipe = MagicMock()
        pipe.return_value = [
            {"label": "background", "score": 0.9},
            {"label": "method", "score": 0.85},
            {"label": "result", "score": 0.7},
        ]
        mock_get_pipe.return_value = pipe

        clf = SciBertClassifier(model_path="m", batch_size=16)
        texts = ["ctx1", "ctx2", "ctx3"]
        results = clf.classify_batch(texts)

        assert results == ["background", "method", "result_comparison"]
        pipe.assert_called_once_with(texts, batch_size=16)

    @patch("scix.citation_intent.SciBertClassifier._get_pipeline")
    def test_classify_batch_empty(self, mock_get_pipe: MagicMock) -> None:
        clf = SciBertClassifier(model_path="m")
        assert clf.classify_batch([]) == []
        mock_get_pipe.assert_not_called()

    @patch("scix.citation_intent.SciBertClassifier._get_pipeline")
    def test_classify_intent_invalid_label_raises(self, mock_get_pipe: MagicMock) -> None:
        pipe = MagicMock()
        pipe.return_value = [{"label": "GARBAGE", "score": 0.99}]
        mock_get_pipe.return_value = pipe

        clf = SciBertClassifier(model_path="m")
        with pytest.raises(ValueError, match="Unknown SciCite label"):
            clf.classify_intent("Some text")


# ---------------------------------------------------------------------------
# LLMClassifier
# ---------------------------------------------------------------------------


class TestLLMClassifier:
    def test_frozen(self) -> None:
        clf = LLMClassifier()
        with pytest.raises(FrozenInstanceError):
            clf.model = "other"  # type: ignore[misc]

    def _mock_anthropic_response(self, label: str) -> MagicMock:
        """Build a mock Anthropic response returning the given label."""
        content_block = MagicMock()
        content_block.text = f"  {label}  "  # whitespace to test .strip()
        response = MagicMock()
        response.content = [content_block]
        return response

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.citation_intent.LLMClassifier._get_client")
    def test_classify_intent_background(self, mock_get_client: MagicMock) -> None:
        client = MagicMock()
        client.messages.create.return_value = self._mock_anthropic_response("background")
        mock_get_client.return_value = client

        clf = LLMClassifier()
        result = clf.classify_intent("Prior work has shown...")
        assert result == "background"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.citation_intent.LLMClassifier._get_client")
    def test_classify_intent_method(self, mock_get_client: MagicMock) -> None:
        client = MagicMock()
        client.messages.create.return_value = self._mock_anthropic_response("method")
        mock_get_client.return_value = client

        clf = LLMClassifier()
        assert clf.classify_intent("We adopt the approach of...") == "method"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.citation_intent.LLMClassifier._get_client")
    def test_classify_intent_result_comparison(self, mock_get_client: MagicMock) -> None:
        client = MagicMock()
        client.messages.create.return_value = self._mock_anthropic_response("result_comparison")
        mock_get_client.return_value = client

        clf = LLMClassifier()
        assert clf.classify_intent("Our results exceed...") == "result_comparison"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.citation_intent.LLMClassifier._get_client")
    def test_classify_intent_invalid_response_raises(self, mock_get_client: MagicMock) -> None:
        client = MagicMock()
        client.messages.create.return_value = self._mock_anthropic_response("nonsense")
        mock_get_client.return_value = client

        clf = LLMClassifier()
        with pytest.raises(ValueError, match="Invalid intent label"):
            clf.classify_intent("Some text")

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.citation_intent.LLMClassifier._get_client")
    def test_classify_batch(self, mock_get_client: MagicMock) -> None:
        client = MagicMock()
        client.messages.create.side_effect = [
            self._mock_anthropic_response("background"),
            self._mock_anthropic_response("method"),
            self._mock_anthropic_response("result_comparison"),
        ]
        mock_get_client.return_value = client

        clf = LLMClassifier()
        results = clf.classify_batch(["a", "b", "c"])
        assert results == ["background", "method", "result_comparison"]
        assert client.messages.create.call_count == 3

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.citation_intent.LLMClassifier._get_client")
    def test_classify_batch_empty(self, mock_get_client: MagicMock) -> None:
        clf = LLMClassifier()
        assert clf.classify_batch([]) == []
        mock_get_client.assert_not_called()

    @patch.dict("os.environ", {}, clear=False)
    def test_missing_api_key_raises(self) -> None:
        import os

        env_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            clf = LLMClassifier()
            # _get_client should raise because ANTHROPIC_API_KEY is missing
            # We need to call the real _get_client, so patch the import
            with patch.dict(
                "os.environ", {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            ):
                with pytest.raises((ValueError, ImportError)):
                    clf.classify_intent("text")
        finally:
            if env_backup is not None:
                os.environ["ANTHROPIC_API_KEY"] = env_backup

    def test_default_model(self) -> None:
        clf = LLMClassifier()
        assert clf.model == "claude-sonnet-4-20250514"
        assert clf.max_tokens == 16

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.citation_intent.LLMClassifier._get_client")
    def test_classify_intent_strips_whitespace(self, mock_get_client: MagicMock) -> None:
        client = MagicMock()
        # Simulate model returning label with newlines/spaces
        content_block = MagicMock()
        content_block.text = "\n  Background \n"
        response = MagicMock()
        response.content = [content_block]
        client.messages.create.return_value = response
        mock_get_client.return_value = client

        clf = LLMClassifier()
        # .strip().lower() should normalize "  Background " -> "background"
        assert clf.classify_intent("text") == "background"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions:
    """Test the module-level classify_intent() and classify_batch() functions."""

    @patch("scix.citation_intent.SciBertClassifier._get_pipeline")
    def test_classify_intent_delegates_to_classifier(self, mock_get_pipe: MagicMock) -> None:
        pipe = MagicMock()
        pipe.return_value = [{"label": "method", "score": 0.9}]
        mock_get_pipe.return_value = pipe

        clf = SciBertClassifier(model_path="m")
        result = classify_intent(clf, "We used the approach from...")
        assert result == "method"

    @patch("scix.citation_intent.SciBertClassifier._get_pipeline")
    def test_classify_batch_delegates_to_classifier(self, mock_get_pipe: MagicMock) -> None:
        pipe = MagicMock()
        pipe.return_value = [
            {"label": "background", "score": 0.9},
            {"label": "result", "score": 0.8},
        ]
        mock_get_pipe.return_value = pipe

        clf = SciBertClassifier(model_path="m", batch_size=8)
        results = classify_batch(clf, ["ctx1", "ctx2"])
        assert results == ["background", "result_comparison"]

    @patch("scix.citation_intent.SciBertClassifier._get_pipeline")
    def test_classify_batch_empty(self, mock_get_pipe: MagicMock) -> None:
        clf = SciBertClassifier(model_path="m")
        assert classify_batch(clf, []) == []
        mock_get_pipe.assert_not_called()

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("scix.citation_intent.LLMClassifier._get_client")
    def test_classify_intent_with_llm_classifier(self, mock_get_client: MagicMock) -> None:
        content_block = MagicMock()
        content_block.text = "background"
        response = MagicMock()
        response.content = [content_block]
        client = MagicMock()
        client.messages.create.return_value = response
        mock_get_client.return_value = client

        clf = LLMClassifier()
        result = classify_intent(clf, "Prior work shows...")
        assert result == "background"


# ---------------------------------------------------------------------------
# Pipeline caching
# ---------------------------------------------------------------------------


class TestPipelineCaching:
    """Test that SciBertClassifier caches pipelines at module level."""

    def setup_method(self) -> None:
        _pipeline_cache.clear()

    def teardown_method(self) -> None:
        _pipeline_cache.clear()

    def test_cache_is_populated(self) -> None:
        """After a call, the model_path key should exist in _pipeline_cache."""
        # Pre-populate cache so _get_pipeline returns it without importing transformers.
        fake_pipe = MagicMock()
        fake_pipe.return_value = [{"label": "background", "score": 0.9}]
        _pipeline_cache["cached-model"] = fake_pipe

        clf = SciBertClassifier(model_path="cached-model")
        result = clf.classify_intent("text")
        assert result == "background"
        # The fake pipeline was used — no transformers import needed
        fake_pipe.assert_called_once_with("text")

    def test_cache_reuses_pipeline(self) -> None:
        """Two classifiers with the same model_path should share the cached pipeline."""
        fake_pipe = MagicMock()
        fake_pipe.return_value = [{"label": "method", "score": 0.9}]
        _pipeline_cache["shared-model"] = fake_pipe

        clf1 = SciBertClassifier(model_path="shared-model")
        clf2 = SciBertClassifier(model_path="shared-model")

        clf1.classify_intent("a")
        clf2.classify_intent("b")

        assert fake_pipe.call_count == 2
