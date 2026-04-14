"""Unit tests for src/scix/sources/licensing.py.

Covers ADR-006 acceptance criteria:

- Snippet budget enforcement (default, override, env var)
- canonical_url is a hard requirement
- Truncation appends ellipsis and total length stays within budget
- Env var resolution is lazy (per-call, not per-import)
- Malformed env var raises ValueError at call time
- Frozen dataclass return contract
- Original length is preserved on the payload for observability

No database, no network. Pure unit tests.
"""

from __future__ import annotations

import dataclasses

import pytest

from scix.sources.licensing import (
    DEFAULT_SNIPPET_BUDGET,
    SNIPPET_BUDGET_ENV_VAR,
    SnippetPayload,
    enforce_snippet_budget,
)

ARXIV_URL = "https://arxiv.org/abs/2601.12345"


class TestDefaults:
    def test_default_budget_is_500(self) -> None:
        assert DEFAULT_SNIPPET_BUDGET == 500

    def test_env_var_name(self) -> None:
        assert SNIPPET_BUDGET_ENV_VAR == "SCIX_LATEX_SNIPPET_BUDGET"


class TestReturnContract:
    def test_returns_snippet_payload(self) -> None:
        result = enforce_snippet_budget("hello", ARXIV_URL)
        assert isinstance(result, SnippetPayload)

    def test_payload_is_frozen(self) -> None:
        result = enforce_snippet_budget("hello", ARXIV_URL)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.snippet = "mutated"  # type: ignore[misc]

    def test_payload_fields(self) -> None:
        result = enforce_snippet_budget("hello", ARXIV_URL)
        assert result.snippet == "hello"
        assert result.canonical_url == ARXIV_URL
        assert result.truncated is False
        assert result.original_length == 5
        assert result.budget == DEFAULT_SNIPPET_BUDGET


class TestUnderBudget:
    def test_short_text_passes_through(self) -> None:
        body = "The Hubble constant is approximately 70 km/s/Mpc."
        result = enforce_snippet_budget(body, ARXIV_URL)
        assert result.snippet == body
        assert result.truncated is False
        assert result.original_length == len(body)

    def test_exactly_at_budget(self) -> None:
        body = "a" * DEFAULT_SNIPPET_BUDGET
        result = enforce_snippet_budget(body, ARXIV_URL)
        assert result.snippet == body
        assert result.truncated is False
        assert len(result.snippet) == DEFAULT_SNIPPET_BUDGET

    def test_empty_body_allowed(self) -> None:
        result = enforce_snippet_budget("", ARXIV_URL)
        assert result.snippet == ""
        assert result.truncated is False
        assert result.original_length == 0


class TestOverBudget:
    def test_over_budget_is_truncated(self) -> None:
        body = "x" * (DEFAULT_SNIPPET_BUDGET + 100)
        result = enforce_snippet_budget(body, ARXIV_URL)
        assert result.truncated is True
        assert result.original_length == DEFAULT_SNIPPET_BUDGET + 100

    def test_truncated_snippet_fits_within_budget(self) -> None:
        body = "x" * (DEFAULT_SNIPPET_BUDGET + 100)
        result = enforce_snippet_budget(body, ARXIV_URL)
        assert len(result.snippet) <= DEFAULT_SNIPPET_BUDGET

    def test_truncated_snippet_ends_with_ellipsis(self) -> None:
        body = "x" * (DEFAULT_SNIPPET_BUDGET + 100)
        result = enforce_snippet_budget(body, ARXIV_URL)
        assert result.snippet.endswith("...")

    def test_truncation_preserves_leading_text(self) -> None:
        body = "LEADING" + ("x" * DEFAULT_SNIPPET_BUDGET)
        result = enforce_snippet_budget(body, ARXIV_URL)
        assert result.snippet.startswith("LEADING")
        assert result.truncated is True


class TestCustomBudget:
    def test_explicit_budget_overrides_default(self) -> None:
        body = "a" * 200
        result = enforce_snippet_budget(body, ARXIV_URL, budget=50)
        assert result.budget == 50
        assert len(result.snippet) <= 50
        assert result.truncated is True

    def test_explicit_budget_larger_than_body(self) -> None:
        body = "short text"
        result = enforce_snippet_budget(body, ARXIV_URL, budget=1000)
        assert result.snippet == body
        assert result.truncated is False
        assert result.budget == 1000

    def test_zero_budget_yields_empty_snippet(self) -> None:
        result = enforce_snippet_budget("some text", ARXIV_URL, budget=0)
        assert result.snippet == ""
        assert result.truncated is True
        assert result.original_length == len("some text")
        assert result.budget == 0

    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            enforce_snippet_budget("any", ARXIV_URL, budget=-1)

    def test_budget_smaller_than_ellipsis_yields_truncated_prefix(self) -> None:
        # Budget of 2 can't fit "..." (3 chars). The contract is
        # "total length <= budget", so the snippet falls back to a plain
        # head-slice with no ellipsis marker.
        result = enforce_snippet_budget("abcdef", ARXIV_URL, budget=2)
        assert len(result.snippet) <= 2
        assert result.truncated is True


class TestCanonicalUrl:
    def test_canonical_url_required_non_empty(self) -> None:
        with pytest.raises(ValueError, match="canonical_url"):
            enforce_snippet_budget("body", "")

    def test_canonical_url_whitespace_rejected(self) -> None:
        with pytest.raises(ValueError, match="canonical_url"):
            enforce_snippet_budget("body", "   ")

    def test_canonical_url_echoed_verbatim(self) -> None:
        url = "https://arxiv.org/abs/2601.98765"
        result = enforce_snippet_budget("body", url)
        assert result.canonical_url == url


class TestEnvVarBudget:
    def test_env_var_sets_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SNIPPET_BUDGET_ENV_VAR, "50")
        body = "a" * 200
        result = enforce_snippet_budget(body, ARXIV_URL)
        assert result.budget == 50
        assert len(result.snippet) <= 50

    def test_env_var_resolved_lazily(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # First call with env var = 100
        monkeypatch.setenv(SNIPPET_BUDGET_ENV_VAR, "100")
        first = enforce_snippet_budget("x" * 300, ARXIV_URL)
        assert first.budget == 100

        # Change env var between calls — helper must pick up new value
        monkeypatch.setenv(SNIPPET_BUDGET_ENV_VAR, "200")
        second = enforce_snippet_budget("x" * 300, ARXIV_URL)
        assert second.budget == 200

    def test_explicit_budget_beats_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SNIPPET_BUDGET_ENV_VAR, "9999")
        result = enforce_snippet_budget("a" * 100, ARXIV_URL, budget=10)
        assert result.budget == 10
        assert len(result.snippet) <= 10
        assert result.truncated is True

    def test_missing_env_var_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SNIPPET_BUDGET_ENV_VAR, raising=False)
        result = enforce_snippet_budget("short", ARXIV_URL)
        assert result.budget == DEFAULT_SNIPPET_BUDGET

    def test_malformed_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SNIPPET_BUDGET_ENV_VAR, "not-a-number")
        with pytest.raises(ValueError, match=SNIPPET_BUDGET_ENV_VAR):
            enforce_snippet_budget("body", ARXIV_URL)

    def test_negative_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SNIPPET_BUDGET_ENV_VAR, "-5")
        with pytest.raises(ValueError, match="budget"):
            enforce_snippet_budget("body", ARXIV_URL)

    def test_empty_env_var_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SNIPPET_BUDGET_ENV_VAR, "")
        result = enforce_snippet_budget("short", ARXIV_URL)
        assert result.budget == DEFAULT_SNIPPET_BUDGET
