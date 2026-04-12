"""Unit tests for scix.resolve_entities — one test per internal lane."""

from __future__ import annotations

from dataclasses import replace

import pytest

from scix import resolve_entities as re_mod
from scix.resolve_entities import (
    EntityResolveContext,
    ResolutionFailed,
    candidate_set_hash,
    resolve_entities,
)


@pytest.fixture(autouse=True)
def reset_mocks():
    re_mod._reset_mocks()
    yield
    re_mod._reset_mocks()


def _ctx(mode: str, ids: frozenset[int] = frozenset({1, 2, 3})) -> EntityResolveContext:
    return EntityResolveContext(candidate_set=ids, mode=mode, model_version="v1")


# ---------------------------------------------------------------------------
# Lane 1: static
# ---------------------------------------------------------------------------


def test_static_lane_returns_seeded_ids():
    re_mod._seed_static("2024ApJ...1A", frozenset({10, 20}))
    result = resolve_entities("2024ApJ...1A", _ctx("static"))
    assert result.lane == "static"
    assert result.entity_ids() == frozenset({10, 20})
    assert result.model_version == "v1"


def test_static_lane_miss_raises():
    with pytest.raises(ResolutionFailed):
        resolve_entities("missing", _ctx("static"))


# ---------------------------------------------------------------------------
# Lane 2: jit_cache_hit
# ---------------------------------------------------------------------------


def test_jit_cache_lane_returns_seeded_ids():
    ctx = _ctx("jit", frozenset({5, 6}))
    cset_hash = candidate_set_hash(ctx)
    re_mod._seed_jit_cache("2024ApJ...2B", cset_hash, "v1", frozenset({42, 43}))
    result = resolve_entities("2024ApJ...2B", ctx)
    assert result.lane == "jit_cache_hit"
    assert result.entity_ids() == frozenset({42, 43})
    assert result.candidate_set_hash == cset_hash


def test_jit_cache_lane_miss_on_wrong_model_version():
    ctx = _ctx("jit", frozenset({5}))
    cset_hash = candidate_set_hash(ctx)
    re_mod._seed_jit_cache("2024X", cset_hash, "v2", frozenset({99}))
    # ctx.model_version is v1 but cache is seeded under v2
    with pytest.raises(ResolutionFailed):
        resolve_entities("2024X", ctx)


# ---------------------------------------------------------------------------
# Lane 3: live_jit
# ---------------------------------------------------------------------------


def test_live_jit_lane_returns_seeded_ids():
    ctx = _ctx("live_jit", frozenset({7, 8}))
    cset_hash = candidate_set_hash(ctx)
    re_mod._seed_live_jit("2024ApJ...3C", cset_hash, frozenset({100, 200}))
    result = resolve_entities("2024ApJ...3C", ctx)
    assert result.lane == "live_jit"
    assert result.entity_ids() == frozenset({100, 200})


# ---------------------------------------------------------------------------
# Lane 4: local_ner
# ---------------------------------------------------------------------------


def test_local_ner_lane_returns_seeded_ids():
    re_mod._seed_local_ner("2024ApJ...4D", frozenset({300, 301, 302}))
    result = resolve_entities("2024ApJ...4D", _ctx("local_ner"))
    assert result.lane == "local_ner"
    assert result.entity_ids() == frozenset({300, 301, 302})


# ---------------------------------------------------------------------------
# Auto mode prefers static, then falls back through the chain.
# ---------------------------------------------------------------------------


def test_auto_mode_prefers_static():
    re_mod._seed_static("2024Z", frozenset({1}))
    re_mod._seed_local_ner("2024Z", frozenset({9999}))
    result = resolve_entities("2024Z", _ctx("auto"))
    assert result.lane == "static"
    assert result.entity_ids() == frozenset({1})


def test_auto_mode_falls_back_to_local_ner():
    re_mod._seed_local_ner("2024Y", frozenset({77}))
    result = resolve_entities("2024Y", _ctx("auto"))
    assert result.lane == "local_ner"
    assert result.entity_ids() == frozenset({77})


def test_auto_mode_all_miss_raises():
    with pytest.raises(ResolutionFailed):
        resolve_entities("nobody_home", _ctx("auto"))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_bibcode_rejected():
    with pytest.raises(ValueError):
        resolve_entities("", _ctx("auto"))


def test_non_context_rejected():
    with pytest.raises(TypeError):
        resolve_entities("2024A", {"candidate_set": frozenset(), "mode": "static"})  # type: ignore[arg-type]


def test_candidate_set_hash_is_deterministic_within_process():
    ctx_a = EntityResolveContext(candidate_set=frozenset({1, 2, 3}), mode="jit")
    ctx_b = EntityResolveContext(candidate_set=frozenset({3, 2, 1}), mode="jit")
    assert candidate_set_hash(ctx_a) == candidate_set_hash(ctx_b)
    # differ only in model_version
    ctx_c = replace(ctx_a, model_version="v2")
    assert candidate_set_hash(ctx_a) != candidate_set_hash(ctx_c)
