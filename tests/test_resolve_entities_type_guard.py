"""Type-guard tests for EntityLinkSet.

These tests confirm that :class:`scix.entity_link_set.EntityLinkSet` cannot
be constructed from outside :mod:`scix.resolve_entities`. They import the
class through :mod:`scix.entity_link_set` (the same module a downstream
caller would import from) and verify:

1. Omitting the sentinel raises ``TypeError``.
2. Passing a non-sentinel value for the token raises ``TypeError``.
3. A call from :mod:`scix.resolve_entities` that *does* have the sentinel
   succeeds — demonstrating the gate is crossable only by the authorized
   entry point.
"""

from __future__ import annotations

import pytest

from scix.entity_link_set import EntityLink, EntityLinkSet
from scix.resolve_entities import EntityResolveContext, resolve_entities
from scix import resolve_entities as re_mod


def test_entity_link_set_rejects_direct_construction_missing_token():
    # No positional sentinel — Python raises TypeError for the missing
    # required positional argument. Spec requires ``TypeError``; the message
    # the stdlib produces satisfies the acceptance criterion.
    with pytest.raises(TypeError):
        EntityLinkSet(  # type: ignore[call-arg]
            bibcode="x",
            entities=frozenset(),
            lane="static",
            model_version="v1",
        )


def test_entity_link_set_rejects_fake_sentinel():
    with pytest.raises(TypeError):
        EntityLinkSet(
            object(),  # type: ignore[arg-type]
            bibcode="x",
            entities=frozenset(),
            lane="static",
            model_version="v1",
        )


def test_entity_link_set_rejects_none_sentinel():
    with pytest.raises(TypeError):
        EntityLinkSet(
            None,  # type: ignore[arg-type]
            bibcode="x",
            entities=frozenset(),
            lane="static",
            model_version="v1",
        )


def test_entity_link_set_rejects_non_frozenset_entities():
    from scix._resolver_token import _RESOLVER_INTERNAL

    with pytest.raises(TypeError):
        EntityLinkSet(
            _RESOLVER_INTERNAL,
            bibcode="x",
            entities=set(),  # type: ignore[arg-type]
            lane="static",
            model_version="v1",
        )


def test_resolve_entities_can_construct_link_set():
    re_mod._reset_mocks()
    re_mod._seed_static("2024A", frozenset({42}))
    result = resolve_entities(
        "2024A",
        EntityResolveContext(candidate_set=frozenset({42}), mode="static"),
    )
    assert isinstance(result, EntityLinkSet)
    assert result.entity_ids() == frozenset({42})
    assert all(isinstance(link, EntityLink) for link in result.entities)
    re_mod._reset_mocks()
