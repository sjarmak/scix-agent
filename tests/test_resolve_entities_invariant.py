"""Hypothesis property test for M13 cross-lane set equality.

Spec (acceptance criterion 5): for a fixed ``(bibcode, candidate_set_hash,
model_version)`` triple, the four resolution lanes must return
:class:`EntityLinkSet` objects with identical entity-id sets. Confidence
values may drift by ≤ 0.01.

We force each lane via ``context.mode`` so the routing in
:func:`scix.resolve_entities.resolve_entities` picks exactly one backend per
call, and we seed every lane with the same deterministic id-set derived
from the randomized ``candidate_set``.
"""

from __future__ import annotations

import string
from dataclasses import replace

import hypothesis.strategies as st
from hypothesis import given, settings

from scix import resolve_entities as re_mod
from scix.resolve_entities import (
    EntityResolveContext,
    candidate_set_hash,
    resolve_entities,
)

_BIBCODE_ALPHABET = string.ascii_letters + string.digits + "."


def _seed_ids_for(bibcode: str, candidate_ids: frozenset[int]) -> frozenset[int]:
    """Deterministic id set derived from the inputs.

    We mix in ``candidate_ids`` so different draws produce different seed
    sets, and we use ``zlib.crc32`` so the function is stable across process
    restarts (Python's builtin ``hash`` is salted).
    """
    import zlib

    if not candidate_ids:
        # Must be non-empty — the invariant is trivial on empty sets but we
        # want real coverage.
        return frozenset({1})
    raw = bibcode.encode() + b"|" + ",".join(str(i) for i in sorted(candidate_ids)).encode()
    base = zlib.crc32(raw)
    return frozenset({(base + i) & 0xFFFF for i in sorted(candidate_ids)})


@given(
    bibcode=st.text(alphabet=_BIBCODE_ALPHABET, min_size=5, max_size=19),
    candidate_ids=st.frozensets(st.integers(min_value=1, max_value=10_000), min_size=0, max_size=8),
    model_version=st.sampled_from(["v1", "v2", "v3"]),
)
@settings(max_examples=120, deadline=None)
def test_lanes_return_equal_entity_id_sets(bibcode, candidate_ids, model_version):
    re_mod._reset_mocks()

    # Pin lane latencies to zero so Hypothesis runs fast.
    for k in re_mod._LANE_LATENCIES:
        re_mod._LANE_LATENCIES[k] = 0.0

    ctx_base = EntityResolveContext(
        candidate_set=candidate_ids,
        mode="static",
        model_version=model_version,
    )
    cset_hash = candidate_set_hash(ctx_base)
    seed_ids = _seed_ids_for(bibcode, candidate_ids)

    re_mod._seed_static(bibcode, seed_ids)
    re_mod._seed_jit_cache(bibcode, cset_hash, model_version, seed_ids)
    re_mod._seed_live_jit(bibcode, cset_hash, seed_ids)
    re_mod._seed_local_ner(bibcode, seed_ids)

    results = {
        "static": resolve_entities(bibcode, replace(ctx_base, mode="static")),
        "jit": resolve_entities(bibcode, replace(ctx_base, mode="jit")),
        "live_jit": resolve_entities(bibcode, replace(ctx_base, mode="live_jit")),
        "local_ner": resolve_entities(bibcode, replace(ctx_base, mode="local_ner")),
    }

    # All four lanes must agree on entity-id set.
    id_sets = {name: r.entity_ids() for name, r in results.items()}
    assert all(s == seed_ids for s in id_sets.values()), id_sets

    # Confidence drift across lanes for any single entity ≤ 0.01.
    for eid in seed_ids:
        confs = [r.confidence_for(eid) for r in results.values()]
        drift = max(confs) - min(confs)
        assert drift <= 0.01 + 1e-9, f"confidence drift {drift} > 0.01 for eid={eid}"

    re_mod._reset_mocks()
