"""Tests for entity link policy classification.

Unit tests for ``src/scix/link_policy.py`` plus integration tests that
exercise ``scripts/set_link_policy.py`` against scix_test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg
import pytest

from scix.link_policy import determine_link_policy
from tests.helpers import get_test_dsn

# Make scripts/ importable (needed for ``import set_link_policy`` in
# integration tests below).
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Unit tests — pure determine_link_policy()
# ---------------------------------------------------------------------------


class TestDetermineLinkPolicy:
    """Rule coverage for determine_link_policy()."""

    def test_rule1_banned_ambiguity_class_returns_banned(self) -> None:
        result = determine_link_policy(
            source="physh",
            canonical_name="Stress",
            ambiguity_class="banned",
            properties={},
        )
        assert result == "banned"

    def test_rule1_banned_takes_precedence_over_gcmd_provider(self) -> None:
        """ambiguity_class='banned' wins even for GCMD providers."""
        result = determine_link_policy(
            source="gcmd",
            canonical_name="DOC/NOAA/NESDIS",
            ambiguity_class="banned",
            properties={"gcmd_scheme": "providers"},
        )
        assert result == "banned"

    def test_rule1_banned_takes_precedence_over_spase(self) -> None:
        result = determine_link_policy(
            source="spase",
            canonical_name="Frequency",
            ambiguity_class="banned",
            properties={},
        )
        assert result == "banned"

    def test_rule2_gcmd_providers_scheme_returns_banned(self) -> None:
        result = determine_link_policy(
            source="gcmd",
            canonical_name="DOC/NOAA/NESDIS/STAR",
            ambiguity_class="unique",
            properties={"gcmd_scheme": "providers"},
        )
        assert result == "banned"

    def test_rule2_gcmd_providers_with_none_ambiguity(self) -> None:
        result = determine_link_policy(
            source="gcmd",
            canonical_name="CA/NFLD/FRA/SOIL",
            ambiguity_class=None,
            properties={"gcmd_scheme": "providers"},
        )
        assert result == "banned"

    def test_rule2_does_not_ban_gcmd_instruments(self) -> None:
        result = determine_link_policy(
            source="gcmd",
            canonical_name="MODIS",
            ambiguity_class="unique",
            properties={"gcmd_scheme": "instruments"},
        )
        assert result == "open"

    def test_rule3_gcmd_hierarchy_returns_context_required(self) -> None:
        result = determine_link_policy(
            source="gcmd",
            canonical_name="SEA ICE > SALINITY",
            ambiguity_class="unique",
            properties={"gcmd_scheme": "sciencekeywords"},
        )
        assert result == "context_required"

    def test_rule3_gcmd_hierarchy_with_deep_nesting(self) -> None:
        result = determine_link_policy(
            source="gcmd",
            canonical_name="EROSION/SEDIMENTATION > DEGRADATION",
            ambiguity_class="domain_safe",
            properties={"gcmd_scheme": "sciencekeywords"},
        )
        assert result == "context_required"

    def test_rule3_gcmd_hierarchy_takes_precedence_over_open(self) -> None:
        """Even domain_safe GCMD hierarchicals need context."""
        result = determine_link_policy(
            source="gcmd",
            canonical_name="CRYOSPHERIC INDICATORS > SALINITY",
            ambiguity_class="domain_safe",
            properties={"gcmd_scheme": "sciencekeywords"},
        )
        assert result == "context_required"

    def test_rule4_spase_returns_context_required(self) -> None:
        result = determine_link_policy(
            source="spase",
            canonical_name="ModeAmplitude",
            ambiguity_class="unique",
            properties={},
        )
        assert result == "context_required"

    def test_rule4_spase_domain_safe_still_context_required(self) -> None:
        result = determine_link_policy(
            source="spase",
            canonical_name="MagneticField",
            ambiguity_class="domain_safe",
            properties={},
        )
        assert result == "context_required"

    def test_rule5_default_open_for_normal_entity(self) -> None:
        result = determine_link_policy(
            source="physh",
            canonical_name="X-ray photoelectron spectroscopy",
            ambiguity_class="domain_safe",
            properties={},
        )
        assert result == "open"

    def test_rule5_open_for_none_ambiguity_class(self) -> None:
        result = determine_link_policy(
            source="ascl",
            canonical_name="Astropy",
            ambiguity_class=None,
            properties={},
        )
        assert result == "open"

    def test_rule5_open_for_homograph_non_gcmd_non_spase(self) -> None:
        """Homographs from non-GCMD/SPASE sources are open for tier-1."""
        result = determine_link_policy(
            source="physh",
            canonical_name="Cavity resonators",
            ambiguity_class="homograph",
            properties={},
        )
        assert result == "open"

    def test_empty_properties_does_not_crash(self) -> None:
        result = determine_link_policy(
            source="gcmd",
            canonical_name="MODIS",
            ambiguity_class="unique",
            properties={},
        )
        assert result == "open"


# ---------------------------------------------------------------------------
# Integration tests — set_link_policy.py against scix_test
# ---------------------------------------------------------------------------

# Seed entities covering each rule branch.
_SEED_ENTITIES: list[dict] = [
    # Rule 1: banned ambiguity_class
    {
        "canonical_name": "test_lp_stress",
        "entity_type": "method",
        "source": "physh",
        "ambiguity_class": "banned",
        "properties": {},
        "expected_policy": "banned",
    },
    # Rule 2: GCMD providers
    {
        "canonical_name": "test_lp_DOC/NOAA/NESDIS",
        "entity_type": "mission",
        "source": "gcmd",
        "ambiguity_class": "unique",
        "properties": {"gcmd_scheme": "providers"},
        "expected_policy": "banned",
    },
    # Rule 3: GCMD hierarchical
    {
        "canonical_name": "test_lp_SEA ICE > SALINITY",
        "entity_type": "observable",
        "source": "gcmd",
        "ambiguity_class": "domain_safe",
        "properties": {"gcmd_scheme": "sciencekeywords"},
        "expected_policy": "context_required",
    },
    # Rule 4: SPASE
    {
        "canonical_name": "test_lp_ModeAmplitude",
        "entity_type": "observable",
        "source": "spase",
        "ambiguity_class": "unique",
        "properties": {},
        "expected_policy": "context_required",
    },
    # Rule 5: normal entity -> open
    {
        "canonical_name": "test_lp_GALFA-HI",
        "entity_type": "survey",
        "source": "ascl",
        "ambiguity_class": "domain_safe",
        "properties": {},
        "expected_policy": "open",
    },
]


@pytest.fixture()
def integration_conn():
    dsn = get_test_dsn()
    if dsn is None:
        pytest.skip("SCIX_TEST_DSN not set or points at production")
    conn = psycopg.connect(dsn)
    conn.autocommit = False
    yield conn
    # Clean up committed seed rows (rollback only undoes uncommitted work).
    seed_names = [ent["canonical_name"] for ent in _SEED_ENTITIES]
    try:
        conn.rollback()  # clear any in-error transaction state
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM entities WHERE canonical_name = ANY(%s)",
                (seed_names,),
            )
        conn.commit()
    finally:
        conn.close()


def _seed_link_policy_entities(conn: psycopg.Connection) -> dict[str, int]:
    """Insert seed entities; return canonical_name -> id.

    Deletes any leftover rows from prior runs first (the test commits,
    so rollback-only teardown does not remove them).
    """
    seed_names = [ent["canonical_name"] for ent in _SEED_ENTITIES]
    ids: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM entities WHERE canonical_name = ANY(%s)",
            (seed_names,),
        )
        for ent in _SEED_ENTITIES:
            cur.execute(
                "INSERT INTO entities "
                "(canonical_name, entity_type, source, ambiguity_class, properties) "
                "VALUES (%s, %s, %s, %s::entity_ambiguity_class, %s::jsonb) "
                "RETURNING id",
                (
                    ent["canonical_name"],
                    ent["entity_type"],
                    ent["source"],
                    ent["ambiguity_class"],
                    json.dumps(ent["properties"]),
                ),
            )
            row = cur.fetchone()
            assert row is not None
            ids[ent["canonical_name"]] = row[0]
    return ids


@pytest.mark.integration
def test_set_link_policy_populates_correct_values(
    integration_conn: psycopg.Connection,
) -> None:
    import set_link_policy

    ids = _seed_link_policy_entities(integration_conn)
    integration_conn.commit()

    set_link_policy.set_all_link_policies(integration_conn, batch_size=100)

    with integration_conn.cursor() as cur:
        for ent in _SEED_ENTITIES:
            entity_id = ids[ent["canonical_name"]]
            cur.execute(
                "SELECT link_policy::text FROM entities WHERE id = %s",
                (entity_id,),
            )
            row = cur.fetchone()
            assert row is not None, f"entity {ent['canonical_name']} not found"
            assert row[0] == ent["expected_policy"], (
                f"{ent['canonical_name']} got link_policy={row[0]!r}, "
                f"expected {ent['expected_policy']!r}"
            )
