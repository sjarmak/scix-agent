"""Tests for :mod:`scix.research_scope` (PRD MH-5)."""

from __future__ import annotations

import pytest

from scix.research_scope import (
    ResearchScope,
    scope_from_dict,
    scope_to_dict,
    scope_to_sql_clauses,
)


# -- Acceptance criteria 4(a): empty scope ------------------------------------


def test_empty_scope_produces_empty_clause_and_params() -> None:
    scope = ResearchScope()
    clause, params = scope_to_sql_clauses(
        scope, {"papers": "p", "paper_metrics": "pm"}
    )
    assert clause == ""
    assert params == []


# -- Acceptance criteria 4(b): year_window ------------------------------------


def test_year_window_clause_uses_inclusive_bounds() -> None:
    scope = ResearchScope(year_window=(2018, 2024))
    clause, params = scope_to_sql_clauses(scope, {"papers": "p"})
    assert clause == "p.year >= %s AND p.year <= %s"
    assert params == [2018, 2024]


def test_year_window_uses_caller_supplied_alias() -> None:
    scope = ResearchScope(year_window=(2018, 2024))
    clause, params = scope_to_sql_clauses(scope, {"papers": "papers_v"})
    assert clause == "papers_v.year >= %s AND papers_v.year <= %s"
    assert params == [2018, 2024]


def test_year_window_swapped_bounds_raises() -> None:
    with pytest.raises(ValueError, match="year_window start"):
        ResearchScope(year_window=(2024, 2018))


# -- Acceptance criteria 4(c): community_ids -> ANY(%s) -----------------------


def test_community_ids_clause_uses_any_array() -> None:
    scope = ResearchScope(community_ids=[1, 2, 3])
    clause, params = scope_to_sql_clauses(scope, {"paper_metrics": "pm"})
    # default leiden_resolution is 'medium' when community_ids set
    assert clause == "pm.community_id_medium = ANY(%s)"
    assert params == [[1, 2, 3]]


def test_community_ids_respects_explicit_resolution() -> None:
    scope = ResearchScope(community_ids=[7], leiden_resolution="fine")
    clause, params = scope_to_sql_clauses(scope, {"paper_metrics": "pm"})
    assert clause == "pm.community_id_fine = ANY(%s)"
    assert params == [[7]]


# -- Acceptance criteria 4(d): leiden_resolution round-trip -------------------


def test_leiden_resolution_round_trip_through_dict() -> None:
    scope = ResearchScope(leiden_resolution="medium")
    payload = scope_to_dict(scope)
    assert payload["leiden_resolution"] == "medium"
    rebuilt = scope_from_dict(payload)
    assert rebuilt == scope
    assert rebuilt.leiden_resolution == "medium"


def test_full_round_trip_preserves_all_fields() -> None:
    scope = ResearchScope(
        community_ids=[1, 2],
        year_window=(2018, 2024),
        methodology_class="observational",
        instruments=["JWST", "ALMA"],
        exclude_authors=["J. Doe"],
        exclude_funders=["NSF"],
        min_venue_tier=2,
        leiden_resolution="coarse",
    )
    payload = scope_to_dict(scope)
    # year_window must be JSON-friendly (list, not tuple)
    assert payload["year_window"] == [2018, 2024]
    rebuilt = scope_from_dict(payload)
    assert rebuilt == scope


# -- Acceptance criteria 4(e): unknown leiden_resolution raises ---------------


def test_unknown_leiden_resolution_raises_value_error() -> None:
    with pytest.raises(ValueError, match="leiden_resolution"):
        ResearchScope(leiden_resolution="ultra-fine")  # type: ignore[arg-type]


def test_unknown_leiden_resolution_via_from_dict_raises() -> None:
    with pytest.raises(ValueError, match="leiden_resolution"):
        scope_from_dict({"leiden_resolution": "bogus"})


def test_unknown_top_level_key_via_from_dict_raises() -> None:
    with pytest.raises(ValueError, match="Unknown ResearchScope keys"):
        scope_from_dict({"not_a_field": 42})


# -- Acceptance criteria 4(f): exclude_authors / exclude_funders subqueries ---


def test_exclude_authors_uses_not_exists_subquery() -> None:
    scope = ResearchScope(exclude_authors=["J. Doe", "K. Roe"])
    clause, params = scope_to_sql_clauses(scope, {"papers": "p"})
    # Documented choice: NOT EXISTS against papers_authors join table.
    assert "NOT EXISTS" in clause
    assert "papers_authors" in clause
    assert "p.id" in clause
    assert "= ANY(%s)" in clause
    assert params == [["J. Doe", "K. Roe"]]


def test_exclude_funders_uses_not_exists_subquery() -> None:
    scope = ResearchScope(exclude_funders=["NSF", "NASA"])
    clause, params = scope_to_sql_clauses(scope, {"papers": "p"})
    assert "NOT EXISTS" in clause
    assert "papers_funders" in clause
    assert "p.id" in clause
    assert "= ANY(%s)" in clause
    assert params == [["NSF", "NASA"]]


# -- Additional coverage of remaining fields ----------------------------------


def test_methodology_class_clause() -> None:
    scope = ResearchScope(methodology_class="observational")
    clause, params = scope_to_sql_clauses(scope, {"papers": "p"})
    assert clause == "p.methodology_class = %s"
    assert params == ["observational"]


def test_instruments_clause_uses_array_overlap() -> None:
    scope = ResearchScope(instruments=["JWST", "ALMA"])
    clause, params = scope_to_sql_clauses(scope, {"papers": "p"})
    assert clause == "p.instruments && %s::text[]"
    assert params == [["JWST", "ALMA"]]


def test_min_venue_tier_clause_filters_by_threshold() -> None:
    scope = ResearchScope(min_venue_tier=2)
    clause, params = scope_to_sql_clauses(scope, {"papers": "p"})
    assert clause == "p.venue_tier <= %s"
    assert params == [2]


def test_combined_clauses_joined_with_and_in_declaration_order() -> None:
    scope = ResearchScope(
        year_window=(2018, 2024),
        community_ids=[1, 2],
        leiden_resolution="fine",
    )
    clause, params = scope_to_sql_clauses(
        scope, {"papers": "p", "paper_metrics": "pm"}
    )
    assert (
        clause
        == "p.year >= %s AND p.year <= %s AND pm.community_id_fine = ANY(%s)"
    )
    assert params == [2018, 2024, [1, 2]]


def test_default_aliases_used_when_caller_passes_empty_dict() -> None:
    scope = ResearchScope(year_window=(2020, 2021))
    # Defaults: papers -> 'p'
    clause, params = scope_to_sql_clauses(scope, {})
    assert clause == "p.year >= %s AND p.year <= %s"
    assert params == [2020, 2021]


def test_scope_from_dict_normalises_year_window_list_to_tuple() -> None:
    scope = scope_from_dict({"year_window": [2010, 2020]})
    assert scope.year_window == (2010, 2020)
    # tuple, not list — the dataclass field is typed tuple[int, int]
    assert isinstance(scope.year_window, tuple)


def test_scope_from_dict_rejects_non_dict() -> None:
    with pytest.raises(TypeError):
        scope_from_dict([("year_window", [2010, 2020])])  # type: ignore[arg-type]


def test_scope_to_dict_preserves_none_fields() -> None:
    scope = ResearchScope(year_window=(2018, 2024))
    payload = scope_to_dict(scope)
    # Stable wire shape: every field is present, even when None
    assert "community_ids" in payload
    assert payload["community_ids"] is None
    assert payload["leiden_resolution"] is None


# -- Bonus: PRD-quoted acceptance ("scope={year_window:[2018,2024]} filters") -


def test_year_window_acceptance_excludes_2017_via_clause_shape() -> None:
    """PRD MH-5 acceptance: scope={year_window:[2018,2024]} filters out
    2017 papers. We assert the clause shape that produces this filtering;
    a DB-level integration test is the responsibility of the threading
    work unit, not this type-definition unit.
    """
    scope = scope_from_dict({"year_window": [2018, 2024]})
    clause, params = scope_to_sql_clauses(scope, {"papers": "p"})
    # 2017 < 2018, so the clause `year >= 2018` rules it out.
    assert ">= %s" in clause
    assert params[0] == 2018
