"""Tests for the NER mention denylist (bead scix_experiments-eq95).

Covers:
- The bead's MUST-include entries are denylisted.
- Match is case-insensitive on canonical_name and exact on entity_type.
- Disjoint (name, type) pairs are independent — a hypothetical
  ``protein`` ``method`` entry is NOT denylisted just because
  ``protein`` ``gene`` is.
- ``filter_denylisted_rows`` doesn't mutate the input list.
"""
from __future__ import annotations

from scix.extract.ner_denylist import (
    _DENYLIST,
    filter_denylisted_rows,
    is_denylisted,
)


class TestRequiredEntries:
    """The bead's acceptance criteria pin three must-include entries."""

    def test_data_dataset_denylisted(self) -> None:
        assert is_denylisted("data", "dataset") is True

    def test_experimental_data_dataset_denylisted(self) -> None:
        assert is_denylisted("experimental data", "dataset") is True

    def test_method_method_denylisted(self) -> None:
        assert is_denylisted("method", "method") is True


class TestMatchSemantics:
    def test_case_insensitive_on_canonical_name(self) -> None:
        assert is_denylisted("Data", "dataset") is True
        assert is_denylisted("DATA", "dataset") is True
        assert is_denylisted("Experimental Data", "dataset") is True

    def test_exact_on_entity_type(self) -> None:
        # ``protein`` is denylisted as a ``gene`` but not as anything else.
        # An entry under another type would need its own denylist row.
        assert is_denylisted("protein", "gene") is True
        # 'protein' under 'chemical' is NOT in the denylist seed —
        # keep this assertion as a guard against accidental over-inclusion.
        assert is_denylisted("protein", "chemical") is False

    def test_strips_surrounding_whitespace(self) -> None:
        assert is_denylisted("  data  ", "dataset") is True

    def test_empty_or_none_inputs_return_false(self) -> None:
        assert is_denylisted(None, "dataset") is False
        assert is_denylisted("data", None) is False
        assert is_denylisted("", "dataset") is False
        assert is_denylisted("data", "") is False

    def test_off_denylist_pair_returns_false(self) -> None:
        assert is_denylisted("JWST", "instrument") is False
        assert is_denylisted("Hubble Space Telescope", "instrument") is False


class TestFilterDenylistedRows:
    def test_drops_denylisted_rows_only(self) -> None:
        rows = [
            {"canonical_name": "JWST", "entity_type": "instrument"},
            {"canonical_name": "data", "entity_type": "dataset"},
            {"canonical_name": "TRAPPIST-1", "entity_type": "celestial_object"},
            {"canonical_name": "method", "entity_type": "method"},
        ]
        out = filter_denylisted_rows(rows)
        assert len(out) == 2
        names = {r["canonical_name"] for r in out}
        assert names == {"JWST", "TRAPPIST-1"}

    def test_does_not_mutate_input(self) -> None:
        rows = [{"canonical_name": "data", "entity_type": "dataset"}]
        before_len = len(rows)
        _ = filter_denylisted_rows(rows)
        assert len(rows) == before_len

    def test_supports_alternate_key_names(self) -> None:
        rows = [
            {"name": "JWST", "type": "instrument"},
            {"name": "data", "type": "dataset"},
        ]
        out = filter_denylisted_rows(rows, name_key="name", type_key="type")
        assert len(out) == 1
        assert out[0]["name"] == "JWST"

    def test_handles_rows_missing_keys(self) -> None:
        # Rows without the keyed fields should pass through unfiltered —
        # absence of evidence isn't evidence of denylist match.
        rows = [
            {"canonical_name": "JWST"},  # no entity_type
            {"entity_type": "dataset"},  # no canonical_name
            {},
        ]
        out = filter_denylisted_rows(rows)
        assert len(out) == 3


class TestDenylistShape:
    def test_all_entries_are_lowercase_canonicals(self) -> None:
        # The is_denylisted lookup casefolds incoming canonical_name; entries
        # in the set must already be normalized so the lookup actually hits.
        for canonical, _entity_type in _DENYLIST:
            assert canonical == canonical.casefold(), (
                f"denylist canonical {canonical!r} is not casefolded"
            )

    def test_no_blank_or_none_entries(self) -> None:
        for canonical, entity_type in _DENYLIST:
            assert canonical, "blank canonical_name in denylist"
            assert entity_type, "blank entity_type in denylist"
