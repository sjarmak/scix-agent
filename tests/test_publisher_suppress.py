"""Tests for src/scix/publisher_suppress.py (PRD R16).

Covers the runtime-loadable contract: missing file tolerance, case-insensitive
matching, None publisher handling, frozenset return type, and — critically —
that edits to the config file are picked up on the next call (no stale
module-level cache).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scix.publisher_suppress import (
    DEFAULT_CONFIG_PATH,
    is_suppressed,
    load_suppress_list,
)


def _write_yaml(path: Path, entries: list[str]) -> None:
    """Write a minimal suppressed_publishers YAML file."""
    if entries:
        body = "suppressed_publishers:\n" + "\n".join(f"  - {e!r}" for e in entries) + "\n"
    else:
        body = "suppressed_publishers: []\n"
    path.write_text(body, encoding="utf-8")


def test_returns_frozenset_of_lowercase_strings(tmp_path: Path) -> None:
    cfg = tmp_path / "suppress.yaml"
    _write_yaml(cfg, ["Predatory Press", "BOGUS Journal Ltd"])

    result = load_suppress_list(cfg)

    assert isinstance(result, frozenset)
    assert result == frozenset({"predatory press", "bogus journal ltd"})


def test_missing_file_returns_empty_frozenset_no_exception(tmp_path: Path) -> None:
    nonexistent = tmp_path / "does_not_exist.yaml"

    result = load_suppress_list(nonexistent)

    assert result == frozenset()
    assert isinstance(result, frozenset)


def test_empty_list_returns_empty_frozenset(tmp_path: Path) -> None:
    cfg = tmp_path / "suppress.yaml"
    _write_yaml(cfg, [])

    assert load_suppress_list(cfg) == frozenset()


def test_is_suppressed_case_insensitive(tmp_path: Path) -> None:
    cfg = tmp_path / "suppress.yaml"
    _write_yaml(cfg, ["Bad Publisher"])
    suppress_set = load_suppress_list(cfg)

    assert is_suppressed("Bad Publisher", suppress_set) is True
    assert is_suppressed("bad publisher", suppress_set) is True
    assert is_suppressed("BAD PUBLISHER", suppress_set) is True
    assert is_suppressed("  Bad Publisher  ", suppress_set) is True
    assert is_suppressed("Good Publisher", suppress_set) is False


def test_is_suppressed_none_publisher_returns_false() -> None:
    suppress_set = frozenset({"anything"})

    assert is_suppressed(None, suppress_set) is False
    # And with an empty set, still False.
    assert is_suppressed(None, frozenset()) is False


def test_reload_picks_up_edits_no_module_cache(tmp_path: Path) -> None:
    """PRD R16: runtime-loadable — edits take effect on next call, no cache."""
    cfg = tmp_path / "suppress.yaml"

    # First write: one publisher.
    _write_yaml(cfg, ["Initial Publisher"])
    first = load_suppress_list(cfg)
    assert first == frozenset({"initial publisher"})
    assert is_suppressed("Initial Publisher", first) is True
    assert is_suppressed("Added Later", first) is False

    # Edit: add a new publisher, remove the old one.
    _write_yaml(cfg, ["Added Later"])
    second = load_suppress_list(cfg)
    assert second == frozenset({"added later"})
    assert is_suppressed("Added Later", second) is True
    assert is_suppressed("Initial Publisher", second) is False

    # Edit again: clear the list entirely.
    _write_yaml(cfg, [])
    third = load_suppress_list(cfg)
    assert third == frozenset()


def test_malformed_yaml_raises_value_error(tmp_path: Path) -> None:
    cfg = tmp_path / "broken.yaml"
    # Unterminated flow sequence → YAMLError.
    cfg.write_text("suppressed_publishers: [unterminated, \n  - bogus\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed YAML"):
        load_suppress_list(cfg)


def test_wrong_top_level_type_raises_value_error(tmp_path: Path) -> None:
    cfg = tmp_path / "wrong.yaml"
    cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level mapping"):
        load_suppress_list(cfg)


def test_suppressed_publishers_not_a_list_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("suppressed_publishers: 'not a list'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a list"):
        load_suppress_list(cfg)


def test_default_path_points_at_repo_config() -> None:
    """Sanity check: default path resolves to <repo-root>/config/publisher_suppress_list.yaml."""
    assert DEFAULT_CONFIG_PATH.name == "publisher_suppress_list.yaml"
    assert DEFAULT_CONFIG_PATH.parent.name == "config"


def test_default_path_loads_when_none_passed() -> None:
    """load_suppress_list(None) (default) reads the repo config without error."""
    result = load_suppress_list()  # path=None → default
    assert isinstance(result, frozenset)
    # The shipped config has an empty list, so result should be empty. We only
    # assert type + that it didn't raise; the file content may evolve.
    assert all(isinstance(e, str) for e in result)
