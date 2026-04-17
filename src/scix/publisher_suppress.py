"""Publisher suppress list loader (PRD R16).

Runtime-loadable suppression list. Every call to :func:`load_suppress_list`
reads the YAML file fresh from disk — there is no module-level cache, so
edits to the config file take effect on the next call without a redeploy
or process restart.

Public API:
    - load_suppress_list(path) -> frozenset[str]
    - is_suppressed(publisher, suppress_set) -> bool
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

__all__ = ["load_suppress_list", "is_suppressed", "DEFAULT_CONFIG_PATH"]


def _default_config_path() -> Path:
    """Resolve the default config path: <repo-root>/config/publisher_suppress_list.yaml.

    This module lives at <repo-root>/src/scix/publisher_suppress.py, so the
    repo root is two ``parent`` hops above ``src/``.
    """
    # .../src/scix/publisher_suppress.py -> .../src/scix -> .../src -> .../<repo-root>
    return Path(__file__).resolve().parent.parent.parent / "config" / "publisher_suppress_list.yaml"


# Exposed as a convenience for callers/tests that want to know where the
# loader looks by default. Computed eagerly — path resolution only, no IO.
DEFAULT_CONFIG_PATH: Path = _default_config_path()


def load_suppress_list(path: Union[str, Path, None] = None) -> frozenset[str]:
    """Load the publisher suppress list from YAML.

    Args:
        path: Path to the YAML config file. If None, resolves to
            ``<repo-root>/config/publisher_suppress_list.yaml``.

    Returns:
        A ``frozenset[str]`` of lowercase publisher identifiers. Empty
        frozenset if the file does not exist or the list is empty.

    Raises:
        ValueError: If the YAML is malformed or does not have the expected
            top-level shape (``suppressed_publishers: <list>``).

    Notes:
        - No module-level cache. Every call re-reads the file, so edits are
          picked up immediately (PRD R16: runtime-loadable without redeploy).
        - Matching is case-insensitive; all entries are lowercased on load.
    """
    config_path = Path(path) if path is not None else _default_config_path()

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        return frozenset()
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Malformed YAML in publisher suppress list at {config_path}: {exc}"
        ) from exc

    # Empty file → yaml.safe_load returns None. Treat as empty list.
    if data is None:
        return frozenset()

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected top-level mapping in {config_path}, got {type(data).__name__}"
        )

    raw_list = data.get("suppressed_publishers", [])
    if raw_list is None:
        return frozenset()

    if not isinstance(raw_list, list):
        raise ValueError(
            f"'suppressed_publishers' in {config_path} must be a list, "
            f"got {type(raw_list).__name__}"
        )

    entries: list[str] = []
    for item in raw_list:
        if item is None:
            continue
        if not isinstance(item, str):
            raise ValueError(
                f"Entries in 'suppressed_publishers' must be strings, "
                f"got {type(item).__name__}: {item!r}"
            )
        stripped = item.strip()
        if stripped:
            entries.append(stripped.lower())

    return frozenset(entries)


def is_suppressed(publisher: Union[str, None], suppress_set: frozenset[str]) -> bool:
    """Return True if ``publisher`` appears in ``suppress_set`` (case-insensitive).

    Args:
        publisher: The publisher string from a paper record. ``None`` is
            tolerated and always returns False.
        suppress_set: The set returned by :func:`load_suppress_list`.

    Returns:
        True if publisher (case-folded) is in the suppress set; False otherwise.
    """
    if publisher is None:
        return False
    if not isinstance(publisher, str):
        return False
    return publisher.strip().lower() in suppress_set
