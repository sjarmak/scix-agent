"""GCMD (NASA Global Change Master Directory) keyword loader.

Pulls four GCMD keyword schemes — sciencekeywords, instruments, platforms,
locations — from the public ``adiwg/gcmd-keywords`` GitHub mirror, and
loads them as a single ``gcmd`` vocabulary in the concepts substrate. The
existing ``harvest_gcmd.py`` script seeds entity rows (instruments,
missions, observables); this loader provides the concept hierarchy that
``concept_search`` needs for routing earth-science queries.

License: NASA GCMD keywords are public domain (US Government work).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import psycopg

from scix.concepts import (
    Concept,
    ConceptRelationship,
    Vocabulary,
    load_vocabulary,
)
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

VOCAB_KEY = "gcmd"
GITHUB_BASE = "https://raw.githubusercontent.com/adiwg/gcmd-keywords/master/resources/json"
HOMEPAGE_URL = "https://gcmd.earthdata.nasa.gov/"
LICENSE_URL = "https://www.usa.gov/government-works"

SCHEMES: tuple[str, ...] = (
    "sciencekeywords",
    "instruments",
    "platforms",
)
DEFAULT_DEST_DIR = Path("data/concepts/gcmd")


def _vocabulary() -> Vocabulary:
    return Vocabulary(
        vocabulary=VOCAB_KEY,
        name="GCMD Keywords",
        description=(
            "NASA Global Change Master Directory keyword vocabularies "
            "(science keywords, instruments, platforms, locations) "
            "spanning earth-science discovery metadata."
        ),
        license="Public Domain (US Government work)",
        license_url=LICENSE_URL,
        homepage_url=HOMEPAGE_URL,
        source_url=GITHUB_BASE,
        properties={"schemes": list(SCHEMES)},
    )


def _download_scheme(client: ResilientClient, scheme: str, dest_dir: Path) -> Path:
    dest = dest_dir / f"{scheme}.json"
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("gcmd: using cached %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GITHUB_BASE}/{scheme}.json"
    resp = client.get(url)
    dest.write_bytes(resp.content if hasattr(resp, "content") else resp.text.encode())
    logger.info("gcmd: downloaded %s (%d bytes)", scheme, dest.stat().st_size)
    return dest


def _walk(
    node: dict[str, Any],
    scheme: str,
    parent_id: str | None,
    level: int,
    out_concepts: dict[str, Concept],
    out_rels: list[ConceptRelationship],
) -> None:
    label = (node.get("label") or "").strip()
    uuid = (node.get("uuid") or "").strip()

    # Top-level placeholders (e.g. {"broader": null, "children": [{label: "EARTH SCIENCE"}]})
    # have no uuid; descend without recording.
    if uuid and label:
        if uuid not in out_concepts:
            properties: dict[str, object] = {"scheme": scheme}
            short_name = (node.get("short_name") or "").strip() or None
            if short_name and short_name != label:
                properties["short_name"] = short_name
            out_concepts[uuid] = Concept(
                vocabulary=VOCAB_KEY,
                concept_id=uuid,
                preferred_label=label,
                alternate_labels=(short_name,) if short_name and short_name != label else (),
                definition=node.get("definition") or None,
                external_uri=f"https://gcmd.earthdata.nasa.gov/kms/concept/{uuid}",
                level=level,
                properties=properties,
            )
        if parent_id and parent_id in out_concepts:
            out_rels.append(
                ConceptRelationship(vocabulary=VOCAB_KEY, parent_id=parent_id, child_id=uuid)
            )
        new_parent = uuid
        new_level = level + 1
    else:
        new_parent = parent_id
        new_level = level

    for child in node.get("children", []) or []:
        _walk(child, scheme, new_parent, new_level, out_concepts, out_rels)


def _parse_scheme(
    path: Path,
    scheme: str,
    out_concepts: dict[str, Concept],
    out_rels: list[ConceptRelationship],
) -> None:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"gcmd: expected list at top of {path}, got {type(data)}")
    for root in data:
        _walk(root, scheme, None, 0, out_concepts, out_rels)


def _dedupe_relationships(
    rels: list[ConceptRelationship],
) -> list[ConceptRelationship]:
    seen: set[tuple[str, str]] = set()
    out: list[ConceptRelationship] = []
    for r in rels:
        key = (r.parent_id, r.child_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load(
    conn: psycopg.Connection,
    *,
    client: ResilientClient | None = None,
    dest_dir: Path = DEFAULT_DEST_DIR,
    schemes: tuple[str, ...] = SCHEMES,
) -> dict[str, int]:
    """Download all schemes and load GCMD concepts."""
    if client is None:
        client = ResilientClient(user_agent="scix-experiments/1.0", max_retries=3)

    out_concepts: dict[str, Concept] = {}
    out_rels: list[ConceptRelationship] = []
    for scheme in schemes:
        path = _download_scheme(client, scheme, dest_dir)
        _parse_scheme(path, scheme, out_concepts, out_rels)

    rels = _dedupe_relationships(out_rels)
    concepts = list(out_concepts.values())
    logger.info("gcmd: parsed %d concepts, %d relationships", len(concepts), len(rels))

    n_c, n_r = load_vocabulary(conn, _vocabulary(), concepts, rels)
    conn.commit()
    return {"concepts": n_c, "relationships": n_r}
