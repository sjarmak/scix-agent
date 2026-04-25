"""PhySH (Physics Subject Headings) loader.

Pulls the canonical APS PhySH JSON-LD bundle (~3,900 concepts) and
materializes the full graph into the ``concepts`` substrate. The existing
``harvest_physh.py`` script seeds the ``entities`` table with technique
concepts as ``method`` entities, but never persisted the full hierarchy
needed for concept_search routing — that is what this loader provides.

License: CC BY 4.0 (per https://physh.org).
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import deque
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

VOCAB_KEY = "physh"
DEFAULT_URL = "https://raw.githubusercontent.com/physh-org/PhySH/master/physh.json.gz"
HOMEPAGE_URL = "https://physh.org/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DEFAULT_DEST = Path("data/concepts/physh.json.gz")

SKOS_BROADER = "http://www.w3.org/2004/02/skos/core#broader"
SKOS_PREFLABEL = "http://www.w3.org/2004/02/skos/core#prefLabel"
SKOS_ALTLABEL = "http://www.w3.org/2004/02/skos/core#altLabel"
SKOS_DEFINITION = "http://www.w3.org/2004/02/skos/core#definition"
PHYSH_PREFLABEL = "https://physh.org/rdf/2018/01/01/core#prefLabel"


def _vocabulary(source_url: str) -> Vocabulary:
    return Vocabulary(
        vocabulary=VOCAB_KEY,
        name="Physics Subject Headings (PhySH)",
        description=(
            "APS PhySH: physics-discipline taxonomy spanning Concepts, "
            "Disciplines, Methods/Techniques, Properties and Subjects."
        ),
        license="CC BY 4.0",
        license_url=LICENSE_URL,
        homepage_url=HOMEPAGE_URL,
        source_url=source_url,
    )


def _download(client: ResilientClient, url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("physh: using cached %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = client.get(url)
    dest.write_bytes(resp.content if hasattr(resp, "content") else resp.text.encode())
    logger.info("physh: downloaded %d bytes -> %s", dest.stat().st_size, dest)
    return dest


def _label_first(values: list[dict[str, Any]] | None) -> str | None:
    if not values:
        return None
    for v in values:
        if v.get("@language", "en") == "en" and v.get("@value"):
            return v["@value"].strip()
    first = values[0]
    return first.get("@value")


def _alt_labels(values: list[dict[str, Any]] | None) -> tuple[str, ...]:
    if not values:
        return ()
    out: list[str] = []
    for v in values:
        if v.get("@language", "en") == "en" and v.get("@value"):
            out.append(v["@value"].strip())
    return tuple(dict.fromkeys(out))


def _parse(path: Path) -> tuple[list[Concept], list[ConceptRelationship]]:
    raw_bytes = path.read_bytes()
    data = json.loads(gzip.decompress(raw_bytes).decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError("physh: expected JSON-LD array as top level")

    raw: dict[str, dict[str, Any]] = {}
    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {}

    for item in data:
        cid = item.get("@id")
        if not cid:
            continue
        types = item.get("@type") or []
        if not any(
            t.endswith("Concept") or t == "http://www.w3.org/2004/02/skos/core#Concept"
            for t in types
        ):
            continue
        pref = _label_first(item.get(SKOS_PREFLABEL)) or _label_first(item.get(PHYSH_PREFLABEL))
        if not pref:
            continue
        alt = _alt_labels(item.get(SKOS_ALTLABEL))
        definition = _label_first(item.get(SKOS_DEFINITION))
        raw[cid] = {
            "preferred_label": pref,
            "alternate_labels": alt,
            "definition": definition,
        }
        for b in item.get(SKOS_BROADER, []) or []:
            pid = b.get("@id")
            if pid:
                parents.setdefault(cid, []).append(pid)
                children.setdefault(pid, []).append(cid)

    all_ids = set(raw.keys())
    root_ids = all_ids - set(parents.keys())
    levels: dict[str, int] = {rid: 0 for rid in root_ids}
    queue: deque[str] = deque(root_ids)
    while queue:
        cur = queue.popleft()
        for child in children.get(cur, []):
            if child not in raw:
                continue
            new_level = levels[cur] + 1
            if child not in levels or new_level < levels[child]:
                levels[child] = new_level
                queue.append(child)

    concepts = [
        Concept(
            vocabulary=VOCAB_KEY,
            concept_id=cid,
            preferred_label=info["preferred_label"],
            alternate_labels=info["alternate_labels"],
            definition=info["definition"],
            external_uri=cid,
            level=levels.get(cid),
        )
        for cid, info in raw.items()
    ]

    seen: set[tuple[str, str]] = set()
    rels: list[ConceptRelationship] = []
    for child_id, parent_ids in parents.items():
        if child_id not in raw:
            continue
        for parent_id in parent_ids:
            if parent_id not in raw:
                continue
            key = (parent_id, child_id)
            if key in seen:
                continue
            seen.add(key)
            rels.append(
                ConceptRelationship(vocabulary=VOCAB_KEY, parent_id=parent_id, child_id=child_id)
            )

    logger.info(
        "physh: parsed %d concepts, %d relationships, %d roots",
        len(concepts),
        len(rels),
        len(root_ids),
    )
    return concepts, rels


def load(
    conn: psycopg.Connection,
    *,
    client: ResilientClient | None = None,
    source_url: str = DEFAULT_URL,
    dest: Path = DEFAULT_DEST,
) -> dict[str, int]:
    """Download and load PhySH into the concepts substrate."""
    if client is None:
        client = ResilientClient(user_agent="scix-experiments/1.0", max_retries=3)

    path = _download(client, source_url, dest)
    concepts, rels = _parse(path)
    n_c, n_r = load_vocabulary(conn, _vocabulary(source_url), concepts, rels)
    conn.commit()
    return {"concepts": n_c, "relationships": n_r}
