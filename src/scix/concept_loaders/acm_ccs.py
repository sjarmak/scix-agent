"""ACM CCS 2012 loader.

The ACM Computing Classification System (2012 revision) is a SKOS taxonomy
of ~2,100 concepts spanning computing topics. ACM publishes the canonical
XML at https://www.acm.org/publications/class-2012 (web request behind a
WAF / 403 to non-browser clients), so we mirror a public copy.

License: per the ACM CCS web page, the taxonomy is free to use for
educational, academic, and bibliographic purposes. We record that as the
license text in the ``vocabularies`` row.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import psycopg

from scix.concepts import (
    Concept,
    ConceptRelationship,
    Vocabulary,
    load_vocabulary,
)
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

VOCAB_KEY = "acm_ccs"

SKOS = "{http://www.w3.org/2004/02/skos/core#}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"

DEFAULT_URL = (
    "https://raw.githubusercontent.com/s246wv/ISARC-bibtex-rdf/"
    "main/acm_ccs_emb/acm_ccs2012-1626988337597.xml"
)
HOMEPAGE_URL = "https://www.acm.org/publications/class-2012"
DEFAULT_DEST = Path("data/concepts/acm_ccs2012.xml")


def _vocabulary(source_url: str) -> Vocabulary:
    return Vocabulary(
        vocabulary=VOCAB_KEY,
        name="ACM Computing Classification System (CCS) 2012",
        description=(
            "ACM's hierarchical taxonomy of computing topics (2012 revision). "
            "Used as the standard subject classification across ACM publications."
        ),
        license="ACM CCS — free for educational, academic, and bibliographic use",
        license_url=HOMEPAGE_URL,
        homepage_url=HOMEPAGE_URL,
        source_url=source_url,
        version="2012",
    )


def _download(client: ResilientClient, url: str, dest: Path) -> Path:
    """Download the XML if the cached copy is missing or empty."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("acm_ccs: using cached %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = client.get(url)
    dest.write_bytes(resp.content if hasattr(resp, "content") else resp.text.encode())
    logger.info("acm_ccs: downloaded %d bytes -> %s", dest.stat().st_size, dest)
    return dest


def _parse(path: Path) -> tuple[list[Concept], list[ConceptRelationship]]:
    tree = ET.parse(path)
    root = tree.getroot()

    raw: dict[str, dict[str, Any]] = {}
    children_map: dict[str, list[str]] = {}
    parents_map: dict[str, list[str]] = {}

    for elem in root.iter(f"{SKOS}Concept"):
        concept_id = elem.get(f"{RDF}about")
        if not concept_id:
            continue
        pref_el = elem.find(f"{SKOS}prefLabel")
        if pref_el is None or not (pref_el.text or "").strip():
            continue
        alt_labels = tuple(
            a.text.strip() for a in elem.findall(f"{SKOS}altLabel") if a.text and a.text.strip()
        )
        definition_el = elem.find(f"{SKOS}definition")
        definition = (
            definition_el.text.strip() if definition_el is not None and definition_el.text else None
        )
        raw[concept_id] = {
            "preferred_label": pref_el.text.strip(),
            "alternate_labels": alt_labels,
            "definition": definition,
        }

        # broader → parent edge
        for b in elem.findall(f"{SKOS}broader"):
            parent_id = b.get(f"{RDF}resource")
            if parent_id:
                parents_map.setdefault(concept_id, []).append(parent_id)
                children_map.setdefault(parent_id, []).append(concept_id)
        # narrower → child edge (capture for completeness, dedupe later)
        for n in elem.findall(f"{SKOS}narrower"):
            child_id = n.get(f"{RDF}resource")
            if child_id:
                children_map.setdefault(concept_id, []).append(child_id)
                parents_map.setdefault(child_id, []).append(concept_id)

    # Compute level via BFS from concepts that have no broader parent
    all_ids = set(raw.keys())
    root_ids = all_ids - set(parents_map.keys())
    levels: dict[str, int] = {rid: 0 for rid in root_ids}
    queue: deque[str] = deque(root_ids)
    while queue:
        cur = queue.popleft()
        for child in children_map.get(cur, []):
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
            external_uri=f"https://dl.acm.org/ccs/{cid}",
            level=levels.get(cid),
        )
        for cid, info in raw.items()
    ]

    seen: set[tuple[str, str]] = set()
    rels: list[ConceptRelationship] = []
    for child_id, parent_ids in parents_map.items():
        if child_id not in raw:
            continue
        for parent_id in parent_ids:
            if parent_id not in raw:
                continue
            if (parent_id, child_id) in seen:
                continue
            seen.add((parent_id, child_id))
            rels.append(
                ConceptRelationship(vocabulary=VOCAB_KEY, parent_id=parent_id, child_id=child_id)
            )

    logger.info(
        "acm_ccs: parsed %d concepts, %d relationships, %d roots",
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
    """Download and load ACM CCS 2012 into the concepts substrate."""
    if client is None:
        client = ResilientClient(user_agent="scix-experiments/1.0", max_retries=3)

    path = _download(client, source_url, dest)
    concepts, rels = _parse(path)
    n_c, n_r = load_vocabulary(conn, _vocabulary(source_url), concepts, rels)
    conn.commit()
    return {"concepts": n_c, "relationships": n_r}
