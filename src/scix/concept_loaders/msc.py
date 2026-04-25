"""MSC2020 (Mathematics Subject Classification 2020) loader.

Pulls the SKOS Turtle published by TIB Hannover (CC BY-NC-SA), which is
itself the SKOS conversion of the joint AMS / zbMATH MSC2020. ~6,600
concepts in a 3-tier hierarchy (top: 00-XX, then xxYxx, then 5-char
codes like 03B05).

License: CC BY-NC-SA 4.0 (TIB conversion). The underlying MSC2020 itself
is © AMS / FIZ Karlsruhe with permission to reproduce for indexing /
classification purposes.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import psycopg

from scix.concepts import (
    Concept,
    ConceptRelationship,
    Vocabulary,
    load_vocabulary,
)
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

VOCAB_KEY = "msc"
DEFAULT_URL = (
    "https://raw.githubusercontent.com/TIBHannover/MSC2020_SKOS/" "main/msc-2020-suggestion4.ttl"
)
HOMEPAGE_URL = "https://msc2020.org/"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
DEFAULT_DEST = Path("data/concepts/msc2020.ttl")


def _vocabulary(source_url: str) -> Vocabulary:
    return Vocabulary(
        vocabulary=VOCAB_KEY,
        name="Mathematics Subject Classification 2020",
        description=(
            "MSC2020: AMS / zbMATH joint subject classification for "
            "mathematics, in SKOS form (TIB Hannover conversion)."
        ),
        license="CC BY-NC-SA 4.0 (TIB SKOS conversion)",
        license_url=LICENSE_URL,
        homepage_url=HOMEPAGE_URL,
        source_url=source_url,
        version="2020",
    )


def _download(client: ResilientClient, url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("msc: using cached %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = client.get(url)
    dest.write_bytes(resp.content if hasattr(resp, "content") else resp.text.encode())
    logger.info("msc: downloaded %d bytes -> %s", dest.stat().st_size, dest)
    return dest


def _parse(path: Path) -> tuple[list[Concept], list[ConceptRelationship]]:
    import rdflib  # local import — only needed inside the loader

    g = rdflib.Graph()
    g.parse(path, format="turtle")

    skos = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")

    raw: dict[str, dict[str, object]] = {}
    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {}

    for s in g.subjects(rdflib.RDF.type, skos.Concept):
        cid = str(s)
        notation_node = next(g.objects(s, skos.notation), None)
        notation = str(notation_node) if notation_node is not None else None
        pref_node = next(g.objects(s, skos.prefLabel), None)
        if pref_node is None:
            continue
        pref_label = str(pref_node)
        scope_nodes = list(g.objects(s, skos.scopeNote))
        definition = str(scope_nodes[0]) if scope_nodes else None
        raw[cid] = {
            "preferred_label": pref_label,
            "notation": notation,
            "definition": definition,
        }
        for parent in g.objects(s, skos.broader):
            pid = str(parent)
            parents.setdefault(cid, []).append(pid)
            children.setdefault(pid, []).append(cid)

    # Compute level via BFS
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
            concept_id=info["notation"] or cid,  # use the human notation as PK
            preferred_label=info["preferred_label"],
            alternate_labels=(),
            definition=info["definition"],
            external_uri=cid,
            level=levels.get(cid),
        )
        for cid, info in raw.items()
        if info["notation"]
    ]

    notation_by_uri = {cid: info["notation"] for cid, info in raw.items()}
    seen: set[tuple[str, str]] = set()
    rels: list[ConceptRelationship] = []
    for child_uri, parent_uris in parents.items():
        child_notation = notation_by_uri.get(child_uri)
        if not child_notation:
            continue
        for parent_uri in parent_uris:
            parent_notation = notation_by_uri.get(parent_uri)
            if not parent_notation:
                continue
            key = (parent_notation, child_notation)
            if key in seen:
                continue
            seen.add(key)
            rels.append(
                ConceptRelationship(
                    vocabulary=VOCAB_KEY,
                    parent_id=parent_notation,
                    child_id=child_notation,
                )
            )

    logger.info(
        "msc: parsed %d concepts, %d relationships, %d roots",
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
    """Download and load MSC2020 into the concepts substrate."""
    if client is None:
        client = ResilientClient(user_agent="scix-experiments/1.0", max_retries=3)

    path = _download(client, source_url, dest)
    concepts, rels = _parse(path)
    n_c, n_r = load_vocabulary(conn, _vocabulary(source_url), concepts, rels)
    conn.commit()
    return {"concepts": n_c, "relationships": n_r}
