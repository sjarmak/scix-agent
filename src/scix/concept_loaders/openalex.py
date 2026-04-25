"""OpenAlex Topics loader (CC0).

Pulls the full 4-level OpenAlex topic taxonomy: domain → field → subfield
→ topic. As of 2026-04 the taxonomy is ~4 domains, 26 fields, 252
subfields, 4,516 topics → ~4,800 nodes.

License: CC0 (https://docs.openalex.org/about-the-data/license).
Source : https://api.openalex.org/topics
"""

from __future__ import annotations

import logging
import time
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

VOCAB_KEY = "openalex"
TOPICS_URL = "https://api.openalex.org/topics"
PER_PAGE = 200
USER_AGENT = "scix-experiments/1.0 (mailto:stephanie.jarmak@gmail.com)"


def _vocabulary() -> Vocabulary:
    return Vocabulary(
        vocabulary=VOCAB_KEY,
        name="OpenAlex Topics",
        description=(
            "OpenAlex 4-level scientific topic taxonomy: domains, fields, "
            "subfields, and topics. Built from clustered citation neighborhoods."
        ),
        license="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        homepage_url="https://docs.openalex.org/api-entities/topics",
        source_url=TOPICS_URL,
    )


def _fetch_all_topics(client: ResilientClient) -> list[dict[str, Any]]:
    """Fetch all OpenAlex topics across paginated API."""
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        resp = client.get(
            TOPICS_URL,
            params={
                "per-page": PER_PAGE,
                "page": page,
                "select": ("id,display_name,description,keywords,ids,subfield,field,domain"),
            },
        )
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        out.extend(results)
        meta = data.get("meta", {})
        total = meta.get("count", 0)
        if page * PER_PAGE >= total:
            break
        page += 1
        time.sleep(0.05)  # be polite
    logger.info("openalex: fetched %d topics", len(out))
    return out


def _parse(topics: list[dict[str, Any]]) -> tuple[list[Concept], list[ConceptRelationship]]:
    """Build domain/field/subfield/topic concepts and parent edges from API rows."""
    concepts: dict[str, Concept] = {}
    rels: set[tuple[str, str]] = set()

    for t in topics:
        topic_id = t["id"]
        sub = t.get("subfield") or {}
        field = t.get("field") or {}
        domain = t.get("domain") or {}

        domain_id = domain.get("id")
        field_id = field.get("id")
        sub_id = sub.get("id")

        # Domain (level 0)
        if domain_id and domain_id not in concepts:
            concepts[domain_id] = Concept(
                vocabulary=VOCAB_KEY,
                concept_id=domain_id,
                preferred_label=domain.get("display_name", "") or "",
                external_uri=domain_id,
                level=0,
            )

        # Field (level 1)
        if field_id and field_id not in concepts:
            concepts[field_id] = Concept(
                vocabulary=VOCAB_KEY,
                concept_id=field_id,
                preferred_label=field.get("display_name", "") or "",
                external_uri=field_id,
                level=1,
            )
            if domain_id:
                rels.add((domain_id, field_id))

        # Subfield (level 2)
        if sub_id and sub_id not in concepts:
            concepts[sub_id] = Concept(
                vocabulary=VOCAB_KEY,
                concept_id=sub_id,
                preferred_label=sub.get("display_name", "") or "",
                external_uri=sub_id,
                level=2,
            )
            if field_id:
                rels.add((field_id, sub_id))

        # Topic (level 3) — keywords become alternate labels
        keywords = tuple(t.get("keywords") or ())
        ids = t.get("ids") or {}
        wiki = ids.get("wikipedia")
        props: dict[str, object] = {}
        if wiki:
            props["wikipedia"] = wiki
        concepts[topic_id] = Concept(
            vocabulary=VOCAB_KEY,
            concept_id=topic_id,
            preferred_label=t.get("display_name", "") or "",
            alternate_labels=keywords,
            definition=t.get("description"),
            external_uri=topic_id,
            level=3,
            properties=props,
        )
        if sub_id:
            rels.add((sub_id, topic_id))

    rel_records = [
        ConceptRelationship(vocabulary=VOCAB_KEY, parent_id=p, child_id=c) for p, c in rels
    ]
    return list(concepts.values()), rel_records


def load(conn: psycopg.Connection, *, client: ResilientClient | None = None) -> dict[str, int]:
    """Download and load OpenAlex Topics into the concepts substrate."""
    if client is None:
        client = ResilientClient(user_agent=USER_AGENT, max_retries=3, rate_limit=10.0)

    raw = _fetch_all_topics(client)
    concepts, rels = _parse(raw)
    n_c, n_r = load_vocabulary(conn, _vocabulary(), concepts, rels)
    conn.commit()
    return {"concepts": n_c, "relationships": n_r}
