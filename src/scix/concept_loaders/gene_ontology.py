"""Gene Ontology (GO) loader (CC-BY 4.0).

Pulls the canonical OBO ontology from the OBO Foundry (~45k terms across
three sub-ontologies: biological_process, molecular_function,
cellular_component). Synonyms and alt_ids become alternate labels;
namespace is captured in ``properties.namespace`` so downstream tools can
filter (e.g. only molecular_function for a "what does this protein do?"
query).

License: CC BY 4.0 (http://geneontology.org/docs/go-citation-policy/).
Source : http://purl.obolibrary.org/obo/go.obo
"""

from __future__ import annotations

import logging
from pathlib import Path

import psycopg

from scix.concept_loaders._obo import iter_terms
from scix.concepts import (
    Concept,
    ConceptRelationship,
    Vocabulary,
    load_vocabulary,
)
from scix.http_client import ResilientClient

logger = logging.getLogger(__name__)

VOCAB_KEY = "gene_ontology"
DEFAULT_URL = "http://purl.obolibrary.org/obo/go.obo"
HOMEPAGE_URL = "http://geneontology.org/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DEFAULT_DEST = Path("data/concepts/go.obo")


def _vocabulary(source_url: str) -> Vocabulary:
    return Vocabulary(
        vocabulary=VOCAB_KEY,
        name="Gene Ontology",
        description=(
            "GO is a structured vocabulary of biological processes, molecular "
            "functions, and cellular components used to describe gene products."
        ),
        license="CC BY 4.0",
        license_url=LICENSE_URL,
        homepage_url=HOMEPAGE_URL,
        source_url=source_url,
    )


def _download(client: ResilientClient, url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("gene_ontology: using cached %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = client.get(url)
    dest.write_bytes(resp.content if hasattr(resp, "content") else resp.text.encode())
    logger.info("gene_ontology: downloaded %d bytes -> %s", dest.stat().st_size, dest)
    return dest


# Map GO namespace strings to a small integer level so that downstream
# tools can filter by sub-ontology. Levels: 0 = biological_process,
# 1 = molecular_function, 2 = cellular_component. Anything else -> None.
_NAMESPACE_LEVEL = {
    "biological_process": 0,
    "molecular_function": 1,
    "cellular_component": 2,
}


def parse(path: Path) -> tuple[list[Concept], list[ConceptRelationship]]:
    """Parse a GO OBO file into Concept / ConceptRelationship records."""
    concepts: list[Concept] = []
    rels: list[ConceptRelationship] = []
    seen_ids: set[str] = set()

    for term in iter_terms(path):
        if term.is_obsolete:
            continue
        if not term.id or not term.name:
            continue
        seen_ids.add(term.id)

        props: dict[str, object] = {}
        if term.namespace:
            props["namespace"] = term.namespace
        if term.alt_ids:
            props["alt_ids"] = list(term.alt_ids)
        # Capture all xrefs verbatim — go.obo's xref set is already curated
        # (CHEBI, EC, FMA, MetaCyc, KEGG, RHEA, ...) and downstream tools
        # benefit from the cross-ontology pointers.
        if term.xrefs:
            props["xrefs"] = list(term.xrefs)

        concepts.append(
            Concept(
                vocabulary=VOCAB_KEY,
                concept_id=term.id,
                preferred_label=term.name,
                alternate_labels=term.synonyms,
                definition=term.definition,
                external_uri=f"http://purl.obolibrary.org/obo/{term.id.replace(':', '_')}",
                level=_NAMESPACE_LEVEL.get(term.namespace or ""),
                properties=props,
            )
        )
        for parent_id in term.parents:
            rels.append(
                ConceptRelationship(
                    vocabulary=VOCAB_KEY,
                    parent_id=parent_id,
                    child_id=term.id,
                )
            )

    rels = [r for r in rels if r.parent_id in seen_ids]
    return concepts, rels


def load(
    conn: psycopg.Connection,
    *,
    client: ResilientClient | None = None,
    url: str = DEFAULT_URL,
    dest: Path = DEFAULT_DEST,
) -> dict[str, int]:
    """Download (cached) and load Gene Ontology into the concepts substrate."""
    if client is None:
        client = ResilientClient(user_agent="scix-experiments/1.0", max_retries=3)

    path = _download(client, url, dest)
    concepts, rels = parse(path)
    n_c, n_r = load_vocabulary(conn, _vocabulary(url), concepts, rels)
    conn.commit()
    return {"concepts": n_c, "relationships": n_r}
