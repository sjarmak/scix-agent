"""ChEBI (Chemical Entities of Biological Interest) loader (CC-BY 4.0).

Pulls the EBI ChEBI ontology in OBO format. We default to ``chebi_lite.obo``
(~7 MB, ~190k entities, structure data dropped) which is enough for
concept_search; the full ``chebi.obo`` (~150 MB) adds chemical structures
that the concepts substrate doesn't model anyway.

License: CC BY 4.0 (https://www.ebi.ac.uk/chebi/aboutChebiForward.do).
Source : https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.obo
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

VOCAB_KEY = "chebi"
# Default to the FULL chebi.obo (~250 MB) so synonyms and xrefs (CAS,
# ChEMBL, KEGG, DrugBank, PubChem, Wikipedia) are available; the lite
# variant strips both, which makes label-based search anemic and breaks
# the ChEBI↔ChEMBL acceptance criterion. The lite URL is preserved as a
# fallback for low-bandwidth environments.
DEFAULT_URL = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo"
LITE_URL = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.obo"
HOMEPAGE_URL = "https://www.ebi.ac.uk/chebi/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DEFAULT_DEST = Path("data/concepts/chebi.obo")

# Cross-reference prefixes we actively retain in ``properties.xrefs`` so the
# bead's ChEBI↔ChEMBL acceptance criterion can be satisfied without storing
# the entire xref soup.
_RETAINED_XREF_PREFIXES = ("CHEMBL", "CAS", "DrugBank", "KEGG", "PubChem", "Wikipedia")


def _vocabulary(source_url: str) -> Vocabulary:
    return Vocabulary(
        vocabulary=VOCAB_KEY,
        name="Chemical Entities of Biological Interest (ChEBI)",
        description=(
            "ChEBI is a freely available dictionary of molecular entities "
            "focused on small chemical compounds, maintained by EMBL-EBI."
        ),
        license="CC BY 4.0",
        license_url=LICENSE_URL,
        homepage_url=HOMEPAGE_URL,
        source_url=source_url,
    )


def _download(client: ResilientClient, url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("chebi: using cached %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = client.get(url)
    dest.write_bytes(resp.content if hasattr(resp, "content") else resp.text.encode())
    logger.info("chebi: downloaded %d bytes -> %s", dest.stat().st_size, dest)
    return dest


def _filter_xrefs(xrefs: tuple[str, ...]) -> list[str]:
    """Keep only xrefs to a curated set of external chemistry registries.

    Prefix match is case-insensitive (chebi.obo uses ``cas:`` / ``chembl:``
    while chebi_lite uses ``CAS:`` / ``CHEMBL:``).
    """
    lowered_prefixes = tuple(p.lower() + ":" for p in _RETAINED_XREF_PREFIXES)
    out: list[str] = []
    for x in xrefs:
        if any(x.lower().startswith(p) for p in lowered_prefixes):
            out.append(x)
    return out


def parse(path: Path) -> tuple[list[Concept], list[ConceptRelationship]]:
    """Parse a ChEBI OBO file into Concept / ConceptRelationship records.

    Obsolete terms are skipped (they have replaced_by pointers but no
    structural place in the hierarchy).
    """
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
        kept_xrefs = _filter_xrefs(term.xrefs)
        if kept_xrefs:
            props["xrefs"] = kept_xrefs
        if term.alt_ids:
            props["alt_ids"] = list(term.alt_ids)

        concepts.append(
            Concept(
                vocabulary=VOCAB_KEY,
                concept_id=term.id,
                preferred_label=term.name,
                alternate_labels=term.synonyms,
                definition=term.definition,
                external_uri=f"http://purl.obolibrary.org/obo/{term.id.replace(':', '_')}",
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

    # Drop edges whose parent was filtered (obsolete) — load_relationships
    # will FK-skip them, but pre-filtering keeps the warning count clean.
    rels = [r for r in rels if r.parent_id in seen_ids]
    return concepts, rels


def load(
    conn: psycopg.Connection,
    *,
    client: ResilientClient | None = None,
    url: str = DEFAULT_URL,
    dest: Path = DEFAULT_DEST,
) -> dict[str, int]:
    """Download (cached) and load ChEBI into the concepts substrate."""
    if client is None:
        client = ResilientClient(user_agent="scix-experiments/1.0", max_retries=3)

    path = _download(client, url, dest)
    concepts, rels = parse(path)
    n_c, n_r = load_vocabulary(conn, _vocabulary(url), concepts, rels)
    conn.commit()
    return {"concepts": n_c, "relationships": n_r}
