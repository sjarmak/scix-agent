"""MeSH (Medical Subject Headings) loader (Public Domain — NLM).

Pulls the descriptor file from NLM (~313 MB XML for 2026, ~30k descriptor
records carrying ~300k entry terms across all concepts within each
descriptor). Uses ``iterparse`` so peak memory stays in the low-hundreds
of MB even on the full file.

Hierarchy: MeSH organizes descriptors into a tree via ``TreeNumber`` codes
(e.g. ``A11.284.180.520`` is a child of ``A11.284.180``). We build a
tree-number -> descriptor-UI map in the first pass, then derive parent
edges in the second.

License: Public domain (https://www.nlm.nih.gov/databases/download/terms_and_conditions.html).
Source : https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc{year}.xml
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date
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

VOCAB_KEY = "mesh"
HOMEPAGE_URL = "https://www.nlm.nih.gov/mesh/"
LICENSE_URL = "https://www.nlm.nih.gov/databases/download/terms_and_conditions.html"
DEFAULT_DEST_DIR = Path("data/concepts")


def default_url(year: int | None = None) -> str:
    """NLM publishes ``desc{year}.xml`` annually (Jan/Feb release)."""
    y = year or date.today().year
    return f"https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc{y}.xml"


def _vocabulary(source_url: str, year: int) -> Vocabulary:
    return Vocabulary(
        vocabulary=VOCAB_KEY,
        name="Medical Subject Headings (MeSH)",
        description=(
            "MeSH is the NLM controlled vocabulary used to index biomedical "
            "literature in PubMed. Descriptors carry tree numbers that "
            "define a polyhierarchical taxonomy."
        ),
        license="Public Domain (US Government work)",
        license_url=LICENSE_URL,
        homepage_url=HOMEPAGE_URL,
        source_url=source_url,
        version=str(year),
    )


def _download(client: ResilientClient, url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("mesh: using cached %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    # MeSH XML is ~300 MB. ResilientClient.get streams via httpx and writes
    # the whole body into memory; for a 300 MB file that is acceptable here
    # but not ideal. The download runs once per year so we accept it.
    resp = client.get(url)
    dest.write_bytes(resp.content if hasattr(resp, "content") else resp.text.encode())
    logger.info("mesh: downloaded %d bytes -> %s", dest.stat().st_size, dest)
    return dest


def _iter_descriptor_records(path: Path) -> Iterator[ET.Element]:
    """Yield each ``<DescriptorRecord>`` element, clearing as we go.

    The full ``DescriptorRecordSet`` document is too large to hold in
    memory; ``iterparse`` lets us pop each top-level child after we've
    pulled what we need.
    """
    context = ET.iterparse(path, events=("end",))
    for _, elem in context:
        if elem.tag != "DescriptorRecord":
            continue
        yield elem
        elem.clear()


def _collect_synonyms_and_xrefs(
    record: ET.Element, preferred_label: str
) -> tuple[list[str], list[str]]:
    """Return (alternate_labels, xrefs) for one DescriptorRecord.

    Alternate labels: every distinct ``Term/String`` across all Concepts
    in the record, excluding the preferred descriptor label itself.

    XRefs: ``RegistryNumber`` + ``RelatedRegistryNumber`` from each Concept
    (CAS / EC / UNII / etc. — left as raw strings).
    """
    seen_alts: set[str] = set()
    alts: list[str] = []
    xrefs: list[str] = []

    for concept in record.iterfind("ConceptList/Concept"):
        for reg in concept.iterfind("RegistryNumber"):
            if reg.text and reg.text.strip() and reg.text.strip() != "0":
                xrefs.append(reg.text.strip())
        for reg in concept.iterfind("RelatedRegistryNumberList/RelatedRegistryNumber"):
            if reg.text and reg.text.strip():
                xrefs.append(reg.text.strip())
        for term_string in concept.iterfind("TermList/Term/String"):
            text = (term_string.text or "").strip()
            if not text or text == preferred_label or text in seen_alts:
                continue
            seen_alts.add(text)
            alts.append(text)

    return alts, xrefs


def _parent_tree_number(tn: str) -> str | None:
    """Return the immediate parent tree number, or None for tree roots."""
    if "." not in tn:
        return None
    return tn.rsplit(".", 1)[0]


def parse(path: Path) -> tuple[list[Concept], list[ConceptRelationship]]:
    """Two-pass parse: collect concepts + tree-number map, then derive edges."""
    concepts: list[Concept] = []
    # tree_number -> descriptor_ui (multiple tree numbers may map to the
    # same descriptor, but each tree number maps to exactly one descriptor)
    tree_to_ui: dict[str, str] = {}
    descriptor_tree_numbers: list[tuple[str, list[str]]] = []  # (ui, [tree_numbers])

    for record in _iter_descriptor_records(path):
        ui_el = record.find("DescriptorUI")
        name_el = record.find("DescriptorName/String")
        if ui_el is None or name_el is None or not ui_el.text or not name_el.text:
            continue

        ui = ui_el.text.strip()
        preferred_label = name_el.text.strip()
        tree_numbers = [
            tn.text.strip()
            for tn in record.iterfind("TreeNumberList/TreeNumber")
            if tn.text and tn.text.strip()
        ]

        # Definition: ScopeNote on the preferred concept (PreferredConceptYN=Y).
        definition: str | None = None
        for concept in record.iterfind("ConceptList/Concept"):
            if concept.get("PreferredConceptYN") == "Y":
                scope = concept.find("ScopeNote")
                if scope is not None and scope.text:
                    definition = scope.text.strip()
                break

        alts, xrefs = _collect_synonyms_and_xrefs(record, preferred_label)

        props: dict[str, object] = {}
        if tree_numbers:
            props["tree_numbers"] = tree_numbers
        if xrefs:
            props["xrefs"] = xrefs

        concepts.append(
            Concept(
                vocabulary=VOCAB_KEY,
                concept_id=ui,
                preferred_label=preferred_label,
                alternate_labels=tuple(alts),
                definition=definition,
                external_uri=f"https://meshb.nlm.nih.gov/record/ui?ui={ui}",
                properties=props,
            )
        )

        for tn in tree_numbers:
            tree_to_ui[tn] = ui
        descriptor_tree_numbers.append((ui, tree_numbers))

    # Derive parent edges: for every tree number, find its parent tree
    # number (drop the last segment) and look up the parent descriptor UI.
    rels_set: set[tuple[str, str]] = set()
    for ui, tree_numbers in descriptor_tree_numbers:
        for tn in tree_numbers:
            parent_tn = _parent_tree_number(tn)
            if parent_tn is None:
                continue
            parent_ui = tree_to_ui.get(parent_tn)
            if parent_ui is None or parent_ui == ui:
                continue
            rels_set.add((parent_ui, ui))

    rels = [ConceptRelationship(vocabulary=VOCAB_KEY, parent_id=p, child_id=c) for p, c in rels_set]
    return concepts, rels


def load(
    conn: psycopg.Connection,
    *,
    client: ResilientClient | None = None,
    year: int | None = None,
    url: str | None = None,
    dest: Path | None = None,
) -> dict[str, int]:
    """Download (cached) and load MeSH into the concepts substrate."""
    if client is None:
        client = ResilientClient(user_agent="scix-experiments/1.0", max_retries=3)
    resolved_year = year or date.today().year
    resolved_url = url or default_url(resolved_year)
    resolved_dest = dest or (DEFAULT_DEST_DIR / f"desc{resolved_year}.xml")

    path = _download(client, resolved_url, resolved_dest)
    concepts, rels = parse(path)
    n_c, n_r = load_vocabulary(conn, _vocabulary(resolved_url, resolved_year), concepts, rels)
    conn.commit()
    return {"concepts": n_c, "relationships": n_r}
