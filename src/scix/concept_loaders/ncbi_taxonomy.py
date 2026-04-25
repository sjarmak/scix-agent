"""NCBI Taxonomy loader (Public Domain — NCBI).

Pulls the standard ``taxdump.tar.gz`` (~73 MB compressed, ~370 MB
uncompressed across ``names.dmp``, ``nodes.dmp``, etc.) and builds:

  - One concept per ``tax_id`` with the scientific name as preferred_label
  - All synonyms / common names / genbank common names as alternate_labels
  - ``parent_tax_id`` -> parent edge (skipping self-loop on root tax_id 1)
  - Rank ("species", "genus", ...) captured in ``properties.rank``

The file is large but the streaming layout keeps peak memory in the
hundreds of MB. We pull only ``names.dmp`` and ``nodes.dmp`` from the tar
to skip merged / deleted node files.

License: Public domain (https://www.ncbi.nlm.nih.gov/home/about/policies/).
Source : https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
"""

from __future__ import annotations

import logging
import tarfile
from collections.abc import Iterator
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

VOCAB_KEY = "ncbi_tax"
DEFAULT_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
HOMEPAGE_URL = "https://www.ncbi.nlm.nih.gov/taxonomy"
LICENSE_URL = "https://www.ncbi.nlm.nih.gov/home/about/policies/"
DEFAULT_DEST = Path("data/concepts/ncbi_taxdump.tar.gz")

# Name classes (from names.dmp) to keep as alternate labels.
_ALT_NAME_CLASSES = frozenset(
    {
        "synonym",
        "common name",
        "genbank common name",
        "equivalent name",
        "acronym",
        "blast name",
    }
)


def _vocabulary(source_url: str) -> Vocabulary:
    return Vocabulary(
        vocabulary=VOCAB_KEY,
        name="NCBI Taxonomy",
        description=(
            "NCBI Taxonomy is the curated classification of all organisms "
            "with sequence data in NCBI databases. Hierarchical from cellular "
            "organisms / viruses down to strains."
        ),
        license="Public Domain (US Government work)",
        license_url=LICENSE_URL,
        homepage_url=HOMEPAGE_URL,
        source_url=source_url,
    )


def _download(client: ResilientClient, url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("ncbi_tax: using cached %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = client.get(url)
    dest.write_bytes(resp.content if hasattr(resp, "content") else resp.text.encode())
    logger.info("ncbi_tax: downloaded %d bytes -> %s", dest.stat().st_size, dest)
    return dest


def _iter_dmp_lines(tar_path: Path, member_name: str) -> Iterator[list[str]]:
    """Yield dmp rows split into stripped fields.

    The dmp format uses ``\\t|\\t`` between fields and ``\\t|\\n`` at row end;
    after splitting on ``|`` we strip each field of leading/trailing tabs
    and whitespace.
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.getmember(member_name)
        fh = tar.extractfile(member)
        if fh is None:
            raise RuntimeError(f"could not extract {member_name} from {tar_path}")
        for raw in fh:
            line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\t|")
            if not line:
                continue
            yield [field.strip("\t ") for field in line.split("|")]


def _read_names(tar_path: Path) -> dict[str, tuple[str | None, list[str]]]:
    """Return ``tax_id -> (scientific_name, [alt_names])`` from names.dmp."""
    out: dict[str, tuple[str | None, list[str]]] = {}
    seen_alts: dict[str, set[str]] = {}
    n_rows = 0
    for fields in _iter_dmp_lines(tar_path, "names.dmp"):
        # tax_id | name_txt | unique_name | name_class
        if len(fields) < 4:
            continue
        tax_id, name_txt, _unique, name_class = fields[0], fields[1], fields[2], fields[3]
        if not tax_id or not name_txt:
            continue
        n_rows += 1
        scientific, alts = out.get(tax_id, (None, []))
        if name_class == "scientific name":
            scientific = name_txt
        elif name_class in _ALT_NAME_CLASSES:
            seen = seen_alts.setdefault(tax_id, set())
            if name_txt not in seen and name_txt != scientific:
                seen.add(name_txt)
                alts.append(name_txt)
        out[tax_id] = (scientific, alts)
    logger.info("ncbi_tax: read %d name rows -> %d distinct taxa", n_rows, len(out))
    return out


def _iter_nodes(
    tar_path: Path,
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(tax_id, parent_tax_id, rank)`` triples from nodes.dmp."""
    for fields in _iter_dmp_lines(tar_path, "nodes.dmp"):
        # tax_id | parent_tax_id | rank | ...
        if len(fields) < 3:
            continue
        yield fields[0], fields[1], fields[2]


def parse(path: Path) -> tuple[list[Concept], list[ConceptRelationship]]:
    """Build Concept + ConceptRelationship records from a taxdump tarball.

    Memory note: peak ~500 MB for the ~2.5M-tax full dump. Run via
    ``scix-batch`` to stay under the systemd-oomd budget on this machine.
    """
    names = _read_names(path)
    concepts: list[Concept] = []
    rels: list[ConceptRelationship] = []
    n_skipped_no_name = 0

    for tax_id, parent_tax_id, rank in _iter_nodes(path):
        scientific, alts = names.get(tax_id, (None, []))
        if not scientific:
            n_skipped_no_name += 1
            continue
        props: dict[str, object] = {}
        if rank:
            props["rank"] = rank
        concepts.append(
            Concept(
                vocabulary=VOCAB_KEY,
                concept_id=tax_id,
                preferred_label=scientific,
                alternate_labels=tuple(alts),
                external_uri=(
                    f"https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={tax_id}"
                ),
                properties=props,
            )
        )
        # tax_id 1 (root) lists itself as its own parent — skip the self-loop.
        if parent_tax_id and parent_tax_id != tax_id:
            rels.append(
                ConceptRelationship(
                    vocabulary=VOCAB_KEY,
                    parent_id=parent_tax_id,
                    child_id=tax_id,
                )
            )

    if n_skipped_no_name:
        logger.warning("ncbi_tax: skipped %d nodes with no scientific name", n_skipped_no_name)
    logger.info("ncbi_tax: built %d concepts and %d edges", len(concepts), len(rels))
    return concepts, rels


def load(
    conn: psycopg.Connection,
    *,
    client: ResilientClient | None = None,
    url: str = DEFAULT_URL,
    dest: Path = DEFAULT_DEST,
) -> dict[str, int]:
    """Download (cached) and load NCBI Taxonomy into the concepts substrate."""
    if client is None:
        client = ResilientClient(user_agent="scix-experiments/1.0", max_retries=3)

    path = _download(client, url, dest)
    concepts, rels = parse(path)
    n_c, n_r = load_vocabulary(conn, _vocabulary(url), concepts, rels)
    conn.commit()
    return {"concepts": n_c, "relationships": n_r}
