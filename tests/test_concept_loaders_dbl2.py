"""Unit tests for the dbl.2 biomed vocabulary parsers.

Each parser is exercised against a small inline fixture written to a
``tmp_path`` file; no network access. The integration round-trip already
covered by ``test_concepts.py`` is not duplicated per loader.
"""

from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

from scix.concept_loaders import chebi, gene_ontology, mesh, ncbi_taxonomy
from scix.concept_loaders._obo import iter_terms

# ---------------------------------------------------------------------------
# OBO parser
# ---------------------------------------------------------------------------


_OBO_FIXTURE = """\
format-version: 1.2
ontology: chebi

[Term]
id: CHEBI:15377
name: water
def: "An oxygen hydride consisting of an oxygen atom bonded to two hydrogen atoms." [ChEBI:1234]
synonym: "H2O" EXACT [IUPAC]
synonym: "dihydrogen oxide" RELATED [ChEBI]
xref: CAS:7732-18-5
xref: KEGG:C00001
is_a: CHEBI:33579 ! main group molecular entity

[Term]
id: CHEBI:33579
name: main group molecular entity

[Term]
id: CHEBI:OBSOLETE
name: obsolete thing
is_obsolete: true

[Typedef]
id: has_part
name: has part
"""


class TestOboParser:
    def test_skips_typedef_and_keeps_terms(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.obo"
        path.write_text(_OBO_FIXTURE)
        terms = list(iter_terms(path))
        ids = {t.id for t in terms}
        assert ids == {"CHEBI:15377", "CHEBI:33579", "CHEBI:OBSOLETE"}

    def test_term_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.obo"
        path.write_text(_OBO_FIXTURE)
        water = next(t for t in iter_terms(path) if t.id == "CHEBI:15377")
        assert water.name == "water"
        assert water.definition.startswith("An oxygen hydride")
        assert "H2O" in water.synonyms
        assert "dihydrogen oxide" in water.synonyms
        assert water.parents == ("CHEBI:33579",)
        assert "CAS:7732-18-5" in water.xrefs
        assert "KEGG:C00001" in water.xrefs
        assert water.is_obsolete is False

    def test_obsolete_flag(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.obo"
        path.write_text(_OBO_FIXTURE)
        ob = next(t for t in iter_terms(path) if t.id == "CHEBI:OBSOLETE")
        assert ob.is_obsolete is True

    def test_gzip_open(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.obo.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(_OBO_FIXTURE)
        ids = {t.id for t in iter_terms(path)}
        assert "CHEBI:15377" in ids


# ---------------------------------------------------------------------------
# ChEBI loader
# ---------------------------------------------------------------------------


_CHEBI_REAL_FORMAT = """\
[Term]
id: CHEBI:27732
name: caffeine
synonym: "Coffein" RELATED [chemidplus]
xref: cas:58-08-2 {source="cas"}
xref: chembl:CHEMBL113
xref: KEGG.COMPOUND:C07481
xref: pubchem:2519
is_a: CHEBI:33672
"""


class TestChebiParse:
    def test_lowercase_xref_prefixes_kept(self, tmp_path: Path) -> None:
        """Real chebi.obo uses lowercase prefixes and ``{source=...}`` trailers."""
        path = tmp_path / "chebi_full.obo"
        path.write_text(_CHEBI_REAL_FORMAT)
        concepts, _ = chebi.parse(path)
        caf = next(c for c in concepts if c.concept_id == "CHEBI:27732")
        kept = caf.properties["xrefs"]
        # Lowercase cas + chembl must survive case-insensitive filter.
        assert any(x == "cas:58-08-2" for x in kept), kept
        assert any(x.lower().startswith("chembl:") for x in kept), kept
        # Trailing {source="cas"} modifier stripped during OBO parse.
        assert all("{" not in x for x in kept)

    def test_filters_obsolete_and_xrefs(self, tmp_path: Path) -> None:
        path = tmp_path / "chebi.obo"
        path.write_text(_OBO_FIXTURE)
        concepts, rels = chebi.parse(path)
        ids = {c.concept_id for c in concepts}
        assert "CHEBI:OBSOLETE" not in ids
        water = next(c for c in concepts if c.concept_id == "CHEBI:15377")
        # Only retained xrefs (CAS, KEGG) are kept; everything else dropped.
        assert "xrefs" in water.properties
        kept = water.properties["xrefs"]
        assert any(x.startswith("CAS:") for x in kept)
        assert any(x.startswith("KEGG:") for x in kept)
        # Synonyms become alternate labels.
        assert "H2O" in water.alternate_labels
        # External URI uses obolibrary canonical form.
        assert water.external_uri == "http://purl.obolibrary.org/obo/CHEBI_15377"
        # Hierarchy edge survives.
        assert any(r.parent_id == "CHEBI:33579" and r.child_id == "CHEBI:15377" for r in rels)


# ---------------------------------------------------------------------------
# Gene Ontology loader
# ---------------------------------------------------------------------------


_GO_FIXTURE = """\
format-version: 1.2
ontology: go

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process
def: "A biological process is..." [GOC:pdt]

[Term]
id: GO:0006950
name: response to stress
namespace: biological_process
synonym: "stress response" EXACT
is_a: GO:0008150 ! biological_process
alt_id: GO:0006951

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function
"""


class TestGoParse:
    def test_levels_and_alt_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "go.obo"
        path.write_text(_GO_FIXTURE)
        concepts, rels = gene_ontology.parse(path)

        bp = next(c for c in concepts if c.concept_id == "GO:0008150")
        mf = next(c for c in concepts if c.concept_id == "GO:0003674")
        stress = next(c for c in concepts if c.concept_id == "GO:0006950")

        assert bp.level == 0  # biological_process
        assert mf.level == 1  # molecular_function
        assert stress.properties.get("namespace") == "biological_process"
        assert stress.properties.get("alt_ids") == ["GO:0006951"]
        assert "stress response" in stress.alternate_labels
        assert any(r.parent_id == "GO:0008150" and r.child_id == "GO:0006950" for r in rels)


# ---------------------------------------------------------------------------
# MeSH loader
# ---------------------------------------------------------------------------


_MESH_FIXTURE = """\
<?xml version="1.0" encoding="UTF-8"?>
<DescriptorRecordSet>
  <DescriptorRecord>
    <DescriptorUI>D000001</DescriptorUI>
    <DescriptorName><String>Calcimycin</String></DescriptorName>
    <TreeNumberList>
      <TreeNumber>D03.633.100.221.173</TreeNumber>
    </TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <ConceptUI>M0000001</ConceptUI>
        <ConceptName><String>Calcimycin</String></ConceptName>
        <ScopeNote>An ionophore that selectively binds Ca++.</ScopeNote>
        <RegistryNumber>37H9VM9WZL</RegistryNumber>
        <RelatedRegistryNumberList>
          <RelatedRegistryNumber>52665-69-7</RelatedRegistryNumber>
        </RelatedRegistryNumberList>
        <TermList>
          <Term><String>Calcimycin</String></Term>
          <Term><String>A-23187</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord>
    <DescriptorUI>D000123</DescriptorUI>
    <DescriptorName><String>Antibiotics</String></DescriptorName>
    <TreeNumberList>
      <TreeNumber>D03.633.100.221</TreeNumber>
    </TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <ConceptUI>M0000123</ConceptUI>
        <ConceptName><String>Antibiotics</String></ConceptName>
        <RegistryNumber>0</RegistryNumber>
        <TermList>
          <Term><String>Antibiotics</String></Term>
          <Term><String>Antibiotic Drugs</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
</DescriptorRecordSet>
"""


class TestMeshParse:
    def test_parses_descriptors_and_tree_edges(self, tmp_path: Path) -> None:
        path = tmp_path / "desc.xml"
        path.write_text(_MESH_FIXTURE)
        concepts, rels = mesh.parse(path)

        ids = {c.concept_id for c in concepts}
        assert ids == {"D000001", "D000123"}

        cal = next(c for c in concepts if c.concept_id == "D000001")
        assert cal.preferred_label == "Calcimycin"
        assert "A-23187" in cal.alternate_labels
        assert cal.preferred_label not in cal.alternate_labels  # de-duped
        assert cal.definition.startswith("An ionophore")
        assert "37H9VM9WZL" in cal.properties["xrefs"]
        assert "52665-69-7" in cal.properties["xrefs"]
        assert "D03.633.100.221.173" in cal.properties["tree_numbers"]

        # Tree-derived edge: D000123 (Antibiotics, .221) parent of D000001 (.173).
        assert any(r.parent_id == "D000123" and r.child_id == "D000001" for r in rels)

    def test_skips_zero_registry_number(self, tmp_path: Path) -> None:
        path = tmp_path / "desc.xml"
        path.write_text(_MESH_FIXTURE)
        concepts, _ = mesh.parse(path)
        ab = next(c for c in concepts if c.concept_id == "D000123")
        # RegistryNumber=0 is the conventional "no registry" marker; should not
        # leak into xrefs.
        assert "xrefs" not in ab.properties or "0" not in ab.properties.get("xrefs", [])


# ---------------------------------------------------------------------------
# NCBI Taxonomy loader
# ---------------------------------------------------------------------------


def _make_taxdump(path: Path, names: str, nodes: str) -> Path:
    """Build a minimal taxdump.tar.gz with names.dmp and nodes.dmp inline."""
    with tarfile.open(path, "w:gz") as tar:
        for member_name, contents in (("names.dmp", names), ("nodes.dmp", nodes)):
            data = contents.encode("utf-8")
            info = tarfile.TarInfo(name=member_name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


_NCBI_NAMES = (
    "1\t|\troot\t|\t\t|\tscientific name\t|\n"
    "2\t|\tBacteria\t|\t\t|\tscientific name\t|\n"
    "2\t|\teubacteria\t|\t\t|\tsynonym\t|\n"
    "2\t|\tBacteria <bacteria>\t|\t\t|\tequivalent name\t|\n"
    "9606\t|\tHomo sapiens\t|\t\t|\tscientific name\t|\n"
    "9606\t|\thuman\t|\t\t|\tcommon name\t|\n"
)


_NCBI_NODES = (
    "1\t|\t1\t|\tno rank\t|\t\t|\t8\t|\t0\t|\t1\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t\t|\n"
    "2\t|\t1\t|\tsuperkingdom\t|\t\t|\t0\t|\t0\t|\t11\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t\t|\n"
    "9606\t|\t9605\t|\tspecies\t|\tHS\t|\t0\t|\t1\t|\t1\t|\t1\t|\t0\t|\t1\t|\t1\t|\t0\t|\t\t|\n"
)


class TestNcbiTaxParse:
    def test_parses_names_and_edges(self, tmp_path: Path) -> None:
        tar_path = _make_taxdump(tmp_path / "tax.tar.gz", _NCBI_NAMES, _NCBI_NODES)
        concepts, rels = ncbi_taxonomy.parse(tar_path)

        by_id = {c.concept_id: c for c in concepts}
        assert "1" in by_id
        assert by_id["1"].preferred_label == "root"
        assert by_id["2"].preferred_label == "Bacteria"
        # 'eubacteria' (synonym) and the equivalent-name form are alt labels.
        assert "eubacteria" in by_id["2"].alternate_labels
        assert "Bacteria <bacteria>" in by_id["2"].alternate_labels
        assert by_id["9606"].preferred_label == "Homo sapiens"
        assert "human" in by_id["9606"].alternate_labels
        # Rank survives in properties.
        assert by_id["9606"].properties["rank"] == "species"

        # Root self-loop must be filtered.
        assert not any(r.parent_id == "1" and r.child_id == "1" for r in rels)
        # Non-root edges survive.
        assert any(r.parent_id == "1" and r.child_id == "2" for r in rels)

    def test_skips_node_with_no_scientific_name(self, tmp_path: Path) -> None:
        # A node row with no matching scientific-name entry should be dropped.
        names = "1\t|\troot\t|\t\t|\tscientific name\t|\n"
        nodes = (
            "1\t|\t1\t|\tno rank\t|\t\t|\t8\t|\t0\t|\t1\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t\t|\n"
            "999\t|\t1\t|\tspecies\t|\t\t|\t0\t|\t0\t|\t1\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t\t|\n"
        )
        tar_path = _make_taxdump(tmp_path / "tax.tar.gz", names, nodes)
        concepts, _ = ncbi_taxonomy.parse(tar_path)
        ids = {c.concept_id for c in concepts}
        assert ids == {"1"}
