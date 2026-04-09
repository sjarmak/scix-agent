#!/usr/bin/env python3
"""Create gold-annotated JSONL for extraction quality gate evaluation.

Reads the 50-paper sample, applies human gold annotations (entity types:
datasets, instruments, materials, methods), and writes the annotated file.

Rubric applied:
- datasets:     Named data collections, surveys, catalogs, databases, archives.
- instruments:  Specifically named instruments, telescopes, detectors, satellites.
- materials:    Chemical compounds, minerals, substances, material systems.
                Theoretical constructs (fields, states) are NOT materials.
- methods:      Named analytical, computational, or experimental techniques.
                Must be a recognizable procedure, not just a phenomenon or concept.
"""

from __future__ import annotations

import json
from pathlib import Path

SAMPLE_PATH = Path("build-artifacts/manual-eval-sample.jsonl")
OUTPUT_PATH = Path("build-artifacts/manual-eval-annotated.jsonl")

# Gold annotations keyed by bibcode.
# Where a bibcode appears twice (duplicate in sample), same gold applies.
GOLD: dict[str, dict[str, list[str]]] = {
    "1937Phy.....4.1058H": {
        # van der Waals forces between spherical particles (theoretical)
        # "spherical particles" is a geometric abstraction, not a material
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": [],
    },
    "1965JChPh..43..139A": {
        # Molecular-kinetic theory of glass-forming liquids
        "datasets": [],
        "instruments": [],
        "materials": ["glass-forming liquids"],
        "methods": [
            "molecular-kinetic theory",
            "viscosimetric experiments",
            "statistical-mechanical theory",
        ],
    },
    "1965SSASJ..29..677W": {
        # Murphy-Riley P determination in soil extracts
        # NaHCO3 is NOT in the abstract
        "datasets": [],
        "instruments": [],
        "materials": [
            "ascorbic acid",
            "ammonium molybdiphosphate",
            "antimony",
            "SnCl2",
            "phosphorus",
        ],
        "methods": ["ascorbic acid reduction method"],
    },
    "1973ApJ...182L..85K": {
        # Short gamma-ray bursts observed by spacecraft
        "datasets": [],
        "instruments": ["spacecraft"],
        "materials": [],
        "methods": [],
    },
    "1976PhRvD..14.3432T": {
        # Tunneling, Belavin-Polyakov-Schwarz-Tyupkin, chiral symmetry
        # "scalar/spinor/vector fields" are theoretical physics constructs, not materials
        # "Adler-Bell-Jackiw anomaly" is a phenomenon, not a method
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": [],
    },
    "1977JOSA...67..423Y": {
        # Electromagnetic radiation in periodically stratified media
        "datasets": [],
        "instruments": [],
        "materials": ["periodically stratified media"],
        "methods": ["Bloch wave analysis", "band structure analysis"],
    },
    "1981RvMP...53..497D": {
        # 1/f noise in condensed matter (metals)
        "datasets": [],
        "instruments": [],
        "materials": ["metals"],
        "methods": ["spectral density analysis"],
    },
    "1986NuPhB.274..285D": {
        # String propagation on manifold quotients
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["modular invariance analysis"],
    },
    "1986Tectp.123..241D": {
        # Palinspastic maps of Tethys belt evolution
        "datasets": [],
        "instruments": [],
        "materials": ["ophiolites"],
        "methods": ["kinematic synthesis", "paleomagnetic synthesis"],
    },
    "1987QSRv....6..183S": {
        # Sea level and oxygen isotope records over glacial cycle
        # "benthonic records" etc. are data descriptions, not named datasets
        "datasets": [],
        "instruments": [],
        "materials": ["oxygen isotopes"],
        "methods": ["oxygen isotope analysis"],
    },
    "1990JGR....95.2661J": {
        # Rare earth elements in abyssal peridotites via ion microprobe
        "datasets": [],
        "instruments": ["ion microprobe"],
        "materials": [
            "diopsides",
            "abyssal peridotites",
            "garnet peridotite",
            "spinel peridotite",
            "clinopyroxene",
            "rare earth elements",
            "ocean floor basalts",
        ],
        "methods": [
            "ion microprobe analysis",
            "trace element analysis",
            "Rayleigh fractionation modeling",
        ],
    },
    "1990PhR...195..127B": {
        # Brownian particles in disordered environments (theory)
        # "Brownian particles" and "disordered environments" are abstract concepts
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["Green function formalism", "renormalization group methods"],
    },
    "1992ARA&A..30..705B": {
        # Review: dynamics of interacting galaxies
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["numerical modeling"],
    },
    "1992ESRv...32..235L": {
        # Environmental geochemistry sampling and AAS analysis
        "datasets": [],
        "instruments": ["AAS"],
        "materials": [
            "marine sediments",
            "suspended particulate matter",
            "major and trace metals",
            "calcium carbonate",
            "organic matter",
        ],
        "methods": [
            "grain size determinations",
            "AAS determination",
            "chemical partition of metals",
            "determination of readily oxidizable organic matter",
            "calcium carbonate determination",
            "normalization of trace metal data",
        ],
    },
    "1993ApJ...406..122A": {
        # Rho Ophiuchi cloud core, VLA 1623
        # "VLA" here is a source name prefix (VLA 1623 = YSO), NOT the instrument
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["submillimeter continuum mapping"],
    },
    "1993PhRvD..48.3743S": {
        # Black hole complementarity (theoretical)
        # "semiclassical general relativity" is a framework, not a method
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": [],
    },
    "1993Radcb..35..215S": {
        # CALIB radiocarbon calibration program
        "datasets": ["CALIB", "14C calibration data set"],
        "instruments": [],
        "materials": ["Carbon-14"],
        "methods": ["radiocarbon calibration"],
    },
    "1994Ecol...75....2T": {
        # Spatial competition hypothesis, Cedar Creek
        "datasets": ["Cedar Creek Natural History Area"],
        "instruments": [],
        "materials": [],
        "methods": ["spatial competition hypothesis", "experimental propagule addition"],
    },
    "1994RvGeo..32....1C": {
        # Large igneous provinces (LIPs) review
        # "scientific drilling" is a method, not an instrument
        "datasets": [],
        "instruments": [],
        "materials": ["continental flood basalts", "intrusive rocks"],
        "methods": ["seismic imaging", "scientific drilling"],
    },
    "1996Natur.384..335W": {
        # Phase-contrast imaging with polychromatic X-ray sources
        "datasets": [],
        "instruments": ["micro-focus X-ray tube"],
        "materials": [],
        "methods": ["phase-contrast imaging", "phase-contrast radiography"],
    },
    "1997Geo....25..483A": {
        # 8.2 ka Holocene event in Greenland ice cores
        "datasets": ["Greenland ice-core proxies"],
        "instruments": [],
        "materials": ["methane"],
        "methods": [],
    },
    "1997JGR...102.5005Z": {
        # GPS precise point positioning
        "datasets": [],
        "instruments": ["Global Positioning System", "GPS receivers"],
        "materials": [],
        "methods": ["precise point positioning"],
    },
    "1998PhRvB..58.6779G": {
        # Subwavelength hole arrays, surface plasmons
        # "surface plasmons" are quasiparticles, not materials
        "datasets": [],
        "instruments": [],
        "materials": ["metal films"],
        "methods": [],
    },
    "1998PhRvL..80.1121B": {
        # Quantum teleportation
        # "linearly/elliptically polarized state" are quantum states, not materials
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["Bell measurement"],
    },
    "1999ApJ...514....1C": {
        # Baryons in CDM cosmological simulations
        # "cold dark matter", "baryons", "warm/hot gas" are astro concepts, not materials
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["cosmological hydrodynamic simulations"],
    },
    "1999JPcgy..35..403H": {
        # Microalgal biovolume calculation
        "datasets": [],
        "instruments": ["microscopy"],
        "materials": [
            "pelagic microalgae",
            "benthic microalgae",
            "marine microalgae",
            "freshwater microalgae",
        ],
        "methods": [
            "biovolume calculation",
            "geometric shape modeling",
            "microscopical measurements",
        ],
    },
    "2000ApJ...533..631K": {
        # BLR reverberation mapping of PG quasars
        "datasets": ["Palomar-Green quasar sample"],
        "instruments": [],
        "materials": [],
        "methods": ["spectrophotometry", "reverberation mapping", "correlation analysis"],
    },
    "2000PhRvL..84.5094F": {
        # Dark-state polaritons, EIT
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["electromagnetically induced transparency"],
    },
    "2000PhRvL..85.5468C": {
        # Percolation on networks with general degree distributions
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["percolation models", "site percolation", "bond percolation"],
    },
    "2002PhyE...14..115N": {
        # Quantum dot solar cells
        "datasets": [],
        "instruments": [],
        "materials": [
            "quantum dots",
            "InP quantum dots",
            "nanocrystalline TiO2",
            "electron-conducting polymers",
            "hole-conducting polymers",
        ],
        "methods": [],
    },
    "2005ApJ...622..759G": {
        # HEALPix pixelization
        "datasets": [],
        "instruments": [
            "BOOMERANG",
            "WMAP",
            "Planck",
            "Herschel",
            "SAFIR",
            "Beyond Einstein inflation probe",
        ],
        "materials": [],
        "methods": ["Hierarchical Equal Area isoLatitude Pixelization", "pixelization"],
    },
    "2007Bioin..23.2633B": {
        # TASSEL software for association analysis
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": [
            "general linear model",
            "mixed linear model",
            "linkage disequilibrium statistics",
            "principal component analysis",
            "imputation",
        ],
    },
    "2007MRRMR.636..178R": {
        # Disinfection by-products review (many chemicals)
        "datasets": [],
        "instruments": [],
        "materials": [
            "chlorine",
            "ozone",
            "chlorine dioxide",
            "chloramines",
            "naturally occurring organic matter",
            "bromide",
            "iodide",
            "halonitromethanes",
            "iodo-acids",
            "iodo-trihalomethanes",
            "halomethanes",
            "halofuranones",
            "MX",
            "brominated MX",
            "haloamides",
            "haloacetonitriles",
            "tribromopyrrole",
            "aldehydes",
            "N-nitrosodimethylamine",
            "nitrosamines",
            "bromodichloromethane",
            "dichloroacetic acid",
            "dibromoacetic acid",
            "bromate",
            "formaldehyde",
            "acetaldehyde",
            "chloral hydrate",
            "chloroacetaldehyde",
            "brominated DBPs",
            "chlorinated compounds",
            "iodinated DBPs",
            "trihalomethanes",
            "total organic halogen",
            "assimilable organic carbon",
        ],
        "methods": [
            "genotoxicity assessment",
            "mutagenicity testing",
            "epidemiologic study",
        ],
    },
    "2007Sci...315..361P": {
        # Litter decomposition and nitrogen release
        "datasets": [],
        "instruments": [],
        "materials": ["leaf litter", "roots"],
        "methods": [],
    },
    "2007Sci...317.1513L": {
        # Coupled human and natural systems
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["case studies"],
    },
    "2008Sci...319..948H": {
        # Human impacts on marine ecosystems
        # "global data sets of anthropogenic drivers" is descriptive, not a named dataset
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["ecosystem-specific multiscale spatial model"],
    },
    "2009PhRvD..79f4016C": {
        # Quasinormal modes and null geodesics
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["Lyapunov exponent computation"],
    },
    "2010Sci...327.1607M": {
        # Complex oxide interfaces
        "datasets": [],
        "instruments": [],
        "materials": ["complex oxides"],
        "methods": [],
    },
    "2011arXiv1101.0593L": {
        # LHC Higgs Cross Sections Working Group report
        "datasets": [],
        "instruments": ["LHC"],
        "materials": [],
        "methods": ["higher-order corrections"],
    },
    "2012RvMP...84.1419M": {
        # Wannier functions review
        # "Wannier functions" and "Bloch orbitals" are mathematical constructs but also methods
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": [
            "Wannier functions",
            "first-principles calculation",
            "Wannier interpolation",
        ],
    },
    "2013JHEP...12..089B": {
        # Higgs potential, SM parameter extraction
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": [
            "2-loop NNLO precision calculation",
            "3-loop NNLO RGE precision",
            "renormalization group equations",
        ],
    },
    "2013NatNa...8..634J": {
        # Valley coherence in monolayer WSe2
        # "excitons" and "trions" are quasiparticles, not materials
        "datasets": [],
        "instruments": [],
        "materials": ["monolayer WSe2"],
        "methods": ["photoluminescence spectroscopy"],
    },
    "2014A&A...562A..71B": {
        # 714 F and G dwarfs, Galactic disk stellar abundances
        "datasets": ["CDS"],
        "instruments": [
            "FEROS spectrograph",
            "ESO 1.5 m telescope",
            "ESO 2.2 m telescope",
            "SOFIN spectrograph",
            "FIES spectrograph",
            "Nordic Optical Telescope",
            "UVES spectrograph",
            "ESO Very Large Telescope",
            "HARPS spectrograph",
            "ESO 3.6 m telescope",
            "MIKE spectrograph",
            "Magellan Clay telescope",
        ],
        "materials": [
            "oxygen",
            "sodium",
            "magnesium",
            "aluminum",
            "silicon",
            "calcium",
            "titanium",
            "chromium",
            "iron",
            "nickel",
            "zinc",
            "yttrium",
            "barium",
        ],
        "methods": [
            "high-resolution spectroscopy",
            "equivalent widths analysis",
            "local thermodynamical equilibrium",
            "non-LTE corrections",
            "kinematical selection",
            "ionisation and excitation balance",
        ],
    },
    "2014NanoL..14.3347B": {
        # Few-layer black phosphorus photodetection
        # "2D material" is too generic
        "datasets": [],
        "instruments": [],
        "materials": ["black phosphorus", "few-layer black phosphorus"],
        "methods": ["mechanical exfoliation"],
    },
    "2014RPPh...77i4001R": {
        # Quantum non-Markovianity review
        # "divisibility property" is a mathematical concept, not a method
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": [],
    },
    "2015NatMa..14..193X": {
        # Perovskite switchable photocurrent
        # "organometal trihalide perovskite" NOT in this abstract
        "datasets": [],
        "instruments": [],
        "materials": ["organic-inorganic perovskite", "hybrid perovskites"],
        "methods": ["electric field poling"],
    },
    "2017NatEn...217102Z": {
        # Perovskite defect passivation
        "datasets": [],
        "instruments": [],
        "materials": [
            "quaternary ammonium halides",
            "organic-inorganic halide perovskite",
            "hybrid perovskite",
        ],
        "methods": ["density-function-theory calculation"],
    },
    "2017NatMa..16..572H": {
        # Garnet solid-state electrolyte with ALD Al2O3 interface
        "datasets": [],
        "instruments": [],
        "materials": [
            "garnet-type solid-state electrolytes",
            "aluminium oxide",
            "Al2O3",
            "Li7La2.75Ca0.25Zr1.75Nb0.25O12",
            "LLCZN",
            "lithium metal anode",
            "lithiated-alumina",
        ],
        "methods": ["atomic layer deposition"],
    },
    "2017Natur.549...43L": {
        # Satellite-based quantum key distribution
        # "weak coherent pulses" is a quantum optics concept, not a material
        "datasets": [],
        "instruments": ["low-Earth-orbit satellite"],
        "materials": ["optical fibres"],
        "methods": ["quantum key distribution", "decoy-state QKD"],
    },
    "2022PTEP.2022h3C01W": {
        # PDG Review of Particle Physics
        # Fundamental particles are NOT materials in the chemical sense
        # "Particle Detectors" and "Colliders" are generic categories, not specific instruments
        "datasets": [],
        "instruments": [],
        "materials": [],
        "methods": ["Machine Learning", "Probability and Statistics"],
    },
}


EMPTY_GOLD: dict[str, list[str]] = {
    "datasets": [],
    "instruments": [],
    "materials": [],
    "methods": [],
}


def main() -> None:
    records: list[dict] = []
    with SAMPLE_PATH.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))

    print(f"Read {len(records)} records from {SAMPLE_PATH}")

    missing: list[str] = []
    annotated: list[dict] = []
    for r in records:
        bib = r.get("bibcode")
        if bib is None:
            print(f"WARNING: record missing bibcode, skipping: {r}")
            continue
        gold = GOLD.get(bib, EMPTY_GOLD)
        if bib not in GOLD:
            missing.append(bib)
        annotated.append({**r, "gold": gold})

    if missing:
        print(f"WARNING: {len(missing)} bibcodes have no gold annotation: {missing}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in annotated:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(annotated)} annotated records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
