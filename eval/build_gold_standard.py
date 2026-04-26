"""Helper that authors the gold standard JSONL.

Run once to produce eval/claim_extraction_gold_standard.jsonl.
This script is a build artifact for the gold standard; the authoritative
file is the JSONL output. Kept in repo so future curators can extend.
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "claim_extraction_gold_standard.jsonl"


def make_entry(bibcode, section_index, paragraph_index, paragraph_text,
               discipline, claim_specs):
    """Build an entry; claim_specs is a list of dicts with keys:
       claim_text, claim_type, subject, predicate, object, anchor (substring of paragraph_text).

    The anchor's start/end are computed by str.index on paragraph_text.
    """
    expected_claims = []
    for spec in claim_specs:
        anchor = spec["anchor"]
        start = paragraph_text.index(anchor)
        end = start + len(anchor)
        assert end > start, f"empty span for {spec}"
        assert paragraph_text[start:end] == anchor
        expected_claims.append({
            "claim_text": spec["claim_text"],
            "claim_type": spec["claim_type"],
            "subject": spec["subject"],
            "predicate": spec["predicate"],
            "object": spec["object"],
            "char_span_start": start,
            "char_span_end": end,
        })
    return {
        "bibcode": bibcode,
        "section_index": section_index,
        "paragraph_index": paragraph_index,
        "paragraph_text": paragraph_text,
        "expected_claims": expected_claims,
        "discipline": discipline,
    }


ENTRIES = []

# ---------------- ASTROPHYSICS (5 entries) ----------------

p1 = (
    "We measured the Hubble constant from the analysis of 42 Type Ia supernovae "
    "in the local distance ladder, obtaining H0 = 73.04 +/- 1.04 km/s/Mpc. "
    "This value is in 5-sigma tension with the Planck 2018 inference of "
    "67.4 +/- 0.5 km/s/Mpc derived from the cosmic microwave background. "
    "We hypothesize that early dark energy could reconcile the two measurements."
)
ENTRIES.append(make_entry(
    "GOLD2024ApJ...001A...01", 3, 0, p1, "astrophysics",
    [
        {
            "claim_text": "H0 measured to be 73.04 +/- 1.04 km/s/Mpc from 42 Type Ia SNe",
            "claim_type": "factual",
            "subject": "Hubble constant",
            "predicate": "measured_as",
            "object": "73.04 +/- 1.04 km/s/Mpc",
            "anchor": "H0 = 73.04 +/- 1.04 km/s/Mpc",
        },
        {
            "claim_text": "Planck 2018 inferred H0 = 67.4 +/- 0.5 km/s/Mpc from CMB",
            "claim_type": "cited_from_other",
            "subject": "Planck 2018",
            "predicate": "inferred",
            "object": "H0 = 67.4 +/- 0.5 km/s/Mpc",
            "anchor": "Planck 2018 inference of "
                      "67.4 +/- 0.5 km/s/Mpc",
        },
        {
            "claim_text": "Early dark energy could reconcile the H0 tension",
            "claim_type": "speculative",
            "subject": "early dark energy",
            "predicate": "could_reconcile",
            "object": "Hubble tension",
            "anchor": "early dark energy could reconcile the two measurements",
        },
    ],
))

p2 = (
    "We used the JWST NIRSpec instrument with the G395M grating to obtain "
    "moderate-resolution spectra of 18 galaxies at z > 7. Redshifts were "
    "determined by simultaneous fitting of [OIII] 5007 and Hbeta emission "
    "lines using the BAGPIPES pipeline. Stellar masses were inferred via "
    "SED fitting with a Chabrier IMF."
)
ENTRIES.append(make_entry(
    "GOLD2024ApJ...001A...02", 2, 1, p2, "astrophysics",
    [
        {
            "claim_text": "JWST NIRSpec G395M used to observe 18 z>7 galaxies",
            "claim_type": "methodological",
            "subject": "authors",
            "predicate": "used",
            "object": "JWST NIRSpec G395M grating",
            "anchor": "We used the JWST NIRSpec instrument with the G395M grating",
        },
        {
            "claim_text": "Redshifts fit with [OIII]5007 and Hbeta via BAGPIPES",
            "claim_type": "methodological",
            "subject": "redshifts",
            "predicate": "determined_by",
            "object": "BAGPIPES line fitting",
            "anchor": "fitting of [OIII] 5007 and Hbeta emission "
                      "lines using the BAGPIPES pipeline",
        },
    ],
))

p3 = (
    "Our convolutional neural network classifier achieves 94.2% accuracy on "
    "the Galaxy Zoo morphology test set, outperforming the previous "
    "state-of-the-art ResNet-50 baseline (89.7%) by 4.5 percentage points. "
    "Training required 32 GPU-hours on an NVIDIA A100. The improvement is "
    "most pronounced for edge-on spirals, where confusion with lenticulars "
    "drops from 18% to 6%."
)
ENTRIES.append(make_entry(
    "GOLD2024ApJ...001A...03", 4, 0, p3, "astrophysics",
    [
        {
            "claim_text": "CNN classifier reaches 94.2% accuracy on Galaxy Zoo test set",
            "claim_type": "factual",
            "subject": "our CNN classifier",
            "predicate": "achieves",
            "object": "94.2% accuracy",
            "anchor": "94.2% accuracy on "
                      "the Galaxy Zoo morphology test set",
        },
        {
            "claim_text": "Our model outperforms ResNet-50 baseline by 4.5 points",
            "claim_type": "comparative",
            "subject": "our CNN",
            "predicate": "outperforms",
            "object": "ResNet-50 baseline by 4.5 percentage points",
            "anchor": "outperforming the previous "
                      "state-of-the-art ResNet-50 baseline (89.7%) by 4.5 percentage points",
        },
    ],
))

p4 = (
    "The detection of a continuous gravitational wave signal at 250 Hz from "
    "the direction of the Galactic Center would constrain neutron star "
    "ellipticity below 10^-7. Future observing runs of LIGO A+ and Einstein "
    "Telescope are expected to probe this regime. We plan to extend the "
    "F-statistic search to the full O5 dataset once available."
)
ENTRIES.append(make_entry(
    "GOLD2024ApJ...001A...04", 6, 2, p4, "astrophysics",
    [
        {
            "claim_text": "A 250 Hz CW detection from Galactic Center constrains NS ellipticity below 1e-7",
            "claim_type": "speculative",
            "subject": "continuous GW detection at 250 Hz",
            "predicate": "would_constrain",
            "object": "neutron star ellipticity < 1e-7",
            "anchor": "would constrain neutron star "
                      "ellipticity below 10^-7",
        },
        {
            "claim_text": "Plan to extend F-statistic search to LIGO O5 data",
            "claim_type": "speculative",
            "subject": "authors",
            "predicate": "plan_to",
            "object": "extend F-statistic search to O5",
            "anchor": "We plan to extend the "
                      "F-statistic search to the full O5 dataset",
        },
    ],
))

p5 = (
    "Spectroscopic confirmation places the host galaxy of FRB 20200120E at "
    "a distance of 3.6 Mpc, making it the closest known fast radio burst "
    "source. Bhardwaj et al. (2021) localized the burst to a globular "
    "cluster in M81. We follow their astrometric solution and adopt the "
    "same fiducial distance."
)
ENTRIES.append(make_entry(
    "GOLD2024ApJ...001A...05", 1, 3, p5, "astrophysics",
    [
        {
            "claim_text": "FRB 20200120E host distance is 3.6 Mpc",
            "claim_type": "factual",
            "subject": "FRB 20200120E host galaxy",
            "predicate": "located_at",
            "object": "3.6 Mpc",
            "anchor": "host galaxy of FRB 20200120E at "
                      "a distance of 3.6 Mpc",
        },
        {
            "claim_text": "Bhardwaj et al. 2021 localized the FRB to an M81 globular cluster",
            "claim_type": "cited_from_other",
            "subject": "Bhardwaj et al. (2021)",
            "predicate": "localized",
            "object": "FRB 20200120E to M81 globular cluster",
            "anchor": "Bhardwaj et al. (2021) localized the burst to a globular "
                      "cluster in M81",
        },
    ],
))

# ---------------- PLANETARY SCIENCE (5 entries) ----------------

p6 = (
    "Mass spectrometer measurements from the Cassini INMS instrument show "
    "molecular hydrogen abundance of 0.4-1.4% in the Enceladus plume. This "
    "is consistent with serpentinization in the subsurface ocean. Waite "
    "et al. (2017) first reported this detection. We refine the mixing "
    "ratios using updated calibration."
)
ENTRIES.append(make_entry(
    "GOLD2024Icar...002P...01", 3, 0, p6, "planetary_science",
    [
        {
            "claim_text": "INMS measured H2 abundance of 0.4-1.4% in Enceladus plume",
            "claim_type": "factual",
            "subject": "Cassini INMS",
            "predicate": "measured",
            "object": "H2 = 0.4-1.4% in Enceladus plume",
            "anchor": "molecular hydrogen abundance of 0.4-1.4% in the Enceladus plume",
        },
        {
            "claim_text": "Waite et al. 2017 first reported H2 in Enceladus plume",
            "claim_type": "cited_from_other",
            "subject": "Waite et al. (2017)",
            "predicate": "first_reported",
            "object": "H2 detection in Enceladus plume",
            "anchor": "Waite "
                      "et al. (2017) first reported this detection",
        },
        {
            "claim_text": "We refined mixing ratios with updated calibration",
            "claim_type": "methodological",
            "subject": "authors",
            "predicate": "refined",
            "object": "mixing ratios with updated calibration",
            "anchor": "We refine the mixing "
                      "ratios using updated calibration",
        },
    ],
))

p7 = (
    "We compared crater size-frequency distributions from CTX imagery across "
    "three Hesperian terrains in Arabia Terra. The mean model age of "
    "3.4 +/- 0.2 Gyr is older than the 2.9 Gyr estimate of Tanaka et al. "
    "(2014) by approximately 500 Myr. Differences arise primarily from our "
    "use of the updated Hartmann production function."
)
ENTRIES.append(make_entry(
    "GOLD2024Icar...002P...02", 2, 1, p7, "planetary_science",
    [
        {
            "claim_text": "Mean Hesperian Arabia Terra model age is 3.4 +/- 0.2 Gyr",
            "claim_type": "factual",
            "subject": "Arabia Terra Hesperian terrains",
            "predicate": "have_age",
            "object": "3.4 +/- 0.2 Gyr",
            "anchor": "mean model age of "
                      "3.4 +/- 0.2 Gyr",
        },
        {
            "claim_text": "Our age is 500 Myr older than Tanaka et al. 2014",
            "claim_type": "comparative",
            "subject": "our model age",
            "predicate": "is_older_than",
            "object": "Tanaka et al. 2014 by ~500 Myr",
            "anchor": "older than the 2.9 Gyr estimate of Tanaka et al. "
                      "(2014) by approximately 500 Myr",
        },
        {
            "claim_text": "Used updated Hartmann production function for chronology",
            "claim_type": "methodological",
            "subject": "authors",
            "predicate": "used",
            "object": "updated Hartmann production function",
            "anchor": "use of the updated Hartmann production function",
        },
    ],
))

p8 = (
    "The Perseverance rover collected 24 sample tubes from the Jezero "
    "crater delta deposits between February 2021 and December 2023. Sample "
    "021 (Lefroy Bay) contains organic molecules detected by the SHERLOC "
    "instrument at concentrations of 10-100 ppb. Future Mars Sample Return "
    "missions will enable Earth-laboratory analysis of these tubes."
)
ENTRIES.append(make_entry(
    "GOLD2024Icar...002P...03", 1, 0, p8, "planetary_science",
    [
        {
            "claim_text": "Perseverance collected 24 sample tubes from Jezero delta",
            "claim_type": "factual",
            "subject": "Perseverance rover",
            "predicate": "collected",
            "object": "24 sample tubes from Jezero delta",
            "anchor": "Perseverance rover collected 24 sample tubes",
        },
        {
            "claim_text": "SHERLOC detected organics at 10-100 ppb in Lefroy Bay sample",
            "claim_type": "factual",
            "subject": "SHERLOC instrument",
            "predicate": "detected",
            "object": "organic molecules at 10-100 ppb",
            "anchor": "organic molecules detected by the SHERLOC "
                      "instrument at concentrations of 10-100 ppb",
        },
        {
            "claim_text": "Future MSR will enable Earth-lab analysis",
            "claim_type": "speculative",
            "subject": "Mars Sample Return",
            "predicate": "will_enable",
            "object": "Earth-laboratory analysis",
            "anchor": "Future Mars Sample Return "
                      "missions will enable Earth-laboratory analysis",
        },
    ],
))

p9 = (
    "We applied a thermal inertia retrieval algorithm to OSIRIS-REx OTES "
    "thermal emission spectra of asteroid Bennu. The derived global mean "
    "thermal inertia of 320 +/- 30 J m^-2 K^-1 s^-1/2 is consistent with "
    "a regolith of cm-scale particles. This contrasts with the higher "
    "values (600-700) reported for asteroid Ryugu by Okada et al. (2020)."
)
ENTRIES.append(make_entry(
    "GOLD2024Icar...002P...04", 4, 2, p9, "planetary_science",
    [
        {
            "claim_text": "Thermal inertia retrieval applied to OSIRIS-REx OTES spectra of Bennu",
            "claim_type": "methodological",
            "subject": "authors",
            "predicate": "applied",
            "object": "thermal inertia retrieval algorithm to OTES",
            "anchor": "We applied a thermal inertia retrieval algorithm to OSIRIS-REx OTES "
                      "thermal emission spectra",
        },
        {
            "claim_text": "Bennu mean thermal inertia is 320 +/- 30 SI units",
            "claim_type": "factual",
            "subject": "Bennu",
            "predicate": "has_thermal_inertia",
            "object": "320 +/- 30 J m^-2 K^-1 s^-1/2",
            "anchor": "global mean "
                      "thermal inertia of 320 +/- 30 J m^-2 K^-1 s^-1/2",
        },
        {
            "claim_text": "Bennu thermal inertia is lower than Ryugu (Okada et al. 2020)",
            "claim_type": "comparative",
            "subject": "Bennu thermal inertia",
            "predicate": "contrasts_with",
            "object": "Ryugu values 600-700 (Okada 2020)",
            "anchor": "contrasts with the higher "
                      "values (600-700) reported for asteroid Ryugu by Okada et al. (2020)",
        },
    ],
))

p10 = (
    "Atmospheric methane variability on Mars has been a subject of debate "
    "since the early 2000s. Webster et al. (2021) reported diurnal "
    "variations of 0.2-0.7 ppbv from MSL TLS measurements at Gale crater. "
    "Our independent retrieval from ExoMars TGO data finds an upper limit "
    "of 0.05 ppbv at 3-sigma confidence in the same region."
)
ENTRIES.append(make_entry(
    "GOLD2024Icar...002P...05", 5, 1, p10, "planetary_science",
    [
        {
            "claim_text": "Webster et al. 2021 reported 0.2-0.7 ppbv methane diurnal variations at Gale",
            "claim_type": "cited_from_other",
            "subject": "Webster et al. (2021)",
            "predicate": "reported",
            "object": "0.2-0.7 ppbv diurnal CH4 variations",
            "anchor": "Webster et al. (2021) reported diurnal "
                      "variations of 0.2-0.7 ppbv from MSL TLS measurements at Gale crater",
        },
        {
            "claim_text": "Our TGO retrieval finds upper limit of 0.05 ppbv CH4 at 3-sigma",
            "claim_type": "factual",
            "subject": "our TGO retrieval",
            "predicate": "finds",
            "object": "CH4 upper limit 0.05 ppbv at 3-sigma",
            "anchor": "upper limit "
                      "of 0.05 ppbv at 3-sigma confidence",
        },
        {
            "claim_text": "Our TGO upper limit contradicts MSL TLS detections",
            "claim_type": "comparative",
            "subject": "our TGO retrieval",
            "predicate": "contrasts_with",
            "object": "MSL TLS Webster 2021 detections",
            "anchor": "Our independent retrieval from ExoMars TGO data finds an upper limit "
                      "of 0.05 ppbv at 3-sigma confidence in the same region",
        },
    ],
))

# ---------------- EARTH SCIENCE (5 entries) ----------------

p11 = (
    "Global mean sea level rose by 4.62 +/- 0.10 mm/yr between 2014 and "
    "2023, as measured by the Jason-3 and Sentinel-6 altimeter time series. "
    "This represents an acceleration of 0.084 mm/yr^2 relative to the "
    "1993-2013 mean rate of 3.1 mm/yr. Continued ice sheet mass loss is the "
    "dominant contributor."
)
ENTRIES.append(make_entry(
    "GOLD2024JGR....003E...01", 2, 0, p11, "earth_science",
    [
        {
            "claim_text": "Global mean sea level rose at 4.62 +/- 0.10 mm/yr 2014-2023",
            "claim_type": "factual",
            "subject": "global mean sea level",
            "predicate": "rose_at",
            "object": "4.62 +/- 0.10 mm/yr (2014-2023)",
            "anchor": "Global mean sea level rose by 4.62 +/- 0.10 mm/yr between 2014 and "
                      "2023",
        },
        {
            "claim_text": "SLR has accelerated by 0.084 mm/yr^2 relative to 1993-2013",
            "claim_type": "comparative",
            "subject": "sea level rise rate",
            "predicate": "accelerated_by",
            "object": "0.084 mm/yr^2 vs 1993-2013 baseline",
            "anchor": "acceleration of 0.084 mm/yr^2 relative to the "
                      "1993-2013 mean rate of 3.1 mm/yr",
        },
    ],
))

p12 = (
    "We trained a U-Net convolutional model on 12,000 Sentinel-2 tiles to "
    "segment surface water extent at 10 m resolution. The model achieves "
    "an F1 score of 0.93 on the held-out test set, exceeding the previously "
    "published JRC Global Surface Water mask (F1 = 0.86) on the same "
    "validation regions. Training used the Adam optimizer with cross-entropy "
    "loss."
)
ENTRIES.append(make_entry(
    "GOLD2024JGR....003E...02", 3, 1, p12, "earth_science",
    [
        {
            "claim_text": "U-Net trained on 12,000 Sentinel-2 tiles for water segmentation",
            "claim_type": "methodological",
            "subject": "authors",
            "predicate": "trained",
            "object": "U-Net on 12,000 Sentinel-2 tiles",
            "anchor": "We trained a U-Net convolutional model on 12,000 Sentinel-2 tiles",
        },
        {
            "claim_text": "Model achieves F1=0.93 on held-out test set",
            "claim_type": "factual",
            "subject": "our U-Net model",
            "predicate": "achieves",
            "object": "F1 = 0.93",
            "anchor": "F1 score of 0.93 on the held-out test set",
        },
        {
            "claim_text": "Our model exceeds JRC Global Surface Water mask F1 (0.86)",
            "claim_type": "comparative",
            "subject": "our U-Net",
            "predicate": "exceeds",
            "object": "JRC GSW mask (F1 = 0.86)",
            "anchor": "exceeding the previously "
                      "published JRC Global Surface Water mask (F1 = 0.86)",
        },
    ],
))

p13 = (
    "Permafrost extent in the Northern Hemisphere has decreased by "
    "approximately 1.6 million km^2 between 1990 and 2020 according to "
    "Obu et al. (2019) and our updated CMIP6 simulations. We project an "
    "additional 4-7 million km^2 loss by 2100 under SSP5-8.5. Methane "
    "release from thawing permafrost remains highly uncertain."
)
ENTRIES.append(make_entry(
    "GOLD2024JGR....003E...03", 4, 0, p13, "earth_science",
    [
        {
            "claim_text": "Obu et al. 2019 reports 1.6M km^2 permafrost loss 1990-2020",
            "claim_type": "cited_from_other",
            "subject": "Obu et al. (2019)",
            "predicate": "reports",
            "object": "1.6M km^2 permafrost loss 1990-2020",
            "anchor": "1.6 million km^2 between 1990 and "
                      "2020 according to "
                      "Obu et al. (2019)",
        },
        {
            "claim_text": "We project 4-7 million km^2 additional permafrost loss by 2100 SSP5-8.5",
            "claim_type": "speculative",
            "subject": "our CMIP6 projections",
            "predicate": "project",
            "object": "4-7 million km^2 loss by 2100 under SSP5-8.5",
            "anchor": "project an "
                      "additional 4-7 million km^2 loss by 2100 under SSP5-8.5",
        },
        {
            "claim_text": "Methane release from thawing permafrost remains uncertain",
            "claim_type": "speculative",
            "subject": "permafrost methane release",
            "predicate": "is",
            "object": "highly uncertain",
            "anchor": "Methane "
                      "release from thawing permafrost remains highly uncertain",
        },
    ],
))

p14 = (
    "We applied principal component analysis to 30 years of monthly NDVI "
    "from MODIS Terra to identify dominant modes of vegetation variability "
    "across the Sahel. The first PC explains 41% of the variance and "
    "correlates strongly (r = 0.78, p < 0.001) with annual rainfall from "
    "the CHIRPS dataset."
)
ENTRIES.append(make_entry(
    "GOLD2024JGR....003E...04", 2, 2, p14, "earth_science",
    [
        {
            "claim_text": "PCA applied to 30 years of MODIS NDVI over the Sahel",
            "claim_type": "methodological",
            "subject": "authors",
            "predicate": "applied",
            "object": "PCA to MODIS Terra monthly NDVI",
            "anchor": "We applied principal component analysis to 30 years of monthly NDVI "
                      "from MODIS Terra",
        },
        {
            "claim_text": "First PC explains 41% of NDVI variance over the Sahel",
            "claim_type": "factual",
            "subject": "first principal component",
            "predicate": "explains",
            "object": "41% of NDVI variance",
            "anchor": "first PC explains 41% of the variance",
        },
        {
            "claim_text": "First PC correlates with CHIRPS rainfall (r=0.78, p<0.001)",
            "claim_type": "factual",
            "subject": "first PC of NDVI",
            "predicate": "correlates_with",
            "object": "CHIRPS annual rainfall (r=0.78)",
            "anchor": "correlates strongly (r = 0.78, p < 0.001) with annual rainfall from "
                      "the CHIRPS dataset",
        },
    ],
))

p15 = (
    "Atmospheric CO2 at Mauna Loa exceeded 420 ppm for the first time in "
    "May 2022, reflecting the continued upward trend documented by NOAA "
    "GML. The 2023 annual mean of 421.08 ppm is consistent with the "
    "Friedlingstein et al. (2023) Global Carbon Budget projection. Future "
    "stabilization will require deep emission cuts beyond current NDCs."
)
ENTRIES.append(make_entry(
    "GOLD2024JGR....003E...05", 1, 3, p15, "earth_science",
    [
        {
            "claim_text": "Mauna Loa CO2 first exceeded 420 ppm in May 2022",
            "claim_type": "factual",
            "subject": "Mauna Loa atmospheric CO2",
            "predicate": "exceeded",
            "object": "420 ppm in May 2022",
            "anchor": "Atmospheric CO2 at Mauna Loa exceeded 420 ppm for the first time in "
                      "May 2022",
        },
        {
            "claim_text": "Friedlingstein et al. 2023 Global Carbon Budget projected ~421 ppm for 2023",
            "claim_type": "cited_from_other",
            "subject": "Friedlingstein et al. (2023)",
            "predicate": "projected",
            "object": "2023 annual mean ~421 ppm",
            "anchor": "Friedlingstein et al. (2023) Global Carbon Budget projection",
        },
        {
            "claim_text": "Stabilization requires deeper emission cuts than current NDCs",
            "claim_type": "speculative",
            "subject": "CO2 stabilization",
            "predicate": "requires",
            "object": "emission cuts beyond NDCs",
            "anchor": "Future "
                      "stabilization will require deep emission cuts beyond current NDCs",
        },
    ],
))


def main() -> None:
    with OUT.open("w", encoding="utf-8") as fh:
        for entry in ENTRIES:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ENTRIES)} entries to {OUT}")


if __name__ == "__main__":
    main()
