"""Tests for u09 Adeft-style per-acronym disambiguators.

Three acronyms are trained on synthetic long-form contexts:

* **HST** — "Hubble Space Telescope" vs "Highest Single Trade"
* **JET** — "Joint European Torus" vs "Jet Engine Test"
* **AI**  — "Artificial Intelligence" vs "Aortic Insufficiency"

For each, we hand-generate 50 positive examples per long-form, train on
80%, and assert ≥ 90% accuracy on the held-out 20%. The u09 acceptance
criterion requires ≥ 90% for at least 3 of 20 acronyms; we verify 3 here
(the 20-acronym corpus arrives with u11's real harvest).
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import pytest

from scix.adeft_disambig import (
    AdeftClassifier,
    load_classifier,
    save_classifier,
    train_classifier,
)

# ---------------------------------------------------------------------------
# Synthetic training corpus generators
# ---------------------------------------------------------------------------


def _make_hst_examples(seed: int = 17) -> list[tuple[str, str]]:
    """50 positives per long-form for HST."""
    rng = random.Random(seed)

    hubble_templates = [
        "We analyzed archival {HST} imaging of distant quasars, finding {x} faint sources.",
        "The {HST} WFC3 camera revealed morphological details of {x} galaxies.",
        "Using deep {HST} exposures, we measured the redshift of {x} clusters.",
        "The cosmic shear signal from {HST} observations constrains dark matter on {x} scales.",
        "Spectroscopic follow-up of {HST} targets showed emission lines at {x} wavelengths.",
        "{HST} photometry in six bands was combined with ground-based spectra.",
        "The ACS camera aboard {HST} resolved individual stars in nearby dwarfs.",
        "Our {HST}-selected sample of high-redshift galaxies has a mean mass of {x} solar.",
        "We combine {HST} parallax measurements with Gaia DR3 to calibrate Cepheid distances.",
        "Ultraviolet imaging with {HST} traces young star-forming complexes in {x} regions.",
    ]
    finance_templates = [
        "The reported {HST} in Q3 was ${x}M, up 12% year over year.",
        "Regulatory filings disclose the {HST} for each trading day in the window.",
        "A {HST} of ${x}B occurred during the Asia-Pacific session.",
        "The exchange publishes the {HST} at market close for transparency.",
        "Risk models use the {HST} as an extreme-tail stress input.",
        "The desk's {HST} for the quarter set a new internal record.",
        "Auditors reconciled the {HST} against counterparty confirmations.",
        "A sharp spike in the {HST} triggered the compliance review.",
        "The {HST} for currency block trades reached ${x}M overnight.",
        "Daily risk reports highlight the {HST} alongside VaR and expected shortfall.",
    ]
    pool_x = ["bright", "unusual", "several", "many", "thousands of"]
    pool_x_finance = ["1.2", "450", "900", "37", "5.6"]

    examples: list[tuple[str, str]] = []
    for _ in range(50):
        tpl = rng.choice(hubble_templates)
        x = rng.choice(pool_x)
        examples.append((tpl.format(HST="HST", x=x), "Hubble Space Telescope"))
    for _ in range(50):
        tpl = rng.choice(finance_templates)
        x = rng.choice(pool_x_finance)
        examples.append((tpl.format(HST="HST", x=x), "Highest Single Trade"))
    rng.shuffle(examples)
    return examples


def _make_jet_examples(seed: int = 29) -> list[tuple[str, str]]:
    """50 positives per long-form for JET."""
    rng = random.Random(seed)

    fusion_templates = [
        "Plasma discharges at {JET} achieved record deuterium-tritium fusion yields.",
        "The {JET} tokamak operated with a tungsten divertor for the {x} campaign.",
        "Neutron flux measurements at {JET} validated ITER projections.",
        "At {JET}, researchers achieved {x} MJ of fusion energy per shot.",
        "The {JET} vacuum vessel was baked to {x} degrees before operation.",
        "Edge-localized modes at {JET} were suppressed via resonant magnetic perturbations.",
        "Deuterium-tritium fusion at {JET} produced {x} megawatts of power.",
        "The {JET} divertor uses a water-cooled tungsten monoblock design.",
        "Gyrokinetic simulations benchmarked against {JET} discharges show strong agreement.",
        "The {JET} neutral beam injectors deliver {x} MW of auxiliary heating.",
    ]
    engine_templates = [
        "The {JET} measured the thrust-to-weight ratio at {x} knots.",
        "Procedural reviews of the {JET} identified a loose compressor blade after {x} hours.",
        "Engineers ran a 48-hour {JET} on the new turbofan core.",
        "A {JET} certified the afterburner reliability at {x} thousand feet.",
        "The {JET} flight log noted an EGT margin of {x} degrees at takeoff.",
        "During the {JET}, fuel burn was {x} pounds per hour at cruise.",
        "Manufacturers require a {JET} before delivering military airframes.",
        "The {JET} recorded a vibration spike during the {x} pass.",
        "Post-{JET} inspections revealed no cracks in the turbine blades.",
        "The {JET} bench simulates flight conditions up to Mach {x}.",
    ]
    pool_x_fusion = ["2021", "record", "sustained", "1.5", "59"]
    pool_x_engine = ["400", "250", "1200", "3.2", "Mach 1"]

    examples: list[tuple[str, str]] = []
    for _ in range(50):
        tpl = rng.choice(fusion_templates)
        x = rng.choice(pool_x_fusion)
        examples.append((tpl.format(JET="JET", x=x), "Joint European Torus"))
    for _ in range(50):
        tpl = rng.choice(engine_templates)
        x = rng.choice(pool_x_engine)
        examples.append((tpl.format(JET="JET", x=x), "Jet Engine Test"))
    rng.shuffle(examples)
    return examples


def _make_ai_examples(seed: int = 41) -> list[tuple[str, str]]:
    """50 positives per long-form for AI."""
    rng = random.Random(seed)

    ml_templates = [
        "Recent {AI} models achieve state-of-the-art on {x} benchmarks.",
        "The {AI} system was trained on {x} billion tokens of scientific text.",
        "A transformer-based {AI} pipeline classified galaxy morphologies.",
        "Large language {AI} models now outperform humans on {x} tasks.",
        "We compared symbolic {AI} approaches with deep learning on {x} datasets.",
        "The {AI} agent used reinforcement learning to solve {x} puzzles.",
        "Our {AI} survey covers text, vision, and multimodal {x} applications.",
        "Benchmarking {AI} models on astrophysical tasks revealed {x} limitations.",
        "The {AI} community debates whether scaling alone drives {x} improvements.",
        "Training an {AI} model on spectroscopic labels required {x} GPU hours.",
    ]
    cardiology_templates = [
        "Severe {AI} was diagnosed via transthoracic echocardiography in {x} patients.",
        "Chronic {AI} leads to left ventricular dilation and volume overload.",
        "Surgical aortic valve replacement was indicated for grade {x} {AI}.",
        "Regurgitant volume in {AI} was measured by cardiac MRI in {x} cases.",
        "Acute {AI} following endocarditis required urgent valve repair.",
        "The ESC guidelines classify {AI} severity as mild, moderate, or severe.",
        "In asymptomatic patients with severe {AI}, serial imaging is recommended.",
        "Echocardiographic markers of {AI} include vena contracta width and jet {x}.",
        "Post-operative outcomes after {AI} repair depend on preoperative EF.",
        "The murmur of {AI} is best heard along the left sternal border.",
    ]
    pool_x_ml = ["GLUE", "ImageNet", "MMLU", "vision", "language"]
    pool_x_card = ["III", "IV", "42", "severe", "7"]

    examples: list[tuple[str, str]] = []
    for _ in range(50):
        tpl = rng.choice(ml_templates)
        x = rng.choice(pool_x_ml)
        examples.append((tpl.format(AI="AI", x=x), "Artificial Intelligence"))
    for _ in range(50):
        tpl = rng.choice(cardiology_templates)
        x = rng.choice(pool_x_card)
        examples.append((tpl.format(AI="AI", x=x), "Aortic Insufficiency"))
    rng.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# Held-out split helper
# ---------------------------------------------------------------------------


def _split(
    examples: list[tuple[str, str]], train_frac: float = 0.8, seed: int = 0
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    return shuffled[:n_train], shuffled[n_train:]


# ---------------------------------------------------------------------------
# Per-acronym accuracy tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "acronym,factory",
    [
        ("HST", _make_hst_examples),
        ("JET", _make_jet_examples),
        ("AI", _make_ai_examples),
    ],
)
def test_adeft_accuracy_ge_90pct(acronym: str, factory) -> None:
    examples = factory()
    train, test = _split(examples, train_frac=0.8, seed=0)
    clf = train_classifier(acronym, train)

    contexts = [c for c, _ in test]
    labels = [lab for _, lab in test]
    acc = clf.score(contexts, labels)
    assert acc >= 0.90, f"{acronym} accuracy {acc:.2f} below 0.90 target"


def test_train_rejects_single_label() -> None:
    with pytest.raises(ValueError):
        train_classifier("HST", [("ctx", "only_label")] * 10)


def test_train_rejects_tiny_label() -> None:
    # 2 labels, but label B only has 1 example.
    examples = [("HST is a telescope", "A")] * 10 + [("HST is a trade", "B")]
    with pytest.raises(ValueError):
        train_classifier("HST", examples)


def test_predict_label_and_proba() -> None:
    examples = _make_hst_examples()
    clf = train_classifier("HST", examples)
    label = clf.predict_label("HST imaging of a distant quasar revealed a jet.")
    assert label in {"Hubble Space Telescope", "Highest Single Trade"}
    probs = clf.predict_proba("HST trading desk reported a record quarter.")
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert "Hubble Space Telescope" in probs
    assert "Highest Single Trade" in probs


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    examples = _make_jet_examples()
    clf = train_classifier("JET", examples)
    path = tmp_path / "jet.pkl"
    save_classifier(clf, path)
    assert path.exists()

    loaded = load_classifier("JET", path)
    assert isinstance(loaded, AdeftClassifier)
    assert loaded.acronym == "JET"
    assert set(loaded.labels) == set(clf.labels)

    # Same input → same prediction.
    sample = "The JET tokamak achieved record deuterium-tritium yields."
    assert loaded.predict_label(sample) == clf.predict_label(sample)


def test_load_rejects_wrong_acronym(tmp_path: Path) -> None:
    clf = train_classifier("HST", _make_hst_examples())
    path = tmp_path / "hst.pkl"
    save_classifier(clf, path)
    with pytest.raises(ValueError):
        load_classifier("JET", path)


def test_classifier_is_pickleable() -> None:
    clf = train_classifier("AI", _make_ai_examples())
    blob = pickle.dumps(clf)
    restored = pickle.loads(blob)
    assert isinstance(restored, AdeftClassifier)
    assert restored.acronym == "AI"
