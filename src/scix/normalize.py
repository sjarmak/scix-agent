"""Deterministic multi-stage normalization pipeline for entity strings.

Stages:
    1. Unicode NFKC + lowercase + strip
    2. Punctuation normalization (hyphens→spaces, remove possessives)
    3. Alias resolution via hand-curated dictionary
    4. Whitespace collapse

All transformations are deterministic — same input always produces same output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Sequence

# ---------------------------------------------------------------------------
# Alias dictionary — keys must be lowercase, post-punctuation-normalization
# forms.  Values are the canonical expanded forms.
# ---------------------------------------------------------------------------

_ALIAS_DICT: dict[str, str] = {
    # Instruments
    "hst": "hubble space telescope",
    "jwst": "james webb space telescope",
    "alma": "atacama large millimeter array",
    "vlt": "very large telescope",
    "vla": "very large array",
    "ska": "square kilometre array",
    "lofar": "low frequency array",
    "eso": "european southern observatory",
    "eso vlt": "very large telescope",
    "sdo": "solar dynamics observatory",
    # X-ray / Gamma-ray
    "cxo": "chandra x ray observatory",
    "xmm": "x ray multi mirror mission",
    "xmm newton": "x ray multi mirror mission",
    "nustar": "nuclear spectroscopic telescope array",
    "lat": "large area telescope",
    "bat": "burst alert telescope",
    # Space missions
    "herschel": "herschel space observatory",
    "spitzer": "spitzer space telescope",
    "fermi": "fermi gamma ray space telescope",
    "tess": "transiting exoplanet survey satellite",
    "kepler": "kepler space telescope",
    "gaia": "gaia space observatory",
    "wmap": "wilkinson microwave anisotropy probe",
    "cobe": "cosmic background explorer",
    "planck": "planck space observatory",
    "euclid": "euclid space telescope",
    "rosat": "roentgen satellite",
    "iras": "infrared astronomical satellite",
    "wise": "wide field infrared survey explorer",
    "neowise": "wide field infrared survey explorer",
    # Surveys and datasets
    "sdss": "sloan digital sky survey",
    "2mass": "two micron all sky survey",
    "lsst": "legacy survey of space and time",
    "des": "dark energy survey",
    "desi": "dark energy spectroscopic instrument",
    "panstarrs": "panoramic survey telescope and rapid response system",
    "pan starrs": "panoramic survey telescope and rapid response system",
    "lamost": "large sky area multi object fiber spectroscopic telescope",
    "askap": "australian square kilometre array pathfinder",
    "nvss": "nrao vla sky survey",
    "first": "faint images of the radio sky at twenty centimeters",
    "boss": "baryon oscillation spectroscopic survey",
    "eboss": "extended baryon oscillation spectroscopic survey",
    "apogee": "apache point observatory galactic evolution experiment",
    # Methods
    "mcmc": "markov chain monte carlo",
    "pca": "principal component analysis",
    "sed": "spectral energy distribution",
    "sed fitting": "spectral energy distribution fitting",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short term memory",
    "gan": "generative adversarial network",
    "mle": "maximum likelihood estimation",
    "em": "expectation maximization",
    "svm": "support vector machine",
    "rf": "random forest",
    "knn": "k nearest neighbors",
    "bnn": "bayesian neural network",
    "gp": "gaussian process",
    "gpr": "gaussian process regression",
    "hmm": "hidden markov model",
    "ica": "independent component analysis",
    "lda": "linear discriminant analysis",
    "nmf": "non negative matrix factorization",
    "svd": "singular value decomposition",
    "fft": "fast fourier transform",
    "psf": "point spread function",
    "snr": "signal to noise ratio",
    "s n": "signal to noise ratio",
    # Cosmology / physics
    "cmb": "cosmic microwave background",
    "bbn": "big bang nucleosynthesis",
    "bao": "baryon acoustic oscillations",
    "agn": "active galactic nucleus",
    "qso": "quasi stellar object",
    "grb": "gamma ray burst",
    "sne": "supernovae",
    "sn ia": "type ia supernova",
    "sn ii": "type ii supernova",
    "ism": "interstellar medium",
    "igm": "intergalactic medium",
    "icm": "intracluster medium",
    "imf": "initial mass function",
    "sfr": "star formation rate",
    "hr diagram": "hertzsprung russell diagram",
    "h r diagram": "hertzsprung russell diagram",
    "cmd": "color magnitude diagram",
    # Software
    "iraf": "image reduction and analysis facility",
    "ds9": "saoimageds9",
    "topcat": "tool for operations on catalogues and tables",
    "astropy": "astropy",
    "sextractor": "source extractor",
    # Units / standards
    "wcs": "world coordinate system",
    "fits": "flexible image transport system",
    "vo": "virtual observatory",
    "ivoa": "international virtual observatory alliance",
}

ALIAS_MAP: MappingProxyType[str, str] = MappingProxyType(_ALIAS_DICT)

# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

_RE_HYPHENS = re.compile(r"[\u2010-\u2015\u2212\-]+")  # all dash variants
_RE_POSSESSIVE = re.compile(r"'s\b|s'\b", re.IGNORECASE)
_RE_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationResult:
    """Immutable result of batch normalization.

    Attributes:
        canonical_forms: Tuple of canonical forms, one per input entity
            (same order and length as the input list).
        original_map: Mapping from each canonical form to the frozenset of
            distinct original strings that produced it (denormalization map).
    """

    canonical_forms: tuple[str, ...]
    original_map: MappingProxyType[str, frozenset[str]]


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _stage_unicode(text: str) -> str:
    """Stage 1: Unicode NFKC normalization + lowercase + strip."""
    return unicodedata.normalize("NFKC", text).casefold().strip()


def _stage_punctuation(text: str) -> str:
    """Stage 2: Hyphens to spaces, remove possessives."""
    text = _RE_HYPHENS.sub(" ", text)
    text = _RE_POSSESSIVE.sub("", text)
    return text


def _stage_alias(text: str) -> str:
    """Stage 3: Alias resolution via ALIAS_MAP."""
    return ALIAS_MAP.get(text, text)


def _stage_whitespace(text: str) -> str:
    """Stage 4: Collapse whitespace and strip."""
    return _RE_WHITESPACE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_entity(entity: str) -> str:
    """Normalize an entity string through the full 4-stage pipeline.

    Args:
        entity: Raw entity string (e.g. 'Density-Functional Theory').

    Returns:
        Canonical normalized form (e.g. 'density functional theory').
    """
    text = _stage_unicode(entity)
    text = _stage_punctuation(text)
    text = _stage_whitespace(text)  # collapse before alias lookup
    text = _stage_alias(text)
    text = _stage_whitespace(text)  # final cleanup after alias expansion
    return text


def normalize_batch(entities: Sequence[str]) -> NormalizationResult:
    """Normalize a batch of entity strings and build a denormalization map.

    Args:
        entities: Sequence of raw entity strings.

    Returns:
        A NormalizationResult containing canonical_forms (same order as
        input) and original_map (canonical -> frozenset of original forms).
    """
    canonical_list: list[str] = []
    reverse_map: dict[str, set[str]] = {}

    for entity in entities:
        canon = normalize_entity(entity)
        canonical_list.append(canon)
        if canon not in reverse_map:
            reverse_map[canon] = set()
        reverse_map[canon].add(entity)

    return NormalizationResult(
        canonical_forms=tuple(canonical_list),
        original_map=MappingProxyType({k: frozenset(v) for k, v in reverse_map.items()}),
    )
