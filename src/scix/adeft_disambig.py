"""Adeft-style per-acronym disambiguators — PRD §M6 / §S2 / u09.

An *Adeft classifier* is a small, per-acronym binary / multi-class model
that reads the abstract context around an ambiguous acronym (e.g. "HST")
and picks the correct long-form (e.g. "Hubble Space Telescope" vs
"Highest Single Trade"). The original Adeft paper (Steppi et al., Bioinf.
2020) trains a char n-gram TF-IDF + LogisticRegression on labeled abstracts
mined from PubMed. We ship the same shape here, seeded from synthetic
long-form examples at u09; a real harvest lands in u11.

This module is pure compute: no DB, no file paths assumed. The caller (the
test suite, or a later harvest CLI) hands in labeled training examples and
receives an :class:`AdeftClassifier` back.

Public API
----------

* :class:`AdeftClassifier` — trained wrapper, pickleable, thread-safe
  inference. Call ``.predict_long_form(context)`` or ``.predict_label
  (context)``.
* :func:`train_classifier` — fit from ``(acronym, [(context, long_form),
  ...])``.
* :func:`save_classifier` / :func:`load_classifier` — pickle IO.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Model shape
# ---------------------------------------------------------------------------

# Character n-gram range used by TfidfVectorizer. ``char_wb`` respects word
# boundaries so we don't overweight n-grams that straddle whitespace.
NGRAM_RANGE: tuple[int, int] = (2, 5)

# LogisticRegression penalty strength. 1.0 is sklearn's default; overridable
# via ``train_classifier(..., C=...)`` if a caller needs to tune.
DEFAULT_C: float = 1.0

# Minimum labels required to train a real multi-class classifier. Below
# this, :func:`train_classifier` raises ValueError rather than training a
# model that will overfit to a trivial one-class target.
MIN_SAMPLES_PER_LABEL: int = 5


# ---------------------------------------------------------------------------
# Trained classifier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdeftClassifier:
    """A trained per-acronym disambiguator.

    Frozen dataclass; the sklearn Pipeline inside is immutable-by-contract
    after training (we never call ``.fit`` twice on the same instance).
    Pickleable via :func:`save_classifier`.
    """

    acronym: str
    labels: tuple[str, ...]
    pipeline: Pipeline

    def predict_label(self, context: str) -> str:
        """Return the predicted long-form label for ``context``."""
        if not isinstance(context, str):
            raise TypeError("context must be a string")
        result = self.pipeline.predict([context])
        return str(result[0])

    def predict_long_form(self, context: str) -> str:
        """Alias for :meth:`predict_label`, matching the Adeft paper's
        vocabulary."""
        return self.predict_label(context)

    def predict_proba(self, context: str) -> dict[str, float]:
        """Return ``{label: probability}`` for ``context``.

        Uses ``LogisticRegression.predict_proba``, so the values are true
        probabilities (softmax over label set).
        """
        if not isinstance(context, str):
            raise TypeError("context must be a string")
        probs = self.pipeline.predict_proba([context])[0]
        return {label: float(p) for label, p in zip(self.labels, probs)}

    def score(self, contexts: Sequence[str], labels: Sequence[str]) -> float:
        """Return accuracy on a held-out set.

        Thin wrapper around ``Pipeline.score`` so the test suite can assert
        ≥ 0.90 without reaching into sklearn internals.
        """
        if len(contexts) != len(labels):
            raise ValueError("contexts and labels must be the same length")
        if not contexts:
            return 0.0
        return float(self.pipeline.score(list(contexts), list(labels)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_classifier(
    acronym: str,
    examples: Sequence[tuple[str, str]],
    *,
    c_inverse_reg: float = DEFAULT_C,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
    random_state: int = 0,
) -> AdeftClassifier:
    """Fit an AdeftClassifier for ``acronym`` on ``(context, long_form)`` pairs.

    Parameters
    ----------
    acronym
        The acronym this classifier disambiguates. Stored on the returned
        instance for downstream routing.
    examples
        Sequence of ``(context, long_form_label)`` tuples. ``context``
        is the abstract substring around the acronym (Adeft uses a
        ±250-char window; synthetic seeds can use the full sentence).
    c_inverse_reg
        Inverse regularization strength passed to LogisticRegression.
    ngram_range
        Char n-gram range. Default ``(2, 5)`` matches the Adeft paper.
    random_state
        Seed for LogisticRegression's solver (for reproducibility).

    Raises
    ------
    ValueError
        If fewer than two distinct labels are present, or if any single
        label has < :data:`MIN_SAMPLES_PER_LABEL` examples.
    """
    if not isinstance(acronym, str) or not acronym:
        raise ValueError("acronym must be a non-empty string")
    if not examples:
        raise ValueError("examples must be non-empty")

    contexts: list[str] = []
    labels: list[str] = []
    for context, label in examples:
        if not isinstance(context, str) or not isinstance(label, str):
            raise TypeError("examples must be (str, str) tuples")
        contexts.append(context)
        labels.append(label)

    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    if len(label_counts) < 2:
        raise ValueError(f"need at least 2 distinct long-form labels, got {list(label_counts)}")
    under_sampled = [label for label, n in label_counts.items() if n < MIN_SAMPLES_PER_LABEL]
    if under_sampled:
        raise ValueError(
            f"labels {under_sampled} have fewer than {MIN_SAMPLES_PER_LABEL} examples each"
        )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=ngram_range,
                    lowercase=True,
                    min_df=1,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    C=c_inverse_reg,
                    max_iter=1000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(contexts, labels)

    # Freeze the label order in the order the fitted pipeline reports it.
    classes = tuple(str(c) for c in pipeline.named_steps["logreg"].classes_)

    return AdeftClassifier(acronym=acronym, labels=classes, pipeline=pipeline)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_classifier(classifier: AdeftClassifier, path: Path) -> None:
    """Pickle ``classifier`` to ``path``.

    The containing directory is created if missing. The file is written
    atomically via a sibling ``.tmp`` rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(classifier, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_classifier(acronym: str, path: Path) -> AdeftClassifier:
    """Load a pickled :class:`AdeftClassifier` from ``path``.

    Verifies the loaded classifier matches ``acronym``; raises ValueError
    if the pickle file is for a different acronym (catches typos and stale
    caches during harvest pipelines).
    """
    path = Path(path)
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, AdeftClassifier):
        raise TypeError(f"{path} does not contain an AdeftClassifier")
    if obj.acronym != acronym:
        raise ValueError(f"{path} is for acronym {obj.acronym!r}, not {acronym!r}")
    return obj


__all__ = [
    "AdeftClassifier",
    "DEFAULT_C",
    "MIN_SAMPLES_PER_LABEL",
    "NGRAM_RANGE",
    "load_classifier",
    "save_classifier",
    "train_classifier",
]
