# Research: Deterministic Normalization

## Codebase Conventions

- Python 3, type annotations on all signatures
- `from __future__ import annotations` at top of every module
- Immutable data: frozen dataclasses, NamedTuples, frozenset
- Testing with pytest, imports from `scix.<module>`
- Tests use `helpers` module for shared fixtures
- Logging via `logging` module, not print

## Related Modules

- `dictionary.py` — entity dictionary with canonical names and aliases (DB-backed)
- `field_mapping.py` — record transformation utilities
- `extract.py` — extraction pipeline

## Entity Types in Scope

Instruments (HST, JWST, ALMA, Chandra), methods (MCMC, PCA, SED fitting),
datasets (SDSS, 2MASS, Gaia), missions, software — all common in ADS metadata.

## Design Decision

This module is purely in-memory, no DB dependency. The alias dict is hand-curated
and shipped as a frozen mapping. The denormalization map tracks original→canonical
so callers can recover surface forms.
