# SciX top-level Makefile.
#
# Keeps the common invocations close to the project root so they are
# discoverable via `make help` and usable from a freshly cloned checkout.
#
# Tooling targets (lint/fmt/test) assume the dev extras are installed:
#   pip install -e '.[dev]'   (or run inside the project .venv)

RUFF  ?= ruff
BLACK ?= black
PYTEST ?= pytest
# CI injects a marker filter (e.g. -m "not integration and not network") so the
# DB/data/model-dependent tests are deselected; local `make check` runs all.
PYTEST_ARGS ?=
# Base ref for the changed-files gate (check-ci). The tree carries pre-existing
# black/ruff debt, so the CI gate lints/format-checks only files changed vs BASE
# (incremental adoption, same philosophy as the pre-commit hooks). Debt is paid
# down as files are touched; full-tree `check` stays as the eventual target.
BASE ?= origin/main

.PHONY: help viz-demo viz-demo-build lint fmt fmt-check test check \
        lint-changed fmt-check-changed check-ci

help:
	@echo "SciX experiments — available targets:"
	@echo "  lint             Ruff lint (E,F,I,W) over src/ scripts/ tests/."
	@echo "  fmt              Auto-fix: ruff --fix + black over src/ scripts/ tests/."
	@echo "  fmt-check        Verify formatting without writing (CI/pre-commit parity)."
	@echo "  test             Run pytest (set SCIX_TEST_DSN to enable write tests)."
	@echo "  check            lint + fmt-check + test — the full local gate."
	@echo "  viz-demo         Build demo data (if missing) and launch the viz server."
	@echo "  viz-demo-build   Build synthetic demo data only (no server)."

lint:
	$(RUFF) check src/ scripts/ tests/

fmt:
	$(RUFF) check --fix src/ scripts/ tests/
	$(BLACK) src/ scripts/ tests/

fmt-check:
	$(RUFF) check src/ scripts/ tests/
	$(BLACK) --check src/ scripts/ tests/

test:
	$(PYTEST) -q $(PYTEST_ARGS)

check: lint fmt-check test

# --- CI gate: lint/format only files changed vs BASE, run the full test suite ---
lint-changed:
	@files=$$(git diff --name-only --diff-filter=ACMR $(BASE)...HEAD -- '*.py'); \
	if [ -z "$$files" ]; then echo "lint-changed: no changed .py files"; \
	else echo "linting changed: $$files"; $(RUFF) check $$files; fi

fmt-check-changed:
	@files=$$(git diff --name-only --diff-filter=ACMR $(BASE)...HEAD -- '*.py'); \
	if [ -z "$$files" ]; then echo "fmt-check-changed: no changed .py files"; \
	else echo "fmt-checking changed: $$files"; $(BLACK) --check $$files; fi

check-ci: lint-changed fmt-check-changed test

viz-demo:
	./scripts/viz/run.sh

viz-demo-build:
	./scripts/viz/run.sh --build-only --synthetic
