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

.PHONY: help viz-demo viz-demo-build lint fmt fmt-check test check

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
	$(PYTEST) -q

check: lint fmt-check test

viz-demo:
	./scripts/viz/run.sh

viz-demo-build:
	./scripts/viz/run.sh --build-only --synthetic
