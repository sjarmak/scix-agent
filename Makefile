# SciX top-level Makefile.
#
# Thin wrapper around the demo runner. Keeps the common invocations close
# to the project root so they are discoverable via `make help` and usable
# from a freshly cloned checkout.

.PHONY: help viz-demo viz-demo-build

help:
	@echo "SciX experiments — available targets:"
	@echo "  viz-demo         Build demo data (if missing) and launch the viz server."
	@echo "  viz-demo-build   Build synthetic demo data only (no server)."

viz-demo:
	./scripts/viz/run.sh

viz-demo-build:
	./scripts/viz/run.sh --build-only --synthetic
