#!/usr/bin/env python3
"""Emit the versioned MCP contract artifact from the live server surface.

Writes ``contract/scix_mcp_v<CONTRACT_VERSION>.json`` (tool names + schemas,
the closed error-code catalog, and the response-envelope shape). Run this after
an *intentional* change to the MCP tool surface; the conformance suite
(``tests/test_mcp_contract_conformance.py``) fails until the committed artifact
matches the live server again.

A breaking change should bump ``CONTRACT_VERSION`` in ``scix.mcp_contract`` so a
new ``scix_mcp_v2.json`` is published alongside the old one — see bead
scix_experiments-x2dp.

Usage:
    python scripts/gen_mcp_contract.py
"""

from __future__ import annotations

from scix.mcp_contract import write_published_contract


def main() -> None:
    path = write_published_contract()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
