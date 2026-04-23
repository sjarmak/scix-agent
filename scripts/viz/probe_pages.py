#!/usr/bin/env python3
"""Drive the three viz pages headlessly and dump diagnostics.

Captures per-page:
  - console errors and warnings
  - failed network requests
  - DOM snapshot (key elements, bounding boxes, counts)
  - a PNG screenshot for eyeballing later

Usage:
    .venv/bin/python scripts/viz/probe_pages.py [--base http://127.0.0.1:8765]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


PAGES = [
    ("sankey", "/viz/sankey.html"),
    ("umap", "/viz/umap_browser.html"),
    ("trace", "/viz/agent_trace.html"),
]


def probe(base: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        for name, path in PAGES:
            page = context.new_page()
            console = []
            requests_failed = []
            page.on("console", lambda msg, _console=console: _console.append(
                {"type": msg.type, "text": msg.text, "location": str(msg.location)}
            ))
            page.on("requestfailed", lambda req, _f=requests_failed: _f.append(
                {"url": req.url, "failure": req.failure}
            ))
            url = base + path
            # 'networkidle' hangs on pages with long-lived SSE streams
            # (agent_trace.html opens /viz/api/trace/stream indefinitely).
            # 'load' fires after window.onload; we then wait a fixed beat.
            page.goto(url, wait_until="load", timeout=30_000)
            page.wait_for_timeout(4_000)
            shot = out_dir / f"{name}.png"
            page.screenshot(path=str(shot), full_page=False)

            dom = page.evaluate("""() => {
                const root = document.getElementById('sankey-root')
                          || document.getElementById('umap-root')
                          || document.body;
                const svg = root?.querySelector('svg');
                const canvas = root?.querySelector('canvas');
                const tracePanel = document.getElementById('trace-panel');
                const bb = root?.getBoundingClientRect();
                return {
                    title: document.title,
                    root_id: root?.id || 'body',
                    root_size: bb ? {w: bb.width, h: bb.height} : null,
                    body_size: {w: document.body.clientWidth, h: document.body.clientHeight},
                    svg_size: svg ? {w: svg.clientWidth, h: svg.clientHeight} : null,
                    svg_nodes: svg ? {
                        rects: svg.querySelectorAll('rect').length,
                        paths: svg.querySelectorAll('path').length,
                        texts: svg.querySelectorAll('text').length,
                        groups: svg.querySelectorAll('g').length,
                    } : null,
                    canvas_size: canvas ? {w: canvas.clientWidth, h: canvas.clientHeight} : null,
                    trace_panel_items: tracePanel ? tracePanel.children.length : null,
                    legend_swatches: document.querySelectorAll('.legend-swatch').length,
                    stats_text: document.getElementById('umap-stats')?.textContent || null,
                    axis_decades: document.querySelectorAll('.sankey-axis text').length,
                    status_text: document.getElementById('sankey-status')?.textContent
                              || document.getElementById('umap-status')?.textContent
                              || null,
                };
            }""")

            results[name] = {
                "url": url,
                "screenshot": str(shot),
                "console": console[-20:],
                "requests_failed": requests_failed[:10],
                "dom": dom,
            }
            page.close()
        browser.close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--out", default="logs/viz_probe")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    results = probe(args.base, out_dir)
    summary_path = out_dir / "results.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    print(f"\n[screenshots in {out_dir.resolve()}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
