"""Capture screenshots of SciX viz pages for slides."""
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("/home/ds/projects/scix_experiments/docs/slides_assets")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "http://127.0.0.1:8765/viz"
PAGES = [
    ("umap_browser", f"{BASE}/umap_browser.html", 4000),
    ("sankey", f"{BASE}/sankey.html", 3500),
    ("heatmap", f"{BASE}/heatmap.html", 3500),
    ("ego", f"{BASE}/ego.html", 3500),
    ("agent_trace", f"{BASE}/agent_trace.html", 3500),
    ("index", f"{BASE}/index.html", 1500),
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    for name, url, wait_ms in PAGES:
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        page = ctx.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
        except Exception as e:
            print(f"goto {name}: {e}")
        page.wait_for_timeout(wait_ms)
        dest = OUT / f"{name}.png"
        page.screenshot(path=str(dest), full_page=False)
        print(f"wrote {dest} ({dest.stat().st_size} bytes)")
        ctx.close()
    browser.close()
