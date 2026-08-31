"""Regenerate the app screenshots embedded in the notebook and README.

The screenshots are documentation, so they have to be reproducible rather than hand-taken:
if the dataset is re-parsed and the flag counts change, a stale screenshot silently
contradicts the notebook. Run this against a locally served app after any pipeline re-run.

    streamlit run app.py --server.headless true --server.port 8555 &
    python tools/capture_screenshots.py --url http://localhost:8555

Requires `playwright` and `playwright install chromium`; it is a documentation tool, not a
runtime dependency, so it is deliberately not in requirements.txt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# name -> (requisition query param, tab label to click or None for the default tab)
SHOTS = [
    ("shot_01_results", "1", None),
    ("shot_02_compare", "1", "Compare candidates"),
    ("shot_03_insights", "1", "Talent pool insights"),
    ("shot_04_quality", "1", "Data quality"),
    ("shot_05_apac", "3", None),
]


def capture(base_url: str) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1300, "height": 887})
        for name, req, tab in SHOTS:
            page.goto(f"{base_url}/?req={req}", wait_until="networkidle")
            page.wait_for_selector("text=Candidate Resume Search Platform", timeout=60_000)
            page.wait_for_timeout(3_000)          # let charts finish their first paint
            if tab:
                page.get_by_role("tab", name=tab).first.click()
                page.wait_for_timeout(3_000)
            target = OUT_DIR / f"{name}.png"
            page.screenshot(path=str(target))
            print(f"wrote {target.relative_to(OUT_DIR.parent)}")
        browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8555", help="base URL of the running app")
    capture(ap.parse_args().url.rstrip("/"))
