"""
Starts Streamlit in the background, waits for it to be ready,
then captures a screenshot of each page and saves them to screenshots/.
"""
import subprocess, time, os, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("screenshots")
OUT.mkdir(exist_ok=True)

PAGES = [
    ("home",        "http://localhost:8501"),
    ("single",      "http://localhost:8501"),
    ("batch",       "http://localhost:8501"),
    ("performance", "http://localhost:8501"),
    ("explorer",    "http://localhost:8501"),
]

NAV_LABELS = [
    "🏠 Home",
    "🔍 Single Prediction",
    "📂 Batch Prediction",
    "📊 Model Performance",
    "📈 Data Explorer",
]

print("Starting Streamlit …")
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
     "--server.headless", "true",
     "--server.port", "8501",
     "--browser.gatherUsageStats", "false"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# Wait until the server is up
import urllib.request, urllib.error
for _ in range(30):
    try:
        urllib.request.urlopen("http://localhost:8501", timeout=2)
        break
    except Exception:
        time.sleep(2)
else:
    proc.terminate()
    raise RuntimeError("Streamlit did not start in time")

print("Streamlit is up. Taking screenshots …")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})

    for label, path in zip(NAV_LABELS, [p[0] for p in PAGES]):
        page.goto("http://localhost:8501", wait_until="networkidle")
        time.sleep(3)

        # Click the sidebar radio option
        try:
            page.locator(f"label:has-text('{label}')").first.click()
            time.sleep(3)
        except Exception:
            pass

        # Scroll to top
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)

        fname = OUT / f"{path}.png"
        page.screenshot(path=str(fname), full_page=True)
        print(f"  Saved {fname}")

    browser.close()

proc.terminate()
print("Done. Screenshots saved to screenshots/")
