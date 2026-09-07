"""Render the Markdown report to PDF using headless Chrome.

Chrome rather than a PDF library because the report is mostly tables and
figures, and a browser already knows how to lay those out, break them across
pages and embed the images at print resolution.

The intermediate HTML is written into ``report/`` rather than a system
temporary directory. The Markdown references its figures as
``../benchmark_results/...``, and those relative paths only resolve if the
page is loaded from the directory the Markdown lives in.

Requires the report extra and a Chromium-family browser::

    pip install -e ".[report]"

Usage
-----
    python scripts/build_report_pdf.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "report"
REPORT_MD = REPORT_DIR / "CV_Benchmarking_Report.md"
REPORT_PDF = REPORT_DIR / "CV_Benchmarking_Report.pdf"

BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]

CSS = """
@page { size: Letter; margin: 0.7in; }

body {
  font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.5;
  color: #0b0b0b;
  max-width: 100%;
}

h1 { font-size: 20pt; margin: 0 0 4px; }
h2 {
  font-size: 13pt;
  margin: 26px 0 8px;
  padding-bottom: 4px;
  border-bottom: 1px solid #e1e0d9;
  /* Keep a heading with the content it introduces. */
  break-after: avoid;
  break-inside: avoid;
}
h3 { font-size: 11.5pt; margin: 18px 0 6px; break-after: avoid; }

p { margin: 8px 0; }
strong { font-weight: 600; }

table {
  border-collapse: collapse;
  font-size: 8.5pt;
  margin: 10px 0;
  width: 100%;
  break-inside: avoid;
}
th, td {
  border: 1px solid #e1e0d9;
  padding: 4px 7px;
  text-align: right;
  /* Digits line up down a column. */
  font-variant-numeric: tabular-nums;
}
th { background: #f4f3ef; font-weight: 600; text-align: right; }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }

img {
  max-width: 100%;
  margin: 10px 0;
  /* A figure split across a page break is unreadable. */
  break-inside: avoid;
}

code {
  background: #f4f3ef;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 9pt;
}
pre {
  background: #f9f9f7;
  border: 1px solid #e1e0d9;
  border-radius: 4px;
  padding: 9px 11px;
  overflow-x: auto;
  break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8.5pt; }

/* The reflection is prose rather than reference material, so it reads at a
   slightly more generous measure. */
h2#reflection + p, h2[id^="9"] ~ p { max-width: 40em; }
"""


def find_browser() -> str:
    """Locate a Chromium-family browser, or explain what to do about it."""
    for candidate in BROWSERS:
        if Path(candidate).exists():
            return candidate
    for name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    sys.exit(
        "No Chrome, Chromium or Edge installation found. Install one, or "
        "export the Markdown with another tool - the report itself is "
        "complete either way at report/CV_Benchmarking_Report.md"
    )


def render_html(markdown_text: str) -> str:
    try:
        import markdown
    except ImportError:
        sys.exit(
            'The `markdown` package is required. Install the report extra:\n'
            '    pip install -e ".[report]"'
        )

    body = markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def main() -> None:
    if not REPORT_MD.is_file():
        sys.exit(
            f"{REPORT_MD.relative_to(REPO_ROOT)} is missing. "
            "Run scripts/generate_report.py first."
        )

    browser = find_browser()
    html = render_html(REPORT_MD.read_text(encoding="utf-8"))

    # Written beside the Markdown so its ../benchmark_results/ image paths
    # resolve; deleted afterwards either way.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", dir=REPORT_DIR, encoding="utf-8", delete=False
    ) as handle:
        handle.write(html)
        html_path = Path(handle.name)

    try:
        completed = subprocess.run(
            [
                browser,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                # Give the figures time to load before the page is captured;
                # without these the PDF can come out with blank images.
                "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=20000",
                f"--print-to-pdf={REPORT_PDF}",
                html_path.as_uri(),
            ],
            capture_output=True,
            text=True,
        )
    finally:
        html_path.unlink(missing_ok=True)

    if not REPORT_PDF.is_file():
        sys.exit(
            f"Chrome did not produce a PDF.\n{completed.stderr.strip()[:800]}"
        )

    size_mb = REPORT_PDF.stat().st_size / 1_000_000
    print(f"Wrote {REPORT_PDF.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
