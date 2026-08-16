"""Render docs/research_paper.md to a self-contained HTML and then to PDF.

Images are inlined as data URIs so the HTML is a single portable file, and
Chrome's headless print engine produces the PDF - no LaTeX or GTK toolchain
required on Windows.

Run with ``python tools/build_paper.py``.
"""
from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC_MD = ROOT / "docs" / "research_paper.md"
OUT_HTML = ROOT / "docs" / "research_paper.html"
OUT_PDF = ROOT / "docs" / "research_paper.pdf"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

CSS = """
:root {
  --ink: #14140f; --muted: #52514e; --rule: #d8d7d2; --surface: #ffffff;
  --accent: #2a78d6; --tint: #f5f7fa;
}
* { box-sizing: border-box; }
body {
  font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
  font-size: 10.6pt; line-height: 1.62; color: var(--ink);
  background: var(--surface); margin: 0 auto; max-width: 44rem;
  padding: 0 1rem; text-rendering: optimizeLegibility;
}
h1, h2, h3, h4 {
  font-family: "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
  line-height: 1.22; color: var(--ink); font-weight: 650;
}
h1 { font-size: 21pt; margin: 0 0 .4rem; letter-spacing: -.015em; }
h2 {
  font-size: 14pt; margin: 2.1rem 0 .7rem; padding-top: .55rem;
  border-top: 1.5px solid var(--rule); page-break-after: avoid;
}
h3 { font-size: 11.6pt; margin: 1.5rem 0 .45rem; page-break-after: avoid; }
h1 + p { font-size: 11.5pt; color: var(--muted); margin-top: 0; }
p.byline {
  font-family: "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 9.4pt; color: var(--muted); line-height: 1.55;
  margin: .9rem 0 1.2rem;
}
p.byline strong { color: var(--ink); font-size: 10.4pt; }
p { margin: 0 0 .75rem; }
strong { font-weight: 650; }
hr { border: 0; border-top: 1.5px solid var(--rule); margin: 1.8rem 0; }
a { color: var(--accent); text-decoration: none; }

blockquote {
  margin: 1rem 0; padding: .7rem 1rem; background: var(--tint);
  border-left: 3px solid var(--accent); color: var(--ink);
}
blockquote p:last-child { margin-bottom: 0; }

table {
  border-collapse: collapse; width: 100%; margin: 1rem 0 1.3rem;
  font-family: "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 8.6pt; page-break-inside: avoid;
}
th, td {
  border-bottom: 1px solid var(--rule); padding: .36rem .5rem;
  text-align: left; vertical-align: top;
}
thead th {
  border-bottom: 1.5px solid var(--ink); font-weight: 650;
  background: var(--tint);
}
tbody tr:last-child td { border-bottom: 1.5px solid var(--rule); }
td:not(:first-child), th:not(:first-child) { text-align: right; }
table.left td, table.left th { text-align: left; }

img {
  max-width: 100%; height: auto; display: block; margin: 1.1rem auto .4rem;
  page-break-inside: avoid;
}
code, pre {
  font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
  font-size: 8.6pt;
}
code { background: var(--tint); padding: .08em .32em; border-radius: 3px; }
pre {
  background: var(--tint); padding: .7rem .9rem; border-radius: 5px;
  overflow-x: auto; border: 1px solid var(--rule); line-height: 1.45;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; }
ol, ul { margin: 0 0 .8rem; padding-left: 1.4rem; }
li { margin-bottom: .28rem; }

@page { size: A4; margin: 17mm 15mm 18mm; }
@media print {
  body { max-width: none; padding: 0; font-size: 9.7pt; }
  h2 { page-break-after: avoid; }
}
"""


def inline_images(html: str, base: Path) -> str:
    """Replace <img src="..."> with data URIs so the HTML stands alone."""
    def repl(m: re.Match) -> str:
        src = m.group(1)
        if src.startswith(("data:", "http://", "https://")):
            return m.group(0)
        path = (base / src).resolve()
        if not path.exists():
            print(f"  WARNING missing image: {src}", file=sys.stderr)
            return m.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return m.group(0).replace(src, f"data:{mime};base64,{data}")

    return re.sub(r'<img[^>]*src="([^"]+)"', repl, html)


def find_chrome() -> Path | None:
    for p in CHROME_CANDIDATES:
        if p.exists():
            return p
    for name in ("chrome", "msedge", "chromium"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def main() -> None:
    text = SRC_MD.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
    )
    body = inline_images(body, SRC_MD.parent)

    html = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,"
        " initial-scale=1\">\n"
        "<title>Machine-Learning Buyer Segmentation and Investment Profiling"
        "</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
    OUT_HTML.write_text(html, encoding="utf-8")
    size_mb = OUT_HTML.stat().st_size / 1e6
    print(f"wrote {OUT_HTML.relative_to(ROOT)} ({size_mb:.2f} MB)")

    chrome = find_chrome()
    if not chrome:
        print("Chrome/Edge not found - HTML written, PDF skipped.",
              file=sys.stderr)
        return

    cmd = [
        str(chrome), "--headless", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUT_PDF}",
        OUT_HTML.as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if OUT_PDF.exists():
        print(f"wrote {OUT_PDF.relative_to(ROOT)} "
              f"({OUT_PDF.stat().st_size / 1e6:.2f} MB)")
    else:
        print("PDF generation failed:", res.stderr[-800:], file=sys.stderr)


if __name__ == "__main__":
    main()
