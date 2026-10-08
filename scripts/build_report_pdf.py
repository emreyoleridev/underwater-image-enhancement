"""Render REPORT.md to REPORT.pdf (Markdown -> styled HTML -> headless Chrome print)."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

FORMULA = ('<div class="formula">I<sub>r</sub>&prime; = I<sub>r</sub> + &alpha; '
           '(<span class="bar">I</span><sub>g</sub> &minus; <span class="bar">I</span><sub>r</sub>) '
           '(1 &minus; I<sub>r</sub>) I<sub>g</sub></div>')

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.5;
       color: #1b1b1a; }
h1 { font-size: 22pt; margin: 0 0 4pt; color: #0b0b0b; }
.subtitle { color: #52514e; margin-bottom: 18pt; border-bottom: 2px solid #2a78d6; padding-bottom: 10pt; }
h2 { font-size: 14pt; margin-top: 20pt; color: #0b0b0b; border-bottom: 1px solid #e4e3df; padding-bottom: 3pt;
     break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 14pt; break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9.2pt; break-inside: avoid; }
th, td { border-bottom: 1px solid #e4e3df; padding: 4pt 6pt; text-align: left; }
th { background: #f3f2ee; font-weight: 600; }
img { max-width: 100%; display: block; margin: 8pt auto; break-inside: avoid; }
code { font-family: Menlo, monospace; font-size: 8.8pt; background: #f3f2ee; padding: 1pt 3pt; border-radius: 3px; }
pre { background: #f3f2ee; padding: 8pt; border-radius: 4px; break-inside: avoid; }
pre code { background: none; padding: 0; }
.formula { text-align: center; font-family: "Times New Roman", serif; font-size: 13pt; margin: 10pt 0;
           font-style: italic; }
.bar { text-decoration: overline; }
li { margin: 2pt 0; }
"""


def main() -> None:
    md = (ROOT / "REPORT.md").read_text()
    md = re.sub(r"\$\$.*?\$\$", FORMULA, md, flags=re.S)
    md = md.replace("$(1 - I_r)\\,I_g$", "(1 &minus; I<sub>r</sub>) I<sub>g</sub>")
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    body = body.replace("</h1>", '</h1>\n<div class="subtitle">Python · OpenCV · UIEB benchmark · '
                        'CLAHE, white balance &amp; histogram equalization</div>', 1)
    html = f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>'

    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=ROOT, delete=False) as f:
        f.write(html)
        tmp = Path(f.name)
    try:
        out = ROOT / "REPORT.pdf"
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out}", tmp.as_uri()], check=True, capture_output=True)
        print("Wrote", out)
    finally:
        tmp.unlink()


if __name__ == "__main__":
    sys.exit(main())
