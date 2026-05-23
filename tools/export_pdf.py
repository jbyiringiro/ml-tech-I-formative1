"""Export ``report/report.md`` to a polished PDF.

Pipeline: markdown -> styled HTML -> Microsoft Edge headless ``--print-to-pdf``.
Edge is bundled with Windows 11, so no extra binaries are needed. Relative
image paths in the markdown (`../results/figures/...`) are preserved because
the HTML is written inside ``report/`` and loaded as a ``file://`` URL.

    python tools/export_pdf.py [output_name_without_extension]
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
MD_PATH = REPORT_DIR / "report.md"

CSS = """
@page { size: A4; margin: 1.6cm 1.5cm; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #1f1f1f; }
h1 { font-size: 22pt; margin: 0 0 6pt; }
h2 { font-size: 16pt; margin-top: 24pt; page-break-after: avoid; }
h3 { font-size: 12.5pt; margin-top: 16pt; page-break-after: avoid; }
h4 { font-size: 11pt; margin-top: 12pt; }
p, li { text-align: justify; }
strong { color: #111; }
table { border-collapse: collapse; margin: 10pt 0; font-size: 9.5pt; width: 100%;
        page-break-inside: avoid; }
th, td { border: 1px solid #9a9a9a; padding: 4px 7px; vertical-align: top; }
th { background: #ececec; text-align: left; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 9.5pt;
       background: #f3f3f3; padding: 0 4px; border-radius: 2px; }
pre { background: #f6f6f6; border: 1px solid #dcdcdc; padding: 8px 10px;
      overflow: auto; font-size: 9pt; page-break-inside: avoid; white-space: pre-wrap; }
blockquote { border-left: 3px solid #999; padding: 4px 12px; color: #444;
             margin: 8pt 0; background: #fafafa; }
img { max-width: 100%; height: auto; display: block; margin: 10pt auto;
      page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #c0c0c0; margin: 16pt 0; }
a { color: #1155cc; text-decoration: none; word-break: break-all; }
"""


def find_edge() -> str:
    """Return the absolute path to msedge.exe (Windows)."""
    candidates = [
        os.path.join(os.environ.get("ProgramFiles(x86)", ""),
                     "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""),
                     "Microsoft", "Edge", "Application", "msedge.exe"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    edge_on_path = shutil.which("msedge")
    if edge_on_path:
        return edge_on_path
    raise SystemExit("msedge.exe not found; install Microsoft Edge.")


def build_html(md_text: str) -> str:
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def main() -> None:
    out_name = sys.argv[1] if len(sys.argv) > 1 \
        else "Josue_Byiringiro-ML_Tech_I-Formative1-Report"
    out_pdf = REPORT_DIR / f"{out_name}.pdf"
    tmp_html = REPORT_DIR / "_report_for_pdf.html"

    md_text = MD_PATH.read_text(encoding="utf-8")
    tmp_html.write_text(build_html(md_text), encoding="utf-8")
    print(f"HTML written -> {tmp_html} ({tmp_html.stat().st_size:,} bytes)")

    edge = find_edge()
    file_url = "file:///" + str(tmp_html).replace("\\", "/")
    cmd = [
        edge, "--headless=new", "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={out_pdf}",
        file_url,
    ]
    print("running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True, timeout=240)

    tmp_html.unlink(missing_ok=True)
    print(f"PDF written -> {out_pdf} "
          f"({out_pdf.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
