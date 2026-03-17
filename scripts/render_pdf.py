#!/usr/bin/env python3
"""
Robust Markdown-to-PDF renderer for Options Quant Engine.

Converts Markdown reports to professional-grade PDFs using a multi-backend
approach:

    1. **WeasyPrint** (primary) — Python-native HTML/CSS→PDF, supports
       CSS Paged Media, @page rules, headers/footers, and the project's
       existing stylesheet.
    2. **Pandoc + Tectonic** (fallback) — Markdown→LaTeX→PDF using the
       project's custom LaTeX header.
    3. **Pandoc + Chrome** (last resort) — the legacy path from
       ``render_research_note.sh``.

Usage
-----
CLI:
    python scripts/render_pdf.py path/to/report.md [-o output.pdf]
    python scripts/render_pdf.py path/to/report.md --backend weasyprint
    python scripts/render_pdf.py --all-daily              # render all daily reports
    python scripts/render_pdf.py --all-research-notes     # render all research notes

Library:
    from scripts.render_pdf import render_markdown_to_pdf
    render_markdown_to_pdf("documentation/daily_reports/report.md")
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Suppress noisy weasyprint warnings about unsupported CSS properties
# (overflow-x, box-shadow, text-rendering, @media screen queries) which
# are harmless in a print context.
logging.getLogger("weasyprint").setLevel(logging.ERROR)
logging.getLogger("weasyprint.css").setLevel(logging.ERROR)

_ROOT = Path(__file__).resolve().parent.parent
_PDF_ASSETS = _ROOT / "documentation" / "_pdf_assets"
_DEFAULT_CSS = _ROOT / "documentation" / "research_notes" / "quant_note_trade_signal_logic_style.css"
_DEFAULT_TEX_HEADER = _PDF_ASSETS / "pandoc_header.tex"
_OQE_PDF_TOOLS = Path(os.environ.get("PDF_TOOLS_ROOT", Path.home() / ".local" / "oqe-pdf-tools"))

# ---------------------------------------------------------------------------
# CSS for daily/signal-evaluation reports (professional quant-report style)
# ---------------------------------------------------------------------------
_REPORT_CSS = r"""
:root {
  --ink: #16202a;
  --muted: #5d6b79;
  --accent: #0e7490;
  --accent-soft: #e6f4f7;
  --rule: #d7e2e8;
  --paper: #ffffff;
  --code-bg: #f6f8fa;
}
@page {
  size: A4;
  margin: 22mm 18mm 20mm 18mm;
  @bottom-center {
    content: counter(page);
    font-size: 9pt;
    color: #5d6b79;
  }
  @top-right {
    content: "Options Quant Engine";
    font-size: 8pt;
    color: #5d6b79;
    font-style: italic;
  }
}
@page :first {
  @top-right { content: none; }
  @bottom-center { content: none; }
}
html {
  font-size: 11pt;
  color: var(--ink);
  background: var(--paper);
}
body {
  font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  line-height: 1.55;
  max-width: none;
  margin: 0;
  padding: 0;
}
h1 { font-size: 1.7em; color: var(--ink); margin: 1.4em 0 0.6em; page-break-after: avoid; }
h2 { font-size: 1.25em; color: var(--ink); border-top: 1px solid var(--rule); padding-top: 0.5em; margin: 1.6em 0 0.6em; page-break-after: avoid; }
h3 { font-size: 1.0em; color: var(--accent); text-transform: uppercase; letter-spacing: 0.04em; margin: 1.2em 0 0.5em; page-break-after: avoid; }
h4 { font-size: 0.95em; color: var(--muted); margin: 1em 0 0.4em; page-break-after: avoid; }
p, ul, ol { margin: 0.5em 0; }
ul, ol { padding-left: 1.4em; }
a { color: var(--accent); text-decoration: none; }
strong { color: #0f1720; }
code {
  background: var(--code-bg);
  border-radius: 3px;
  padding: 0.05em 0.3em;
  font-family: "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 0.88em;
}
pre {
  background: var(--code-bg);
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 10px 12px;
  overflow-x: auto;
  page-break-inside: avoid;
}
pre code { padding: 0; background: transparent; }
blockquote {
  margin: 0.7em 0;
  padding: 8px 14px;
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
  page-break-inside: avoid;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9em;
  margin: 0.8em 0;
  page-break-inside: avoid;
}
th, td {
  padding: 6px 8px;
  border-bottom: 1px solid var(--rule);
  text-align: left;
  vertical-align: top;
}
th {
  color: var(--muted);
  font-size: 0.8em;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
tr:nth-child(even) { background: #fafbfc; }
hr { border: none; border-top: 1px solid var(--rule); margin: 1em 0; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_pandoc() -> str | None:
    """Locate pandoc: oqe-pdf-tools → system PATH."""
    oqe_pandoc = _OQE_PDF_TOOLS / "bin" / "pandoc"
    if oqe_pandoc.is_file():
        return str(oqe_pandoc)
    return shutil.which("pandoc")


def _find_tectonic() -> str | None:
    """Locate tectonic: oqe-pdf-tools → system PATH."""
    oqe = _OQE_PDF_TOOLS / "bin" / "tectonic"
    if oqe.is_file():
        return str(oqe)
    return shutil.which("tectonic")


def _find_chrome() -> str | None:
    """Locate Chrome/Chromium on macOS or Linux."""
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    chromium = shutil.which("chromium-browser") or shutil.which("chromium") or shutil.which("google-chrome")
    if chromium:
        candidates.insert(0, chromium)
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _pandoc_md_to_html(md_path: Path, css_path: Path | None, pandoc: str) -> str:
    """Convert Markdown to standalone HTML via Pandoc."""
    cmd = [
        pandoc,
        str(md_path),
        "--standalone",
        "--from", "gfm+yaml_metadata_block",
        "--to", "html5",
        "--embed-resources",
        "--toc",
        "--toc-depth=2",
        "--number-sections",
    ]
    if css_path and css_path.is_file():
        cmd.extend(["--css", str(css_path)])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"pandoc failed: {result.stderr.strip()}")
    return result.stdout


def _inject_css_into_html(html: str, css: str) -> str:
    """Inject a <style> block into HTML <head>."""
    style_tag = f"<style>\n{css}\n</style>"
    if "<head>" in html:
        return html.replace("<head>", f"<head>\n{style_tag}", 1)
    return f"<html><head>{style_tag}</head><body>{html}</body></html>"


def _oqe_lib_env() -> dict[str, str]:
    """Build environment with library paths for weasyprint C dependencies."""
    env = os.environ.copy()
    lib_dirs = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        lib_dirs.append(os.path.join(conda_prefix, "lib"))
    oqe_lib = _OQE_PDF_TOOLS / "lib"
    if oqe_lib.is_dir():
        lib_dirs.append(str(oqe_lib))
    if lib_dirs:
        existing = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    return env


def _weasyprint_available() -> bool:
    """Check whether weasyprint can be imported (with library paths set)."""
    env = _oqe_lib_env()
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import weasyprint"],
            capture_output=True,
            env=env,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Backend: WeasyPrint (primary)
# ---------------------------------------------------------------------------

def _render_weasyprint(
    html_content: str,
    output_pdf: Path,
) -> Path:
    """Render HTML string to PDF via WeasyPrint."""
    # WeasyPrint must be imported after library paths are set in the process
    # environment.  When called from CLI the env is already set; when called
    # as a library we patch it here.
    env = _oqe_lib_env()
    for k, v in env.items():
        if k.startswith("DYLD"):
            os.environ[k] = v

    import weasyprint  # noqa: E402 — intentional late import

    # Suppress cosmetic warnings about CSS properties unsupported in print
    # (overflow-x, box-shadow, text-rendering, @media screen queries).
    for _wlog in ("weasyprint", "weasyprint.css", "weasyprint.html",
                   "weasyprint.document", "weasyprint.pdf"):
        logging.getLogger(_wlog).setLevel(logging.ERROR)

    doc = weasyprint.HTML(string=html_content)
    doc.write_pdf(str(output_pdf))
    return output_pdf


# ---------------------------------------------------------------------------
# Backend: Pandoc + Tectonic (LaTeX path)
# ---------------------------------------------------------------------------

def _render_pandoc_tectonic(
    md_path: Path,
    output_pdf: Path,
    pandoc: str,
    tectonic: str,
) -> Path:
    """Render Markdown to PDF via Pandoc→LaTeX→Tectonic."""
    cmd = [
        pandoc,
        str(md_path),
        "--from", "gfm+yaml_metadata_block",
        "--pdf-engine", tectonic,
        "--toc",
        "--toc-depth=2",
        "--number-sections",
        "-o", str(output_pdf),
    ]
    if _DEFAULT_TEX_HEADER.is_file():
        cmd.extend(["-H", str(_DEFAULT_TEX_HEADER)])

    env = _oqe_lib_env()
    # Tectonic and fontconfig config from oqe-pdf-tools
    if (_OQE_PDF_TOOLS / "etc" / "fonts").is_dir():
        env["FONTCONFIG_PATH"] = str(_OQE_PDF_TOOLS / "etc" / "fonts")
        env["FC_CONFIG_DIR"] = str(_OQE_PDF_TOOLS / "etc" / "fonts")
        fonts_conf = _OQE_PDF_TOOLS / "etc" / "fonts" / "fonts.conf"
        if fonts_conf.is_file():
            env["FC_CONFIG_FILE"] = str(fonts_conf)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"pandoc+tectonic failed: {result.stderr.strip()}")
    return output_pdf


# ---------------------------------------------------------------------------
# Backend: Pandoc + Chrome headless (legacy)
# ---------------------------------------------------------------------------

def _render_pandoc_chrome(
    md_path: Path,
    output_pdf: Path,
    pandoc: str,
    chrome: str,
    css_path: Path | None = None,
) -> Path:
    """Render Markdown → HTML (Pandoc) → PDF (Chrome headless)."""
    html_str = _pandoc_md_to_html(md_path, css_path, pandoc)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html_str)
        tmp_html = f.name

    try:
        cmd = [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={output_pdf}",
            f"file://{tmp_html}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Chrome headless failed: {result.stderr.strip()}")
        return output_pdf
    finally:
        os.unlink(tmp_html)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_markdown_to_pdf(
    md_path: str | Path,
    output_pdf: str | Path | None = None,
    css_path: str | Path | None = None,
    backend: str | None = None,
) -> Path:
    """Render a Markdown file to PDF.

    Parameters
    ----------
    md_path : str or Path
        Path to the Markdown file.
    output_pdf : str or Path, optional
        Output PDF path.  Defaults to ``<md_stem>.pdf`` next to the source.
    css_path : str or Path, optional
        Custom CSS for the HTML intermediate.  Falls back to the project
        default stylesheet or the built-in report CSS.
    backend : str, optional
        Force a backend: ``"weasyprint"``, ``"tectonic"``, ``"chrome"``.
        When *None* (default), backends are tried in priority order.

    Returns
    -------
    Path
        The path to the generated PDF.

    Raises
    ------
    RuntimeError
        If no backend can produce the PDF.
    """
    md_path = Path(md_path).resolve()
    if not md_path.is_file():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    if output_pdf is None:
        output_pdf = md_path.with_suffix(".pdf")
    output_pdf = Path(output_pdf).resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    css_path = Path(css_path) if css_path else None
    if css_path is None or not css_path.is_file():
        # Use project default if it exists, otherwise use the built-in CSS
        css_path = _DEFAULT_CSS if _DEFAULT_CSS.is_file() else None

    pandoc = _find_pandoc()
    backends_tried: list[str] = []
    errors: list[str] = []

    # Determine backend order
    if backend:
        order = [backend]
    else:
        order = ["weasyprint", "tectonic", "chrome"]

    for name in order:
        try:
            if name == "weasyprint":
                if not pandoc:
                    errors.append("weasyprint: pandoc not found for Markdown→HTML conversion")
                    backends_tried.append(name)
                    continue
                if not _weasyprint_available():
                    errors.append("weasyprint: not importable (missing C libraries?)")
                    backends_tried.append(name)
                    continue

                html_str = _pandoc_md_to_html(md_path, css_path, pandoc)
                # Always inject the built-in report CSS for consistent styling
                html_str = _inject_css_into_html(html_str, _REPORT_CSS)
                _render_weasyprint(html_str, output_pdf)

            elif name == "tectonic":
                tectonic = _find_tectonic()
                if not pandoc:
                    errors.append("tectonic: pandoc not found")
                    backends_tried.append(name)
                    continue
                if not tectonic:
                    errors.append("tectonic: tectonic not found")
                    backends_tried.append(name)
                    continue
                _render_pandoc_tectonic(md_path, output_pdf, pandoc, tectonic)

            elif name == "chrome":
                chrome = _find_chrome()
                if not pandoc:
                    errors.append("chrome: pandoc not found")
                    backends_tried.append(name)
                    continue
                if not chrome:
                    errors.append("chrome: Chrome/Chromium not found")
                    backends_tried.append(name)
                    continue
                _render_pandoc_chrome(md_path, output_pdf, pandoc, chrome, css_path)

            else:
                errors.append(f"unknown backend: {name}")
                backends_tried.append(name)
                continue

            logger.info("PDF rendered via %s: %s", name, output_pdf)
            return output_pdf

        except Exception as exc:
            errors.append(f"{name}: {exc}")
            backends_tried.append(name)
            logger.warning("Backend %s failed: %s", name, exc)
            continue

    msg = "All PDF backends failed:\n" + "\n".join(f"  - {e}" for e in errors)
    raise RuntimeError(msg)


def render_all_daily_reports(output_dir: str | Path | None = None) -> list[Path]:
    """Render all Markdown daily reports to PDF."""
    reports_dir = _ROOT / "documentation" / "daily_reports"
    if not reports_dir.is_dir():
        logger.warning("No daily_reports directory found at %s", reports_dir)
        return []

    results = []
    for md_file in sorted(reports_dir.glob("*.md")):
        out = Path(output_dir) / md_file.with_suffix(".pdf").name if output_dir else None
        try:
            pdf = render_markdown_to_pdf(md_file, output_pdf=out)
            results.append(pdf)
            print(f"  OK: {pdf.name}")
        except Exception as exc:
            print(f"  FAIL: {md_file.name} — {exc}", file=sys.stderr)
    return results


def render_all_research_notes(output_dir: str | Path | None = None) -> list[Path]:
    """Render all Markdown research notes to PDF."""
    notes_dir = _ROOT / "documentation" / "research_notes"
    if not notes_dir.is_dir():
        logger.warning("No research_notes directory found at %s", notes_dir)
        return []

    results = []
    for md_file in sorted(notes_dir.glob("*.md")):
        out = Path(output_dir) / md_file.with_suffix(".pdf").name if output_dir else None
        try:
            pdf = render_markdown_to_pdf(md_file, output_pdf=out)
            results.append(pdf)
            print(f"  OK: {pdf.name}")
        except Exception as exc:
            print(f"  FAIL: {md_file.name} — {exc}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render Markdown reports to professional PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/render_pdf.py documentation/daily_reports/report.md
  python scripts/render_pdf.py report.md -o output.pdf --backend tectonic
  python scripts/render_pdf.py --all-daily
  python scripts/render_pdf.py --all-research-notes
  python scripts/render_pdf.py --all-daily --all-research-notes
""",
    )
    parser.add_argument("input", nargs="?", help="Markdown file to render")
    parser.add_argument("-o", "--output", help="Output PDF path")
    parser.add_argument("--css", help="Custom CSS stylesheet path")
    parser.add_argument(
        "--backend",
        choices=["weasyprint", "tectonic", "chrome"],
        help="Force a specific rendering backend",
    )
    parser.add_argument("--all-daily", action="store_true", help="Render all daily reports")
    parser.add_argument("--all-research-notes", action="store_true", help="Render all research notes")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.input and not args.all_daily and not args.all_research_notes:
        parser.error("Provide a Markdown file or use --all-daily / --all-research-notes")

    rendered: list[Path] = []

    if args.all_daily:
        print("Rendering daily reports...")
        rendered.extend(render_all_daily_reports())

    if args.all_research_notes:
        print("Rendering research notes...")
        rendered.extend(render_all_research_notes())

    if args.input:
        pdf = render_markdown_to_pdf(
            args.input,
            output_pdf=args.output,
            css_path=args.css,
            backend=args.backend,
        )
        rendered.append(pdf)
        print(f"PDF: {pdf}")

    if rendered:
        print(f"\n{len(rendered)} PDF(s) rendered successfully.")
    else:
        print("No PDFs rendered.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Bootstrap project imports
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    main()
