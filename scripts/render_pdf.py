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
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PDF_COVER_AUTHOR = "Pramit Dutta"
PDF_COVER_ORGANIZATION = "Quant Engines"


BACKEND_CATEGORY = {
    "tectonic": "latex",
    "weasyprint": "html_css",
    "chrome": "browser",
}

CATEGORY_RENDER_TIME_THRESHOLDS_SECONDS = {
    "latex": 6.0,
    "html_css": 2.5,
    "browser": 4.0,
    "unknown": 3.0,
}

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

.oqe-cover {
    min-height: 88vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    padding: 8vh 0;
    page-break-after: always;
}
.oqe-cover-kicker {
    font-size: 0.9rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
}
.oqe-cover-title {
    font-size: 2.1rem;
    line-height: 1.2;
    margin: 0 0 1.4rem;
    color: var(--ink);
}
.oqe-cover-meta {
    font-size: 1rem;
    color: #23303d;
    margin: 0.18rem 0;
}
.oqe-cover-rule {
    width: 78%;
    border-top: 1px solid var(--rule);
    margin: 1.2rem 0;
}
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


def _derive_report_title(md_path: Path, md_text: str) -> str:
    """Derive report title from first H1 or fallback to filename."""
    for line in md_text.splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return md_path.stem.replace("_", " ").replace("-", " ").strip().title()


def _cover_date_string() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _build_cover_html(title: str) -> str:
    """Generate a deterministic branded cover page for HTML-based backends."""
    date_str = _cover_date_string()
    return (
        '<section class="oqe-cover">'
        '<div class="oqe-cover-kicker">Quant Research Document</div>'
        f'<h1 class="oqe-cover-title">{title}</h1>'
        '<div class="oqe-cover-rule"></div>'
        f'<p class="oqe-cover-meta"><strong>Author:</strong> {PDF_COVER_AUTHOR}</p>'
        f'<p class="oqe-cover-meta"><strong>Organisation:</strong> {PDF_COVER_ORGANIZATION}</p>'
        f'<p class="oqe-cover-meta"><strong>Generated:</strong> {date_str}</p>'
        '</section>'
    )


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
        "--variable=secnumdepth:2",
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


def _inject_cover_into_html(html: str, title: str) -> str:
    """Inject branded cover page right after <body>."""
    cover_html = _build_cover_html(title)
    if "<body" in html:
        body_match = re.search(r"<body[^>]*>", html, flags=re.IGNORECASE)
        if body_match:
            insert_at = body_match.end()
            return html[:insert_at] + "\n" + cover_html + "\n" + html[insert_at:]
    return f"<html><body>{cover_html}{html}</body></html>"


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
    title: str,
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
        "--variable=secnumdepth:2",
        "-M", f"title={title}",
        "-M", f"author=Author: {PDF_COVER_AUTHOR}",
        "-M", f"subtitle=Organisation: {PDF_COVER_ORGANIZATION}",
        "-M", f"date=Generated: {_cover_date_string()}",
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
    title: str = "",
) -> Path:
    """Render Markdown → HTML (Pandoc) → PDF (Chrome headless)."""
    html_str = _pandoc_md_to_html(md_path, css_path, pandoc)
    html_str = _inject_css_into_html(html_str, _REPORT_CSS)
    html_str = _inject_cover_into_html(html_str, title)

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
    return_details: bool = False,
) -> Path | dict[str, Any]:
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
    Path or dict
        By default returns the generated PDF path.
        When ``return_details=True``, returns a dict with
        ``pdf_path``, ``backend``, ``backend_category``, ``has_math``,
        and ``backend_order``.

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

    md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    math_patterns = (
        r"\$\$",            # display math
        r"\\\(",          # inline math \(...\)
        r"\\\[",          # display math \[...\]
        r"(?<!\\)\$(?:[^\n$]|\\\$){1,200}(?<!\\)\$",  # inline $...$
    )
    has_math = any(re.search(pattern, md_text, flags=re.MULTILINE) for pattern in math_patterns)
    report_title = _derive_report_title(md_path, md_text)

    # Determine backend order
    if backend:
        order = [backend]
    else:
        # Prefer LaTeX pipeline for math-heavy docs; keep HTML pipeline first otherwise.
        order = ["tectonic", "weasyprint", "chrome"] if has_math else ["weasyprint", "tectonic", "chrome"]

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
                html_str = _inject_cover_into_html(html_str, report_title)
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
                _render_pandoc_tectonic(md_path, output_pdf, pandoc, tectonic, report_title)

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
                _render_pandoc_chrome(md_path, output_pdf, pandoc, chrome, css_path, report_title)

            else:
                errors.append(f"unknown backend: {name}")
                backends_tried.append(name)
                continue

            backend_category = BACKEND_CATEGORY.get(name, "unknown")
            logger.info("PDF rendered via %s: %s", name, output_pdf)
            if return_details:
                return {
                    "pdf_path": output_pdf,
                    "backend": name,
                    "backend_category": backend_category,
                    "has_math": has_math,
                    "backend_order": order,
                    "cover_applied": True,
                    "cover_author": PDF_COVER_AUTHOR,
                    "cover_organization": PDF_COVER_ORGANIZATION,
                }
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


def _discover_documentation_markdown(include_archive: bool = False) -> list[Path]:
    """Discover all Markdown files under documentation, optionally including archive paths."""
    docs_dir = _ROOT / "documentation"
    if not docs_dir.is_dir():
        return []

    md_files = sorted(docs_dir.rglob("*.md"))
    if include_archive:
        return md_files

    filtered: list[Path] = []
    for md_file in md_files:
        rel = md_file.relative_to(docs_dir).as_posix()
        if rel.startswith("archive/"):
            continue
        filtered.append(md_file)
    return filtered


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
    parser.add_argument("--all-docs", action="store_true", help="Render all Markdown files under documentation/")
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Include documentation/archive/** when used with --all-docs",
    )
    parser.add_argument(
        "--report-json",
        help="Write render results to a JSON report file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any render target fails",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.input and not args.all_daily and not args.all_research_notes and not args.all_docs:
        parser.error("Provide a Markdown file or use --all-daily / --all-research-notes / --all-docs")

    rendered: list[Path] = []
    failures: list[dict[str, str]] = []
    report_rows: list[dict[str, str | float]] = []

    def _record_render(
        md_path: Path,
        out_pdf: Path,
        ok: bool,
        error: str | None = None,
        *,
        backend: str = "",
        backend_category: str = "unknown",
        has_math: bool = False,
        cover_applied: bool = True,
        cover_author: str = PDF_COVER_AUTHOR,
        cover_organization: str = PDF_COVER_ORGANIZATION,
    ):
        expected_category = "latex" if has_math else "html_css"
        threshold_seconds = CATEGORY_RENDER_TIME_THRESHOLDS_SECONDS.get(
            backend_category,
            CATEGORY_RENDER_TIME_THRESHOLDS_SECONDS["unknown"],
        )
        row = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "markdown": str(md_path),
            "pdf": str(out_pdf),
            "ok": ok,
            "error": error or "",
            "backend": backend,
            "backend_category": backend_category,
            "has_math": bool(has_math),
            "cover_applied": bool(cover_applied),
            "cover_author": cover_author,
            "cover_organization": cover_organization,
            "expected_backend_category": expected_category,
            "render_time_threshold_seconds": float(threshold_seconds),
            "performance_alert": False,
            "quality_alert": False,
            "alerts": [],
        }
        report_rows.append(row)

    def _render_many(md_files: list[Path]):
        for md_file in md_files:
            out_pdf = md_file.with_suffix(".pdf")
            started = time.time()
            try:
                result = render_markdown_to_pdf(
                    md_file,
                    output_pdf=out_pdf,
                    css_path=args.css,
                    backend=args.backend,
                    return_details=True,
                )
                elapsed = round(time.time() - started, 3)
                pdf = result["pdf_path"]
                rendered.append(pdf)
                _record_render(
                    md_file,
                    pdf,
                    True,
                    backend=str(result.get("backend", "")),
                    backend_category=str(result.get("backend_category", "unknown")),
                    has_math=bool(result.get("has_math", False)),
                    cover_applied=bool(result.get("cover_applied", True)),
                    cover_author=str(result.get("cover_author", PDF_COVER_AUTHOR)),
                    cover_organization=str(result.get("cover_organization", PDF_COVER_ORGANIZATION)),
                )
                report_rows[-1]["elapsed_seconds"] = elapsed
                threshold = float(report_rows[-1]["render_time_threshold_seconds"])
                if elapsed > threshold:
                    report_rows[-1]["performance_alert"] = True
                    report_rows[-1]["alerts"].append(
                        f"render_time_exceeded:{elapsed}>{threshold}"
                    )
                if bool(report_rows[-1]["has_math"]) and report_rows[-1]["backend_category"] != "latex":
                    report_rows[-1]["quality_alert"] = True
                    report_rows[-1]["alerts"].append("math_doc_rendered_without_latex_backend")
                print(f"  OK: {md_file}")
            except Exception as exc:
                elapsed = round(time.time() - started, 3)
                _record_render(md_file, out_pdf, False, str(exc))
                report_rows[-1]["elapsed_seconds"] = elapsed
                report_rows[-1]["performance_alert"] = True
                report_rows[-1]["alerts"].append("render_failure")
                failures.append({"markdown": str(md_file), "error": str(exc)})
                print(f"  FAIL: {md_file} — {exc}", file=sys.stderr)

    if args.all_daily:
        print("Rendering daily reports...")
        _render_many(sorted((_ROOT / "documentation" / "daily_reports").glob("*.md")))

    if args.all_research_notes:
        print("Rendering research notes...")
        _render_many(sorted((_ROOT / "documentation" / "research_notes").glob("*.md")))

    if args.all_docs:
        print("Rendering all documentation markdown files...")
        _render_many(_discover_documentation_markdown(include_archive=args.include_archive))

    if args.input:
        started = time.time()
        try:
            result = render_markdown_to_pdf(
                args.input,
                output_pdf=args.output,
                css_path=args.css,
                backend=args.backend,
                return_details=True,
            )
            elapsed = round(time.time() - started, 3)
            pdf = result["pdf_path"]
            rendered.append(pdf)
            _record_render(
                Path(args.input).resolve(),
                pdf,
                True,
                backend=str(result.get("backend", "")),
                backend_category=str(result.get("backend_category", "unknown")),
                has_math=bool(result.get("has_math", False)),
                cover_applied=bool(result.get("cover_applied", True)),
                cover_author=str(result.get("cover_author", PDF_COVER_AUTHOR)),
                cover_organization=str(result.get("cover_organization", PDF_COVER_ORGANIZATION)),
            )
            report_rows[-1]["elapsed_seconds"] = elapsed
            threshold = float(report_rows[-1]["render_time_threshold_seconds"])
            if elapsed > threshold:
                report_rows[-1]["performance_alert"] = True
                report_rows[-1]["alerts"].append(
                    f"render_time_exceeded:{elapsed}>{threshold}"
                )
            if bool(report_rows[-1]["has_math"]) and report_rows[-1]["backend_category"] != "latex":
                report_rows[-1]["quality_alert"] = True
                report_rows[-1]["alerts"].append("math_doc_rendered_without_latex_backend")
            print(f"PDF: {pdf}")
        except Exception as exc:
            elapsed = round(time.time() - started, 3)
            target = Path(args.input).resolve()
            out_pdf = Path(args.output).resolve() if args.output else target.with_suffix(".pdf")
            _record_render(target, out_pdf, False, str(exc))
            report_rows[-1]["elapsed_seconds"] = elapsed
            report_rows[-1]["performance_alert"] = True
            report_rows[-1]["alerts"].append("render_failure")
            failures.append({"markdown": str(target), "error": str(exc)})
            print(f"FAIL: {target} — {exc}", file=sys.stderr)

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "rendered_count": len(rendered),
            "failure_count": len(failures),
            "performance_alert_count": sum(1 for row in report_rows if row.get("performance_alert")),
            "quality_alert_count": sum(1 for row in report_rows if row.get("quality_alert")),
            "strict": bool(args.strict),
            "thresholds": {
                "backend_category_seconds": CATEGORY_RENDER_TIME_THRESHOLDS_SECONDS,
            },
            "rows": report_rows,
        }
        report_path.write_text(json.dumps(report_payload, indent=2))
        print(f"Render report: {report_path}")

    if rendered:
        print(f"\n{len(rendered)} PDF(s) rendered successfully.")
        if failures:
            print(f"{len(failures)} target(s) failed.", file=sys.stderr)
            if args.strict:
                sys.exit(2)
    else:
        print("No PDFs rendered.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Bootstrap project imports
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    main()
