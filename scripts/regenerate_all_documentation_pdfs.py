"""
Batch Markdown to PDF Converter — Documentation Suite
====================================================
Converts all Markdown files in /documentation to professionally styled PDFs
using consistent navy/gold branding.

Usage:
    python scripts/regenerate_all_documentation_pdfs.py

Applies to:
    - research_notes/*.md
    - parameter_tuning/*.md
    - audits/*.md
    - system_docs/*.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    HRFlowable,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    Preformatted,
)
from reportlab.platypus.frames import Frame
from reportlab.lib.colors import HexColor
import markdown

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DOC_DIRS = [
    ROOT / "documentation" / "research_notes",
    ROOT / "documentation" / "parameter_tuning",
    ROOT / "documentation" / "audits",
    ROOT / "documentation" / "system_docs",
    ROOT / "documentation" / "daily_reports",
]

# ── Brand Colours ─────────────────────────────────────────────────────────────
NAVY      = HexColor("#0D1B2A")
NAVY_MID  = HexColor("#1A3A5C")
GOLD      = HexColor("#E8A020")
GOLD_LIGHT= HexColor("#F5C842")
SLATE     = HexColor("#3A4A5C")
WHITE     = colors.white
OFF_WHITE = HexColor("#F7F9FC")
LIGHT_GREY= HexColor("#E8ECF0")
MID_GREY  = HexColor("#9AA8B8")
DARK_TEXT = HexColor("#1A2535")

PAGE_W, PAGE_H = A4

# ── Styles ────────────────────────────────────────────────────────────────────
def build_styles() -> dict:
    base = getSampleStyleSheet()
    s = {}

    s["title"] = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=NAVY,
        spaceAfter=12,
        alignment=TA_LEFT,
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle",
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=SLATE,
        spaceAfter=8,
        alignment=TA_LEFT,
    )
    s["h1"] = ParagraphStyle(
        "h1",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=6,
    )
    s["h2"] = ParagraphStyle(
        "h2",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=NAVY_MID,
        spaceBefore=8,
        spaceAfter=4,
    )
    s["h3"] = ParagraphStyle(
        "h3",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=SLATE,
        spaceBefore=6,
        spaceAfter=3,
    )
    s["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=DARK_TEXT,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    )
    s["bullet"] = ParagraphStyle(
        "bullet",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=DARK_TEXT,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=2,
    )
    s["code"] = ParagraphStyle(
        "code",
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=DARK_TEXT,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=4,
        backColor=OFF_WHITE,
    )
    s["table_header"] = ParagraphStyle(
        "table_header",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=WHITE,
        alignment=TA_CENTER,
    )
    s["table_cell"] = ParagraphStyle(
        "table_cell",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=DARK_TEXT,
        alignment=TA_LEFT,
    )
    s["caption"] = ParagraphStyle(
        "caption",
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10,
        textColor=MID_GREY,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    s["footer"] = ParagraphStyle(
        "footer",
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=MID_GREY,
        alignment=TA_CENTER,
    )

    return s


# ── Page Templates ────────────────────────────────────────────────────────────
def on_page(canvas, doc):
    """Footer for body pages."""
    canvas.saveState()
    w, h = A4
    # bottom rule
    canvas.setStrokeColor(LIGHT_GREY)
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 1.5*cm, w - 2*cm, 1.5*cm)
    # footer text
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(2*cm, 1.1*cm, "Quant Engines  ·  Options Quantitative Engine")
    canvas.drawRightString(w - 2*cm, 1.1*cm, f"Page {doc.page}")
    # top thin gold bar
    canvas.setFillColor(GOLD)
    canvas.rect(0, h - 3*mm, w, 3*mm, fill=1, stroke=0)
    canvas.restoreState()


def markdown_to_html(text: str) -> str:
    """Convert markdown formatting to simple HTML safe for reportlab.
    
    Processes in order: code (backticks) → bold (**) → italic (*) → links
    This prevents underscores in code from being converted to italic tags.
    """
    if not text:
        return text
    
    # Step 1: Extract and protect code blocks (backticks) with placeholders
    code_blocks = []
    def protect_code(match):
        code_blocks.append(f'<font face="Courier">{match.group(1)}</font>')
        return f'__CODE_BLOCK_{len(code_blocks)-1}__'
    
    text = re.sub(r'`([^`]+)`', protect_code, text)
    
    # Step 2: Handle bold (must come before italic to avoid conflicts)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__([^_]+)__', r'<b>\1</b>', text)
    
    # Step 3: Handle italic (* and _ are different - only match within word boundaries
    # or surrounded by spaces to avoid matching underscores in identifiers)
    text = re.sub(r'\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    # Only match underscores if surrounded by spaces or at boundaries
    text = re.sub(r'(?:^|\s)_([^_\s]+)_(?:\s|$)', lambda m: f' <i>{m.group(1)}</i> ', text)
    
    # Step 4: Handle links - just keep the text part
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\1', text)
    
    # Step 5: Restore code blocks
    for i, code_block in enumerate(code_blocks):
        text = text.replace(f'__CODE_BLOCK_{i}__', code_block)
    
    return text


# ── Markdown parsing ──────────────────────────────────────────────────────────
def parse_markdown_to_flowables(content: str, styles: dict) -> list:
    """Convert markdown content to reportlab flowables."""
    story = []
    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        # Headers
        if line.startswith('# ') and not line.startswith('##'):
            story.append(Spacer(1, 4*mm))
            story.append(Paragraph(line[2:].strip(), styles["title"]))
            story.append(Spacer(1, 3*mm))
            i += 1
        elif line.startswith('## '):
            story.append(Spacer(1, 3*mm))
            story.append(Paragraph(line[3:].strip(), styles["h1"]))
            story.append(Spacer(1, 2*mm))
            i += 1
        elif line.startswith('### '):
            story.append(Paragraph(line[4:].strip(), styles["h2"]))
            story.append(Spacer(1, 1.5*mm))
            i += 1
        elif line.startswith('#### '):
            story.append(Paragraph(line[5:].strip(), styles["h3"]))
            i += 1

        # Bullet points
        elif line.startswith('- ') or line.startswith('* '):
            bullet_text = line[2:].strip()
            bullet_text = markdown_to_html(bullet_text)
            story.append(Paragraph(bullet_text, styles["bullet"]))
            i += 1

        # Code blocks
        elif line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            code_text = '\n'.join(code_lines).strip()
            story.append(Preformatted(code_text, styles["code"]))
            story.append(Spacer(1, 2*mm))
            i += 1

        # Horizontal rule
        elif line.strip() in ['---', '***', '___']:
            story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GREY, spaceAfter=4))
            i += 1

        # Blockquote
        elif line.startswith('> '):
            quote = line[2:].strip()
            quote = markdown_to_html(quote)
            story.append(Paragraph(f'<i>{quote}</i>', ParagraphStyle(
                "blockquote", fontName="Helvetica-Oblique", fontSize=9,
                leading=13, textColor=SLATE, leftIndent=20, spaceAfter=4
            )))
            i += 1

        # Regular paragraph
        elif line.strip():
            # Collect consecutive non-empty lines as one paragraph
            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i][0] in ['#', '-', '>', '|', '`']:
                para_lines.append(lines[i])
                i += 1

            para_text = ' '.join(para_lines).strip()
            # Apply markdown formatting using proper regex patterns
            para_text = markdown_to_html(para_text)

            story.append(Paragraph(para_text, styles["body"]))

        else:
            # Empty line = spacer
            story.append(Spacer(1, 2*mm))
            i += 1

    return story


# ── PDF Builder ───────────────────────────────────────────────────────────────
def build_pdf_from_markdown(md_path: Path, styles: dict) -> Path:
    """Convert a markdown file to a polished PDF."""
    try:
        content = md_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  ✗ Error reading {md_path}: {e}")
        return None

    # Extract title from first # heading
    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else md_path.stem

    # Output PDF path
    pdf_path = md_path.with_suffix('.pdf')

    # Parse content to flowables
    try:
        flowables = parse_markdown_to_flowables(content, styles)
    except Exception as e:
        print(f"  ✗ Error parsing markdown in {md_path}: {e}")
        return None

    # Create PDF
    try:
        body_margin_l = 2.0*cm
        body_margin_r = 2.0*cm
        body_margin_t = 2.2*cm
        body_margin_b = 2.0*cm

        def on_page_wrapper(canvas, doc):
            on_page(canvas, doc)

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=body_margin_l,
            rightMargin=body_margin_r,
            topMargin=body_margin_t,
            bottomMargin=body_margin_b,
            title=title,
            author="Quant Engines",
        )

        # Add footer
        template = PageTemplate(
            frames=[Frame(
                body_margin_l, body_margin_b,
                PAGE_W - body_margin_l - body_margin_r,
                PAGE_H - body_margin_t - body_margin_b,
                id="frame"
            )],
            onPage=on_page_wrapper
        )
        doc.addPageTemplates([template])

        # Add title page spacer and content
        doc.build(flowables)
        return pdf_path

    except Exception as e:
        print(f"  ✗ Error building PDF for {md_path}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    print("\n" + "=" * 80)
    print("  BATCH DOCUMENTATION PDF REGENERATION")
    print("=" * 80)

    styles = build_styles()
    count_success = 0
    count_fail = 0

    for doc_dir in DOC_DIRS:
        if not doc_dir.exists():
            continue

        print(f"\n📁 {doc_dir.name}:")
        md_files = sorted(doc_dir.glob("*.md"))

        # Skip daily_reports if already there
        if "daily_reports" in doc_dir.name:
            md_files = [f for f in md_files if f.name != "README.md"]

        for md_file in md_files:
            # Skip files that are not documentation reports
            if md_file.name in ["README.md"]:
                continue

            pdf_path = build_pdf_from_markdown(md_file, styles)
            if pdf_path:
                size_kb = pdf_path.stat().st_size / 1024
                print(f"  ✅ {md_file.name:50s} → {size_kb:6.1f} KB")
                count_success += 1
            else:
                print(f"  ✗ {md_file.name:50s}")
                count_fail += 1

    print("\n" + "=" * 80)
    print(f"  COMPLETE: {count_success} success, {count_fail} failed")
    print("=" * 80 + "\n")

    return 0 if count_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
