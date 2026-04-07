#!/usr/bin/env python3
"""
Convert Signal Evaluation Reports to PDF via Pandoc + Markdown to HTML
=======================================================================

Uses pandoc to convert markdown directly to PDF without external deps.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "documentation" / "daily_reports"


def convert_md_to_pdf_direct(md_path: Path, pdf_path: Path) -> bool:
    """Convert markdown directly to PDF using pandoc's default engine."""
    try:
        print(f"Converting: {md_path.name}")
        
        # Use pandoc to convert markdown directly to PDF
        # Using the default PDF engine without specifying pdflatex or xelatex
        cmd = [
            "pandoc",
            str(md_path),
            "-o", str(pdf_path),
            "--from", "markdown",
            "-V", "geometry:margin=1in",
            "--standalone",
            "--highlight-style=pygments",
            "-M", "title=Signal Evaluation Report",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"✓ Successfully generated: {pdf_path.name}")
            return True
        else:
            print(f"✗ Error with default engine: {result.stderr[:200]}")
            return convert_via_html_intermediate(md_path, pdf_path)
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout converting {md_path.name}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def convert_via_html_intermediate(md_path: Path, pdf_path: Path) -> bool:
    """Fallback: convert markdown to HTML, then HTML to PDF."""
    try:
        print(f"  → Trying HTML intermediate format...")
        
        # Convert markdown to HTML
        html_path = pdf_path.with_stem(pdf_path.stem + "_temp").with_suffix('.html')
        
        cmd_html = [
            "pandoc",
            str(md_path),
            "-o", str(html_path),
            "--from", "markdown",
            "--to", "html5",
            "--standalone",
            "--self-contained",
        ]
        
        result = subprocess.run(cmd_html, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  ✗ HTML conversion failed")
            return False
        
        # Read and enhance the HTML
        html_content = html_path.read_text(encoding='utf-8')
        
        # Inject better CSS
        enhanced_html = html_content.replace(
            '</head>',
            '''<style>
            body { margin: 1in; font-size: 11pt; line-height: 1.5; }
            h1 { border-bottom: 2pt solid #0066cc; padding-bottom: 0.5em; }
            h2 { color: #0066cc; margin-top: 1.5em; }
            table { border-collapse: collapse; margin: 1em 0; }
            th { background-color: #e8f0f8; padding: 0.5em; border: 1pt solid #999; }
            td { padding: 0.4em; border: 1pt solid #ddd; }
            blockquote { border-left: 3pt solid #0066cc; padding-left: 1em; }
            code { background-color: #f4f4f4; padding: 2pt 4pt; }
            @media print { h1, h2 { page-break-after: avoid; } table { page-break-inside: avoid; } }
            </style>
            </head>'''
        )
        
        html_path.write_text(enhanced_html, encoding='utf-8')
        
        # Convert HTML to PDF using pandoc
        cmd_pdf = [
            "pandoc",
            str(html_path),
            "-o", str(pdf_path),
            "--from", "html",
        ]
        
        result = subprocess.run(cmd_pdf, capture_output=True, text=True, timeout=60)
        html_path.unlink(missing_ok=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully generated: {pdf_path.name}")
            return True
        else:
            print(f"  ✗ PDF conversion failed: {result.stderr[:200]}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    base_dir = REPORTS_DIR
    
    reports = [
        ("signal_research_report_20260324.md", "signal_research_report_20260324.pdf"),
        ("signal_research_report_cumulative.md", "signal_research_report_cumulative.pdf"),
    ]
    
    print("=" * 80)
    print("SIGNAL EVALUATION REPORT PDF GENERATION (via Pandoc)")
    print("=" * 80)
    print()
    
    success_count = 0
    
    for md_name, pdf_name in reports:
        md_path = base_dir / md_name
        pdf_path = base_dir / pdf_name
        
        if not md_path.exists():
            print(f"✗ File not found: {md_path}")
            continue
        
        # Remove existing PDF to force fresh generation
        pdf_path.unlink(missing_ok=True)
        
        if convert_md_to_pdf_direct(md_path, pdf_path):
            success_count += 1
        print()
    
    print("=" * 80)
    print(f"PDF GENERATION COMPLETE — {success_count}/{len(reports)} reports generated")
    print("=" * 80)
    print()
    
    if success_count > 0:
        print("Generated PDFs:")
        for _, pdf_name in reports:
            pdf_path = base_dir / pdf_name
            if pdf_path.exists():
                size_kb = pdf_path.stat().st_size / 1024
                print(f"  ✓ {pdf_name} ({size_kb:.1f} KB)")
    
    return success_count == len(reports)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
