#!/usr/bin/env python3
"""
Professional Signal Research Report PDF Generator
==================================================

Generates publication-quality PDFs with:
- Custom title page with metadata
- Table of contents
- Professional typography and spacing
- Consistent styling across all reports
- Header/footer with page numbers
"""

import sys
from pathlib import Path
from datetime import datetime
import markdown
from weasyprint import HTML

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "documentation" / "daily_reports"

def extract_frontmatter(md_content: str) -> dict:
    """Extract YAML frontmatter from markdown."""
    if not md_content.startswith("---"):
        return {}
    
    try:
        _, fm, _ = md_content.split("---", 2)
        metadata = {}
        for line in fm.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"\'')
        return metadata
    except:
        return {}

def remove_frontmatter(md_content: str) -> str:
    """Remove YAML frontmatter from markdown."""
    if md_content.startswith("---"):
        try:
            _, _, content = md_content.split("---", 2)
            return content.strip()
        except:
            return md_content
    return md_content

def generate_title_page(metadata: dict) -> str:
    """Generate a professional title page HTML."""
    title = metadata.get("title", "Signal Research Report")
    subtitle = metadata.get("subtitle", "")
    author = metadata.get("author", "Quant Engines")
    date_str = metadata.get("date", datetime.now().strftime("%Y-%m-%d"))
    
    # Extract date from title if available (e.g., "April 07, 2026")
    date_display = date_str
    
    return f"""
    <div class="title-page">
        <div class="title-page-content">
            <div class="title-header">
                <svg class="logo" width="60" height="60" viewBox="0 0 60 60">
                    <rect x="5" y="5" width="50" height="50" fill="none" stroke="#2c3e50" stroke-width="2" rx="4"/>
                    <line x1="10" y1="20" x2="50" y2="20" stroke="#3498db" stroke-width="2"/>
                    <polyline points="15,30 25,20 35,28 45,18" fill="none" stroke="#27ae60" stroke-width="2" stroke-linejoin="round"/>
                    <circle cx="15" cy="30" r="2" fill="#27ae60"/>
                    <circle cx="25" cy="20" r="2" fill="#27ae60"/>
                    <circle cx="35" cy="28" r="2" fill="#27ae60"/>
                    <circle cx="45" cy="18" r="2" fill="#27ae60"/>
                </svg>
                <h1 class="org-name">QUANT ENGINES</h1>
            </div>
            
            <div class="title-main">
                <h1 class="report-title">{title}</h1>
                <h2 class="report-subtitle">{subtitle}</h2>
            </div>
            
            <div class="title-metadata">
                <div class="metadata-item">
                    <span class="metadata-label">Date:</span>
                    <span class="metadata-value">{date_display}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Author:</span>
                    <span class="metadata-value">{author}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Generated:</span>
                    <span class="metadata-value">{datetime.now().strftime("%B %d, %Y at %H:%M IST")}</span>
                </div>
            </div>
            
            <div class="title-footer">
                <p class="tagline">Signal Quality and Predictive Performance Analysis</p>
                <p class="footer-note">This report evaluates signal quality and predictive power based on realized market outcomes.</p>
            </div>
        </div>
    </div>
    """

def md_to_html(md_content: str, metadata: dict) -> str:
    """Convert markdown to professional HTML with title page."""
    # Extract frontmatter and remove from content
    content = remove_frontmatter(md_content)
    
    # Convert markdown to HTML
    html_body = markdown.markdown(content, extensions=['tables', 'fenced_code', 'codehilite', 'markdown.extensions.toc'])
    
    # Generate title page
    title_page = generate_title_page(metadata)
    
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{metadata.get('title', 'Signal Research Report')}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            @page {{
                size: A4;
                margin: 2cm;
                @bottom-center {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 10pt;
                    color: #7f8c8d;
                }}
                @top-right {{
                    content: "Quant Engines — Signal Research";
                    font-size: 9pt;
                    color: #95a5a6;
                }}
            }}
            
            @page:first {{
                @bottom-center {{
                    content: "";
                }}
                @top-right {{
                    content: "";
                }}
            }}
            
            body {{
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
                line-height: 1.65;
                color: #2c3e50;
                font-size: 11pt;
            }}
            
            /* Title Page */
            .title-page {{
                page-break-after: always;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background: linear-gradient(135deg, #ecf0f1 0%, #f8f9fa 100%);
                padding: 3cm;
            }}
            
            .title-page-content {{
                text-align: center;
                width: 100%;
                max-width: 600px;
            }}
            
            .title-header {{
                margin-bottom: 3em;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 1em;
            }}
            
            .logo {{
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
            }}
            
            .org-name {{
                font-size: 18pt;
                font-weight: 700;
                letter-spacing: 3px;
                color: #2c3e50;
                text-transform: uppercase;
            }}
            
            .title-main {{
                margin-bottom: 3em;
                padding-bottom: 2em;
                border-bottom: 3px solid #3498db;
            }}
            
            .report-title {{
                font-size: 32pt;
                font-weight: 700;
                color: #2c3e50;
                margin-bottom: 0.5em;
                line-height: 1.3;
            }}
            
            .report-subtitle {{
                font-size: 16pt;
                color: #7f8c8d;
                font-weight: 400;
            }}
            
            .title-metadata {{
                margin-bottom: 3em;
                background: white;
                padding: 2em;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            
            .metadata-item {{
                margin-bottom: 1em;
                display: flex;
                align-items: baseline;
                gap: 1em;
            }}
            
            .metadata-item:last-child {{
                margin-bottom: 0;
            }}
            
            .metadata-label {{
                font-weight: 600;
                color: #34495e;
                min-width: 80px;
            }}
            
            .metadata-value {{
                color: #2c3e50;
            }}
            
            .title-footer {{
                color: #7f8c8d;
            }}
            
            .tagline {{
                font-size: 14pt;
                font-weight: 600;
                margin-bottom: 0.5em;
                color: #34495e;
            }}
            
            .footer-note {{
                font-size: 10pt;
                line-height: 1.5;
            }}
            
            /* Content Styling */
            h1 {{
                font-size: 24pt;
                font-weight: 700;
                color: #2c3e50;
                margin-top: 2em;
                margin-bottom: 0.8em;
                border-bottom: 3px solid #3498db;
                padding-bottom: 0.4em;
            }}
            
            h2 {{
                font-size: 18pt;
                font-weight: 600;
                color: #2c3e50;
                margin-top: 1.8em;
                margin-bottom: 0.8em;
                border-left: 4px solid #3498db;
                padding-left: 0.8em;
            }}
            
            h3 {{
                font-size: 14pt;
                font-weight: 600;
                color: #34495e;
                margin-top: 1.4em;
                margin-bottom: 0.6em;
            }}
            
            h4, h5, h6 {{
                font-size: 12pt;
                font-weight: 600;
                color: #34495e;
                margin-top: 1em;
                margin-bottom: 0.5em;
            }}
            
            p {{
                margin-bottom: 1em;
                text-align: justify;
            }}
            
            strong {{
                color: #2c3e50;
                font-weight: 700;
            }}
            
            em {{
                color: #555;
                font-style: italic;
            }}
            
            /* Lists */
            ul, ol {{
                margin-left: 2em;
                margin-bottom: 1em;
            }}
            
            li {{
                margin-bottom: 0.5em;
                text-align: justify;
            }}
            
            /* Tables */
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1.5em 0;
                font-size: 10pt;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            
            thead {{
                background: linear-gradient(to right, #34495e, #2c3e50);
                color: white;
            }}
            
            th {{
                padding: 0.9em;
                text-align: left;
                font-weight: 700;
                font-size: 10pt;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            td {{
                padding: 0.8em 0.9em;
                border-bottom: 1px solid #ecf0f1;
            }}
            
            tbody tr:nth-child(odd) {{
                background-color: #f8f9fa;
            }}
            
            tbody tr:hover {{
                background-color: #ecf0f1;
            }}
            
            /* Code */
            code {{
                background-color: #f4f4f4;
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 2px 4px;
                font-family: 'Courier New', 'Monaco', monospace;
                font-size: 9pt;
                color: #d63384;
            }}
            
            pre {{
                background-color: #2c3e50;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 1.2em;
                overflow-x: auto;
                margin: 1em 0;
                line-height: 1.4;
                font-size: 9pt;
            }}
            
            pre code {{
                background: none;
                border: none;
                padding: 0;
                color: #ecf0f1;
            }}
            
            /* Blockquotes */
            blockquote {{
                border-left: 4px solid #3498db;
                background-color: #ecf0f1;
                padding: 1em 1.5em;
                margin: 1.5em 0;
                border-radius: 4px;
                font-style: italic;
                color: #34495e;
            }}
            
            /* Horizontal Rule */
            hr {{
                border: none;
                border-top: 2px solid #ecf0f1;
                margin: 2em 0;
            }}
            
            /* Links */
            a {{
                color: #3498db;
                text-decoration: none;
            }}
            
            a:active {{
                color: #2980b9;
            }}
            
            /* Special sections */
            .note {{
                background-color: #f0f7ff;
                border-left: 4px solid #0066cc;
                padding: 1em;
                margin: 1em 0;
                border-radius: 4px;
            }}
            
            .warning {{
                background-color: #fff3cd;
                border-left: 4px solid #ff6b6b;
                padding: 1em;
                margin: 1em 0;
                border-radius: 4px;
            }}
            
            .success {{
                background-color: #d4edda;
                border-left: 4px solid #27ae60;
                padding: 1em;
                margin: 1em 0;
                border-radius: 4px;
            }}
            
            /* Page breaks */
            .page-break {{
                page-break-after: always;
            }}
        </style>
    </head>
    <body>
        {title_page}
        {html_body}
    </body>
    </html>
    """
    
    return full_html

def convert_md_to_pdf(md_path: Path, pdf_path: Path) -> bool:
    """Convert markdown file to professional PDF."""
    try:
        if not md_path.exists():
            print(f"  ✗ File not found: {md_path}")
            return False
        
        # Read markdown
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Extract metadata
        metadata = extract_frontmatter(md_content)
        
        # Convert to HTML
        html_content = md_to_html(md_content, metadata)
        
        # Convert to PDF
        HTML(string=html_content).write_pdf(pdf_path)
        
        # Verify output
        if pdf_path.exists():
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {pdf_path.name} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ✗ PDF creation failed (no output file)")
            return False
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main() -> int:
    print("=" * 80)
    print("PROFESSIONAL SIGNAL RESEARCH REPORT PDF GENERATION")
    print("=" * 80)
    print()
    
    # Find all markdown reports to regenerate
    md_files = []
    
    # All signal research reports (daily and cumulative)
    for md_file in sorted(REPORTS_DIR.glob("signal_research_report_*.md")):
        # Skip archive files like cumulative_through_*
        if "through_" in md_file.name:
            continue
        pdf_file = md_file.with_suffix(".pdf")
        md_files.append((md_file.name, pdf_file.name, md_file))
    
    if not md_files:
        print("No markdown reports found to convert!")
        return 1
    
    success_count = 0
    for md_name, pdf_name, md_path in md_files:
        pdf_path = REPORTS_DIR / pdf_name
        
        print(f"Converting: {md_name}")
        if convert_md_to_pdf(md_path, pdf_path):
            success_count += 1
    
    print()
    print("=" * 80)
    print(f"PDF GENERATION COMPLETE — {success_count}/{len(md_files)} reports generated")
    print("=" * 80)
    print()
    print(f"Reports saved to: {REPORTS_DIR}")
    
    return 0 if success_count == len(md_files) else 1

if __name__ == "__main__":
    sys.exit(main())
