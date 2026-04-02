#!/usr/bin/env python3
"""
Generate a professional PDF version of the Trading Guide.

This script reads the markdown file and uses weasyprint to create a beautiful, 
printable PDF directly from HTML.
"""

import sys
import re
from pathlib import Path
from datetime import datetime

try:
    from weasyprint import HTML
except ImportError:
    print("ERROR: weasyprint not found.")
    sys.exit(1)


def simple_markdown_to_html(md_text: str) -> str:
    """Simple markdown to HTML converter without external dependencies."""
    
    lines = md_text.split('\n')
    html_lines = []
    in_code = False
    in_table = False
    
    for line in lines:
        # Code blocks
        if line.startswith('```'):
            if in_code:
                html_lines.append('</pre>')
                in_code = False
            else:
                html_lines.append('<pre><code>')
                in_code = True
            continue
        
        if in_code:
            html_lines.append(line)
            continue
        
        # Headings
        if line.startswith('# '):
            html_lines.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('### '):
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('#### '):
            html_lines.append(f'<h4>{line[5:]}</h4>')
        elif line.startswith('##### '):
            html_lines.append(f'<h5>{line[6:]}</h5>')
        elif line.startswith('###### '):
            html_lines.append(f'<h6>{line[7:]}</h6>')
        
        # Tables
        elif line.startswith('|'):
            if not in_table:
                html_lines.append('<table>')
                in_table = True
            
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            
            # Check if separator row
            if all(re.match(r'^-+$', cell) for cell in cells):
                html_lines.append('</thead><tbody>')
                continue
            
            if in_table and not any(re.match(r'^-+$', cell) for cell in cells):
                if 'thead' not in ''.join(html_lines[-5:]):  # Rough check if we're in header
                    row_html = '<tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr></thead>'
                else:
                    row_html = '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
                html_lines.append(row_html)
        
        elif in_table and line.strip() == '':
            html_lines.append('</tbody></table>')
            in_table = False
        
        # Bullet lists
        elif line.startswith('- '):
            html_lines.append(f'<li>{line[2:]}</li>')
        
        # Paragraphs
        elif line.strip():
            # Inline formatting
            text = line
            text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)  # Italic
            text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)  # Code
            text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)  # Links
            html_lines.append(f'<p>{text}</p>')
        else:
            if line.strip() == '':
                html_lines.append('')
    
    return '\n'.join(html_lines)


def markdown_to_pdf(md_file: Path, output_pdf: Path) -> None:
    """Convert markdown file to PDF via HTML."""
    
    # Read markdown file
    print(f"📖 Reading: {md_file}")
    md_content = md_file.read_text(encoding="utf-8")
    
    # Convert to HTML
    print("🔄 Converting markdown to HTML...")
    html_body = simple_markdown_to_html(md_content)
    
    # Create styled HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>The Options Quant Engine: A Mathematical Guide to Trading</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            html {{
                font-size: 10pt;
                line-height: 1.6;
            }}
            
            body {{
                font-family: 'Segoe UI', 'Gill Sans', sans-serif;
                color: #333;
                background: white;
                padding: 1in;
                line-height: 1.7;
            }}
            
            h1 {{
                font-size: 2.2em;
                color: #003d99;
                margin: 1.5em 0 0.5em 0;
                page-break-after: avoid;
                border-bottom: 3px solid #0066cc;
                padding-bottom: 0.3em;
            }}
            
            h2 {{
                font-size: 1.8em;
                color: #0052a3;
                margin: 1.2em 0 0.4em 0;
                page-break-after: avoid;
                border-bottom: 2px solid #0066cc;
                padding-bottom: 0.2em;
            }}
            
            h3 {{
                font-size: 1.4em;
                color: #0066cc;
                margin: 1em 0 0.3em 0;
                page-break-after: avoid;
            }}
            
            h4, h5, h6 {{
                font-size: 1.1em;
                color: #004499;
                margin: 0.8em 0 0.2em 0;
            }}
            
            p {{
                margin-bottom: 0.8em;
            }}
            
            ul, ol {{
                margin: 0 0 0.8em 1.5em;
            }}
            
            li {{
                margin-bottom: 0.4em;
            }}
            
            code {{
                background: #f5f5f5;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 0.95em;
                color: #d73a49;
            }}
            
            pre {{
                background: #f8f8f8;
                padding: 12px;
                border-left: 3px solid #0066cc;
                border-radius: 4px;
                margin: 0.8em 0;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                line-height: 1.4;
                page-break-inside: avoid;
                overflow: auto;
            }}
            
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 1em 0;
                font-size: 0.95em;
                page-break-inside: avoid;
            }}
            
            table th {{
                background: #0066cc;
                color: white;
                padding: 8px;
                text-align: left;
                font-weight: 600;
            }}
            
            table td {{
                padding: 8px;
                border: 1px solid #ddd;
            }}
            
            table tr:nth-child(even) {{
                background: #f9f9f9;
            }}
            
            strong {{
                font-weight: 600;
                color: #0052a3;
            }}
            
            em {{
                font-style: italic;
                color: #555;
            }}
            
            a {{
                color: #0066cc;
                text-decoration: underline;
            }}
            
            @page {{
                size: A4;
                margin: 1in;
                @bottom-center {{
                    content: "Page " counter(page);
                    font-size: 9pt;
                    color: #999;
                }}
            }}
        </style>
    </head>
    <body>
        <h1>The Options Quant Engine</h1>
        <h3>A Mathematical Guide to Trading Using Systematic Signals</h3>
        <p style="text-align: center; color: #666; margin-top: 2em;">Version 1.0 | April 2, 2026</p>
        <hr style="margin: 2em 0; border: none; border-top: 2px solid #0066cc;">
        
        {html_body}
        
        <hr style="margin: 2em 0 1em 0; border: none; border-top: 2px solid #0066cc;">
        <p style="text-align: center; color: #999; font-size: 9pt;">
            Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}<br>
            The Options Quant Engine: A Mathematical Guide to Trading Using Systematic Signals
        </p>
    </body>
    </html>
    """
    
    # Generate PDF
    print(f"📄 Generating PDF: {output_pdf}")
    HTML(string=full_html).write_pdf(output_pdf)
    print(f"✅ PDF created successfully: {output_pdf}")
    print(f"📊 Size: {output_pdf.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    workspace_root = Path(__file__).parent
    md_file = workspace_root / "TRADING_GUIDE.md"
    pdf_file = workspace_root / "TRADING_GUIDE_v1.pdf"
    
    if not md_file.exists():
        print(f"ERROR: {md_file} not found!")
        sys.exit(1)
    
    try:
        markdown_to_pdf(md_file, pdf_file)
    except Exception as e:
        print(f"ERROR during PDF generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
