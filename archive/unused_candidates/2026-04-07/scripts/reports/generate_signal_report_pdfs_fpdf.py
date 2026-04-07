#!/usr/bin/env python3
"""
Convert Signal Evaluation Reports to PDF using fpdf2
======================================================

Pure Python PDF generation without external system dependencies.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "documentation" / "daily_reports"

try:
    from fpdf import FPDF
except ImportError as e:
    print(f"Missing fpdf2: {e}")
    sys.exit(1)


class MarkdownPDFConverter:
    """Convert markdown to PDF with formatting."""
    
    def __init__(self):
        self.pdf = FPDF()
        # Use built-in Helvetica font (always available)
        self.font_name = 'Helvetica'
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.set_font(self.font_name, '', 10)
    
    def _sanitize_text(self, text: str) -> str:
        """Remove or replace problematic Unicode characters."""
        # Replace common Unicode characters with ASCII equivalents
        replacements = {
            '\u2014': '-',  # em dash
            '\u2013': '-',  # en dash
            '\u2010': '-',  # hyphen
            '\u2012': '-',  # figure dash
            '\u2011': '-',  # non-breaking hyphen
            '\u2212': '-',  # minus
            '\u00b1': '+/-',  # plus-minus
            '\u2018': "'",  # left single quote
            '\u2019': "'",  # right single quote
            '\u201c': '"',  # left double quote
            '\u201d': '"',  # right double quote
            '\u2026': '...',  # ellipsis
            '\u20ac': 'EUR',  # euro
            '\u00d7': 'x',  # multiplication
            '\u00f7': '/',  # division
            '\u2192': '->',  # rightwards arrow
            '\u2190': '<-',  # leftwards arrow
            '\u2191': '^',  # upwards arrow
            '\u2193': 'v',  # downwards arrow
            '\u2713': '[OK]',  # checkmark (213)
            '\u2717': '[X]',  # cross mark
            '\u2705': '[OK]',  # white checkmark
            '\u274c': '[X]',  # cross mark
            '\u00e9': 'e',  # é
            '\u00e8': 'e',  # è
            '\u00ea': 'e',  # ê
        }
        for unicode_char, ascii_equiv in replacements.items():
            text = text.replace(unicode_char, ascii_equiv)
        # Strip any remaining non-Latin-1 characters
        return text.encode('latin-1', 'ignore').decode('latin-1')
    
    def add_title_page(self, title: str, date: str):
        """Add a title page."""
        self.pdf.add_page()
        self.pdf.set_font(self.font_name, 'B', 20)
        self.pdf.ln(40)
        self.pdf.cell(0, 10, self._sanitize_text(title), new_x='LMARGIN', new_y='NEXT', align='C')
        self.pdf.set_font(self.font_name, '', 12)
        self.pdf.ln(10)
        self.pdf.cell(0, 10, f"Generated: {date}", new_x='LMARGIN', new_y='NEXT', align='C')
        self.pdf.cell(0, 10, "Signal Evaluation Report", new_x='LMARGIN', new_y='NEXT', align='C')
        self.pdf.ln(20)
    
    def add_heading(self, text: str, level: int = 1):
        """Add a heading."""
        if level == 1:
            self.pdf.set_font(self.font_name, 'B', 14)
            self.pdf.ln(3)
        elif level == 2:
            self.pdf.set_font(self.font_name, 'B', 12)
            self.pdf.ln(2)
        else:
            self.pdf.set_font(self.font_name, 'B', 11)
            self.pdf.ln(1)
        
        self.pdf.cell(0, 8, self._sanitize_text(text), new_x='LMARGIN', new_y='NEXT')
        self.pdf.set_font(self.font_name, '', 10)
        self.pdf.ln(1)
    
    def add_paragraph(self, text: str):
        """Add a paragraph of text."""
        self.pdf.set_font(self.font_name, '', 10)
        usable_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin
        self.pdf.set_x(self.pdf.l_margin)
        self.pdf.multi_cell(usable_width, 5, self._sanitize_text(text))
        self.pdf.ln(2)

    @staticmethod
    def _is_markdown_table_separator(line: str) -> bool:
        stripped = line.strip()
        if not stripped.startswith('|'):
            return False
        return all(ch in "|-: " for ch in stripped)

    @staticmethod
    def _strip_markdown_inline(text: str) -> str:
        return text.replace('**', '').replace('*', '').replace('`', '')
    
    def add_table_from_markdown_text(self, md_text: str):
        """Extract and add a markdown table as wrapped full text rows."""
        if '|' not in md_text:
            return False
        
        lines = md_text.strip().split('\n')
        rows = []
        for line in lines:
            if '|' in line:
                if self._is_markdown_table_separator(line):
                    continue
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    rows.append(cells)
        
        if not rows:
            return False

        header = rows[0]
        usable_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

        self.pdf.set_font(self.font_name, 'B', 9)
        self.pdf.set_x(self.pdf.l_margin)
        self.pdf.multi_cell(usable_width, 5, self._sanitize_text(' | '.join(header)))
        self.pdf.set_font(self.font_name, '', 9)

        for row in rows[1:]:
            row_text = ' | '.join(str(cell) for cell in row)
            self.pdf.set_x(self.pdf.l_margin)
            self.pdf.multi_cell(usable_width, 5, self._sanitize_text(row_text))
        
        self.pdf.ln(3)
        return True

    def _extract_front_matter(self, lines: list[str]) -> tuple[dict[str, str], list[str]]:
        metadata = {}
        body_lines = lines

        if lines and lines[0].strip() == '---':
            end_idx = None
            for idx in range(1, len(lines)):
                if lines[idx].strip() == '---':
                    end_idx = idx
                    break

            if end_idx is not None:
                front_lines = lines[1:end_idx]
                body_lines = lines[end_idx + 1:]
                for line in front_lines:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        metadata[key.strip().lower()] = val.strip().strip('"')

        return metadata, body_lines
    
    def convert_markdown_file(self, md_path: Path, pdf_path: Path):
        """Convert markdown file to PDF."""
        md_content = md_path.read_text(encoding='utf-8')
        lines = md_content.split('\n')
        metadata, body_lines = self._extract_front_matter(lines)
        
        # Parse YAML front matter for title/date
        title = metadata.get('title', 'Signal Evaluation Report')
        date = metadata.get('date', '2026-03-24')
        
        self.add_title_page(title, date)
        
        # Process content
        current_block = []
        in_table = False
        
        for line in body_lines:
            if line.startswith('#'):
                # Process previous block
                if current_block:
                    text = '\n'.join(current_block).strip()
                    if text and not in_table:
                        self.add_paragraph(text)
                    current_block = []
                    in_table = False
                
                # Add heading
                level = len(line) - len(line.lstrip('#'))
                heading_text = line.lstrip('#').strip()
                self.add_heading(heading_text, level)
            
            elif line.startswith('|'):
                # Handle table
                if current_block and not in_table:
                    text = '\n'.join(current_block).strip()
                    if text:
                        self.add_paragraph(text)
                    current_block = []
                
                in_table = True
                current_block.append(line)
            
            elif line.startswith('>'):
                # Blockquote
                if current_block and not line.startswith('>'):
                    text = '\n'.join(current_block).strip()
                    if text:
                        self.add_paragraph(text)
                    current_block = []
                current_block.append(line)
            
            elif line.strip() == '':
                # Empty line - process block
                if current_block:
                    if in_table:
                        table_text = '\n'.join(current_block)
                        self.add_table_from_markdown_text(table_text)
                    else:
                        text = '\n'.join(current_block).strip()
                        if text:
                            text = self._strip_markdown_inline(text)
                            self.add_paragraph(text)
                    current_block = []
                    in_table = False
            
            else:
                current_block.append(line)
        
        # Process final block
        if current_block:
            if in_table:
                table_text = '\n'.join(current_block)
                self.add_table_from_markdown_text(table_text)
            else:
                text = '\n'.join(current_block).strip()
                if text:
                    text = self._strip_markdown_inline(text)
                    self.add_paragraph(text)
        
        # Output PDF
        self.pdf.output(str(pdf_path))


def main():
    base_dir = REPORTS_DIR
    
    reports = [
        ("signal_research_report_20260324.md", "signal_research_report_20260324.pdf"),
        ("signal_research_report_cumulative.md", "signal_research_report_cumulative.pdf"),
    ]
    
    print("=" * 80)
    print("SIGNAL EVALUATION REPORT PDF GENERATION (via fpdf2)")
    print("=" * 80)
    print()
    
    success_count = 0
    
    for md_name, pdf_name in reports:
        md_path = base_dir / md_name
        pdf_path = base_dir / pdf_name
        
        if not md_path.exists():
            print(f"✗ File not found: {md_path}")
            continue
        
        print(f"Converting: {md_name}")
        
        try:
            pdf_path.unlink(missing_ok=True)
            rendered_with_robust_engine = False
            try:
                from scripts.render_pdf import render_markdown_to_pdf

                render_markdown_to_pdf(md_path, output_pdf=pdf_path)
                rendered_with_robust_engine = True
            except Exception:
                # Fall back to native fpdf rendering when robust renderer is unavailable.
                converter = MarkdownPDFConverter()
                converter.convert_markdown_file(md_path, pdf_path)
            
            if pdf_path.exists():
                size_kb = pdf_path.stat().st_size / 1024
                if rendered_with_robust_engine:
                    print(f"✓ Successfully generated (robust renderer): {pdf_name} ({size_kb:.1f} KB)")
                else:
                    print(f"✓ Successfully generated (fpdf fallback): {pdf_name} ({size_kb:.1f} KB)")
                success_count += 1
            else:
                print(f"✗ PDF file not created")
        
        except Exception as e:
            print(f"✗ Error converting {md_name}: {e}")
            import traceback
            traceback.print_exc()
        
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
