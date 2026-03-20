"""
PDF Report Generator — Holistic 8-Method Predictor Trial Report
================================================================
Generates a polished, publication-quality PDF from the trial report data.

Usage:
    python scripts/generate_trial_report_pdf.py

Output:
    documentation/daily_reports/holistic_8method_trial_report_20260320.pdf
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    FrameBreak,
    HRFlowable,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)
from reportlab.platypus.frames import Frame
from reportlab.lib.colors import HexColor

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "research" / "ml_evaluation" / "predictor_comparison"
OUT_DIR = ROOT / "documentation" / "daily_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_OUT = OUT_DIR / "holistic_8method_trial_report_20260320.pdf"

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
GREEN     = HexColor("#2E7D32")
GREEN_LIGHT=HexColor("#E8F5E9")
RED       = HexColor("#C62828")
RED_LIGHT = HexColor("#FFEBEE")
AMBER     = HexColor("#E65100")
AMBER_LIGHT=HexColor("#FFF3E0")
BLUE      = HexColor("#1565C0")
BLUE_LIGHT= HexColor("#E3F2FD")

PAGE_W, PAGE_H = A4

# ── Styles ────────────────────────────────────────────────────────────────────
def build_styles() -> dict:
    base = getSampleStyleSheet()
    s = {}

    s["cover_title"] = ParagraphStyle(
        "cover_title",
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        textColor=WHITE,
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    s["cover_subtitle"] = ParagraphStyle(
        "cover_subtitle",
        fontName="Helvetica",
        fontSize=14,
        leading=18,
        textColor=GOLD_LIGHT,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    s["cover_meta"] = ParagraphStyle(
        "cover_meta",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=MID_GREY,
        alignment=TA_LEFT,
    )
    s["cover_kpi_label"] = ParagraphStyle(
        "cover_kpi_label",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=MID_GREY,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    s["cover_kpi_value"] = ParagraphStyle(
        "cover_kpi_value",
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        textColor=GOLD,
        alignment=TA_CENTER,
    )
    s["section_heading"] = ParagraphStyle(
        "section_heading",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=6,
        leftIndent=0,
    )
    s["sub_heading"] = ParagraphStyle(
        "sub_heading",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=NAVY_MID,
        spaceBefore=10,
        spaceAfter=4,
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
    s["body_bold"] = ParagraphStyle(
        "body_bold",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=DARK_TEXT,
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
        alignment=TA_CENTER,
    )
    s["table_cell_left"] = ParagraphStyle(
        "table_cell_left",
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
    s["callout"] = ParagraphStyle(
        "callout",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=DARK_TEXT,
        leftIndent=12,
        rightIndent=12,
        spaceAfter=6,
        spaceBefore=6,
        borderPad=8,
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


# ── Page Templates ─────────────────────────────────────────────────────────────
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
    canvas.drawString(2*cm, 1.1*cm, "Quant Engines  ·  Options Quantitative Engine  ·  RESEARCH ONLY — not financial advice")
    canvas.drawRightString(w - 2*cm, 1.1*cm, f"Page {doc.page}")
    # top thin gold bar
    canvas.setFillColor(GOLD)
    canvas.rect(0, h - 3*mm, w, 3*mm, fill=1, stroke=0)
    canvas.restoreState()


def on_cover(canvas, doc):
    """Full-bleed cover background."""
    canvas.saveState()
    w, h = A4
    # Full navy background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    # Gold accent bar at top
    canvas.setFillColor(GOLD)
    canvas.rect(0, h - 10*mm, w, 10*mm, fill=1, stroke=0)
    # Dark panel gradient illusion: slightly lighter navy strip
    canvas.setFillColor(NAVY_MID)
    canvas.rect(0, h * 0.38, w, h * 0.62 - 10*mm, fill=1, stroke=0)
    # Subtle diagonal accent line
    canvas.setStrokeColor(GOLD)
    canvas.setLineWidth(0.5)
    canvas.setDash(4, 6)
    canvas.line(0, h * 0.38, w * 0.6, h * 0.38)
    canvas.setDash()
    canvas.restoreState()


# ── Table Helpers ──────────────────────────────────────────────────────────────
TS = TableStyle

def header_row_style(n_cols: int, bg=NAVY, fg=WHITE) -> list:
    return [
        ("BACKGROUND",   (0, 0), (n_cols - 1, 0), bg),
        ("TEXTCOLOR",    (0, 0), (n_cols - 1, 0), fg),
        ("FONTNAME",     (0, 0), (n_cols - 1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (n_cols - 1, 0), 8),
        ("ALIGN",        (0, 0), (n_cols - 1, 0), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("GRID",         (0, 0), (-1, -1), 0.4, LIGHT_GREY),
        ("LINEBELOW",    (0, 0), (-1, 0), 1.5, GOLD),
    ]


def zebra_rows(n_rows: int, start: int = 1) -> list:
    cmds = []
    for i in range(start, n_rows):
        bg = OFF_WHITE if i % 2 == 0 else WHITE
        cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
    return cmds


# ── Cover Page ────────────────────────────────────────────────────────────────
def build_cover(styles: dict) -> list:
    """Returns flowables for the cover page."""
    story = []

    # Spacer to push below gold top bar
    story.append(Spacer(1, 1.2*cm))

    # Tag line above title
    story.append(Paragraph(
        "QUANTITATIVE RESEARCH  ·  INTERNAL REPORT",
        ParagraphStyle("tag", fontName="Helvetica", fontSize=8,
                       textColor=GOLD_LIGHT, spaceBefore=0, spaceAfter=10,
                       alignment=TA_LEFT)
    ))

    # Main title
    story.append(Paragraph("Holistic 8-Method<br/>Predictor Trial", styles["cover_title"]))
    story.append(Spacer(1, 4*mm))

    # Subtitle
    story.append(Paragraph(
        "Full Historical Backtest + Live Signal Validation  ·  NIFTY",
        styles["cover_subtitle"]
    ))
    story.append(Spacer(1, 6*mm))

    # Horizontal gold rule
    story.append(HRFlowable(width="100%", thickness=1.5, color=GOLD, spaceAfter=8))

    # Meta block
    story.append(Paragraph(
        "Date: <b>March 20, 2026</b> &nbsp;&nbsp; | &nbsp;&nbsp; "
        "Run ID: backtest_comparison_results_20260320_103615 &nbsp;&nbsp; | &nbsp;&nbsp; "
        "Author: <b>Pramit Dutta</b>  ·  Quant Engines",
        ParagraphStyle("meta2", fontName="Helvetica", fontSize=9,
                       textColor=MID_GREY, spaceBefore=0, spaceAfter=0, alignment=TA_LEFT)
    ))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "Scope: NIFTY 2016-01-01 → 2026-03-16 &nbsp;·&nbsp; 2,518 trading days &nbsp;·&nbsp; "
        "Max 3 expiries/day &nbsp;·&nbsp; 7,554 signals per method &nbsp;·&nbsp; "
        "8 prediction methods evaluated",
        ParagraphStyle("meta3", fontName="Helvetica", fontSize=9,
                       textColor=MID_GREY, spaceBefore=0, spaceAfter=0, alignment=TA_LEFT)
    ))

    # KPI tiles
    story.append(Spacer(1, 14*mm))

    kpi_data = [
        [
            Paragraph("METHODS TESTED", styles["cover_kpi_label"]),
            Paragraph("SIGNALS / METHOD", styles["cover_kpi_label"]),
            Paragraph("BACKTEST WINDOW", styles["cover_kpi_label"]),
            Paragraph("TOP COMPOSITE", styles["cover_kpi_label"]),
            Paragraph("BEST HIT RATE*", styles["cover_kpi_label"]),
        ],
        [
            Paragraph("8", styles["cover_kpi_value"]),
            Paragraph("7,554", styles["cover_kpi_value"]),
            Paragraph("10 years", styles["cover_kpi_value"]),
            Paragraph("60.80", styles["cover_kpi_value"]),
            Paragraph("75%", styles["cover_kpi_value"]),
        ],
        [
            Paragraph("blended → research_\nuncertainty_adjusted", styles["cover_kpi_label"]),
            Paragraph("identical universe,\ndifferent filters", styles["cover_kpi_label"]),
            Paragraph("2016 – Mar 2026", styles["cover_kpi_label"]),
            Paragraph("research_rank_gate", styles["cover_kpi_label"]),
            Paragraph("*offline, research_\nrank_gate", styles["cover_kpi_label"]),
        ],
    ]
    kpi_col_w = (PAGE_W - 4*cm) / 5
    kpi_table = Table(kpi_data, colWidths=[kpi_col_w] * 5, rowHeights=[14, 22, 14])
    kpi_table.setStyle(TS([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY_MID),
        ("TEXTCOLOR",  (0, 0), (-1, -1), WHITE),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("LINEAFTER",  (0, 0), (3, -1), 0.5, SLATE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(kpi_table)

    story.append(Spacer(1, 12*mm))

    # Key findings on cover
    story.append(HRFlowable(width="100%", thickness=0.5, color=SLATE, spaceAfter=6))
    story.append(Paragraph(
        "KEY FINDINGS AT A GLANCE",
        ParagraphStyle("kf_label", fontName="Helvetica-Bold", fontSize=8,
                       textColor=GOLD, spaceAfter=8)
    ))

    findings = [
        ("🏆 Top performer (composite)", "research_rank_gate — score 60.80, expiry accuracy 48.1%, MFE +265.7 bps"),
        ("🎯 Best 1D directional accuracy", "research_decision_policy — 50.04%, the only method exceeding the 50% random baseline"),
        ("📊 Highest signal quality", "research_uncertainty_adjusted — avg strength 88.61, though MAE wider at -288 bps"),
        ("⚙️  Production baseline", "blended — strongest TP/SL ratio (43.9% / 51.5%), full live coverage, HR=0.58 live"),
        ("⚠️  Critical live finding", "Only 37% of live signals carry ML scores; research methods fire on n=5 live trades"),
        ("✅ Research quality confirmed", "Offline dataset: rank_gate HR=0.75, +19.6 bps Sharpe 0.37 on 1,044 trades"),
    ]

    find_data = [[
        Paragraph(label, ParagraphStyle("fl", fontName="Helvetica-Bold", fontSize=8,
                                        textColor=GOLD_LIGHT)),
        Paragraph(val, ParagraphStyle("fv", fontName="Helvetica", fontSize=8,
                                      textColor=OFF_WHITE))
    ] for label, val in findings]

    find_table = Table(find_data, colWidths=[5.5*cm, PAGE_W - 4*cm - 5.5*cm])
    find_table.setStyle(TS([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW",     (0, 0), (-1, -4), 0.3, SLATE),
    ]))
    story.append(find_table)

    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=SLATE, spaceAfter=6))
    story.append(Paragraph(
        "RESEARCH ONLY — All results are for internal research and evaluation purposes. "
        "No production changes should be made based solely on this document without additional validation.",
        ParagraphStyle("disc", fontName="Helvetica-Oblique", fontSize=7.5,
                       textColor=MID_GREY, alignment=TA_LEFT)
    ))

    story.append(NextPageTemplate("body"))
    story.append(PageBreak())
    return story


# ── Section Heading Helper ─────────────────────────────────────────────────────
def section_heading(text: str, styles: dict) -> list:
    bar = Table([[""]], colWidths=[PAGE_W - 4*cm], rowHeights=[3])
    bar.setStyle(TS([
        ("BACKGROUND", (0, 0), (0, 0), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return [Spacer(1, 4*mm), bar, Paragraph(text, styles["section_heading"])]


def callout_box(text: str, bg=BLUE_LIGHT, border=BLUE) -> Table:
    """A coloured callout / alert box."""
    p = Paragraph(text, ParagraphStyle(
        "cb", fontName="Helvetica", fontSize=9, leading=13,
        textColor=DARK_TEXT, leftIndent=0
    ))
    t = Table([[p]], colWidths=[PAGE_W - 4*cm])
    t.setStyle(TS([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("LINEAFTER",     (0, 0), (0, -1), 3, border),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


# ── Data ──────────────────────────────────────────────────────────────────────
METHODS_FULL = [
    # name, trades, rate, strength, composite, acc1d, expiry, tp, sl, mfe, mae
    ("blended",                    4071, "53.9%", 83.64, 58.06, "49.1%", "45.8%", "43.9%", "51.5%", "+238.0", "-259.9"),
    ("pure_rule",                  3983, "52.7%", 82.53, 57.96, "49.0%", "45.6%", "43.8%", "51.7%", "+238.9", "-262.3"),
    ("pure_ml",                    3408, "45.1%", 83.14, 58.44, "48.5%", "45.3%", "42.2%", "52.8%", "+248.9", "-271.8"),
    ("research_dual_model",        3154, "41.8%", 83.04, 58.86, "48.6%", "46.1%", "42.5%", "53.2%", "+250.7", "-274.3"),
    ("ev_sizing",                  3150, "41.7%", 83.06, 58.87, "48.6%", "46.1%", "42.5%", "53.2%", "+250.6", "-273.8"),
    ("research_decision_policy",   2602, "34.5%", 85.99, 60.38, "50.0%", "47.9%", "43.6%", "52.2%", "+261.5", "-274.7"),
    ("research_rank_gate",         2335, "30.9%", 87.53, 60.80, "49.6%", "48.1%", "43.3%", "53.3%", "+265.7", "-280.2"),
    ("research_uncertainty_adj",   2148, "28.4%", 88.61, 59.56, "48.2%", "46.9%", "42.1%", "54.7%", "+264.6", "-288.3"),
]


OFFLINE_COMP = [
    # method, n_trade, hit_rate, avg_ret, sharpe
    ("blended",           2697, "0.50", "-2.4",  "-0.030"),
    ("pure_rule",         2546, "0.50", "-2.5",  "-0.033"),
    ("pure_ml",           1603, "0.58", "+5.0",  "+0.078"),
    ("research_dual",     1264, "0.67", "+10.9", "+0.172"),
    ("ev_sizing",         1264, "0.67", "+10.9", "+0.172"),
    ("decision_policy",   1105, "0.74", "+19.0", "+0.348"),
    ("research_rank_gate",1044, "0.75", "+19.6", "+0.370"),
    ("uncertainty_adj",    526, "0.65", "+8.6",  "+0.147"),
]

LIVE_COMP = [
    ("blended",           253,  "0.58", "+0.5",  "+0.012"),
    ("pure_rule",         194,  "0.52", "-2.5",  "-0.061"),
    ("pure_ml",           240,  "0.57", "-0.6",  "-0.016"),
    ("research_dual",      47,  "0.21", "-21.8", "-0.952"),
    ("ev_sizing",          47,  "0.21", "-21.8", "-0.952"),
    ("decision_policy",     5,  "0.00", "-57.8", "—"),
    ("research_rank_gate",  5,  "0.00", "-57.8", "—"),
    ("uncertainty_adj",     0,  "—",    "—",     "—"),
]

# ── Flowable builders ─────────────────────────────────────────────────────────

def build_main_metrics_table(styles: dict) -> Table:
    headers = [
        Paragraph(h, styles["table_header"]) for h in [
            "Method", "Trades", "Rate", "Strength", "Composite",
            "1D Acc", "Expiry", "TP%", "SL%", "MFE(bps)", "MAE(bps)"
        ]
    ]
    rows = [headers]
    best_composite = max(r[4] for r in METHODS_FULL)
    best_strength  = max(r[3] for r in METHODS_FULL)

    for m in METHODS_FULL:
        name, trades, rate, strength, composite, acc1d, expiry, tp, sl, mfe, mae = m
        comp_bold = composite == best_composite
        str_bold  = strength == best_strength

        row = [
            Paragraph(f"<b>{name}</b>", ParagraphStyle("tl", fontName="Helvetica-Bold",
                        fontSize=8, leading=10, textColor=NAVY_MID)),
            Paragraph(f"{trades:,}", styles["table_cell"]),
            Paragraph(rate, styles["table_cell"]),
            Paragraph(f"<b>{strength:.2f}</b>" if str_bold else f"{strength:.2f}", styles["table_cell"]),
            Paragraph(f"<b>{composite:.2f}</b>" if comp_bold else f"{composite:.2f}", styles["table_cell"]),
            Paragraph(acc1d, styles["table_cell"]),
            Paragraph(expiry, styles["table_cell"]),
            Paragraph(tp, styles["table_cell"]),
            Paragraph(sl, styles["table_cell"]),
            Paragraph(f"<b>{mfe}</b>", styles["table_cell"]),
            Paragraph(mae, styles["table_cell"]),
        ]
        rows.append(row)

    col_w = [3.8*cm, 1.4*cm, 1.3*cm, 1.5*cm, 1.5*cm, 1.3*cm, 1.3*cm, 1.2*cm, 1.2*cm, 1.6*cm, 1.6*cm]
    t = Table(rows, colWidths=col_w, repeatRows=1)

    style_cmds = header_row_style(11) + zebra_rows(len(rows))
    # Highlight best composite row (index 7 = rank_gate)
    style_cmds += [
        ("BACKGROUND",  (0, 7), (-1, 7), HexColor("#E8F0FE")),
        ("LINEABOVE",   (0, 7), (-1, 7), 1.0, BLUE),
        ("LINEBELOW",   (0, 7), (-1, 7), 1.0, BLUE),
        # Highlight best 1D acc (row 6 = decision_policy)
        ("BACKGROUND",  (5, 6), (5, 6), GREEN_LIGHT),
        # Highlight best expiry (row 7)
        ("BACKGROUND",  (6, 7), (6, 7), GREEN_LIGHT),
        # Highlight best MFE (row 7)
        ("BACKGROUND",  (9, 7), (9, 7), GREEN_LIGHT),
        # Highlight worst MAE (row 8)
        ("BACKGROUND",  (10, 8), (10, 8), RED_LIGHT),
        ("ALIGN",       (0, 0), (0, -1), "LEFT"),
    ]
    t.setStyle(TS(style_cmds))
    return t


def build_promotion_table(styles: dict) -> Table:
    headers = [
        Paragraph(h, styles["table_header"]) for h in
        ["Method", "Composite Δ", "1D Acc Δ", "Retention Floor", "Status"]
    ]
    data = [
        ("pure_rule",               "-0.10", "-0.10pp", "53%", "❌  No improvement"),
        ("pure_ml",                 "+0.38", "-0.60pp", "45%", "❌  Worse accuracy"),
        ("research_dual_model",     "+0.80", "-0.50pp", "42%", "🟡  Marginal — monitor"),
        ("ev_sizing",               "+0.81", "-0.50pp", "42%", "🟡  Needs sizing audit"),
        ("research_decision_policy","+2.32", "+0.94pp", "34%", "🟢  Candidate"),
        ("research_rank_gate",      "+2.74", "+0.49pp", "31%", "🟢  Primary candidate"),
        ("research_uncertainty_adj","+1.50", "-0.88pp", "28%", "🔴  Recalibration needed"),
    ]
    rows = [headers]
    for name, cd, ad, rf, status in data:
        rows.append([
            Paragraph(f"<b>{name}</b>", ParagraphStyle("tl2", fontName="Helvetica-Bold",
                        fontSize=8, leading=10, textColor=NAVY_MID)),
            Paragraph(cd, styles["table_cell"]),
            Paragraph(ad, styles["table_cell"]),
            Paragraph(rf, styles["table_cell"]),
            Paragraph(status, styles["table_cell_left"]),
        ])

    col_w = [4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 5.7*cm]
    t = Table(rows, colWidths=col_w, repeatRows=1)

    status_colors = {
        "🟢": GREEN_LIGHT, "🟡": AMBER_LIGHT, "❌": RED_LIGHT, "🔴": RED_LIGHT,
    }
    style_cmds = header_row_style(5) + zebra_rows(len(rows))
    for i, (_, __, ___, ____, status) in enumerate(data, start=1):
        for char, bg in status_colors.items():
            if char in status:
                style_cmds.append(("BACKGROUND", (4, i), (4, i), bg))
                break
    style_cmds += [("ALIGN", (0, 0), (0, -1), "LEFT")]
    t.setStyle(TS(style_cmds))
    return t


def build_comparison_table(data: list, title: str, styles: dict) -> Table:
    headers = [
        Paragraph(h, styles["table_header"]) for h in
        ["Method", "n TRADE", "Hit Rate", "Avg Ret (bps)", "Sharpe"]
    ]
    rows = [headers]
    for name, n, hr, ret, sharpe in data:
        rows.append([
            Paragraph(f"<b>{name}</b>", ParagraphStyle("tl3", fontName="Helvetica-Bold",
                        fontSize=8, leading=10, textColor=NAVY_MID)),
            Paragraph(f"{n:,}" if isinstance(n, int) else str(n), styles["table_cell"]),
            Paragraph(str(hr), styles["table_cell"]),
            Paragraph(str(ret), styles["table_cell"]),
            Paragraph(str(sharpe), styles["table_cell"]),
        ])
    col_w = [5.0*cm, 2.5*cm, 2.5*cm, 3.0*cm, 3.0*cm]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    style_cmds = header_row_style(5) + zebra_rows(len(rows))
    style_cmds += [("ALIGN", (0, 0), (0, -1), "LEFT")]
    t.setStyle(TS(style_cmds))
    return t


def build_coverage_table(styles: dict) -> Table:
    headers = [
        Paragraph(h, styles["table_header"]) for h in
        ["Method", "Requires ML Scores", "Live Evaluable Trades", "Coverage Impact"]
    ]
    rows_data = [
        ("blended",          "No",  "253", "None — full coverage"),
        ("pure_rule",        "No",  "194", "None"),
        ("pure_ml",          "No",  "240", "None"),
        ("research_dual",    "Yes",  "47", "−81% vs blended"),
        ("ev_sizing",        "Yes",  "47", "−81% vs blended"),
        ("decision_policy",  "Yes",   "5", "−98% vs blended"),
        ("research_rank_gate","Yes",  "5", "−98% vs blended"),
        ("uncertainty_adj",  "Yes",   "0", "100% blocked"),
    ]

    rows = [headers]
    for name, req, n, impact in rows_data:
        bold_impact = "−98%" in impact or "100%" in impact
        rows.append([
            Paragraph(f"<b>{name}</b>", ParagraphStyle("tl4", fontName="Helvetica-Bold",
                        fontSize=8, leading=10, textColor=NAVY_MID)),
            Paragraph(req, styles["table_cell"]),
            Paragraph(n, styles["table_cell"]),
            Paragraph(
                f"<b>{impact}</b>" if bold_impact else impact,
                ParagraphStyle("ci", fontName="Helvetica-Bold" if bold_impact else "Helvetica",
                               fontSize=8, leading=10, textColor=RED if bold_impact else DARK_TEXT,
                               alignment=1)
            ),
        ])
    col_w = [4.5*cm, 3.5*cm, 4.0*cm, 5.2*cm]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    style_cmds = header_row_style(4) + zebra_rows(len(rows))
    for i, (_, req, n, impact) in enumerate(rows_data, start=1):
        if req == "Yes":
            style_cmds.append(("BACKGROUND", (1, i), (1, i), AMBER_LIGHT))
        if "100%" in impact or "−98%" in impact:
            style_cmds.append(("BACKGROUND", (3, i), (3, i), RED_LIGHT))
    style_cmds += [("ALIGN", (0, 0), (0, -1), "LEFT")]
    t.setStyle(TS(style_cmds))
    return t


def maybe_image(path: Path, width: float) -> Image | Spacer:
    if path.exists():
        try:
            img = Image(str(path))
            aspect = img.drawHeight / img.drawWidth
            return Image(str(path), width=width, height=width * aspect)
        except Exception:
            pass
    return Spacer(1, 2*mm)


# ── Main story builder ────────────────────────────────────────────────────────
def build_story(styles: dict) -> list:
    story = []
    body_w = PAGE_W - 4*cm

    # ── COVER ──────────────────────────────────────────────────────────────────
    story += build_cover(styles)

    # ── TOC-style intro ────────────────────────────────────────────────────────
    story += section_heading("1. Executive Summary", styles)
    story.append(Paragraph(
        "All 8 prediction methods completed successfully over the full 10-year NIFTY historical window "
        "(2016-01-01 to 2026-03-16). The trial covers 7,554 signals per method across 2,518 trading days "
        "with a maximum of 3 expiries evaluated per day.",
        styles["body"]
    ))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "The core finding is a clear <b>quality-vs-volume trade-off</b>: high-volume methods (blended, "
        "pure_rule) preserve maximum trade throughput but capture lower-quality signals, while aggressive "
        "filtering methods (research_rank_gate, research_decision_policy) trade 15–25% of volume for "
        "meaningfully higher signal quality and directional precision.",
        styles["body"]
    ))
    story.append(Spacer(1, 4*mm))

    # Key metrics callouts
    kpi2 = [
        ["Best Composite Score",  "research_rank_gate",           "60.80"],
        ["Best 1D Dir. Accuracy", "research_decision_policy",     "50.04% (only >50%)"],
        ["Best Expiry Accuracy",  "research_rank_gate",           "48.05%"],
        ["Highest Signal Strength","research_uncertainty_adj",    "88.61"],
        ["Best TP / SL Ratio",    "blended",                      "43.9% TP / 51.5% SL"],
        ["Best Live Hit Rate",    "blended",                      "0.58 (n=253 live outcomes)"],
    ]
    kpi_tbl = Table(kpi2, colWidths=[4.5*cm, 6.0*cm, body_w - 10.5*cm])
    kpi_tbl.setStyle(TS([
        ("BACKGROUND",    (0, 0), (-1, -1), OFF_WHITE),
        ("BACKGROUND",    (0, 0), (0, -1), NAVY),
        ("TEXTCOLOR",     (0, 0), (0, -1), GOLD_LIGHT),
        ("TEXTCOLOR",     (1, 0), (1, -1), NAVY_MID),
        ("TEXTCOLOR",     (2, 0), (2, -1), DARK_TEXT),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.4, LIGHT_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 4*mm))

    story.append(callout_box(
        "⚠️  <b>Critical live-data finding</b>: Testing on the live <i>signals_dataset_cumul</i> "
        "(499 signals, Mar 13–20) reveals only 37% of live signals carry ML scores. Research predictors "
        "fire on just n=5 live trades — statistically void. Research method quality is confirmed "
        "<i>offline</i> (rank_gate HR=0.75, +19.6 bps) but an ML inference coverage fix is required "
        "before live evaluation is meaningful.",
        bg=AMBER_LIGHT, border=AMBER
    ))

    # ── Section 2: Full Metrics ────────────────────────────────────────────────
    story += section_heading("2. Full Metrics Table — Holistic Backtest (2016–2026)", styles)
    story.append(Paragraph(
        "All 8 methods evaluated on the identical 7,554-signal universe. "
        "Bold values denote the best result for each column. "
        "Blue-highlighted row = highest composite score.",
        styles["body"]
    ))
    story.append(Spacer(1, 3*mm))
    story.append(KeepTogether([build_main_metrics_table(styles)]))

    # ── Section 3: Method Tiers ────────────────────────────────────────────────
    story += section_heading("3. Method Tier Classification", styles)

    tier_data = [
        [
            Paragraph("TIER 1", ParagraphStyle("t1l", fontName="Helvetica-Bold", fontSize=9,
                        textColor=WHITE)),
            Paragraph("Production-Proven Baselines — Highest Volume, Lowest MAE",
                        ParagraphStyle("td", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        ],
    ]
    tier_hdr = Table(tier_data, colWidths=[2.0*cm, body_w - 2.0*cm])
    tier_hdr.setStyle(TS([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tier_hdr)
    story.append(Paragraph(
        "<b>blended</b> — Best TP rate (43.9%), lowest SL rate (51.5%), best MFE/MAE ratio (0.915). "
        "Current production default. Provides the cleanest risk-adjusted throughput across the full decade.",
        styles["bullet"]
    ))
    story.append(Paragraph(
        "<b>pure_rule</b> — Nearly identical to blended in all metrics. Confirms rule logic is the dominant "
        "driver; the ML blend contributes marginally at current 65/35 weights.",
        styles["bullet"]
    ))

    story.append(Spacer(1, 4*mm))

    tier_data2 = [[
        Paragraph("TIER 2", ParagraphStyle("t2l", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        Paragraph("ML-Enhanced Filters — Moderate Filtering, Marginal Quality Lift",
                  ParagraphStyle("td2", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
    ]]
    tier_hdr2 = Table(tier_data2, colWidths=[2.0*cm, body_w - 2.0*cm])
    tier_hdr2.setStyle(TS([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY_MID),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tier_hdr2)
    for txt in [
        "<b>pure_ml</b> — 8.8% fewer trades, slight composite lift (+0.38) but worse TP (−1.7pp) and higher SL (+1.3pp). Raw ML signal without the rule backbone underperforms blended.",
        "<b>research_dual_model</b> — 22.5% fewer trades; composite +0.80 vs blended. Slightly better MFE. No decisive quality argument in isolation.",
        "<b>ev_sizing</b> — Virtually identical to research_dual_model (Δ < 0.01 on all metrics). CRT sizing overlay does not differentiate at current 41-cell resolution.",
    ]:
        story.append(Paragraph(txt, styles["bullet"]))

    story.append(Spacer(1, 4*mm))

    tier_data3 = [[
        Paragraph("TIER 3", ParagraphStyle("t3l", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
        Paragraph("Research Candidates — Quality-Concentrated, Aggressive Filtering",
                  ParagraphStyle("td3", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
    ]]
    tier_hdr3 = Table(tier_data3, colWidths=[2.0*cm, body_w - 2.0*cm])
    tier_hdr3.setStyle(TS([
        ("BACKGROUND", (0, 0), (-1, -1), BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tier_hdr3)
    for txt in [
        "<b>research_decision_policy</b> — 34.5% trade rate. <i>Only method crossing the 50% 1D accuracy ceiling</i> (50.04%). Composite +2.32 over blended. Best MFE (+23.5 bps). Valid promotion candidate.",
        "<b>research_rank_gate</b> — 30.9% trade rate. Highest composite (60.80), highest expiry accuracy (48.1%), highest MFE (+27.7 bps). MAE is wider (−280 vs −260) but MFE/MAE ratio improves to 0.949. Primary promotion candidate.",
        "<b>research_uncertainty_adjusted</b> — 28.4% trade rate; highest signal strength (88.61) but composite (59.56) lags rank_gate. Highest SL rate (54.7%) and widest MAE (−288 bps). Ambiguity penalty may be over-penalising.",
    ]:
        story.append(Paragraph(txt, styles["bullet"]))

    # ── Section 4: Key Insights ────────────────────────────────────────────────
    story += section_heading("4. Key Analytical Insights", styles)

    insights = [
        ("4.1  Quality-Volume Gradient",
         "Each filtering step buys ~0.5–1.0pp of composite score at the cost of ~5–10pp trade rate. "
         "The relationship is non-linear: research_decision_policy and research_rank_gate show step-change "
         "improvements, suggesting the 31–35% rate zone captures a qualitatively different signal cohort "
         "rather than simply a smaller random subset."),
        ("4.2  1D vs Expiry Accuracy Divergence",
         "Most methods show a 3–4pp gap between 1D accuracy and expiry accuracy. research_decision_policy "
         "narrows this to 2.1pp (50.0% → 47.9%), suggesting its policy gates improve short-term directional "
         "precision — important for tight TP/SL exit frameworks where overnight decay is not incurred."),
        ("4.3  MFE/MAE Asymmetry",
         "blended exhibits the widest MFE/MAE gap: +238 / −260 → ratio 0.915 (MAE exceeds MFE by 9%). "
         "research_rank_gate closes this to 0.949 (+266 / −280). research_uncertainty_adjusted widens "
         "the absolute MAE to −288 bps without proportionally increasing MFE, making its risk-return "
         "profile less attractive despite the highest nominal signal strength."),
        ("4.4  research_dual_model ≈ ev_sizing (Convergence Warning)",
         "These two methods produce near-identical metrics on every dimension. "
         "This strongly suggests the CRT-based EV overlay in ev_sizing is not feeding into the "
         "trade-selection probability; it may be modifying sizing downstream but not the signal gate pass rate."),
        ("4.5  research_uncertainty_adjusted Concern",
         "Despite highest signal strength (88.61), this method shows the highest SL rate (54.7%) and "
         "widest MAE. The uncertainty multiplier formula (ambiguity weight=0.35) may be clearing nominally "
         "confident but directionally ambiguous signals — blocks should be tighter. Proposed fix: "
         "reduce ambiguity weight 0.35 → 0.20 and re-test."),
    ]
    for title, body in insights:
        story.append(Paragraph(f"<b>{title}</b>", styles["sub_heading"]))
        story.append(Paragraph(body, styles["body"]))

    # ── Section 5: Promotion Assessment ───────────────────────────────────────
    story += section_heading("5. Promotion Assessment", styles)
    story.append(Paragraph(
        "Proposed promotion gate criteria for replacing <b>blended</b> in production: "
        "Composite score > 60.0, 1D accuracy ≥ 49.5%, trade retention ≥ 30%, "
        "MFE/MAE ratio ≥ 0.93, plus successful out-of-sample validation.",
        styles["body"]
    ))
    story.append(Spacer(1, 3*mm))
    story.append(build_promotion_table(styles))
    story.append(Spacer(1, 3*mm))
    story.append(callout_box(
        "🏆 <b>research_rank_gate passes all 4 quantitative gates</b>: composite 60.80 ✓, "
        "1D accuracy 49.6% ✓, retention 30.9% ✓, MFE/MAE ratio 0.949 ✓. "
        "Out-of-sample validation over a recent 60-day window is the remaining requirement "
        "before a production promotion decision.",
        bg=GREEN_LIGHT, border=GREEN
    ))

    # ── Section 6: Next Steps ──────────────────────────────────────────────────
    story += section_heading("6. Recommended Next Steps", styles)

    steps = [
        ("Step 1 — Fix ML inference coverage (Urgent)",
         "63% of live signals carry no ML scores. Diagnose and resolve the inference pipeline gap before "
         "any live research method evaluation. Target: ≥ 80% coverage on live signals. "
         "Likely causes: sklearn version mismatch (1.7.2 → 1.6.1), signal-path gaps in engine_runner, "
         "or silent exception swallowing post-inference."),
        ("Step 2 — Recent-window out-of-sample validation",
         "Run research_rank_gate and research_decision_policy over a recent 60-day window "
         "(2025-H2 or 2026 YTD) against blended. Full-history backtest numbers may include periods "
         "that overlap with model training data."),
        ("Step 3 — Regime-sliced attribution",
         "Decompose composite gains by gamma_regime × volatility_regime. Methods that only outperform "
         "in low-vol trending regimes are not promotable without a regime-conditioning wrapper."),
        ("Step 4 — research_uncertainty_adjusted recalibration",
         "Reduce ambiguity weight in multiplier formula: 0.35 → 0.20. "
         "Verify that MAE narrows while signal strength is maintained. Re-run full comparison."),
        ("Step 5 — ev_sizing CRT audit",
         "Verify whether the EV scale factor from the Conditional Return Table feeds into "
         "trade-selection probability or only downstream position sizing. "
         "The ev_sizing ≈ research_dual convergence suggests the former is not happening."),
        ("Step 6 — Live evaluation milestone",
         "Re-evaluate after accumulating ≥ 500 live TRADE signals with ML scores populated. "
         "Current live n=5 is 99× below a valid evaluation sample. "
         "Current blended live performance (HR=0.58, +0.5 bps) is the valid production baseline."),
    ]
    for i, (title, body) in enumerate(steps, 1):
        row_bg = GREEN_LIGHT if i == 1 else OFF_WHITE
        step_tbl = Table(
            [[Paragraph(f"<b>{title}</b>", ParagraphStyle("sh", fontName="Helvetica-Bold",
                            fontSize=9, textColor=NAVY_MID)),
              Paragraph(body, styles["body"])]],
            colWidths=[5.2*cm, body_w - 5.2*cm]
        )
        step_tbl.setStyle(TS([
            ("BACKGROUND",    (0, 0), (-1, -1), row_bg),
            ("LINEBELOW",     (0, 0), (-1, -1), 0.5, LIGHT_GREY),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(step_tbl)

    # ── Section 7: Live Validation ─────────────────────────────────────────────
    story.append(PageBreak())
    story += section_heading("7. Live Signal Validation — signals_dataset_cumul (Mar 13–20, 2026)", styles)

    story.append(callout_box(
        "<b>Dataset</b>: research/signal_evaluation/signals_dataset_cumul.csv  ·  "
        "<b>499 signals</b>  ·  <b>253 with 60m outcomes</b>  ·  "
        "Date range: 2026-03-13 → 2026-03-20  ·  "
        "Evaluated using the same predictor_comparison_runner.py framework",
        bg=BLUE_LIGHT, border=BLUE
    ))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("<b>7.1  Side-by-Side: Offline Backtest vs Live Dataset</b>", styles["sub_heading"]))

    # Combined comparison table
    combined_headers = [
        Paragraph(h, styles["table_header"]) for h in [
            "Method",
            "Offline\nn TRADE", "Offline\nHit Rate", "Offline\nAvg Ret",
            "Live\nn TRADE",   "Live\nHit Rate",     "Live\nAvg Ret",
        ]
    ]
    combined_data = [combined_headers]
    method_map = {
        "blended": ("blended", 2697, "0.50", "−2.4", 253, "0.58", "+0.5"),
        "pure_rule": ("pure_rule", 2546, "0.50", "−2.5", 194, "0.52", "−2.5"),
        "pure_ml": ("pure_ml", 1603, "0.58", "+5.0", 240, "0.57", "−0.6"),
        "research_dual": ("research_dual", 1264, "0.67", "+10.9", 47, "0.21", "−21.8"),
        "ev_sizing": ("ev_sizing", 1264, "0.67", "+10.9", 47, "0.21", "−21.8"),
        "decision_policy": ("decision_policy", 1105, "0.74", "+19.0", 5, "0.00", "−57.8"),
        "research_rank_gate": ("research_rank_gate", 1044, "0.75", "+19.6", 5, "0.00", "−57.8"),
        "uncertainty_adj": ("uncertainty_adj", 526, "0.65", "+8.6", 0, "—", "—"),
    }
    for key in ["blended", "pure_rule", "pure_ml", "research_dual", "ev_sizing",
                "decision_policy", "research_rank_gate", "uncertainty_adj"]:
        name, on, ohr, oret, ln, lhr, lret = method_map[key]
        combined_data.append([
            Paragraph(f"<b>{name}</b>", ParagraphStyle("cl", fontName="Helvetica-Bold",
                        fontSize=8, leading=10, textColor=NAVY_MID)),
            Paragraph(f"{on:,}" if isinstance(on, int) else str(on), styles["table_cell"]),
            Paragraph(str(ohr), styles["table_cell"]),
            Paragraph(str(oret), styles["table_cell"]),
            Paragraph(f"<b>{ln}</b>" if isinstance(ln, int) and ln <= 5 else
                       (f"{ln:,}" if isinstance(ln, int) else str(ln)), styles["table_cell"]),
            Paragraph(str(lhr), styles["table_cell"]),
            Paragraph(str(lret), styles["table_cell"]),
        ])

    col_w_c = [4.2*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm]
    ct = Table(combined_data, colWidths=col_w_c, repeatRows=1)
    style_c = header_row_style(7) + zebra_rows(len(combined_data))
    # Shade live n=5 cells orange
    for i, key in enumerate(["blended", "pure_rule", "pure_ml", "research_dual", "ev_sizing",
                              "decision_policy", "research_rank_gate", "uncertainty_adj"], start=1):
        ln = method_map[key][5]
        if isinstance(ln, int) and ln <= 5:
            style_c += [("BACKGROUND", (4, i), (6, i), AMBER_LIGHT)]
    # Blue header split
    style_c += [
        ("BACKGROUND", (1, 0), (3, 0), NAVY_MID),
        ("BACKGROUND", (4, 0), (6, 0), HexColor("#8B5500")),
        ("LINEAFTER",  (3, 0), (3, -1), 1.5, GOLD),
        ("ALIGN",      (0, 0), (0, -1), "LEFT"),
    ]
    ct.setStyle(TS(style_c))
    story.append(ct)
    story.append(Paragraph(
        "Orange rows = live n ≤ 5 — statistically void; do not use for promotion decisions.",
        styles["caption"]
    ))

    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("<b>7.2  ML Score Coverage Gap on Live Data</b>", styles["sub_heading"]))
    story.append(build_coverage_table(styles))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("<b>7.3  Offline Dataset — Research Methods Perform Strongly When ML Scores Are Present</b>", styles["sub_heading"]))
    story.append(build_comparison_table(OFFLINE_COMP, "Offline Backtest", styles))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("<b>7.4  Root Cause Candidates for Missing ML Scores (63% of live signals)</b>", styles["sub_heading"]))
    causes = [
        ("1. sklearn version mismatch",
         "Inference loads models trained under sklearn 1.7.2 into 1.6.1. "
         "InconsistentVersionWarning is printed at load time; individual signal inference may silently "
         "fail and return None for rank/confidence scores."),
        ("2. Inference pipeline not triggered",
         "The engine's live run path may not invoke build_inference_row() + infer_single() for all "
         "signal types. Signals with missing raw state fields (gamma_regime classification failures, "
         "incomplete market context) may silently skip ML inference."),
        ("3. Signal source gap",
         "REPLAY-sourced signals (Mar 13 ICICI replay run) may not pass through the same "
         "inference path as real-time LIVE-sourced signals, leaving their ML score columns unpopulated."),
    ]
    for title, body in causes:
        story.append(Paragraph(f"<b>{title}</b>", styles["bullet"]))
        story.append(Paragraph(body, ParagraphStyle(
            "sub_body", fontName="Helvetica", fontSize=9, leading=13,
            textColor=DARK_TEXT, leftIndent=28, spaceAfter=4
        )))

    story.append(Spacer(1, 3*mm))
    story.append(callout_box(
        "📌 <b>MISSING bucket is the strongest live cohort</b>: The 187 signals without ML scores "
        "show HR=0.67, Avg Ret +5.0 bps — the best live performance. This confirms the underlying "
        "blended signal logic is sound. The gap is in the ML inference pipeline, not the base engine.",
        bg=GREEN_LIGHT, border=GREEN
    ))

    # ── Section 8: Charts ──────────────────────────────────────────────────────
    story.append(PageBreak())
    story += section_heading("8. Supporting Charts", styles)

    chart_specs = [
        (PRED_DIR / "predictor_comparison_backtest.png",
         "Figure 1 — Hit Rate, Avg Return, Retention & Sharpe on Offline Backtest Dataset"),
        (PRED_DIR / "predictor_comparison_cumulative.png",
         "Figure 2 — Same Metrics on Live Cumulative Dataset (limited outcomes)"),
        (PRED_DIR / "risk_return_scatter.png",
         "Figure 3 — Sharpe vs Avg Return Risk-Return Scatter (Offline Backtest)"),
        (PRED_DIR / "threshold_sweep.png",
         "Figure 4 — Hit Rate and Avg Return vs Probability Threshold Sweep"),
        (PRED_DIR / "trade_vs_no_trade.png",
         "Figure 5 — TRADE vs NO-TRADE Signal Cohort Comparison (Offline)"),
        (PRED_DIR / "policy_decision_breakdown.png",
         "Figure 6 — Decision Policy ALLOW / DOWNGRADE / BLOCK Breakdown (Offline)"),
    ]

    img_w = body_w - 1*cm
    for path, caption in chart_specs:
        img = maybe_image(path, img_w)
        story.append(KeepTogether([img, Paragraph(caption, styles["caption"])]))
        story.append(Spacer(1, 4*mm))

    # ── Section 9: Artifacts ───────────────────────────────────────────────────
    story += section_heading("9. Artifact References", styles)

    artifacts = [
        ("Holistic backtest JSON",
         "scripts/backtest/backtest_comparison_results_20260320_103615.json"),
        ("Live comparison JSON",
         "research/ml_evaluation/predictor_comparison/predictor_comparison_results.json"),
        ("Live CSV — per method",
         "research/ml_evaluation/predictor_comparison/cumulative_comparison.csv"),
        ("Offline CSV — per method",
         "research/ml_evaluation/predictor_comparison/backtest_comparison.csv"),
        ("Full predictor report (Markdown)",
         "research/ml_evaluation/predictor_comparison/predictor_comparison_report.md"),
        ("This report (source Markdown)",
         "research/ml_evaluation/predictor_comparison/holistic_8method_trial_report.md"),
        ("Run wall-clock time",
         "2026-03-20 07:51 → 10:36 IST (2h 44min, 8/8 methods SUCCESS)"),
    ]
    art_tbl = Table(
        [[Paragraph(k, ParagraphStyle("ak", fontName="Helvetica-Bold", fontSize=8,
                                      textColor=NAVY_MID)),
          Paragraph(v, ParagraphStyle("av", fontName="Helvetica", fontSize=8,
                                      textColor=SLATE))]
         for k, v in artifacts],
        colWidths=[5.5*cm, body_w - 5.5*cm]
    )
    art_tbl.setStyle(TS([
        ("LINEBELOW",     (0, 0), (-1, -2), 0.4, LIGHT_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(art_tbl)

    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GREY, spaceAfter=6))
    story.append(Paragraph(
        "RESEARCH ONLY — All results in this document are for internal research and evaluation purposes. "
        "No production changes should be made based solely on this document without additional out-of-sample validation. "
        "Quant Engines  ·  Options Quantitative Engine  ·  2026",
        ParagraphStyle("final_disc", fontName="Helvetica-Oblique", fontSize=7.5,
                       textColor=MID_GREY, alignment=TA_CENTER)
    ))

    return story


# ── Document Assembly ─────────────────────────────────────────────────────────
def build_doc(story: list) -> None:
    body_margin_l = 2.0*cm
    body_margin_r = 2.0*cm
    body_margin_t = 2.2*cm
    body_margin_b = 2.0*cm

    doc = BaseDocTemplate(
        str(PDF_OUT),
        pagesize=A4,
        leftMargin=body_margin_l,
        rightMargin=body_margin_r,
        topMargin=body_margin_t,
        bottomMargin=body_margin_b,
        title="Holistic 8-Method Predictor Trial Report",
        author="Pramit Dutta — Quant Engines",
        subject="NIFTY Predictor Method Comparison 2016–2026",
    )

    cover_frame = Frame(0, 0, PAGE_W, PAGE_H, leftPadding=2.5*cm, rightPadding=2.5*cm,
                        topPadding=1.2*cm, bottomPadding=2.5*cm, id="cover_frame")

    body_frame = Frame(body_margin_l, body_margin_b,
                       PAGE_W - body_margin_l - body_margin_r,
                       PAGE_H - body_margin_t - body_margin_b,
                       id="body_frame")

    cover_template = PageTemplate(id="cover", frames=[cover_frame], onPage=on_cover)
    body_template  = PageTemplate(id="body",  frames=[body_frame],  onPage=on_page)

    doc.addPageTemplates([cover_template, body_template])
    doc.build(story)
    print(f"\n✅  PDF generated: {PDF_OUT}")
    print(f"    Size: {PDF_OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    print("Building PDF report …")
    styles = build_styles()
    story  = build_story(styles)
    build_doc(story)
