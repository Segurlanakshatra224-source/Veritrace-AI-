"""
PDF Report Generator

VeriTrace previously only wrote .txt and .json reports (see explainability.py /
reports.py) - there was no PDF generation anywhere in the codebase. This module
adds it: same content as the .txt report, formatted as a proper PDF, with the
heatmap images embedded when they exist on disk.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)


def _get_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            spaceBefore=14,
            spaceAfter=6,
            textColor=colors.HexColor("#1a1a2e"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
        )
    )
    return styles


def _escape(text):
    # Paragraph() interprets a small subset of HTML/XML - escape user/model
    # text so stray "<" or "&" characters don't break rendering.
    text = str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def build_pdf_report(explanation, pdf_path, heatmap_paths=None, title="VeriTrace Explainability Report"):
    """
    explanation: the dict produced by explainability.build_explanation()
    heatmap_paths: optional list of (file_path, caption) tuples to embed.
                   Missing files are skipped silently rather than raising -
                   e.g. if Integrated Gradients failed for this run, its
                   heatmap just won't appear in the PDF instead of crashing
                   the whole report.
    """
    styles = _get_styles()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    for key, value in explanation.items():
        if isinstance(value, dict):
            # e.g. a "Timeline" block - render as a small table instead of raw dict text
            story.append(Paragraph(_escape(key), styles["SectionHeading"]))
            rows = [[str(k), str(v)] for k, v in value.items()]
            if rows:
                table = Table(rows, colWidths=[150, 330])
                table.setStyle(
                    TableStyle(
                        [
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ]
                    )
                )
                story.append(table)
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(_escape(key), styles["SectionHeading"]))
            story.append(Paragraph(_escape(value), styles["Body"]))
            story.append(Spacer(1, 6))

    if heatmap_paths:
        available = [(p, c) for p, c in heatmap_paths if p and os.path.exists(p)]
        if available:
            story.append(PageBreak())
            story.append(Paragraph("Explainability Heatmaps", styles["SectionHeading"]))
            story.append(Spacer(1, 6))
            for path, caption in available:
                try:
                    story.append(Paragraph(_escape(caption), styles["Body"]))
                    story.append(RLImage(path, width=4.5 * inch, height=2.8 * inch))
                    story.append(Spacer(1, 12))
                except Exception as e:
                    story.append(Paragraph(f"(could not embed {caption}: {e})", styles["Body"]))

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, title=title)
    doc.build(story)

    return pdf_path


def save_pdf_report(explanation, report_folder, filename_prefix="report", stamp=None, heatmap_paths=None):
    """
    Convenience wrapper matching the naming convention used by
    explainability.save_report()'s .txt/.json outputs.
    """
    from datetime import datetime

    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(report_folder, f"{filename_prefix}_{stamp}.pdf")

    try:
        build_pdf_report(explanation, pdf_path, heatmap_paths=heatmap_paths)
        print(f"Saved PDF report to {pdf_path}")
        return pdf_path
    except Exception as e:
        import traceback
        print(f"[pdf_report] Failed to generate PDF report: {e}")
        traceback.print_exc()
        return None
