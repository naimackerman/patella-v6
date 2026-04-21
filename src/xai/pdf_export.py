"""PDF export helpers for KOA-TriFQ clinical reports."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def save_clinical_report_pdf(
    output_path: str | Path,
    title: str,
    report_text: str,
    overlay_png_bytes: bytes | None = None,
):
    """Write a single-page PDF report with optional annotated image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    margin = 36
    cursor_y = height - margin

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(margin, cursor_y, title)
    cursor_y -= 24

    if overlay_png_bytes:
        image_reader = ImageReader(BytesIO(overlay_png_bytes))
        image_width = width - 2 * margin
        image_height = min(260, image_width * 0.70)
        pdf.drawImage(image_reader, margin, cursor_y - image_height, width=image_width, height=image_height, preserveAspectRatio=True, anchor="n")
        cursor_y -= image_height + 18

    pdf.setFont("Helvetica", 10)
    for line in report_text.splitlines():
        if cursor_y <= margin + 12:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            cursor_y = height - margin
        pdf.drawString(margin, cursor_y, line[:140])
        cursor_y -= 12

    pdf.save()
