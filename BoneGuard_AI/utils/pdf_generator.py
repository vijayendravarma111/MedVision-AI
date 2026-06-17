from fpdf import FPDF
from datetime import datetime
import os

def create_pdf(report):

    os.makedirs(
        "reports",
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        f"reports/BoneGuard_Report_{timestamp}.pdf"
    )

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font(
        "Arial",
        "B",
        18
    )

    pdf.cell(
        0,
        10,
        "BoneGuard AI Report",
        ln=True,
        align="C"
    )

    pdf.ln(10)

    for key, value in report.items():

        pdf.set_font(
            "Arial",
            "B",
            12
        )

        pdf.cell(
            0,
            8,
            key,
            ln=True
        )

        pdf.set_font(
            "Arial",
            "",
            11
        )

        safe_text = str(value)

        safe_text = safe_text.replace("•", "-")
        safe_text = safe_text.replace("–", "-")
        safe_text = safe_text.replace("—", "-")

        pdf.multi_cell(
            0,
            6,
            safe_text
        )

        pdf.ln(3)

    pdf.output(
        output_path
    )

    return output_path