import io
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

# colour for each risk level used in the predictions table
RISK_COLORS = {
    "HIGH":   colors.HexColor("#8B0000"),
    "MEDIUM": colors.HexColor("#8B6914"),
    "LOW":    colors.HexColor("#1a5c38"),
}


def generate_pdf(student_row, recommendations, risk_level=None, predicted_grade=None, pass_fail=None, category=None) -> bytes:
    # we write the pdf into memory instead of saving it to disk
    # this lets streamlit serve it as a download without creating temporary files
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm,  bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    elements = []

    # title section
    title_style = ParagraphStyle("title", fontSize=20, fontName="Helvetica-Bold", spaceAfter=12)
    sub_style   = ParagraphStyle("sub",   fontSize=11, fontName="Helvetica", textColor=colors.grey, spaceBefore=6)

    elements.append(Paragraph("Student Intelligence Platform", title_style))
    elements.append(Paragraph("Academic Performance Report", sub_style))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 0.5 * cm))

    # student profile table
    section_style = ParagraphStyle("section", fontSize=13, fontName="Helvetica-Bold", spaceAfter=6, spaceBefore=10)
    body_style    = ParagraphStyle("body",    fontSize=11, fontName="Helvetica", spaceAfter=4)

    elements.append(Paragraph("Student Profile", section_style))

    info_data = [
        ["Name",           student_row["name"]],
        ["Subject",        student_row["subject"]],
        ["Report Date",    date.today().strftime("%d %B %Y")],
        ["Period 1 Grade", f"{int(student_row['G1'])} / 20"],
        ["Period 2 Grade", f"{int(student_row['G2'])} / 20"],
        ["Final Grade",    f"{int(student_row['G3'])} / 20"],
        ["Absences",       str(int(student_row["absences"]))],
        ["Study Time",     {1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"}[int(student_row["studytime"])] + " / week"],
        ["Past Failures",  str(int(student_row["failures"]))],
    ]

    # alternating row background makes the table easier to read
    info_table = Table(info_data, colWidths=[5 * cm, 11 * cm])
    info_table.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",       (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 11),
        ("TEXTCOLOR",      (0, 0), (0, -1), colors.HexColor("#333333")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f8f8f8"), colors.white]),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",     (0, 0), (-1, -1), 6),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.6 * cm))

    # model predictions table
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    elements.append(Paragraph("Model Predictions", section_style))

    # use the values passed in directly — if not provided, fall back to the row
    risk      = risk_level      if risk_level      is not None else student_row["risk_level"]
    pred_g    = predicted_grade if predicted_grade is not None else student_row["predicted_grade"]
    pf        = pass_fail       if pass_fail       is not None else student_row["pass_fail"]
    cat       = category        if category        is not None else student_row["category"]
    risk_color = RISK_COLORS.get(risk, colors.black)

    pred_data = [
        ["Predicted Final Grade", f"{pred_g} / 20"],
        ["Pass / Fail",           pf],
        ["Performance Category",  cat],
        ["Risk Level",            risk],
    ]

    pred_table = Table(pred_data, colWidths=[5 * cm, 11 * cm])
    pred_table.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",       (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f8f8f8"), colors.white]),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",     (0, 0), (-1, -1), 6),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        # highlight the risk level cell with the matching colour
        ("BACKGROUND",     (1, 3), (1, 3), risk_color),
        ("TEXTCOLOR",      (1, 3), (1, 3), colors.white),
        ("FONTNAME",       (1, 3), (1, 3), "Helvetica-Bold"),
    ]))
    elements.append(pred_table)
    elements.append(Spacer(1, 0.6 * cm))

    # recommendations list
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    elements.append(Paragraph("Recommendations", section_style))

    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", body_style))

    # footer
    elements.append(Spacer(1, 1 * cm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "Generated by Student Intelligence Platform",
        ParagraphStyle("footer", fontSize=9, textColor=colors.grey)
    ))

    doc.build(elements)
    return buffer.getvalue()
