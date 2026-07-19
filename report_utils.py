from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import io
import numpy as np
from PIL import Image as PILImage


def pil_to_rl_image(pil_img, width_mm=80):
    """Converts PIL image to ReportLab image object."""
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    w = width_mm * mm
    aspect = pil_img.height / pil_img.width
    return RLImage(buf, width=w, height=w * aspect)


def generate_report(
    original_pil,
    heatmap_pil,
    filename,
    pneumonia_prob,
    normal_prob,
    predicted_label
):
    """
    Generates a clinical-style PDF report.
    Returns bytes of the PDF.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm
    )

    styles = getSampleStyleSheet()
    elements = []

    # ── Header ────────────────────────────────────────────────
    header_style = ParagraphStyle(
        'header', fontSize=18, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0a0f0d'),
        spaceAfter=2*mm
    )
    sub_style = ParagraphStyle(
        'sub', fontSize=9, fontName='Helvetica',
        textColor=colors.HexColor('#888888'),
        spaceAfter=6*mm
    )

    elements.append(Paragraph("PneumoVision — Clinical Report", header_style))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ·  Model: ResNet18  ·  For educational use only",
        sub_style
    ))

    # ── Divider ───────────────────────────────────────────────
    elements.append(Table(
        [['']],
        colWidths=[170*mm],
        style=TableStyle([('LINEABOVE', (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd'))])
    ))
    elements.append(Spacer(1, 5*mm))

    # ── Verdict banner ────────────────────────────────────────
    verdict_color = '#c0392b' if predicted_label == 'PNEUMONIA' else '#1a7a4a'
    verdict_style = ParagraphStyle(
        'verdict', fontSize=22, fontName='Helvetica-Bold',
        textColor=colors.HexColor(verdict_color),
        spaceAfter=2*mm
    )
    elements.append(Paragraph(f"Diagnosis: {predicted_label}", verdict_style))

    conf = max(pneumonia_prob, normal_prob) * 100
    elements.append(Paragraph(
        f"Model confidence: {conf:.1f}%",
        ParagraphStyle('conf', fontSize=11, fontName='Helvetica',
                       textColor=colors.HexColor('#555555'), spaceAfter=6*mm)
    ))

    # ── Images side by side ───────────────────────────────────
    orig_rl = pil_to_rl_image(original_pil, width_mm=78)
    heat_rl = pil_to_rl_image(heatmap_pil,  width_mm=78)

    img_table = Table(
        [[orig_rl, heat_rl]],
        colWidths=[85*mm, 85*mm]
    )
    img_table.setStyle(TableStyle([
        ('ALIGN',     (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',    (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING',  (0,0), (-1,-1), 2*mm),
        ('RIGHTPADDING', (0,0), (-1,-1), 2*mm),
    ]))
    elements.append(img_table)

    label_table = Table(
        [['Original X-Ray', 'Grad-CAM Heatmap']],
        colWidths=[85*mm, 85*mm]
    )
    label_table.setStyle(TableStyle([
        ('ALIGN',    (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('TEXTCOLOR',(0,0), (-1,-1), colors.HexColor('#888888')),
        ('TOPPADDING',(0,0),(-1,-1), 2*mm),
    ]))
    elements.append(label_table)
    elements.append(Spacer(1, 6*mm))

    # ── Probability table ─────────────────────────────────────
    elements.append(Paragraph(
        "Probability Breakdown",
        ParagraphStyle('sec', fontSize=12, fontName='Helvetica-Bold',
                       textColor=colors.HexColor('#111111'), spaceAfter=3*mm)
    ))

    prob_data = [
        ['Class', 'Probability', 'Confidence'],
        ['PNEUMONIA', f'{pneumonia_prob*100:.2f}%',
         'HIGH' if pneumonia_prob > 0.75 else 'MEDIUM' if pneumonia_prob > 0.4 else 'LOW'],
        ['NORMAL', f'{normal_prob*100:.2f}%',
         'HIGH' if normal_prob > 0.75 else 'MEDIUM' if normal_prob > 0.4 else 'LOW'],
    ]

    prob_table = Table(prob_data, colWidths=[60*mm, 60*mm, 50*mm])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  colors.HexColor('#111111')),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f9f9f9'), colors.white]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 3*mm),
        ('BOTTOMPADDING',(0,0),(-1,-1), 3*mm),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 6*mm))

    # ── Model info ────────────────────────────────────────────
    elements.append(Paragraph(
        "Model Information",
        ParagraphStyle('sec2', fontSize=12, fontName='Helvetica-Bold',
                       textColor=colors.HexColor('#111111'), spaceAfter=3*mm)
    ))

    info_data = [
        ['Architecture', 'ResNet18 (Transfer Learning)'],
        ['Training Dataset', 'Chest X-Ray Images — Kaggle (5,216 images)'],
        ['Test Accuracy', '89.74%'],
        ['Classes', 'NORMAL · PNEUMONIA'],
        ['Input Size', '224 × 224 px'],
        ['Explainability', 'Grad-CAM (layer4.conv2)'],
        ['File Analyzed', filename],
    ]

    info_table = Table(info_data, colWidths=[60*mm, 110*mm])
    info_table.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',    (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('TEXTCOLOR',   (0,0), (0,-1), colors.HexColor('#555555')),
        ('ROWBACKGROUNDS', (0,0), (-1,-1),
         [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#eeeeee')),
        ('TOPPADDING',  (0,0), (-1,-1), 2.5*mm),
        ('BOTTOMPADDING',(0,0),(-1,-1), 2.5*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 3*mm),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 8*mm))

    # ── Disclaimer ────────────────────────────────────────────
    elements.append(Paragraph(
        "⚠ This report is generated by an AI model for educational purposes only. "
        "It is not a certified medical device and should not be used for clinical diagnosis. "
        "Always consult a qualified medical professional.",
        ParagraphStyle('disc', fontSize=8, fontName='Helvetica',
                       textColor=colors.HexColor('#aaaaaa'),
                       borderPad=3*mm)
    ))

    doc.build(elements)
    buf.seek(0)
    return buf.read()