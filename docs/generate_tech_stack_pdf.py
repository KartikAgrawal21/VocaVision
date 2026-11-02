"""
Generates a professional, beautifully styled PDF document containing all
technology stack tables and architecture details for VocaVision.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUTPUT_PDF = os.path.join(os.path.dirname(__file__), "VocaVision_Tech_Stack.pdf")

def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        alignment=0,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0284C7'),
        spaceAfter=12
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=12
    )

    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=6
    )

    th_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    td_tech_style = ParagraphStyle(
        'TableTech',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#0F172A')
    )

    td_desc_style = ParagraphStyle(
        'TableDesc',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#334155')
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("VocaVision — Technology Stack & Architecture", title_style))
    story.append(Paragraph("Assistive AI System for Real-Time Color, Clothing & Food Recognition with Voice Output", subtitle_style))
    story.append(Paragraph("<b>GitHub:</b> github.com/RAJ-A58/VocaVision &nbsp;|&nbsp; <b>Framework:</b> PyTorch (CUDA) + TensorFlow &nbsp;|&nbsp; <b>Python:</b> 3.13", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284C7'), spaceBefore=0, spaceAfter=10))

    def make_table(header_col1, header_col2, data_rows, col1_width=160, col2_width=380):
        table_data = [[
            Paragraph(f"<b>{header_col1}</b>", th_style),
            Paragraph(f"<b>{header_col2}</b>", th_style)
        ]]
        for row in data_rows:
            table_data.append([
                Paragraph(row[0], td_tech_style),
                Paragraph(row[1], td_desc_style)
            ])
        
        t = Table(table_data, colWidths=[col1_width, col2_width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.HexColor('#FFFFFF')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0, 1), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 7),
            ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ]))
        return t

    # Table 1: Deep Learning
    story.append(Paragraph("1. Deep Learning & Machine Learning", section_heading_style))
    dl_rows = [
        ("PyTorch 2.6.0 + cu124", "Core framework for the Clothing Classifier v2. Implements MobileNetV2 architecture with a custom classification head."),
        ("CUDA 12.4 Acceleration", "Hardware-accelerated training and inference on NVIDIA RTX 3050 Laptop GPU (6GB VRAM)."),
        ("torch.cuda.amp", "Automatic Mixed Precision (AMP) using FP16 tensors to accelerate training and reduce GPU memory consumption."),
        ("Data Augmentations", "PyTorch torchvision transforms: RandomHorizontalFlip, RandomRotation, RandomAffine, and ColorJitter to bridge the domain gap."),
        ("TensorFlow / Keras 2.20", "Framework powering the Food Classifier (MobileNetV2 fine-tuned on Food-101) and legacy clothing CNN fallback."),
        ("TensorFlow Datasets (TFDS)", "High-performance data streaming, pipelining, and preprocessing for the Food-101 dataset."),
        ("Scikit-Learn (K-Means)", "Unsupervised color centroid clustering to extract dominant color centers from segmented pixels.")
    ]
    story.append(make_table("Technology / Module", "Role & Engineering Implementation", dl_rows))
    story.append(Spacer(1, 8))

    # Table 2: Computer Vision
    story.append(Paragraph("2. Computer Vision & Image Processing", section_heading_style))
    cv_rows = [
        ("OpenCV (cv2)", "Real-time webcam video stream capture, frame resizing, format conversions, and contour drawing."),
        ("GrabCut Algorithm (OpenCV)", "Foreground-background extraction isolating the clothing article from complex indoor backgrounds."),
        ("rembg & onnxruntime", "Neural background removal using the lightweight U²-Net (u2netp.onnx, 4.5 MB) ONNX model."),
        ("HSV Color Space Mapping", "Maps K-Means centroid BGR clusters into calibrated Hue-Saturation-Value boundaries with human-readable names."),
        ("Test-Time Augmentation (TTA)", "Inference ensembling averaging predictions across multiple geometric transformations to improve accuracy by +2-3%."),
        ("Pillow (PIL)", "Image type conversion layer mediating between OpenCV NumPy arrays, PyTorch Tensors, and Gradio interfaces.")
    ]
    story.append(make_table("Component", "Function & Technical Scope", cv_rows))
    story.append(Spacer(1, 8))

    # Table 3: Voice Output
    story.append(Paragraph("3. Assistive Audio & Voice Synthesis (TTS)", section_heading_style))
    voice_rows = [
        ("pyttsx3 (SAPI5)", "Native offline Text-to-Speech library utilizing Windows SAPI5 synthesizer for standalone desktop execution without internet."),
        ("Python Threading", "Non-blocking, asynchronous speech dispatch (speak_async) allowing the camera stream to stay smooth during voice playback."),
        ("gTTS (Google Text-to-Speech)", "Cloud-based TTS audio synthesis generating MP3 audio files for the web browser interface.")
    ]
    story.append(make_table("Library / Tool", "Application & Integration", voice_rows))
    story.append(Spacer(1, 8))

    # Table 4: Web Interface
    story.append(Paragraph("4. Web Interface & UI Framework", section_heading_style))
    web_rows = [
        ("Gradio 6.0", "Reactive web UI providing image upload, audio output playback, confidence charts, and segmented mask inspection."),
        ("Gradio Live Tunneling", "Generates temporary public URLs (gradio.live) enabling remote browser testing and demonstration."),
        ("Hugging Face Spaces Ready", "Designed for containerized cloud deployment with zero hardware dependency changes.")
    ]
    story.append(make_table("Framework", "Deployment & Features", web_rows))
    story.append(Spacer(1, 8))

    # Table 5: Hardware & Infrastructure
    story.append(Paragraph("5. Development Environment, Hardware & Version Control", section_heading_style))
    infra_rows = [
        ("Language & Environment", "Python 3.13 (64-bit) running on Windows 11 / PowerShell."),
        ("Hardware Accelerator", "NVIDIA GeForce RTX 3050 Laptop GPU (6.4 GB VRAM) running Driver 596.21."),
        ("Git & Git LFS", "Version control with Git Large File Storage tracking binary neural network weights (*.keras, *.pt)."),
        ("Matplotlib & Seaborn", "Generates training accuracy/loss progression plots and real-time horizontal confidence distribution charts.")
    ]
    story.append(make_table("Resource", "Specification", infra_rows))
    story.append(Spacer(1, 8))

    # Table 6: System Design Highlights
    story.append(Paragraph("6. Architectural Highlights & System Design", section_heading_style))
    arch_rows = [
        ("Hybrid Inference Engine", "Coordinates PyTorch and TensorFlow models simultaneously in memory without cross-framework conflicts."),
        ("Hierarchical Decision Logic", "Implements confidence margin thresholding (FOOD_MIN_THRESHOLD = 0.75, FOOD_MARGIN = 0.30) to eliminate false positives."),
        ("Mask-Restricted Color Analysis", "Applies K-Means clustering strictly within GrabCut foreground masks, filtering out shadows and background colors.")
    ]
    story.append(make_table("Architecture Pillar", "Strategic Benefit", arch_rows))

    doc.build(story)
    print(f"PDF successfully created: {OUTPUT_PDF}")

if __name__ == "__main__":
    build_pdf()
