"""
VocaVision — PowerPoint Presentation Generator
Generates VocaVision_Presentation.pptx in the project root.
Run:  python generate_ppt.py
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG        = RGBColor(0x0F, 0x0F, 0x1A)   # near-black navy
C_CARD      = RGBColor(0x16, 0x21, 0x3E)   # dark blue card
C_ACCENT1   = RGBColor(0x2E, 0xCC, 0x71)   # emerald green
C_ACCENT2   = RGBColor(0x3A, 0x9B, 0xD5)   # sky blue
C_ACCENT3   = RGBColor(0xF3, 0x9C, 0x12)   # amber
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT     = RGBColor(0xCC, 0xDD, 0xEE)   # muted blue-white
C_MUTED     = RGBColor(0x88, 0x99, 0xAA)
C_RED       = RGBColor(0xE7, 0x4C, 0x3C)

W, H = Inches(13.33), Inches(7.5)          # 16:9 widescreen

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

BLANK = prs.slide_layouts[6]   # truly blank layout


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_bg(slide, color=C_BG):
    """Fill slide background with a solid colour."""
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def box(slide, l, t, w, h, bg=None, border=None, border_w=Pt(1.5), radius=False):
    """Add a filled rectangle shape."""
    from pptx.util import Emu
    shp = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(l), Inches(t), Inches(w), Inches(h)
    )
    shp.line.fill.background()  # no border by default
    if bg:
        shp.fill.solid()
        shp.fill.fore_color.rgb = bg
    else:
        shp.fill.background()
    if border:
        shp.line.fill.solid()
        shp.line.fill.fore_color.rgb = border
        shp.line.width = border_w
    return shp


def txt(slide, text, l, t, w, h,
        size=18, bold=False, color=C_WHITE, align=PP_ALIGN.LEFT,
        italic=False, wrap=True):
    """Add a text box."""
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size    = Pt(size)
    run.font.bold    = bold
    run.font.italic  = italic
    run.font.color.rgb = color
    return tb


def bullet_box(slide, items, l, t, w, h,
               size=15, color=C_LIGHT, dot=C_ACCENT1, spacing=0.52):
    """Add a vertical list of bullet points with coloured dots."""
    for i, item in enumerate(items):
        # dot
        dot_shp = slide.shapes.add_shape(1,
            Inches(l), Inches(t + i * spacing + 0.06),
            Inches(0.12), Inches(0.12))
        dot_shp.fill.solid(); dot_shp.fill.fore_color.rgb = dot
        dot_shp.line.fill.background()
        # text
        txt(slide, item, l + 0.22, t + i * spacing, w - 0.22, spacing,
            size=size, color=color)


def chip(slide, label, l, t, w=1.7, h=0.38, bg=C_ACCENT1, tc=C_BG):
    """Pill/badge label."""
    b = box(slide, l, t, w, h, bg=bg)
    txt(slide, label, l, t + 0.02, w, h - 0.04,
        size=13, bold=True, color=tc, align=PP_ALIGN.CENTER)


def heading_bar(slide, title, subtitle=None):
    """Top gradient-like heading bar."""
    box(slide, 0, 0, 13.33, 1.35, bg=C_CARD)
    # accent stripe
    box(slide, 0, 1.28, 13.33, 0.07, bg=C_ACCENT1)
    txt(slide, title,    0.4, 0.10, 10, 0.65, size=32, bold=True,  color=C_WHITE)
    if subtitle:
        txt(slide, subtitle, 0.4, 0.72, 10, 0.50, size=16, italic=True, color=C_ACCENT2)


def section_label(slide, text, l, t, color=C_ACCENT1):
    txt(slide, text.upper(), l, t, 4, 0.35, size=11, bold=True, color=color)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — Title
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)

# Left accent bar
box(sl, 0, 0, 0.18, 7.5, bg=C_ACCENT1)

# Main card
box(sl, 0.5, 1.2, 9.0, 5.1, bg=C_CARD, border=C_ACCENT1, border_w=Pt(1.5))

# Eye + mic emoji replaced with text icons
txt(sl, "🎙️  👁️", 1.0, 1.4, 3.0, 1.0, size=44, align=PP_ALIGN.LEFT)

txt(sl, "VocaVision", 1.0, 2.4, 8.5, 1.2,
    size=54, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)

txt(sl, "AI-Based Color, Clothing & Food Recognition",
    1.0, 3.5, 8.2, 0.65, size=22, color=C_ACCENT2, align=PP_ALIGN.LEFT)

txt(sl, "Assistive System for Independent Daily Living",
    1.0, 4.1, 8.2, 0.65, size=18, italic=True, color=C_LIGHT, align=PP_ALIGN.LEFT)

# Chips
chip(sl, "YOLO v3",        1.0, 5.2, bg=C_ACCENT1)
chip(sl, "PyTorch GPU",    2.85, 5.2, bg=C_ACCENT2, tc=C_BG)
chip(sl, "Gradio Live",    4.70, 5.2, bg=C_ACCENT3, tc=C_BG)
chip(sl, "gTTS Voice",     6.55, 5.2, bg=RGBColor(0x9B,0x59,0xB6), tc=C_WHITE)

# Right decorative panel
box(sl, 9.8, 1.4, 3.2, 4.8, bg=RGBColor(0x0A,0x0A,0x14), border=C_ACCENT2, border_w=Pt(1))
txt(sl, "LIVE DETECTION", 9.9, 1.6, 3.0, 0.5, size=11, bold=True,
    color=C_ACCENT1, align=PP_ALIGN.CENTER)
for i, line in enumerate([
    "📷  Webcam feed",
    "🔍  YOLO detection",
    "🎨  K-Means colors",
    "🔊  Voice output",
    "📊  Confidence chart",
]):
    txt(sl, line, 9.9, 2.25 + i*0.7, 3.0, 0.6, size=14, color=C_LIGHT,
        align=PP_ALIGN.LEFT)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — Problem Statement
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)
heading_bar(sl, "Problem Statement",
            "Challenges faced by visually impaired individuals in daily life")

# Left problem cards
problems = [
    ("👕", "Clothing Recognition",
     "Cannot independently identify clothing color or type when dressing."),
    ("🍕", "Food Identification",
     "Difficulty distinguishing food items on a plate or in a kitchen."),
    ("🗣️", "No Feedback Loop",
     "Existing apps require internet APIs, delay responses, or cost money."),
    ("♿", "Accessibility Gap",
     "No single offline tool combines clothing + food + voice output."),
]

for i, (icon, title, desc) in enumerate(problems):
    col = i % 2
    row = i // 2
    lx  = 0.4 + col * 6.45
    ty  = 1.55 + row * 2.6
    box(sl, lx, ty, 6.1, 2.3, bg=C_CARD, border=C_ACCENT2, border_w=Pt(1.2))
    txt(sl, icon,  lx+0.15, ty+0.12, 0.8, 0.8, size=28)
    txt(sl, title, lx+1.0,  ty+0.15, 4.8, 0.5, size=17, bold=True, color=C_WHITE)
    txt(sl, desc,  lx+1.0,  ty+0.65, 4.8, 1.4, size=13, color=C_LIGHT)

# Bottom stat bar
box(sl, 0, 6.7, 13.33, 0.8, bg=RGBColor(0x0A,0x10,0x20))
txt(sl, "253 million people worldwide are visually impaired  (WHO, 2024)",
    0, 6.72, 13.33, 0.7, size=15, italic=True, color=C_ACCENT2, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — Solution Overview
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)
heading_bar(sl, "Our Solution — VocaVision",
            "A single offline, real-time assistive AI combining vision + voice")

# Central pipeline flow
steps = [
    ("📷", "Live Camera\nFeed",     C_ACCENT2),
    ("🔍", "YOLO v3\nDetection",    C_ACCENT1),
    ("🎨", "K-Means\nColor ID",     C_ACCENT3),
    ("🗣️", "Natural Language\nDesc", RGBColor(0x9B,0x59,0xB6)),
    ("🔊", "Voice\nOutput",         RGBColor(0xE7,0x4C,0x3C)),
]

for i, (icon, label, color) in enumerate(steps):
    lx = 0.5 + i * 2.52
    box(sl, lx, 1.7, 2.1, 2.1, bg=C_CARD, border=color, border_w=Pt(2))
    txt(sl, icon,  lx+0.6,  1.85, 0.9, 0.7, size=26, align=PP_ALIGN.CENTER)
    txt(sl, label, lx+0.05, 2.6,  2.0, 0.9,
        size=13, bold=True, color=color, align=PP_ALIGN.CENTER)
    # Arrow (except after last)
    if i < len(steps) - 1:
        txt(sl, "▶", lx+2.12, 2.55, 0.38, 0.5,
            size=18, color=C_MUTED, align=PP_ALIGN.CENTER)

# Key differentiators
box(sl, 0.4, 4.1, 12.5, 2.9, bg=RGBColor(0x0A,0x10,0x20), border=C_ACCENT1, border_w=Pt(1))
txt(sl, "Why VocaVision is Different", 0.7, 4.2, 12, 0.45,
    size=16, bold=True, color=C_ACCENT1)

diffs = [
    "✅  Fully offline — no internet or API key needed for inference",
    "✅  Real-time live webcam feed processed every 1.5 seconds",
    "✅  Single unified model (YOLO v3) detects both food and clothing",
    "✅  Voice output automatically narrates detected objects with color",
    "✅  14 color names including navy, beige, and neutral tones",
]
for i, d in enumerate(diffs):
    txt(sl, d, 0.7, 4.75 + i*0.41, 12, 0.4, size=13, color=C_LIGHT)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — System Architecture
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)
heading_bar(sl, "System Architecture",
            "Three-tier engine with automatic fallback")

# Engine priority diagram
tiers = [
    ("YOLO v3 Engine  [PRIMARY]",
     "vocavision_yolo.pt — Custom YOLOv8s fine-tuned on 20 classes\n"
     "• Single forward pass detects ALL food + clothing objects\n"
     "• Bounding-box crops fed to K-Means color detector\n"
     "• Per-class confidence thresholds cut false positives",
     C_ACCENT1, "🥇"),
    ("Hybrid v2  [FALLBACK]",
     "PyTorch MobileNetV2 (clothing, CUDA) + TF food classifier\n"
     "• Test-Time Augmentation (TTA) for robust predictions\n"
     "• Bag-penalty: reduces frequent false-positive class by 40%\n"
     "• GrabCut segmentation isolates clothing region for color",
     C_ACCENT2, "🥈"),
    ("TF v1 Legacy  [LAST RESORT]",
     "clothing_classifier.keras — original 3-block CNN\n"
     "• Always available as final safety net\n"
     "• Slower but zero additional dependencies",
     C_MUTED,   "🥉"),
]

for i, (title, body, color, medal) in enumerate(tiers):
    ty = 1.55 + i * 1.88
    box(sl, 0.4, ty, 12.5, 1.7, bg=C_CARD, border=color, border_w=Pt(1.5))
    txt(sl, medal, 0.55, ty+0.12, 0.5, 0.5, size=20)
    txt(sl, title, 1.15, ty+0.10, 6.0, 0.45, size=15, bold=True, color=color)
    txt(sl, body,  1.15, ty+0.55, 11.4, 1.0, size=12, color=C_LIGHT)

# Right side mini-legend
box(sl, 9.9, 1.55, 2.95, 1.6, bg=RGBColor(0x0A,0x10,0x20))
txt(sl, "Auto-selection at startup",
    9.95, 1.6, 2.8, 0.4, size=11, bold=True, color=C_ACCENT1, align=PP_ALIGN.CENTER)
txt(sl, "The highest-priority\nmodel that is present on\ndisk is automatically used.",
    9.95, 2.0, 2.8, 0.9, size=11, color=C_LIGHT, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — Technology Stack
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)
heading_bar(sl, "Technology Stack", "Libraries, frameworks, and models used")

techs = [
    ("🧠", "Ultralytics YOLOv8s",
     "Object detection backbone. Fine-tuned on custom 20-class\nfood + clothing dataset.",
     C_ACCENT1),
    ("🔥", "PyTorch + CUDA",
     "MobileNetV2 clothing classifier. GPU-accelerated inference\nwith torchvision transforms.",
     RGBColor(0xEE,0x44,0x22)),
    ("🤖", "TensorFlow / Keras",
     "Food classifier (MobileNetV2 transfer learning).\nFallback clothing CNN.",
     RGBColor(0xFF,0x6F,0x00)),
    ("👁️", "OpenCV",
     "GrabCut segmentation, bounding box rendering,\nframe pre-processing.",
     C_ACCENT2),
    ("🎨", "scikit-learn K-Means",
     "Dominant color extraction from YOLO crop regions.\n14 named colors via HSV mapping.",
     RGBColor(0xF3,0x9C,0x12)),
    ("🌐", "Gradio 6 (Blocks)",
     "Live webcam streaming UI. Processes frame every 1.5s,\nshares annotated output panel.",
     RGBColor(0xFF,0x7C,0x00)),
    ("🔊", "gTTS",
     "Google Text-to-Speech. Converts AI description to MP3.\nThrottled — fires only on change.",
     RGBColor(0x1A,0xBC,0x9C)),
    ("🧹", "rembg (u2netp)",
     "Neural background removal for cleaner color detection\nin fallback (non-YOLO) mode.",
     RGBColor(0x9B,0x59,0xB6)),
]

cols = 4
for i, (icon, title, desc, color) in enumerate(techs):
    col = i % cols
    row = i // cols
    lx  = 0.35 + col * 3.22
    ty  = 1.6  + row * 2.55
    box(sl, lx, ty, 3.0, 2.3, bg=C_CARD, border=color, border_w=Pt(1.5))
    txt(sl, icon,  lx+0.1,  ty+0.10, 0.6, 0.6, size=22)
    txt(sl, title, lx+0.75, ty+0.12, 2.1, 0.5, size=13, bold=True, color=color)
    txt(sl, desc,  lx+0.10, ty+0.68, 2.8, 1.4, size=11, color=C_LIGHT)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6 — How It Works (Pipeline Detail)
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)
heading_bar(sl, "How It Works — Inference Pipeline",
            "Step-by-step flow from webcam frame to voice output")

# Vertical numbered steps
steps = [
    (C_ACCENT1,  "1",
     "Webcam frame captured",
     "Gradio streams the live webcam feed. A frame is grabbed every 1.5 s and\n"
     "converted from PIL RGB to OpenCV BGR for inference."),
    (C_ACCENT2,  "2",
     "YOLO v3 object detection",
     "Custom YOLOv8s runs a single forward pass. Per-class confidence thresholds\n"
     "filter detections: Bag ≥75%, Sneaker ≥70%, all others ≥52%."),
    (C_ACCENT3,  "3",
     "K-Means color extraction",
     "For each clothing bounding box crop, K-Means (k=4) clusters pixel colors.\n"
     "Background clusters (white/black >70%) are discarded automatically."),
    (RGBColor(0x9B,0x59,0xB6), "4",
     "Natural language description built",
     "Oxford-comma sentence generator: 'I can see a red T-shirt and a pizza.'\n"
     "Falls back to classifier pipeline if YOLO finds nothing confident."),
    (RGBColor(0x1A,0xBC,0x9C), "5",
     "Voice output (throttled)",
     "gTTS converts description to MP3. gr.State tracks the last description —\n"
     "audio only fires when the detected object changes, preventing repetition."),
]

for i, (color, num, title, body) in enumerate(steps):
    ty = 1.48 + i * 1.08
    # Number badge
    badge = box(sl, 0.35, ty+0.05, 0.55, 0.55, bg=color)
    txt(sl, num, 0.35, ty+0.05, 0.55, 0.55,
        size=18, bold=True, color=C_BG, align=PP_ALIGN.CENTER)
    # Content
    txt(sl, title, 1.1, ty+0.03, 5.5, 0.4, size=14, bold=True, color=color)
    txt(sl, body,  1.1, ty+0.42, 12.0, 0.55, size=11.5, color=C_LIGHT)
    # Divider (except last)
    if i < len(steps)-1:
        box(sl, 0.55, ty+0.65, 0.15, 0.38, bg=color)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7 — Model Details & Accuracy
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)
heading_bar(sl, "Model Details & Training",
            "Datasets, architectures, and expected performance metrics")

# Table header
header_cols = ["Model", "Dataset", "Architecture", "Classes", "Accuracy"]
col_w = [2.6, 2.6, 2.8, 1.3, 1.7]
col_x = [0.35]
for w in col_w[:-1]:
    col_x.append(col_x[-1]+w)

box(sl, 0.35, 1.55, 11.65, 0.5, bg=C_ACCENT1)
for j, (h, cx, cw) in enumerate(zip(header_cols, col_x, col_w)):
    txt(sl, h, cx+0.05, 1.58, cw, 0.42,
        size=13, bold=True, color=C_BG, align=PP_ALIGN.CENTER)

rows = [
    ("YOLO v3 Detector",    "Custom (food+clothing)",  "YOLOv8s fine-tuned",  "20",  "~mAP50 see runs/"),
    ("Food Classifier",     "Food-101 (10 classes)",   "MobileNetV2 + TL",    "10",  "~75–85%"),
    ("Clothing v2 (PT)",    "Fashion-MNIST",            "MobileNetV2 + TTA",   "10",  "~88–92%"),
    ("Clothing v1 (TF)",    "Fashion-MNIST",            "3-block CNN",         "10",  "~88–90%"),
]

for i, row in enumerate(rows):
    ty   = 2.1 + i * 0.72
    rbg  = C_CARD if i % 2 == 0 else RGBColor(0x12,0x1A,0x30)
    box(sl, 0.35, ty, 11.65, 0.68, bg=rbg)
    for val, cx, cw in zip(row, col_x, col_w):
        txt(sl, val, cx+0.05, ty+0.08, cw-0.1, 0.52,
            size=12, color=C_LIGHT, align=PP_ALIGN.CENTER)

# Class lists
box(sl, 0.35, 5.05, 5.7, 2.2, bg=C_CARD, border=C_ACCENT1, border_w=Pt(1))
txt(sl, "Food Classes (10)",    0.5,  5.1, 5.4, 0.4, size=13, bold=True, color=C_ACCENT1)
txt(sl, "Apple Pie  •  Fried Rice  •  Hamburger  •  Hot Dog  •  Ice Cream\n"
        "Pizza  •  Ramen  •  Samosa  •  Sushi  •  Waffles",
    0.5, 5.55, 5.4, 1.55, size=12, color=C_LIGHT)

box(sl, 6.3, 5.05, 6.8, 2.2, bg=C_CARD, border=C_ACCENT2, border_w=Pt(1))
txt(sl, "Clothing Classes (10)", 6.45, 5.1, 6.5, 0.4, size=13, bold=True, color=C_ACCENT2)
txt(sl, "T-shirt/Top  •  Trouser  •  Pullover  •  Dress  •  Coat\n"
        "Sandal  •  Shirt  •  Sneaker  •  Bag  •  Ankle Boot",
    6.45, 5.55, 6.5, 1.55, size=12, color=C_LIGHT)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8 — Key Features
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)
heading_bar(sl, "Key Features", "What makes VocaVision stand out")

features = [
    ("🔴 Live Video Feed",
     ["Real-time webcam detection via Gradio streaming",
      "Annotated bounding-box output panel updates every 1.5s",
      "No snapshot button needed — fully automatic"],
     C_RED, 0.35, 1.55),
    ("🟢 Multi-Class Detection",
     ["Single YOLO model covers 10 food + 10 clothing classes",
      "Multiple objects in one frame handled simultaneously",
      "Oxford-comma natural sentences for multi-item scenes"],
     C_ACCENT1, 0.35, 4.05),
    ("🔵 Smart Color Detection",
     ["14 named colors incl. navy, beige, pink, brown",
      "Background suppression — ignores white studio backdrops",
      "K-Means clusters on YOLO crop (not full frame)"],
     C_ACCENT2, 6.85, 1.55),
    ("🟡 Voice Narration",
     ["gTTS converts description to spoken MP3 automatically",
      "Audio throttled via gr.State — fires only on change",
      "Natural phrasing: 'I can see a red T-shirt'"],
     C_ACCENT3, 6.85, 4.05),
]

for title, bullets, color, lx, ty in features:
    box(sl, lx, ty, 6.25, 2.25, bg=C_CARD, border=color, border_w=Pt(1.8))
    txt(sl, title, lx+0.15, ty+0.10, 5.9, 0.45, size=15, bold=True, color=color)
    bullet_box(sl, bullets, lx+0.15, ty+0.6, 5.9, 1.5,
               size=12, color=C_LIGHT, dot=color, spacing=0.52)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9 — Live Demo
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)
heading_bar(sl, "Live Demonstration", "VocaVision running on webcam in real time")

# Center demo placeholder
box(sl, 1.5, 1.6, 10.33, 4.6, bg=RGBColor(0x05,0x08,0x12),
    border=C_ACCENT1, border_w=Pt(2.5))
txt(sl, "▶  LIVE DEMO", 1.5, 1.6, 10.33, 4.6,
    size=44, bold=True, color=RGBColor(0x22,0x44,0x22), align=PP_ALIGN.CENTER)

# Side labels
for i, label in enumerate([
    "📷  Camera tab → allow access",
    "🔍  Detection starts automatically",
    "🎨  Colors shown on annotated panel",
    "🔊  Voice fires on new detection",
]):
    box(sl, 0.2, 1.65 + i*1.1, 1.1, 1.0, bg=C_CARD)
    txt(sl, label, 0.25, 1.7 + i*1.1, 1.0, 0.9, size=10, color=C_ACCENT1)

txt(sl, "Open  http://localhost:7860  →  python app.py",
    1.5, 6.35, 10.33, 0.5, size=14, color=C_MUTED, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 10 — Challenges & Solutions
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)
heading_bar(sl, "Challenges Faced & Solutions",
            "Key engineering problems encountered during development")

pairs = [
    ("🛑 Challenge",
     "Bag class false positives — YOLO frequently labeled red T-shirts as 'Bag'",
     "✅ Solution",
     "Per-class confidence thresholds: Bag requires ≥75% confidence vs 52% default.\n"
     "Fallback path applies a 40% penalty to Bag probability.",
     C_RED, C_ACCENT1),
    ("🛑 Challenge",
     "Color detector returning 'white' for studio-background clothing photos",
     "✅ Solution",
     "Background cluster filter: near-white clusters covering >70% of image\n"
     "are skipped; next most-dominant color promoted as result.",
     C_RED, C_ACCENT1),
    ("🛑 Challenge",
     "PyTorch checkpoint saved as wrapped dict (model_state key), not plain state_dict",
     "✅ Solution",
     "Loader detects the wrapper format and unwraps model_state automatically.\n"
     "Backward-compatible with both plain and wrapped checkpoint formats.",
     C_RED, C_ACCENT1),
    ("🛑 Challenge",
     "Gradio 6 breaking changes: theme in Blocks(), mirror_webcam removed",
     "✅ Solution",
     "Migrated theme to launch(). Removed mirror_webcam parameter.\n"
     "Streaming tested with concurrency_limit=1 to avoid GPU contention.",
     C_RED, C_ACCENT1),
]

for i, (ct, cd, st, sd, cc, sc) in enumerate(pairs):
    row = i // 2
    col = i %  2
    lx  = 0.35 + col * 6.5
    ty  = 1.55 + row * 2.75
    box(sl, lx, ty, 6.2, 2.55, bg=C_CARD)
    txt(sl, ct, lx+0.1, ty+0.08, 2.5, 0.35, size=12, bold=True, color=cc)
    txt(sl, cd, lx+0.1, ty+0.43, 5.9, 0.55, size=11.5, color=C_LIGHT)
    box(sl, lx, ty+1.02, 6.2, 0.04, bg=sc)
    txt(sl, st, lx+0.1, ty+1.12, 2.5, 0.35, size=12, bold=True, color=sc)
    txt(sl, sd, lx+0.1, ty+1.48, 5.9, 0.90, size=11.5, color=C_LIGHT)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 11 — Future Enhancements
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)
heading_bar(sl, "Future Enhancements",
            "Planned improvements and extended capabilities")

enhancements = [
    ("📱", "Mobile App (Flutter / React Native)",
     ["On-device ONNX/TFLite model export for real-time phone camera use",
      "Offline TTS with no internet dependency",
      "Haptic + audio feedback modes for accessibility"],
     C_ACCENT1),
    ("🌍", "Multi-Language Voice Output",
     ["Support Hindi, Tamil, French and other languages via gTTS lang codes",
      "Language auto-detection based on device locale",
      "Regional food & clothing terminology adaptation"],
     C_ACCENT2),
    ("👗", "Expanded Class Coverage",
     ["Grow clothing to 50+ classes (jackets, kurtas, sarees, etc.)",
      "Expand food to 50+ classes using Food-101 full dataset",
      "Pattern detection: stripes, floral, plain, checkered"],
     C_ACCENT3),
    ("🧠", "LLM-Enhanced Descriptions",
     ["Pipe YOLO detections into a small on-device LLM (e.g., Gemma 2B)",
      "Generate richer sentences: 'A red striped T-shirt — casual style'",
      "Context-aware suggestions: 'Good for office wear'"],
     RGBColor(0x9B,0x59,0xB6)),
    ("📊", "User Analytics Dashboard",
     ["Track detection history with timestamps",
      "Export daily usage reports (CSV/PDF)",
      "Insights on most-detected items for caregiver review"],
     RGBColor(0x1A,0xBC,0x9C)),
    ("🔐", "Privacy Mode",
     ["All inference stays fully offline with local TTS",
      "No webcam frames stored or transmitted",
      "GDPR-compliant local processing pipeline"],
     RGBColor(0xE7,0x4C,0x3C)),
]

cols = 3
for i, (icon, title, bullets, color) in enumerate(enhancements):
    col = i % cols
    row = i // cols
    lx  = 0.35 + col * 4.3
    ty  = 1.55 + row * 2.7
    box(sl, lx, ty, 4.1, 2.5, bg=C_CARD, border=color, border_w=Pt(1.5))
    txt(sl, icon,  lx+0.1,  ty+0.10, 0.55, 0.55, size=22)
    txt(sl, title, lx+0.75, ty+0.12, 3.2,  0.45, size=13, bold=True, color=color)
    bullet_box(sl, bullets, lx+0.1, ty+0.65, 3.85, 1.7,
               size=11, color=C_LIGHT, dot=color, spacing=0.56)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 12 — Conclusion / Thank You
# ─────────────────────────────────────────────────────────────────────────────
sl = prs.slides.add_slide(BLANK)
add_bg(sl)

# Accent bar
box(sl, 0, 0, 13.33, 0.12, bg=C_ACCENT1)
box(sl, 0, 7.38, 13.33, 0.12, bg=C_ACCENT1)

# Central block
box(sl, 1.8, 0.8, 9.7, 5.9, bg=C_CARD, border=C_ACCENT1, border_w=Pt(1.5))

txt(sl, "🎙️  👁️", 1.8, 1.0, 9.7, 1.0, size=40, align=PP_ALIGN.CENTER)
txt(sl, "VocaVision",
    1.8, 1.95, 9.7, 1.1, size=46, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
txt(sl, "Empowering independence through AI-driven vision and voice",
    1.8, 3.0, 9.7, 0.65, size=17, italic=True, color=C_ACCENT2, align=PP_ALIGN.CENTER)

# Summary chips row
summary = [
    ("20 Classes",  C_ACCENT1),
    ("YOLO v3",     C_ACCENT2),
    ("14 Colors",   C_ACCENT3),
    ("Live Feed",   RGBColor(0x9B,0x59,0xB6)),
    ("Voice Out",   RGBColor(0x1A,0xBC,0x9C)),
]
total_w = sum(1.9 for _ in summary) + 0.2*(len(summary)-1)
start_x = (13.33 - total_w) / 2
for j, (label, color) in enumerate(summary):
    chip(sl, label, start_x + j*2.1, 3.85, w=1.9, bg=color,
         tc=C_BG if color in (C_ACCENT1, C_ACCENT2, C_ACCENT3,
                               RGBColor(0x1A,0xBC,0x9C)) else C_WHITE)

txt(sl, "Thank You",
    1.8, 4.55, 9.7, 0.85, size=34, bold=True, color=C_ACCENT1, align=PP_ALIGN.CENTER)
txt(sl, "Questions & Discussion",
    1.8, 5.35, 9.7, 0.5,  size=16, color=C_MUTED, align=PP_ALIGN.CENTER)

box(sl, 0, 6.85, 13.33, 0.55, bg=RGBColor(0x0A,0x10,0x20))
txt(sl, "github.com/VocaVision  |  python app.py  →  http://localhost:7860",
    0, 6.88, 13.33, 0.45, size=12, color=C_MUTED, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
OUT = "VocaVision_Presentation.pptx"
prs.save(OUT)
print(f"Saved -> {OUT}  ({len(prs.slides)} slides)")
