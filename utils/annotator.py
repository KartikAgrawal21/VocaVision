"""
Clothing region detection and visual annotation using OpenCV GrabCut.

Pipeline:
  1. GrabCut segments the clothing (foreground) from the background
  2. Largest contour gives us the bounding box of the clothing item
  3. A semi-transparent color tint is overlaid on just the clothing region
  4. A colored bounding box + text label is drawn on the image

No additional ML models needed — pure OpenCV.
"""
import cv2
import numpy as np

# Maps human-readable color names → BGR tuples for drawing
_COLOR_BGR = {
    "red":     (  0,   0, 220),
    "orange":  (  0, 140, 255),
    "yellow":  (  0, 220, 220),
    "green":   (  0, 180,   0),
    "blue":    (220,  80,   0),
    "navy":    (128,  20,   0),
    "purple":  (180,   0, 180),
    "pink":    (180,  80, 200),
    "white":   (240, 240, 240),
    "black":   ( 40,  40,  40),
    "gray":    (150, 150, 150),
    "brown":   ( 30,  80, 130),
    "beige":   (180, 210, 220),
    "neutral": (160, 160, 160),
}

def _get_draw_color(color_name: str) -> tuple:
    """Returns a BGR tuple for the first word of a color name string."""
    first = color_name.split()[0].lower()
    return _COLOR_BGR.get(first, (100, 255, 100))


def segment_and_annotate(
    frame: np.ndarray,
    clothing_class: str,
    color_name: str,
    confidence: float,
    fg_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Detects the main clothing item (optionally using a pre-computed mask),
    then draws a semi-transparent tinted mask, bounding box, and label.

    Args:
        frame          : BGR OpenCV image
        clothing_class : Predicted clothing category
        color_name     : Detected dominant color
        confidence     : Model confidence (0–1)
        fg_mask        : Optional pre-computed GrabCut mask (1=fg, 0=bg).
                         If None, GrabCut is run internally.

    Returns:
        Annotated BGR image
    """
    h, w = frame.shape[:2]

    # ── Step 1: Use provided mask or run GrabCut ──────────────────────────────
    if fg_mask is None:
        mask   = np.zeros((h, w), np.uint8)
        bgd    = np.zeros((1, 65), np.float64)
        fgd    = np.zeros((1, 65), np.float64)
        mx, my = int(w * 0.05), int(h * 0.05)
        rect   = (mx, my, w - 2 * mx, h - 2 * my)
        try:
            cv2.grabCut(frame, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            fg_mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
            ).astype(np.uint8)
        except cv2.error:
            return _draw_fullframe_label(frame, clothing_class, color_name, confidence)

    # ── Step 2: Find the largest foreground contour (the clothing item) ───────
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return _draw_fullframe_label(frame, clothing_class, color_name, confidence)

    largest   = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # ── Step 3: Build annotated image ─────────────────────────────────────────
    annotated  = frame.copy()
    draw_color = _get_draw_color(color_name)

    # Semi-transparent tint over the clothing mask
    tint        = annotated.copy()
    tint_color  = np.array(draw_color, dtype=np.uint8)
    tint[fg_mask == 1] = cv2.addWeighted(
        tint[fg_mask == 1], 0.55,
        np.full_like(tint[fg_mask == 1], tint_color), 0.45,
        0
    )
    cv2.addWeighted(tint, 0.5, annotated, 0.5, 0, annotated)

    # Clothing region bounding box
    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), draw_color, 3)

    # Label background pill
    label      = f"{color_name.upper()}  {clothing_class}  {confidence:.0%}"
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.5, min(0.75, w / 700))
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    lx         = x
    ly         = max(y - 12, th + 8)
    cv2.rectangle(annotated, (lx - 4, ly - th - 6), (lx + tw + 8, ly + baseline), draw_color, -1)
    cv2.putText(annotated, label, (lx + 2, ly - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated


def _draw_fullframe_label(frame, clothing_class, color_name, confidence):
    """Fallback: draws just a label on the full frame when GrabCut fails."""
    annotated  = frame.copy()
    draw_color = _get_draw_color(color_name)
    label      = f"{color_name.upper()}  {clothing_class}  {confidence:.0%}"
    cv2.rectangle(annotated, (8, 8), (frame.shape[1] - 8, frame.shape[0] - 8), draw_color, 4)
    cv2.putText(annotated, label, (16, 40), cv2.FONT_HERSHEY_DUPLEX,
                0.8, draw_color, 2, cv2.LINE_AA)
    return annotated
