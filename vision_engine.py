"""
VocaVision — Vision Engine v3 (YOLO-Powered)
=============================================
Priority:
  1. YOLO v3 (best) — Custom YOLOv8s trained on food + clothing   → models/vocavision_yolo.pt
  2. PyTorch v2 (fallback) — MobileNetV2 GPU clothing + TF food   → models/clothing_classifier_v2.pt
  3. TF v1 (legacy fallback) — original CNN                       → models/clothing_classifier.keras

YOLO mode:
  - Single model detects ALL objects in one forward pass (~15ms GPU)
  - Bounding box crops used for precise color analysis
  - Handles multiple objects in one image simultaneously
  - Class-specific confidence thresholds reduce bag/sneaker false positives

Fallback (non-YOLO) mode:
  - Hybrid PyTorch (clothing) + TF (food) classification
  - GrabCut segmentation for color detection
"""
import os
import cv2
import numpy as np
from utils.color_detector import describe_colors

# ── Class Labels (must match training order) ──────────────────────────────────
FOOD_CLASSES = sorted([
    "apple_pie", "fried_rice", "hamburger", "hot_dog", "ice_cream",
    "pizza",     "ramen",      "samosa",    "sushi",   "waffles"
])
CLOTHING_CLASSES = [
    "T-shirt or top", "Trouser", "Pullover", "Dress",      "Coat",
    "Sandal",         "Shirt",   "Sneaker",  "Bag",        "Ankle boot"
]

# YOLO class names (must match training/yolo_classes.py ALL_CLASSES)
YOLO_CLASSES = [
    "apple_pie", "fried_rice", "hamburger", "hot_dog",    "ice_cream",
    "pizza",     "ramen",      "samosa",    "sushi",      "waffles",
    "t_shirt",   "trouser",    "pullover",  "dress",      "coat",
    "sandal",    "shirt",      "sneaker",   "bag",        "ankle_boot"
]
YOLO_FOOD_IDS     = set(range(0,  10))
YOLO_CLOTHING_IDS = set(range(10, 20))

# ── Class-specific confidence thresholds (YOLO) ───────────────────────────────
# Bag and sneaker are the most common false positives on clothing images.
# Raising their individual thresholds significantly cuts wrong detections
# without hurting detection of other classes.
_YOLO_CLASS_THRESHOLDS: dict[int, float] = {
    18: 0.75,   # bag        — very frequent FP on t-shirts / folded garments
    17: 0.70,   # sneaker    — confused with sandal and ankle-boot crops
    15: 0.65,   # sandal     — occasionally confused with ankle-boot
    19: 0.65,   # ankle_boot — sometimes triggered by bag straps
}
_YOLO_DEFAULT_THRESHOLD = 0.52   # base threshold for all other classes

# ── Natural-language display names for YOLO clothing classes ──────────────────
_YOLO_DISPLAY_NAMES: dict[str, str] = {
    "t_shirt":    "T-shirt",
    "trouser":    "trousers",
    "pullover":   "pullover",
    "dress":      "dress",
    "coat":       "coat",
    "sandal":     "sandals",
    "shirt":      "shirt",
    "sneaker":    "sneakers",
    "bag":        "bag",
    "ankle_boot": "ankle boots",
}

_COLOR_BOX_BGR: dict[str, tuple[int, int, int]] = {
    "red":     (  0,   0, 220),
    "orange":  (  0, 140, 255),
    "yellow":  (  0, 220, 220),
    "green":   (  0, 180,   0),
    "blue":    (220,  80,   0),
    "navy":    (128,  20,   0),
    "purple":  (180,   0, 180),
    "pink":    (180,  80, 200),
    "white":   (240, 240, 240),
    "black":   ( 50,  50,  50),
    "gray":    (150, 150, 150),
    "brown":   ( 30,  80, 130),
    "beige":   (180, 210, 220),
    "neutral": (160, 160, 160),
}

# ── Model Paths ───────────────────────────────────────────────────────────────
_DIR          = os.path.dirname(__file__)
_MODELS_DIR   = os.path.join(_DIR, "models")
_YOLO_PATH    = os.path.join(_MODELS_DIR, "vocavision_yolo.pt")   # v3 — YOLO
_FOOD_TF_PATH = os.path.join(_MODELS_DIR, "food_classifier.keras")
_CLTH_PT_PATH = os.path.join(_MODELS_DIR, "clothing_classifier_v2.pt")
_CLTH_TF_PATH = os.path.join(_MODELS_DIR, "clothing_classifier.keras")

# ── Optional: rembg background removal ───────────────────────────────────────
try:
    from rembg import remove as _rembg_remove, new_session as _rembg_new_session
    from PIL import Image as _PIL_Image
    _REMBG_AVAILABLE = True
except Exception:
    _REMBG_AVAILABLE = False

_REMBG_SESSION = None

def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None and _REMBG_AVAILABLE:
        try:
            _REMBG_SESSION = _rembg_new_session("u2netp")
        except Exception:
            _REMBG_SESSION = False
    return _REMBG_SESSION if _REMBG_SESSION else None

# ── Optional: PyTorch ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torchvision.models as tvm
    _TORCH_AVAILABLE = True
    _TORCH_DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH_AVAILABLE = False

# ── TensorFlow ────────────────────────────────────────────────────────────────
import tensorflow as tf


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_models() -> tuple:
    """
    Returns (food_model, clothing_model, using_pytorch, yolo_model, using_yolo)
    where yolo_model is an ultralytics YOLO instance if vocavision_yolo.pt exists.
    """
    # ── Try loading YOLO (v3) ─────────────────────────────────────────────────
    yolo_model  = None
    using_yolo  = False

    if os.path.exists(_YOLO_PATH):
        try:
            from ultralytics import YOLO
            print("[VocaVision] Loading YOLO detector (v3)...")
            yolo_model = YOLO(_YOLO_PATH)
            using_yolo = True
            print(f"[VocaVision] YOLO model loaded | Classes: {len(YOLO_CLASSES)} | Device: GPU" )
        except Exception as e:
            print(f"[VocaVision] YOLO load failed: {e} — falling back to v2/v1")

    # ── Food model (TF — always loaded for fallback) ───────────────────────────
    print("[VocaVision] Loading food classifier (TF)...")
    food_model = tf.keras.models.load_model(_FOOD_TF_PATH)

    # ── Clothing model: PyTorch v2 or TF v1 ──────────────────────────────────
    using_pytorch  = False
    clothing_model = None

    if os.path.exists(_CLTH_PT_PATH) and _TORCH_AVAILABLE:
        try:
            print(f"[VocaVision] Loading clothing classifier v2 (PyTorch, device={_TORCH_DEVICE})...")
            m = tvm.mobilenet_v2(weights=None)
            m.classifier[1] = torch.nn.Linear(m.last_channel, len(CLOTHING_CLASSES))
            checkpoint = torch.load(_CLTH_PT_PATH, map_location=_TORCH_DEVICE,
                                     weights_only=False)
            # Support both a raw state-dict and a wrapped checkpoint
            # (keys: model_state, classes, img_size, architecture)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            else:
                state_dict = checkpoint
            m.load_state_dict(state_dict)
            m.eval().to(_TORCH_DEVICE)
            clothing_model = m
            using_pytorch  = True
        except Exception as e:
            print(f"[VocaVision] PyTorch clothing load failed: {e}")

    if clothing_model is None and os.path.exists(_CLTH_TF_PATH):
        print("[VocaVision] Loading clothing classifier v1 (TF fallback)...")
        clothing_model = tf.keras.models.load_model(_CLTH_TF_PATH)

    if clothing_model is None:
        raise FileNotFoundError(
            "No clothing model found. Train one:\n"
            "  python training/train_clothing_v2.py  (fast, GPU)"
        )

    if using_yolo:
        print(f"[VocaVision] Engine: YOLO v3 [ACTIVE] + TF food fallback")
    else:
        mode = "PyTorch v2 (GPU)" if using_pytorch else "TF v1 (CPU)"
        print(f"[VocaVision] Engine: Hybrid | Clothing: {mode} | Food: TF")

    if _REMBG_AVAILABLE:
        print("[VocaVision] Background removal (rembg): ENABLED [OK]")
    else:
        print("[VocaVision] Background removal (rembg): not installed")

    return food_model, clothing_model, using_pytorch, yolo_model, using_yolo


# ── Background Removal ────────────────────────────────────────────────────────

def _remove_background(frame: np.ndarray) -> np.ndarray:
    session = _get_rembg_session()
    if session is None:
        return frame
    try:
        pil_in  = _PIL_Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_out = _rembg_remove(pil_in, session=session)
        bg      = _PIL_Image.new("RGB", pil_out.size, (255, 255, 255))
        bg.paste(pil_out, mask=pil_out.split()[3])
        return cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
    except Exception:
        return frame


# ── Preprocessing (for fallback classifiers) ──────────────────────────────────

def _preprocess_food(frame: np.ndarray) -> np.ndarray:
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    arr     = resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def _preprocess_clothing(frame: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    arr     = resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=(0, -1))


# ── PyTorch TTA Clothing (fallback) ──────────────────────────────────────────

def _clothing_to_fmnist_style(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Convert a real clothing photo to match Fashion-MNIST training distribution.

    Fashion-MNIST: WHITE object on BLACK background (28x28 grayscale).
    Real photo   : Colored object on WHITE background (after rembg).

    Fix: invert grayscale so clothing = bright, background = dark — matching
    training data appearance. Then auto-crop tight to the object.
    Returns a BGR image ready for the standard torchvision transforms.
    """
    gray     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray                               # invert: dark bg, bright clothing

    # Auto-crop to clothing bounding box (remove white padding)
    _, thresh = cv2.threshold(inverted, 15, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 8
        x1 = max(0, x - pad);          y1 = max(0, y - pad)
        x2 = min(inverted.shape[1], x + w + pad)
        y2 = min(inverted.shape[0], y + h + pad)
        inverted = inverted[y1:y2, x1:x2]

    # Contrast stretch so brightest pixel = 255
    if inverted.max() > 0:
        inverted = (inverted.astype(np.float32) / inverted.max() * 255).astype(np.uint8)

    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)  # back to 3-ch for torchvision


_CLOTHING_PT_TF = (T.Compose([
    T.ToPILImage(),
    T.Grayscale(num_output_channels=3),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]) if _TORCH_AVAILABLE else None)

_TTA_TRANSFORMS = [
    T.Compose([T.ToPILImage(), T.Grayscale(3), T.Resize((224,224)),
               T.RandomHorizontalFlip(p=1), T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.ToPILImage(), T.Grayscale(3),
               T.Resize((256,256)), T.CenterCrop(224), T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.ToPILImage(), T.Grayscale(3), T.Resize((224,224)),
               T.ColorJitter(brightness=0.2), T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
] if _TORCH_AVAILABLE else []

def _predict_clothing_pytorch(model, frame: np.ndarray, use_tta: bool = True) -> np.ndarray:
    # Convert to Fashion-MNIST style FIRST (invert + crop) before TTA pipeline
    fmnist_frame = _clothing_to_fmnist_style(frame)
    rgb = cv2.cvtColor(fmnist_frame, cv2.COLOR_BGR2RGB)
    if use_tta and _TTA_TRANSFORMS:
        preds = []
        for tf_aug in _TTA_TRANSFORMS:
            tensor = tf_aug(rgb).unsqueeze(0).to(_TORCH_DEVICE)
            with torch.no_grad():
                preds.append(F.softmax(model(tensor), dim=1).cpu().numpy()[0])
        return np.mean(preds, axis=0)
    else:
        tensor = _CLOTHING_PT_TF(rgb).unsqueeze(0).to(_TORCH_DEVICE)
        with torch.no_grad():
            return F.softmax(model(tensor), dim=1).cpu().numpy()[0]


# ── GrabCut ──────────────────────────────────────────────────────────────────

def _grabcut_mask(frame: np.ndarray) -> np.ndarray:
    h, w   = frame.shape[:2]
    mask   = np.zeros((h, w), np.uint8)
    bgd    = np.zeros((1, 65), np.float64)
    fgd    = np.zeros((1, 65), np.float64)
    mx, my = int(w * 0.05), int(h * 0.05)
    rect   = (mx, my, w - 2*mx, h - 2*my)
    try:
        cv2.grabCut(frame, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except cv2.error:
        return np.ones((h, w), np.uint8)


# ── YOLO Inference ────────────────────────────────────────────────────────────

def _passes_class_threshold(class_id: int, conf: float) -> bool:
    """
    Returns True only when the detection confidence clears the per-class bar.

    Bag (id=18) and sneaker (id=17) have elevated thresholds because they are
    the most common YOLO false-positives on plain garment images.
    """
    threshold = _YOLO_CLASS_THRESHOLDS.get(class_id, _YOLO_DEFAULT_THRESHOLD)
    return conf >= threshold


def _build_description(food_items: list, cloth_items: list) -> str:
    """
    Generates a natural-language sentence from detected items.

    Examples
    --------
    • 1 food  → "I can see a pizza."
    • 2 foods → "I can see a pizza and a hamburger."
    • 1 cloth → "I can see a red T-shirt."
    • mix     → "I can see a pizza. I can also see a blue coat."
    """
    def _oxford(items: list[str]) -> str:
        """Join item names with commas / 'and', Oxford-comma style."""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _counted(items: list) -> list[str]:
        """Collapse repeats: 3x 'Hamburger' -> '3 Hamburgers' (else 'a <label>')."""
        counts: dict[str, int] = {}
        for d in items:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        return [f"a {l}" if n == 1 else f"{n} {l}{'' if l.endswith('s') else 's'}"
                for l, n in counts.items()]

    parts: list[str] = []

    if food_items:
        parts.append(f"I can see {_oxford(_counted(food_items))}.")

    if cloth_items:
        descs = _counted(cloth_items)
        connector = "I can also see" if parts else "I can see"
        parts.append(f"{connector} {_oxford(descs)}.")

    if not parts:
        return "I can see something, but I am not sure what it is."

    return " ".join(parts)


def _analyze_with_yolo(yolo_model, frame: np.ndarray) -> tuple:
    """
    Runs the custom YOLOv8 model on the frame.

    Returns:
        description    : str   — voice-ready text listing all detected objects
        annotated      : np.ndarray — frame with bounding boxes + labels drawn
        detections     : list  — [{label, conf, bbox, is_food, color}]
    """
    # Run inference with the base threshold; per-class filtering happens below
    results  = yolo_model(frame, conf=_YOLO_DEFAULT_THRESHOLD, verbose=False)[0]
    boxes    = results.boxes

    if boxes is None or len(boxes) == 0:
        return "I could not detect any known objects.", frame.copy(), []

    annotated   = frame.copy()
    detections  = []

    for box in boxes:
        class_id = int(box.cls[0])
        conf     = float(box.conf[0])

        # ── Per-class confidence gate ─────────────────────────────────────────
        if not _passes_class_threshold(class_id, conf):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Clamp coords to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2); y2 = min(frame.shape[0] - 1, y2)

        raw_label = YOLO_CLASSES[class_id]
        is_food   = class_id in YOLO_FOOD_IDS
        is_cloth  = class_id in YOLO_CLOTHING_IDS

        # ── Color detection for clothing only ─────────────────────────────────
        color = None
        if is_cloth and (x2 - x1) > 20 and (y2 - y1) > 20:
            crop  = frame[y1:y2, x1:x2]
            color = describe_colors(crop)

        # ── Human-readable display label ──────────────────────────────────────
        if is_food:
            display_name = raw_label.replace("_", " ").title()
        else:
            display_name = _YOLO_DISPLAY_NAMES.get(raw_label,
                                                    raw_label.replace("_", " ").title())

        display_label = f"{color} {display_name}" if color else display_name

        detections.append({
            "label"   : display_label,
            "raw"     : raw_label,
            "conf"    : conf,
            "bbox"    : (x1, y1, x2, y2),
            "is_food" : is_food,
            "color"   : color,
        })

        # ── Draw bounding box (tinted by detected color) ─────────────────────
        if is_food:
            box_color = (0, 140, 255)  # Food: vibrant orange/gold
        elif color:
            first_c = color.split()[0].lower()
            box_color = _COLOR_BOX_BGR.get(first_c, (50, 220, 80))
        else:
            box_color = (50, 220, 80)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Label background banner above the box
        label_str = f"{display_label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx1, ly1 = x1, max(0, y1 - th - 8)
        cv2.rectangle(annotated, (lx1, ly1), (lx1 + tw + 6, y1), box_color, -1)
        # Use dark text on light color backgrounds (white, yellow, beige)
        txt_color = (20, 20, 20) if color and color.split()[0].lower() in ("white", "yellow", "beige") else (255, 255, 255)
        cv2.putText(annotated, label_str, (lx1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 1, cv2.LINE_AA)

    if not detections:
        return "I could not detect any known objects with enough confidence.", frame.copy(), []

    # ── Build voice description ───────────────────────────────────────────────
    food_items  = [d for d in detections if     d["is_food"]]
    cloth_items = [d for d in detections if not d["is_food"]]
    description = _build_description(food_items, cloth_items)

    return description, annotated, detections


# ── Fallback Inference (v1/v2 non-YOLO) ──────────────────────────────────────

def _analyze_fallback(food_model, clothing_model, frame: np.ndarray,
                      using_pytorch: bool) -> tuple:
    """Classic hybrid classify approach (used when YOLO is not yet trained)."""
    from utils.annotator import segment_and_annotate

    clean_frame = _remove_background(frame)

    # ── Food prediction ───────────────────────────────────────────────────────
    food_probs = food_model.predict(_preprocess_food(frame), verbose=0)[0]
    food_conf  = float(np.max(food_probs))
    food_class = FOOD_CLASSES[int(np.argmax(food_probs))]

    # ── Clothing prediction ───────────────────────────────────────────────────
    if using_pytorch and _TORCH_AVAILABLE and hasattr(clothing_model, "parameters"):
        clothing_probs_raw = _predict_clothing_pytorch(clothing_model, clean_frame, use_tta=True)
    else:
        clothing_probs_raw = clothing_model.predict(_preprocess_clothing(clean_frame), verbose=0)[0]

    # Bag-penalty: "Bag" is the most common false positive class.
    # Reduce its effective confidence by 40% so other classes can overtake it.
    clothing_probs = clothing_probs_raw.copy()
    bag_idx = CLOTHING_CLASSES.index("Bag") if "Bag" in CLOTHING_CLASSES else -1
    if bag_idx >= 0:
        clothing_probs[bag_idx] *= 0.60

    clothing_conf  = float(np.max(clothing_probs))
    clothing_class = CLOTHING_CLASSES[int(np.argmax(clothing_probs))]

    # ── Domain decision ───────────────────────────────────────────────────────
    # Raised food threshold to 0.75: food model can give ~60% on clothing images
    # due to color similarity (red t-shirt → waffles/samosa). Need high certainty.
    if food_conf >= 0.75 and clothing_class == "Bag":
        food_wins = True
    elif food_conf >= 0.75:
        food_wins = True
    elif food_conf > clothing_conf + 0.25:   # food must lead by a clear margin
        food_wins = True
    else:
        food_wins = False

    if food_wins:
        label       = food_class.replace("_", " ")
        description = f"I can see a {label}. I am {food_conf:.0%} confident."
        return description, frame.copy(), food_probs, clothing_probs
    else:
        fg_mask     = _grabcut_mask(clean_frame)
        color       = describe_colors(clean_frame, fg_mask=fg_mask)
        description = (
            f"I can see a {color} {clothing_class.lower()}. "
            f"I am {clothing_conf:.0%} confident."
        )
        annotated   = segment_and_annotate(frame, clothing_class, color, clothing_conf,
                                           fg_mask=fg_mask)
        return description, annotated, food_probs, clothing_probs



# ── CLIP zero-shot fallback ───────────────────────────────────────────────────

def _analyze_clip(food_model, clothing_model, frame: np.ndarray,
                  using_pytorch: bool) -> tuple:
    """Zero-shot CLIP recognition; falls back to the old classifiers on any failure."""
    from utils import clip_classifier as clip_cls
    from utils.annotator import _draw_fullframe_label

    try:
        res = clip_cls.classify(frame)
    except Exception as e:
        print(f"[CLIP] unavailable ({e}) -- using Fashion-MNIST classifier")
        return _analyze_fallback(food_model, clothing_model, frame, using_pytorch)

    food_probs     = np.zeros(len(FOOD_CLASSES),     dtype=np.float32)
    clothing_probs = np.zeros(len(CLOTHING_CLASSES), dtype=np.float32)
    for (name, _, kind, cls), p in zip(res["labels"], res["probs"]):
        if kind == "food":
            food_probs[FOOD_CLASSES.index(name.replace(" ", "_"))] = p
        elif kind == "clothing":
            clothing_probs[CLOTHING_CLASSES.index(cls)] = p

    name, kind, conf = res["name"], res["kind"], res["conf"]
    print(f"[CLIP] best guess: {name} ({kind}) {conf:.0%}")

    if kind == "none" or conf < 0.30:
        return "I am not sure what this is. Please move closer and try again.", \
               frame.copy(), food_probs, clothing_probs

    if kind == "food":
        description = f"I can see {name}. I am {conf:.0%} confident."
        annotated = _draw_fullframe_label(frame, name, "food", conf)
    else:
        try:
            color = clip_cls.classify_colour(frame, name)
        except Exception:
            color = describe_colors(clip_cls.centre_region(frame))
        description = f"I can see a {color} {name}. I am {conf:.0%} confident."
        annotated = _draw_fullframe_label(frame, name, color, conf)
    return description, annotated, food_probs, clothing_probs


def _fallback(food_model, clothing_model, frame, using_pytorch):
    try:
        from utils import clip_classifier as clip_cls
        if clip_cls.is_available():
            return _analyze_clip(food_model, clothing_model, frame, using_pytorch)
    except Exception as e:
        print(f"[CLIP] disabled: {e}")
    return _analyze_fallback(food_model, clothing_model, frame, using_pytorch)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def analyze_image(food_model, clothing_model, frame: np.ndarray,
                  using_pytorch: bool = True,
                  yolo_model=None, using_yolo: bool = False,
                  allow_fallback: bool = True) -> tuple:
    """
    Unified inference entry point.

    Args:
        food_model, clothing_model : loaded models
        frame                      : BGR numpy array
        using_pytorch              : whether clothing model is PyTorch
        yolo_model                 : ultralytics YOLO instance (or None)
        using_yolo                 : True when yolo_model is available
        allow_fallback             : False disables heavy GrabCut/rembg on live video

    Returns:
        (description: str, annotated_frame: np.ndarray,
         food_probs: np.ndarray, clothing_probs: np.ndarray)
    """
    if using_yolo and yolo_model is not None:
        description, annotated, detections = _analyze_with_yolo(yolo_model, frame)

        # If YOLO found nothing confident enough
        if not detections:
            if allow_fallback:
                print("[YOLO] No confident detections -- falling back to classifier")
                return _fallback(food_model, clothing_model, frame, using_pytorch)
            else:
                return "I could not detect any known objects.", frame.copy(), np.zeros(len(FOOD_CLASSES), dtype=np.float32), np.zeros(len(CLOTHING_CLASSES), dtype=np.float32)

        if allow_fallback:
            # Cross-check with CLIP, which is much more reliable on single items
            # (YOLO read a sneaker as "bag" at 67%). Photos that clearly contain
            # several different things keep YOLO's multi-object result.
            best = max(detections, key=lambda d: d["conf"])
            distinct_strong = {d["raw"] for d in detections if d["conf"] >= 0.75}
            if len(distinct_strong) < 2:
                try:
                    from utils import clip_classifier as clip_cls
                    if clip_cls.is_available():
                        res = clip_cls.classify(frame)
                        norm = lambda s: "".join(ch for ch in s.lower() if ch.isalpha())
                        a, b = norm(res["name"]), norm(best["raw"])
                        agrees = a in b or b in a
                        clip_wins = res["kind"] != "none" and not agrees and (
                            res["conf"] >= 0.85 or (best["conf"] < 0.60 and res["conf"] >= best["conf"]))
                        if clip_wins:
                            print(f"[CLIP] overrides YOLO ({best['raw']} {best['conf']:.0%}) -> {res['name']} {res['conf']:.0%}")
                            return _analyze_clip(food_model, clothing_model, frame, using_pytorch)
                except Exception as e:
                    print(f"[CLIP] cross-check skipped: {e}")

        # Build dummy prob arrays for the confidence chart in app.py
        food_probs     = np.zeros(len(FOOD_CLASSES),     dtype=np.float32)
        clothing_probs = np.zeros(len(CLOTHING_CLASSES), dtype=np.float32)
        for det in detections:
            if det["is_food"]:
                idx = FOOD_CLASSES.index(det["raw"]) if det["raw"] in FOOD_CLASSES else -1
                if idx >= 0:
                    food_probs[idx] = max(food_probs[idx], det["conf"])
            else:
                # Map YOLO raw name back to CLOTHING_CLASSES display name for chart
                yolo_to_cloth = {
                    "t_shirt":    "T-shirt or top",
                    "trouser":    "Trouser",
                    "pullover":   "Pullover",
                    "dress":      "Dress",
                    "coat":       "Coat",
                    "sandal":     "Sandal",
                    "shirt":      "Shirt",
                    "sneaker":    "Sneaker",
                    "bag":        "Bag",
                    "ankle_boot": "Ankle boot",
                }
                chart_name = yolo_to_cloth.get(det["raw"])
                if chart_name and chart_name in CLOTHING_CLASSES:
                    idx = CLOTHING_CLASSES.index(chart_name)
                    clothing_probs[idx] = max(clothing_probs[idx], det["conf"])

        return description, annotated, food_probs, clothing_probs

    else:
        # Fallback to v2/v1 classifier pipeline
        return _fallback(food_model, clothing_model, frame, using_pytorch)
