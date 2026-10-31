"""
Zero-shot clothing + food recognition with CLIP (runs locally, no API).

Used when YOLO finds nothing: much better on live webcam frames than the
Fashion-MNIST classifier, which was trained on 28x28 product photos.
Weights (~600 MB) download once from Hugging Face and are then cached.
"""
import cv2
import numpy as np

_MODEL_ID = "openai/clip-vit-base-patch32"

# (spoken name, CLIP prompt, matching name in vision_engine.CLOTHING_CLASSES or None)
_CLOTHING = [
    ("t-shirt",   "a person wearing a t-shirt",                 "T-shirt or top"),
    ("shirt",     "a person wearing a collared button-up shirt", "Shirt"),
    ("pullover",  "a person wearing a sweater or hoodie",       "Pullover"),
    ("coat",      "a person wearing a jacket or coat",          "Coat"),
    ("dress",     "a person wearing a dress",                   "Dress"),
    ("trousers",  "a person wearing trousers or jeans",         "Trouser"),
    ("sneaker",   "a sneaker shoe",                             "Sneaker"),
    ("sandal",    "a sandal",                                   "Sandal"),
    ("boot",      "a boot",                                     "Ankle boot"),
    ("bag",       "a handbag or backpack",                      "Bag"),
]
_FOOD_NAMES = ["apple pie", "fried rice", "hamburger", "hot dog", "ice cream",
               "pizza", "ramen", "samosa", "sushi", "waffles"]

_LABELS = (
    [(n, f"a photo of {p}", "clothing", c) for n, p, c in _CLOTHING]
    + [(n, f"a photo of {n}, a plate of food", "food", None) for n in _FOOD_NAMES]
    + [("nothing", "a photo of a room with no clothing or food", "none", None)]
)

_model = _processor = None
_text_feats = None
_device = "cpu"


def is_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _load():
    global _model, _processor, _text_feats, _device
    if _model is not None:
        return
    import torch
    from transformers import CLIPModel, CLIPProcessor
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[VocaVision] Loading CLIP (zero-shot clothing/food)...")
    _model = CLIPModel.from_pretrained(_MODEL_ID).eval().to(_device)
    _processor = CLIPProcessor.from_pretrained(_MODEL_ID)
    with torch.no_grad():
        tok = _processor(text=[l[1] for l in _LABELS], return_tensors="pt", padding=True).to(_device)
        f = _model.get_text_features(**tok)
        _text_feats = f / f.norm(dim=-1, keepdim=True)


def classify(frame_bgr: np.ndarray) -> dict:
    """
    Returns {"name", "kind" ('clothing'|'food'|'none'), "conf",
             "clothing_class", "probs"} for the best label.
    The frame is scored whole and as a centre crop (where a worn item usually
    is); the two are averaged.
    """
    import torch
    _load()
    h, w = frame_bgr.shape[:2]
    centre = frame_bgr[int(h * .2):int(h * .95), int(w * .2):int(w * .8)]
    images = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in (frame_bgr, centre)]
    with torch.no_grad():
        inp = _processor(images=images, return_tensors="pt").to(_device)
        f = _model.get_image_features(**inp)
        f = f / f.norm(dim=-1, keepdim=True)
        logits = 100.0 * f @ _text_feats.T
        probs = logits.softmax(dim=-1).mean(dim=0).cpu().numpy()
    best = int(np.argmax(probs))
    name, _, kind, cls = _LABELS[best]
    return {"name": name, "kind": kind, "conf": float(probs[best]),
            "clothing_class": cls, "probs": probs, "labels": _LABELS}


_COLOURS = ["white", "black", "gray", "red", "blue", "green", "yellow",
            "orange", "pink", "purple", "brown", "beige", "navy blue"]


def classify_colour(frame_bgr: np.ndarray, garment: str) -> str:
    """Colour of the garment via CLIP; ignores face/background unlike K-means."""
    import torch
    _load()
    h, w = frame_bgr.shape[:2]
    # Lower-centre crop: below the chin, so face/skin/background don't skew colour.
    # (Tuned on a person seated facing the webcam; garment name is unused for
    # the prompt because "close-up of <colour> fabric" scored most reliably.)
    crop = frame_bgr[int(h * .6):int(h * .98), int(w * .28):int(w * .72)]
    prompts = [f"a close-up of {c} fabric" for c in _COLOURS]
    with torch.no_grad():
        tok = _processor(text=prompts, return_tensors="pt", padding=True).to(_device)
        t = _model.get_text_features(**tok)
        t = t / t.norm(dim=-1, keepdim=True)
        img = _processor(images=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                         return_tensors="pt").to(_device)
        f = _model.get_image_features(**img)
        f = f / f.norm(dim=-1, keepdim=True)
        probs = (100.0 * f @ t.T).softmax(dim=-1)[0].cpu().numpy()
    return _COLOURS[int(np.argmax(probs))]


def centre_region(frame_bgr: np.ndarray) -> np.ndarray:
    """Torso-ish crop so face/background pixels don't dominate colour detection."""
    h, w = frame_bgr.shape[:2]
    return frame_bgr[int(h * .35):int(h * .95), int(w * .3):int(w * .7)]
