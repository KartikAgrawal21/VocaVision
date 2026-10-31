"""
Color detection using K-Means clustering on BGR frames.
Maps dominant pixel clusters to human-readable color names via HSV ranges.
No training required — pure computer vision using OpenCV + scikit-learn.
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Each entry: (name, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v)) in HSV space
_COLOR_RANGES = [
    ("red",    (  0,  80,  50), ( 10, 255, 255)),
    ("red",    (165,  80,  50), (180, 255, 255)),  # red wraps in HSV
    ("orange", ( 11,  80,  50), ( 25, 255, 255)),
    ("yellow", ( 26,  80,  50), ( 35, 255, 255)),
    ("green",  ( 36,  50,  50), ( 85, 255, 255)),
    ("blue",   ( 86,  50,  50), (130, 255, 255)),
    ("purple", (131,  50,  50), (155, 255, 255)),
    ("pink",   (156,  40, 150), (164, 255, 255)),
    ("white",  (  0,   0, 200), (180,  30, 255)),
    ("black",  (  0,   0,   0), (180, 255,  40)),
    ("gray",   (  0,   0,  41), (180,  30, 199)),
    ("brown",  ( 10,  60,  20), ( 20, 200, 130)),
]


def _bgr_to_name(bgr_pixel: np.ndarray) -> str:
    """Converts a single BGR pixel array to a human-readable color name."""
    pixel = np.uint8([[bgr_pixel]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Low-saturation pixels are greys: split by brightness. Indoor light tints
    # white shirts slightly, so the old strict white range reported "neutral".
    if s < 45:
        if v >= 150:
            return "white"
        if v <= 60:
            return "black"
        return "gray"

    for name, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v) in _COLOR_RANGES:
        if lo_h <= h <= hi_h and lo_s <= s <= hi_s and lo_v <= v <= hi_v:
            return name
    return "neutral"


def describe_colors(frame: np.ndarray, n_colors: int = 3,
                    fg_mask: np.ndarray = None) -> str:
    """
    Finds the dominant colors using K-Means clustering.

    If fg_mask is provided (a binary uint8 mask where 1 = clothing),
    color detection runs only on those pixels — ignoring background,
    shadows, hangers, and other distractors.

    Only reports a secondary color if it covers at least 25% of the
    analysed pixels.

    Args:
        frame    : OpenCV BGR image (H x W x 3)
        n_colors : Number of K-Means clusters
        fg_mask  : Optional binary mask (H x W), 1 = include, 0 = ignore

    Returns:
        A string such as "red" or "blue and white"
    """
    # Downsample for speed
    small = cv2.resize(frame, (80, 80))

    if fg_mask is not None:
        # Resize mask to match downsampled image and extract only clothing pixels
        small_mask = cv2.resize(fg_mask, (80, 80), interpolation=cv2.INTER_NEAREST)
        pixels = small[small_mask == 1].astype(np.float32)
        if len(pixels) < n_colors * 10:   # too few pixels → fall back to full image
            pixels = small.reshape(-1, 3).astype(np.float32)
    else:
        pixels = small.reshape(-1, 3).astype(np.float32)

    total = len(pixels)

    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels)

    # Sort clusters by size (most dominant first)
    counts = np.bincount(kmeans.labels_)
    order  = np.argsort(-counts)
    sorted_centers = kmeans.cluster_centers_[order]
    sorted_counts  = counts[order]

    # Only include a color if its cluster covers ≥25% of the image
    # → filters out hangers, backgrounds, and other small distractors
    MIN_COVERAGE = 0.25
    significant  = [
        (center, count / total)
        for center, count in zip(sorted_centers, sorted_counts)
        if count / total >= MIN_COVERAGE
    ]

    # Always include at least the most dominant color
    if not significant:
        significant = [(sorted_centers[0], sorted_counts[0] / total)]

    names = [_bgr_to_name(c.astype(np.uint8)) for c, _ in significant]

    # Deduplicate while preserving dominance order
    seen, unique = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)

    return " and ".join(unique)

