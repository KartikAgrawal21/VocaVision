"""
VocaVision — Camera & IP Webcam Handler
========================================
Supports local webcams, virtual cameras (DroidCam, Iriun), and mobile phone
IP Webcam streams (e.g., http://192.168.X.X:8080/video).
"""

import cv2

def parse_camera_source(source):
    """
    Parses a camera input parameter into either:
      - int: Device index for local or virtual webcams (0, 1, 2, ...)
      - str: Stream URL for phone / IP webcams (http://..., rtsp://...)
    """
    if isinstance(source, int):
        return source

    if isinstance(source, str):
        src_clean = source.strip()
        # Numeric string -> device index
        if src_clean.isdigit():
            return int(src_clean)

        # URL or IP string
        if not (src_clean.startswith("http://") or src_clean.startswith("https://") or src_clean.startswith("rtsp://")):
            src_clean = "http://" + src_clean

        # Auto-append /video endpoint for Android IP Webcam if user just entered IP:Port
        if ":8080" in src_clean and not src_clean.endswith("/video") and not src_clean.endswith("/shot.jpg"):
            src_clean = src_clean.rstrip("/") + "/video"
        elif ":4747" in src_clean and not src_clean.endswith("/video"):
            src_clean = src_clean.rstrip("/") + "/video"

        return src_clean

    return 0

def capture_frame(camera_source=0, warmup_frames=4):
    """
    Captures a single frame from the camera or IP stream.
    Releases the capture object immediately after capture.
    Returns: frame (np.ndarray) or raises RuntimeError.
    """
    src = parse_camera_source(camera_source)
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source '{camera_source}'. "
            "For mobile cameras, verify that your phone and PC are on the same Wi-Fi "
            "and the IP Webcam server is running."
        )

    try:
        # Physical webcams need warmup; IP streams generally stream ready frames
        warmups = 1 if isinstance(src, str) else warmup_frames
        for _ in range(warmups):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to grab a frame from camera source '{camera_source}'.")
        return frame
    finally:
        cap.release()

def show_camera_feed(camera_source=0):
    """
    Generator that yields frames from a local webcam or mobile IP stream.
    """
    src = parse_camera_source(camera_source)
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera stream '{camera_source}'. "
            "Check that your camera or IP stream is active and not used by another application."
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            yield frame
    finally:
        cap.release()
