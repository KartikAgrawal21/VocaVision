"""
VocaVision Desktop App
======================
Double-click launcher (use VocaVision.pyw / the desktop shortcut). Starts the local
recognition server and opens the interface in a standalone app window.

Everything runs offline on this PC: the browser window only talks to 127.0.0.1.
"""
import base64
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

PORT = 8765
URL = f"http://127.0.0.1:{PORT}"
DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ROOT), "VocaVision")
os.makedirs(DATA_DIR, exist_ok=True)


# ── pythonw has no console: send prints/warnings to a log file ───────────────
def _setup_logging():
    log = open(os.path.join(DATA_DIR, "vocavision.log"), "a", encoding="utf-8", buffering=1)
    if sys.stdout is None or os.path.basename(sys.executable).lower().startswith("pythonw"):
        sys.stdout = log
        sys.stderr = log
    return log


# ── Model state (loaded in the background so the window opens instantly) ─────
STATE = {"status": "loading", "message": "Starting up...", "engine": "", "error": ""}
MODELS = {}
_infer_lock = threading.Lock()
_last_ping = {"t": time.time(), "seen": False, "bye": None}


def _load_everything():
    try:
        STATE["message"] = "Loading recognition models..."
        from vision_engine import load_models
        (MODELS["food"], MODELS["clothing"], MODELS["pt"],
         MODELS["yolo"], MODELS["use_yolo"]) = load_models()

        STATE["message"] = "Warming up the colour and object engine..."
        try:
            from utils import clip_classifier as clip_cls
            if clip_cls.is_available():
                import numpy as np
                clip_cls.classify(np.zeros((224, 224, 3), np.uint8))
                MODELS["clip"] = True
        except Exception as e:  # CLIP is optional
            print(f"[VocaVision] CLIP warm-up skipped: {e}")
            MODELS["clip"] = False

        parts = []
        if MODELS["use_yolo"]:
            parts.append("YOLOv8")
        if MODELS.get("clip"):
            parts.append("CLIP")
        parts.append("MobileNet")
        STATE["engine"] = " + ".join(parts)
        STATE["status"] = "ready"
        STATE["message"] = "Ready"
        print("[VocaVision] ready:", STATE["engine"])
    except Exception as e:
        traceback.print_exc()
        STATE["status"] = "error"
        STATE["error"] = f"{type(e).__name__}: {e}"


# ── Image helpers ────────────────────────────────────────────────────────────
MAX_SIDE = 1280


def _decode_image(data_url: str):
    import numpy as np
    from PIL import Image, ImageOps
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    img = ImageOps.exif_transpose(img).convert("RGB")      # phone photos: fix rotation
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    import cv2
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _encode_jpeg(bgr, quality=88) -> str:
    import cv2
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _top_matches(food_probs, clothing_probs, n=5):
    from vision_engine import CLOTHING_CLASSES, FOOD_CLASSES
    rows = [{"label": c.replace("_", " ").title(), "kind": "food", "p": float(p)}
            for c, p in zip(FOOD_CLASSES, food_probs)]
    rows += [{"label": c, "kind": "clothing", "p": float(p)}
             for c, p in zip(CLOTHING_CLASSES, clothing_probs)]
    rows = [r for r in rows if r["p"] > 0.01]
    return sorted(rows, key=lambda r: -r["p"])[:n]


def analyze(data_url: str) -> dict:
    from vision_engine import analyze_image
    frame = _decode_image(data_url)
    t0 = time.time()
    with _infer_lock:
        description, annotated, food_p, cloth_p = analyze_image(
            MODELS["food"], MODELS["clothing"], frame,
            using_pytorch=MODELS["pt"],
            yolo_model=MODELS["yolo"], using_yolo=MODELS["use_yolo"],
        )
    return {
        "description": description,
        "annotated": _encode_jpeg(annotated),
        "top": _top_matches(food_p, cloth_p),
        "ms": int((time.time() - t0) * 1000),
    }


# ── Web server ───────────────────────────────────────────────────────────────
def build_app():
    from flask import Flask, jsonify, request, send_from_directory
    app = Flask(__name__, static_folder=os.path.join(ROOT, "ui"), static_url_path="")

    @app.after_request
    def _no_cache(resp):
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/status")
    def status():
        return jsonify(STATE)

    @app.get("/api/ping")
    def ping():
        _last_ping.update(t=time.time(), seen=True, bye=None)
        return jsonify(ok=True)

    @app.post("/api/bye")
    def bye():
        _last_ping["bye"] = time.time()          # window closed; exit after a short grace
        return ("", 204)

    @app.post("/api/analyze")
    def analyze_route():
        if STATE["status"] != "ready":
            return jsonify(error="The models are still loading. Please wait a moment."), 503
        try:
            payload = request.get_json(force=True, silent=True) or {}
            if not payload.get("image"):
                return jsonify(error="No image received."), 400
            return jsonify(analyze(payload["image"]))
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=f"Could not analyse this image ({type(e).__name__})."), 500

    return app


def _watchdog():
    """Quit when the app window has gone away (no more pings)."""
    while True:
        time.sleep(2)
        now = time.time()
        if _last_ping["bye"] and now - _last_ping["bye"] > 6:
            os._exit(0)
        # Minimised windows throttle timers, so be generous before giving up.
        limit = 240 if _last_ping["seen"] else 180
        if now - _last_ping["t"] > limit:
            os._exit(0)


# ── Launching the app window ─────────────────────────────────────────────────
def _find_browser():
    pf, pf86 = os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", "")
    local = os.environ.get("LOCALAPPDATA", "")
    for p in (
        os.path.join(pf86, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(pf, "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(pf, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(pf86, "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(local, "Google", "Chrome", "Application", "chrome.exe"),
    ):
        if os.path.exists(p):
            return p
    return None


def open_window():
    browser = _find_browser()
    if not browser:
        import webbrowser
        webbrowser.open(URL)
        return
    subprocess.Popen([
        browser,
        f"--app={URL}",
        f"--user-data-dir={os.path.join(DATA_DIR, 'window')}",
        "--window-size=1320,900",
        "--use-fake-ui-for-media-stream",     # camera allowed without a prompt (local app only)
        "--no-first-run", "--no-default-browser-check",
        "--disable-features=Translate,msEdgeSidebarV2",
    ])


def _already_running() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/api/status", timeout=1.5) as r:
            return b"status" in r.read()
    except Exception:
        return False


def _port_free() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


def main():
    _setup_logging()
    if _already_running():            # second double-click: just bring up a window
        open_window()
        return
    if not _port_free():
        print(f"Port {PORT} is busy; cannot start.")
        return

    threading.Thread(target=_load_everything, daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()
    app = build_app()
    threading.Timer(0.8, open_window).start()
    app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
