# VocaVision — Project Tasks & Progress Tracker

> **Vision-based Assistive System for Food, Clothing & Color Recognition with Offline Speech Output**

---

## 📌 Executive Summary
VocaVision is an AI-powered assistive system built to empower visually impaired individuals in daily activities. It identifies food items, recognizes clothing articles and their dominant colors/patterns, and announces natural-language descriptions through offline Text-to-Speech (TTS).

All neural networks and pipelines run **100% locally** without external API dependencies or active internet connection requirements.

---

## ✅ Completed Milestones (Till Now)

### 1. Neural Network Architecture & Model Zoo
- [x] **Primary Engine — YOLO v3 (`models/vocavision_yolo.pt`)**:
  - Custom-trained **YOLOv8s** fine-tuned on a merged 20-class dataset (10 food + 10 clothing categories).
  - Trained over 60 epochs, achieving **~65.9% mAP50** and **72.4% precision**.
  - Performs object localization and multi-object classification in a single GPU pass (~15 ms latency).
- [x] **Secondary Fallback — PyTorch MobileNetV2 v2 (`models/clothing_classifier_v2.pt`)**:
  - PyTorch transfer learning model for clothing classification with Test-Time Augmentation (TTA), achieving **~88–92% accuracy**.
- [x] **Food Classifier — MobileNetV2 (`models/food_classifier.keras`)**:
  - Fine-tuned on Food-101 (10 common daily food classes: Apple Pie, Fried Rice, Hamburger, Hot Dog, Ice Cream, Pizza, Ramen, Samosa, Sushi, Waffles).
- [x] **Legacy Fallback — TF CNN (`models/clothing_classifier.keras`)**:
  - 3-block convolutional neural network for baseline fallback.
- [x] **Dynamic Priority Loading (`vision_engine.py`)**:
  - Automatic fallback cascade: `YOLO v3` -> `PyTorch v2 + TF Food` -> `TF CNN Legacy`.
  - Automatic hardware acceleration routing (CUDA GPU / CPU).

### 2. Color & Pattern Analysis Engine
- [x] **Ultra-Fast OpenCV C++ K-Means (`utils/color_detector.py`)**:
  - Replaced scikit-learn `KMeans` with native C++ `cv2.kmeans()`, achieving a **2,720x speedup** (dropped color detection latency from **7,069 ms down to 2.59 ms**).
  - Extracts dominant colors directly from detected bounding box crops in real-time.
  - Recognizes **14 distinct color profiles** (Chromatic: Red, Orange, Yellow, Green, Blue, Navy, Purple, Pink; Achromatic/Earth: White, Black, Gray, Brown, Beige, Neutral).
  - Background/studio backdrop filtering (eliminates near-white lighting glare and near-black shadow clusters).
- [x] **Dynamic Bounding Box Color Marking (`vision_engine.py`)**:
  - Bounding boxes and title banners are **dynamically tinted** with the exact detected clothing color (e.g., blue bounding box for blue shirt, red box for red dress, orange for food).
  - High-contrast text switching (dark text on light colors like white/yellow, white text on dark colors).
- [x] **Background Segmentation (`rembg`)**:
  - Integrated u2netp lightweight background removal for precise garment boundary isolation.

### 3. Audio & Speech Synthesis
- [x] **Offline Text-to-Speech (`audio_output.py`)**:
  - Offline SAPI5 / espeak speech output via `pyttsx3`.
  - Threaded non-blocking asynchronous speech worker (`speak_async`), preventing frame drops or terminal lockups during speech.
  - Throttled smart speech during live video: announces new objects only when they change with a speech cooldown, or on-demand via the Spacebar.

### 4. Real-Time Live Video Stream & Mobile Phone Camera Integration
- [x] **High-Speed Real-Time Live Stream (`--live-view`, 30+ FPS)**:
  - Total end-to-end inference latency optimized down to **~9.9 ms** (YOLOv8s GPU ~14 ms + C++ K-Means ~2.6 ms), delivering smooth 30–60 FPS live video.
  - Live window features real-time color-coded bounding boxes, confidence tags, and real-time FPS counter.
  - Window shortcuts: `[Q]` to Exit, `[SPACE]` to Speak current detection immediately, `[S]` to Save snapshot, `[M]` to Toggle voice mute.
- [x] **Mobile Camera / IP Webcam Support (`camera_handler.py`)**:
  - Use your high-resolution mobile phone camera over Wi-Fi instead of a low-quality laptop webcam.
  - Native support for Android **IP Webcam**, **DroidCam**, and **Iriun Webcam**.
  - Intelligent URL parsing (auto-appends `/video` endpoints for `IP:Port`).
- [x] **Terminal & Headless Mode (`main.py`)**:
  - Interactive terminal menu with camera source configuration (`Option 5`).
  - Command-line flags for automation (`--live-view`, `--image`, `--webcam`, `--monitor`, `--camera`, `--save`, `--no-tts`).

### 5. Documentation & Presentation Assets
- [x] Updated repository [`README.md`](README.md) with live video, mobile cam, and terminal documentation.
- [x] Organized presentation and architecture specifications in [`docs/`](docs/) (`VocaVision_Presentation.pptx` & `VocaVision_Tech_Stack.pdf`).

---

## 📱 How to Use Your Mobile Phone as the Camera (IP Webcam)

If your laptop camera has poor resolution or lighting issues, you can stream from your phone camera in 3 easy steps:

### Option A: Using "IP Webcam" (Android — Recommended)
1. Install **IP Webcam** (by Pavel Khlebovich) from Google Play Store on your phone.
2. Connect your phone and laptop to the **same Wi-Fi network** (or connect laptop to your phone's Wi-Fi Hotspot).
3. Open the app, scroll to the bottom, and tap **"Start server"**.
4. Note the IPv4 address shown at the bottom of your phone screen (e.g. `http://192.168.1.15:8080`).
5. Run VocaVision:
   ```bash
   # Live visual video stream with real-time bounding boxes & colors
   python main.py --live-view --camera http://192.168.1.15:8080/video

   # Or in terminal mode
   python main.py --monitor --camera http://192.168.1.15:8080/video
   ```
   *(You can also simply run `python main.py`, pick option `5`, and enter `192.168.1.15:8080`)*.

### Option B: Using DroidCam or Iriun Webcam (USB or Wi-Fi)
* **DroidCam / Iriun**: Install the desktop client and mobile app. They register as a virtual USB webcam (usually device index `1` or `2`).
  ```bash
  python main.py --live-view --camera 1
  ```

---

## 📋 Open Tasks & Roadmap for Teammates

Here is the assigned task backlog for upcoming development iterations:

| Priority | Task ID | Description | Target Area | Assignee / Status |
|:---:|:---:|---|---|:---:|
| 🔴 **P1** | **TASK-01** | **Lighting & Environment Robustness**: Implement automatic CLAHE / white-balancing on webcam frames before passing to the color detector to prevent warm-light color drift. | `utils/color_detector.py` | Open |
| 🔴 **P1** | **TASK-02** | **Audio Queue Throttling in Monitor Mode**: Ensure rapid changes in camera feed do not overlap overlapping speech threads (add a FIFO speech queue or cancellation token). | `audio_output.py` | Open |
| 🟡 **P2** | **TASK-03** | **Model Quantization (ONNX / OpenVINO / INT8)**: Export `vocavision_yolo.pt` to ONNX and INT8 to optimize CPU-only inference for low-power laptops and Raspberry Pi. | `models/`, `training/` | Open |
| 🟡 **P2** | **TASK-04** | **Spatial & Directional Heuristics**: Use bounding box `(x1, y1, x2, y2)` coordinates to announce object locations (e.g., *"On your left, a blue shirt; in the center, a plate of pizza"*). | `vision_engine.py` | Open |
| 🟡 **P2** | **TASK-05** | **Automated Unit & Regression Tests**: Create a `tests/` directory with `pytest` scripts testing `vision_engine.py` against dummy matrices and benchmark images. | `tests/` | Open |
| 🟢 **P3** | **TASK-06** | **Multi-Language Speech & Accents**: Add support for language selection (Spanish, Hindi, French, etc.) and speech speed presets via CLI args. | `audio_output.py`, `main.py` | Open |
| 🟢 **P3** | **TASK-07** | **Raspberry Pi / Edge Device Integration**: Test camera capture loop with CSI / USB cameras on Linux/ARM environments and document setup. | `camera_handler.py` | Open |

---

## 🚀 Quickstart for Teammates

### 1. Setup Environment
```bash
git pull origin main
pip install -r requirements.txt
```

### 2. Launch Interactive Menu
```bash
python main.py
```

### 3. Direct Commands
```bash
# Live video feed with real-time bounding box color tags
python main.py --live-view

# Live video feed using mobile phone camera over Wi-Fi
python main.py --live-view --camera http://192.168.1.15:8080/video

# Terminal headless camera monitoring
python main.py --monitor --interval 2.0

# Analyze a static photo
python main.py --image path/to/sample.jpg
```

---

## 👥 Contribution Guidelines
1. Create a feature branch for your task: `git checkout -b feature/TASK-ID-description`.
2. Keep the codebase modular and compatible with both headless terminal and live video modes.
3. Update this file (`task.md`) when you complete or pick up a task.
4. Verify your changes with `python test_image.py --image <image_path>` before pushing.
