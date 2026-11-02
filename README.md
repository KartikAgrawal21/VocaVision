# VocaVision 👁️🔊

**AI-Based Real-Time Color, Clothing & Food Recognition Assistive System**

VocaVision is an assistive computer vision system designed to empower visually impaired individuals in their daily lives. It detects clothing items and food, determines the dominant clothing colors and patterns, draws real-time dynamically tinted bounding boxes, and speaks natural-language descriptions aloud via offline Text-to-Speech (TTS).

> **100% Local Inference**: Runs entirely offline using custom-trained neural networks. No external cloud APIs, recurring costs, or internet access required for inference.

---

## ⚡ Architecture & Engine Hierarchy

VocaVision uses an automatic fallback cascade, prioritizing the highest performance engine available:

```
1. YOLO v3 (Primary)  ─── models/vocavision_yolo.pt        (Custom YOLOv8s, 20 classes, ~15 ms GPU)
2. PyTorch v2         ─── models/clothing_classifier_v2.pt (MobileNetV2 GPU + TTA) + TF Food
3. TF v1 (Legacy)     ─── models/clothing_classifier.keras + food_classifier.keras
```

| Component | Technology | Purpose |
|---|---|---|
| **Object Detector** | Custom YOLOv8s (20 classes) | Detects food & clothing with high-speed bounding boxes |
| **Color Detector** | OpenCV + K-Means Clustering | Identifies dominant colors (14 profiles) & tints bounding boxes |
| **Voice Output** | `pyttsx3` (Offline SAPI5 / espeak) | Non-blocking asynchronous audio announcements |
| **Camera Engine** | OpenCV VideoCapture + Stream Parser | Supports Built-in Cams, USB Cams, and Phone IP Webcams |
| **Interface** | CLI & Real-Time Live Video Feed | Headless terminal mode + 30 FPS live visual video stream |

---

## 📦 Project Structure

```
VocaVision/
├── main.py                       # Unified entry point (Interactive menu + CLI flags)
├── test_image.py                 # Quick headless terminal diagnostic tool
├── vision_engine.py              # Unified inference engine (YOLO v3 + Fallback cascade)
├── camera_handler.py             # Headless camera grabber & mobile IP stream parser
├── audio_output.py               # pyttsx3 offline Text-to-Speech worker
│
├── utils/
│   ├── color_detector.py         # K-Means dominant color detection (14 colors)
│   └── annotator.py              # GrabCut segmentation + drawing utilities
│
├── training/
│   ├── train_yolo.py             # Fine-tune YOLOv8s on food + clothing
│   ├── train_clothing_v2.py      # Train clothing classifier v2 (PyTorch GPU)
│   ├── train_food.py             # Train food classifier (MobileNetV2 TF)
│   ├── prepare_yolo_dataset.py   # Dataset builder
│   └── evaluate.py               # Evaluation & confusion matrices
│
├── models/                       # Trained neural network weights
│   ├── vocavision_yolo.pt        # Primary YOLO v3 model (60 epochs, 20 classes)
│   ├── clothing_classifier_v2.pt # PyTorch fallback
│   ├── clothing_classifier.keras # TF legacy fallback
│   └── food_classifier.keras     # TF food classifier
│
├── docs/                         # Presentation & technical stack documentation
├── task.md                       # Teammate progress tracker & task roadmap
├── requirements.txt              # Cleaned production dependencies
└── README.md
```

---

## Desktop App (recommended)

Double-click **VocaVision** on the desktop (or run `VocaVision.pyw`). No terminal is needed.
It opens a standalone window with photo upload (drag-and-drop or paste), a live camera mode,
on-screen results with confidence scores, and spoken output. Everything runs offline.

- Shortcuts: `U` upload, `C` camera, `Space` capture, `R` repeat voice, `M` mute
- Recognition: YOLOv8 detector, cross-checked by a local CLIP model (pre-trained, by OpenAI) for single items. Needs internet once to download about 600 MB of CLIP weights
- Logs: `%LOCALAPPDATA%\VocaVision\vocavision.log`
ocavision.log`

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/RAJ-A58/VocaVision.git
cd VocaVision
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Requires Python 3.10+; PyTorch with CUDA recommended for GPU acceleration)*.

---

## 🎮 How to Run VocaVision

VocaVision can be operated in multiple flexible modes:

### Mode 1: Interactive Terminal Menu (Recommended)
Run without arguments to launch the interactive terminal interface:
```bash
python main.py
```
You will be greeted with an interactive console menu:
```text
Select an action:
  [1] Live Video Stream (Real-Time Bounding Boxes & Color Marking Window)
  [2] Snap & Analyze Camera (Single Shot in Terminal)
  [3] Continuous Headless Monitor (Voice Alerts only, No Window)
  [4] Analyze Image File
  [5] Configure Camera Source (Current: 0)
  [6] Toggle Voice Output (Currently: ENABLED)
  [7] Exit
```

---

### Mode 2: Live Video Stream with Real-Time Bounding Boxes (`--live-view`)
Pops up an interactive 30 FPS video feed window that tracks objects in real-time, marks bounding boxes tinted with the detected clothing color, and speaks descriptions when new objects appear:
```bash
# Using default webcam:
python main.py --live-view

# Using a mobile phone camera:
python main.py --live-view --camera http://192.168.1.15:8080/video
```

#### Controls inside the Live Video Window:
| Key | Action |
|---|---|
| **`[SPACE]`** | **Speak Now** — Immediately announces whatever is currently detected |
| **`[S]`** | **Save Snapshot** — Saves the current annotated frame to disk |
| **`[M]`** | **Toggle Mute** — Turns voice output on or off |
| **`[Q]` / `[ESC]`** | **Quit** — Closes the video stream and returns to the terminal |

---

### Mode 3: Using Your Mobile Phone as the Camera (IP Webcam)
If your laptop webcam is low quality, you can use your smartphone's camera over Wi-Fi in 3 simple steps:

1. **Install IP Webcam**: Download **IP Webcam** (by Pavel Khlebovich) from Google Play Store on your phone.
2. **Connect**: Ensure your phone and PC are connected to the same Wi-Fi network (or laptop connected to phone hotspot).
3. **Start Stream**:
   - Open the app, scroll to the bottom, and tap **"Start server"**.
   - Note the IP address displayed on your phone (e.g., `http://192.168.1.15:8080`).
4. **Launch VocaVision**:
   ```bash
   python main.py --live-view --camera http://192.168.1.15:8080/video
   ```
   *(Alternatively, run `python main.py`, pick option `[5]`, enter `192.168.1.15:8080`, and select option `[1]`)*.

> *Note: Virtual USB camera drivers like **DroidCam** or **Iriun Webcam** are also supported by passing their device index (e.g., `--camera 1`).*

---

### Mode 4: Headless Continuous Monitor Mode (`--monitor`)
Scans camera frames in a background loop without opening any graphical windows. It speaks aloud whenever detected objects change and writes the latest annotated frame to `monitor_latest.jpg`:
```bash
# Monitor default camera every 2 seconds:
python main.py --monitor --interval 2.0

# Monitor mobile camera:
python main.py --monitor --camera http://192.168.1.15:8080/video --interval 2.5
```

---

### Mode 5: Single Image File Analysis (`--image`)
Processes an image file directly from the terminal, prints ASCII confidence bars, speaks the description, and saves the annotated result:
```bash
# Analyze image and speak description:
python main.py --image path/to/sample.jpg

# Analyze image, save custom result, and mute voice:
python main.py --image path/to/sample.jpg --save output.jpg --no-tts
```

---

### Mode 6: Quick Headless Diagnostic Tool (`test_image.py`)
Run quick tests on image files or webcam snapshots:
```bash
# Test an image file:
python test_image.py --image path/to/sample.jpg

# Test webcam snapshot:
python test_image.py --webcam
```

---

## 🏷️ Recognized Categories (20 Classes)

| Category Group | Classes Recognized |
|---|---|
| **Food (10)** | Apple Pie • Fried Rice • Hamburger • Hot Dog • Ice Cream • Pizza • Ramen • Samosa • Sushi • Waffles |
| **Clothing (10)** | T-shirt/top • Trouser • Pullover • Dress • Coat • Sandal • Shirt • Sneaker • Bag • Ankle Boot |

---

## 🎨 Color Detection (14 Color Profiles)

The K-Means clustering algorithm filters studio backdrops and shadows, identifying **14 dominant colors** and dynamically tinting bounding boxes:

| Palette | Colors |
|---|---|
| **Chromatic** | Red, Orange, Yellow, Green, Blue, Navy, Purple, Pink |
| **Achromatic & Earth** | White, Black, Gray, Brown, Beige, Neutral |

---

## 📊 Model Performance

| Model | Architecture | Dataset | Performance Metrics |
|---|---|---|---|
| **Object Detector (v3)** | YOLOv8s fine-tuned | Custom Food + Clothing | **65.9% mAP50**, **72.4% precision** |
| **Clothing Classifier (v2)** | MobileNetV2 (PyTorch + TTA) | Fashion-MNIST | **88–92% accuracy** |
| **Food Classifier** | MobileNetV2 (TensorFlow) | Food-101 | **75–85% accuracy** |
| **Legacy Clothing (v1)** | 3-Block CNN (TensorFlow) | Fashion-MNIST | **88–90% accuracy** |

---

## 🤝 For Teammates & Contributing

Please refer to [`task.md`](task.md) for:
* Full project roadmap and milestone breakdown.
* Open development tasks (P1–P3) and issue assignments.
* Contribution workflow and branching guidelines.
