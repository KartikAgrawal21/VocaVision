# VocaVision 

**AI-Based Color, Clothing and Food Recognition Assistive System**

A vision-based assistive system that identifies **clothing colors/patterns** and **food items** and describes them through **voice output** to support independent daily activities.

> **All recognition runs locally using custom-trained neural networks. No external API or internet connection required.**

---

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Food Classifier | MobileNetV2 (Transfer Learning) | Identifies 10 food categories |
| Clothing Classifier | Custom CNN (trained from scratch) | Identifies 10 clothing categories |
| Color Detector | OpenCV + K-Means Clustering | Detects dominant clothing colors |
| Voice Output | pyttsx3 (offline TTS) | Speaks descriptions aloud |
| Camera | OpenCV (webcam) | Captures live video feed |

---

## Project Structure

```
VocaVision/
├── main.py                       # App entry point
├── vision_engine.py              # Local model inference
├── audio_output.py               # Text-to-Speech module
├── camera_handler.py             # Webcam capture
│
├── utils/
│   └── color_detector.py         # K-Means dominant color detection
│
├── training/
│   ├── train_food.py             # Train food classifier (MobileNetV2)
│   ├── train_clothing.py         # Train clothing classifier (CNN)
│   └── evaluate.py              # Evaluate models & plot confusion matrices
│
├── models/                       # Trained .keras model files (generated after training)
├── requirements.txt
└── .env.example
```

---

## Desktop App (recommended)

Double-click **VocaVision** on the desktop (or run `VocaVision.pyw`). No terminal is needed.
It opens a standalone window with photo upload (drag-and-drop or paste), a live camera mode,
on-screen results with confidence scores, and spoken output. Everything runs offline.

- Shortcuts: `U` upload, `C` camera, `Space` capture, `R` repeat voice, `M` mute
- Recognition: YOLOv8 detector, cross-checked by a local CLIP model for single items
- Logs: `%LOCALAPPDATA%\VocaVisionocavision.log`

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models

**Clothing Classifier** (~30 MB download, ~10–20 min training):
```bash
python training/train_clothing.py
```

**Food Classifier** (~4.6 GB download first time, ~30–60 min training):
```bash
python training/train_food.py
```

### 3. Evaluate (optional)
```bash
python training/evaluate.py
```
This generates confusion matrix plots in `models/`.

### 4. Run the application
```bash
python main.py
```

A webcam window will open. Point it at **clothing or food** and:
- Press **`Space`** → AI analyzes and speaks the description
- Press **`Q`** → Quit

---

## Model Details

| Model | Dataset | Architecture | Expected Accuracy |
|-------|---------|-------------|-------------------|
| Food Classifier | Food-101 (10 classes) | MobileNetV2 + fine-tuning | ~75–85% |
| Clothing Classifier | Fashion MNIST (10 classes) | 3-block CNN from scratch | ~90–92% |

### Food Categories (10)
Apple Pie, Fried Rice, Hamburger, Hot Dog, Ice Cream, Pizza, Ramen, Samosa, Sushi, Waffles

### Clothing Categories (10)
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot
