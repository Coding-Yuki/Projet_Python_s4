# Mask Detection Pro G11

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

A real-time face mask detection system using **MobileNetV2 + TensorFlow Lite** and **OpenCV**. Features a premium Sci-Fi HUD with live diagnostics, pulsing detection overlays, confidence meters, and 30+ FPS performance on a standard CPU.

> **Model accuracy**: 96.33% — exceeds the 95% target.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.9+** installed
- A webcam connected to your computer

### 2. Setup Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (Linux / macOS)
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Run the Detector

```bash
python main.py run
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Q** | Quit application |
| **SPACE** | Pause / Resume detection |
| **S** | Save screenshot to `screenshots/` |

---

## 📁 Project Structure

```
Mask_Detection_Pro_G11/
├── main.py                  # CLI entry point (train / evaluate / export / run)
├── requirements.txt
│
├── app/
│   ├── realtime_app.py      # Live webcam detection loop
│   └── hud_utils.py         # Sci-Fi HUD rendering
│
├── src/
│   ├── config.py            # All hyperparameters, paths, and color palette
│   ├── model.py             # MobileNetV2 architecture builder
│   ├── dataset.py           # Data loading, cleaning, augmentation, tf.data pipeline
│   ├── train.py             # Training pipeline (2-phase: frozen → fine-tune)
│   ├── evaluate.py          # Metrics, confusion matrix, training curves
│   └── export.py            # TFLite conversion & benchmarking
│
├── models/
│   ├── best_mask_detector.keras   # Best Keras model (saved by ModelCheckpoint)
│   └── mask_detector_int8.tflite  # Quantized model used for real-time inference
│
├── data/
│   └── raw/
│       ├── with_mask/       # Training images — faces with masks
│       └── without_mask/    # Training images — faces without masks
│
├── plots/                   # Auto-generated training curves, confusion matrix
├── screenshots/             # Screenshots saved with the S key
└── report/                  # Any exported reports or analysis
```

---

## 📦 Dataset Setup

The training pipeline expects images in this layout:

```
data/raw/
    with_mask/
        image001.jpg
        image002.jpg
        ...
    without_mask/
        image001.jpg
        image002.jpg
        ...
```

A popular source is the [Face Mask Detection dataset on Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset).  
Download and extract into `data/raw/` so the two class folders are present.

---

## 🛠️ CLI Commands

All operations are run through `main.py`:

```bash
# Show current project configuration
python main.py info

# Train the model (50 epochs + fine-tuning by default)
python main.py train
python main.py train --epochs 30 --batch_size 32 --no_fine_tune

# Evaluate a trained model and generate plots
python main.py evaluate
python main.py evaluate --model models/best_mask_detector.keras

# Export the Keras model to TFLite (dynamic quantization by default)
python main.py export
python main.py export --quantize int8

# Launch real-time webcam detection
python main.py run
python main.py run --camera 1      # use a secondary camera

# Run the full pipeline: train → evaluate → export
python main.py all
python main.py all --data_dir ./data/raw --epochs 50
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | **96.33%** |
| Backbone | MobileNetV2 (ImageNet) |
| TFLite Model Size | ~2.6 MB (int8) |
| Target FPS | ≥ 30 FPS (CPU) |
| Input Size | 224 × 224 × 3 |

---

## 🔧 Troubleshooting

**Camera not found**
```
RuntimeError: Could not open camera (id=0)
```
→ Make sure your webcam is plugged in and not open in another application. Try `--camera 1` for a secondary camera.

**`data/raw` not found during training**
→ See the [Dataset Setup](#-dataset-setup) section above. The `data/raw/with_mask/` and `data/raw/without_mask/` folders must exist.

**MediaPipe import error**
→ MediaPipe is optional. The app falls back to Haar Cascade detection automatically. To fix MediaPipe, run:
```bash
pip install --upgrade mediapipe
```

**Low FPS**
→ Close other CPU-heavy applications. The TFLite int8 model is already optimized for CPU. Set `TFLITE_NUM_THREADS` in `src/config.py` to match your core count.

---

## 📄 License

This project is for educational purposes — PFE / Master, G11 Team.
