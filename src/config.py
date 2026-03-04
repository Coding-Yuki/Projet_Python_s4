"""
==========================================================================
  Mask Detection Pro G11 — Configuration Module
==========================================================================
  Centralizes all project hyperparameters, directory paths, color schemes,
  and runtime settings used throughout the pipeline.

  Author  : G11 Team
  Project : PFE / Master — Face Mask Detection (Edge AI)
  Stack   : TensorFlow 2.x, MobileNetV2, TFLite, OpenCV, MediaPipe
==========================================================================
"""

import os
from pathlib import Path

# ── Project Root ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Directory Structure ──────────────────────────────────────────────────
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"           # Original Kaggle dataset
CLEAN_DATA_DIR  = DATA_DIR / "clean"         # Cleaned & verified images
MODELS_DIR      = PROJECT_ROOT / "models"
PLOTS_DIR       = PROJECT_ROOT / "plots"
ASSETS_DIR      = PROJECT_ROOT / "assets"
NOTEBOOKS_DIR   = PROJECT_ROOT / "notebooks"
REPORT_DIR      = PROJECT_ROOT / "report"

# ── Dataset Configuration ────────────────────────────────────────────────
IMG_SIZE: int          = 224                  # MobileNetV2 default input
IMG_SIZE_LITE: int     = 160                  # Lite variant for faster inference
BATCH_SIZE: int        = 32
VALIDATION_SPLIT: float = 0.20
RANDOM_SEED: int       = 42
SUPPORTED_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Model Configuration ──────────────────────────────────────────────────
BACKBONE: str          = "MobileNetV2"
DENSE_UNITS: int       = 128
DROPOUT_RATE: float    = 0.4
LEARNING_RATE: float   = 1e-4
ACTIVATION_HIDDEN: str = "swish"              # Swish > ReLU for MobileNetV2
ACTIVATION_OUTPUT: str = "sigmoid"

# ── Training Configuration ───────────────────────────────────────────────
EPOCHS: int                   = 50
EARLY_STOPPING_PATIENCE: int  = 6
REDUCE_LR_PATIENCE: int       = 3
REDUCE_LR_FACTOR: float       = 0.5
REDUCE_LR_MIN: float          = 1e-7
MIN_DELTA: float              = 1e-4

# ── Class Labels ─────────────────────────────────────────────────────────
CLASS_NAMES = {0: "No Mask", 1: "Mask"}
LABEL_MAP   = {"without_mask": 0, "with_mask": 1}

# ── Performance Targets ──────────────────────────────────────────────────
TARGET_ACCURACY: float  = 0.95                # >= 95% validation accuracy
TARGET_FPS: int         = 20                  # >= 20 FPS on standard CPU
TARGET_MODEL_SIZE_MB    = 10                  # < 10 MB for TFLite

# ── Model Save Paths ────────────────────────────────────────────────────
BEST_MODEL_PATH         = MODELS_DIR / "best_mask_detector.keras"
FINAL_MODEL_PATH        = MODELS_DIR / "final_mask_detector.keras"
TFLITE_MODEL_PATH       = MODELS_DIR / "mask_detector.tflite"
TFLITE_QUANT_PATH       = MODELS_DIR / "mask_detector_int8.tflite"
HISTORY_PATH            = MODELS_DIR / "training_history.json"

# ── HUD Color Scheme (BGR for OpenCV) ────────────────────────────────────
class Colors:
    """Color palette for the real-time HUD interface."""
    MASK_GREEN      = (128, 255, 0)           # Neon Green — Mask detected
    NO_MASK_RED     = (0, 40, 255)            # Scarlet Red — No mask
    HUD_CYAN        = (200, 255, 0)           # Cyan accent for HUD elements
    HUD_AMBER       = (0, 200, 255)           # Amber for warnings
    TEXT_WHITE       = (255, 255, 255)         # Primary text
    TEXT_GRAY        = (180, 180, 180)         # Secondary text
    BG_DARK          = (20, 20, 30)            # Overlay background
    BG_PANEL         = (30, 30, 45)            # Panel background
    CONFIDENCE_LOW   = (0, 80, 255)            # Red-ish for low confidence
    CONFIDENCE_HIGH  = (0, 255, 128)           # Green for high confidence

# ── MediaPipe / Face Detection ───────────────────────────────────────────
FACE_DETECTION_CONFIDENCE: float = 0.5
FACE_DETECTION_MODEL: int        = 0          # 0 = short-range, 1 = full-range

# ── TFLite Optimization ─────────────────────────────────────────────────
TFLITE_NUM_THREADS: int = 4

# ── Visualization Settings ───────────────────────────────────────────────
PLOT_STYLE: str       = "seaborn-v0_8-darkgrid"
PLOT_DPI: int         = 150
FIGURE_SIZE_TRAINING  = (14, 5)
FIGURE_SIZE_CM        = (8, 7)

# ── Ensure all directories exist ─────────────────────────────────────────
for _dir in [
    DATA_DIR, RAW_DATA_DIR, CLEAN_DATA_DIR,
    MODELS_DIR, PLOTS_DIR, ASSETS_DIR,
    NOTEBOOKS_DIR, REPORT_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)


def print_config() -> None:
    """Display the current configuration summary."""
    print("=" * 60)
    print("  MASK DETECTION PRO G11 — Configuration")
    print("=" * 60)
    print(f"  Image Size       : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch Size       : {BATCH_SIZE}")
    print(f"  Backbone         : {BACKBONE}")
    print(f"  Dense Units      : {DENSE_UNITS}")
    print(f"  Dropout Rate     : {DROPOUT_RATE}")
    print(f"  Learning Rate    : {LEARNING_RATE}")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Early Stopping   : patience={EARLY_STOPPING_PATIENCE}")
    print(f"  Activation       : {ACTIVATION_HIDDEN} / {ACTIVATION_OUTPUT}")
    print(f"  Target Accuracy  : {TARGET_ACCURACY * 100:.0f}%")
    print(f"  Target FPS       : >= {TARGET_FPS}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
