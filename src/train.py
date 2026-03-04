"""
==========================================================================
  Mask Detection Pro G11 — Training Script
==========================================================================
  Orchestrates the full training pipeline:

    1. Load and prepare data via the dataset module
    2. Build MobileNetV2 model via the model module
    3. Train with callbacks (EarlyStopping, ReduceLR, Checkpoint)
    4. Save best model and training history
    5. Optionally fine-tune top backbone layers

  Usage:
    python -m src.train --data_dir ./data/raw --epochs 50

  Author  : G11 Team
  Project : PFE / Master — Face Mask Detection
==========================================================================
"""

import os
import json
import time
import logging
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.config import (
    EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN,
    MIN_DELTA, BEST_MODEL_PATH, FINAL_MODEL_PATH,
    HISTORY_PATH, RAW_DATA_DIR, MODELS_DIR,
)
from src.dataset import build_data_pipeline
from src.model import build_mobilenetv2_model, unfreeze_top_layers

# ── Logger ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Trainer")


# =========================================================================
#  1. CALLBACK FACTORY
# =========================================================================

def create_callbacks() -> list:
    """
    Create training callbacks for optimal convergence.

    Callbacks:
        - EarlyStopping      : Stop when val_loss stops improving
        - ReduceLROnPlateau  : Reduce LR when val_loss plateaus
        - ModelCheckpoint    : Save best model based on val_accuracy

    Returns:
        List of Keras callbacks.
    """
    callbacks = [
        # Early Stopping — prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce Learning Rate — adaptive optimization
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
            verbose=1,
        ),
        # Model Checkpoint — save best weights
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    logger.info(f"Created {len(callbacks)} training callbacks.")
    return callbacks


# =========================================================================
#  2. TRAINING PIPELINE
# =========================================================================

def train(
    data_dir: str = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    fine_tune: bool = True,
    fine_tune_epochs: int = 15,
) -> dict:
    """
    Execute the complete training pipeline.

    Steps:
        1. Build data pipeline (clean, load, split, augment)
        2. Build MobileNetV2 model (frozen backbone)
        3. Phase 1: Train with frozen backbone
        4. Phase 2: Fine-tune top backbone layers (optional)
        5. Save model and history

    Args:
        data_dir: Path to dataset root directory.
        epochs: Number of training epochs (Phase 1).
        batch_size: Batch size.
        fine_tune: Whether to fine-tune after initial training.
        fine_tune_epochs: Additional epochs for fine-tuning.

    Returns:
        Training history dictionary.
    """
    if data_dir is None:
        data_dir = str(RAW_DATA_DIR)

    logger.info("=" * 60)
    logger.info("  MASK DETECTION PRO G11 — TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Data Pipeline ───────────────────────────────────────────
    logger.info("\n[PHASE 0] Building data pipeline...")
    train_ds, val_ds, train_size, val_size = build_data_pipeline(
        data_dir=data_dir, clean=True,
    )

    # ── Step 2: Build Model ─────────────────────────────────────────────
    logger.info("\n[PHASE 1] Building MobileNetV2 model...")
    model = build_mobilenetv2_model()

    # ── Step 3: Phase 1 Training (Frozen Backbone) ──────────────────────
    logger.info(f"\n[PHASE 2] Training (frozen backbone) for {epochs} epochs...")
    start_time = time.time()

    callbacks = create_callbacks()

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    phase1_time = time.time() - start_time
    logger.info(f"Phase 1 complete in {phase1_time:.1f}s")

    # Merge history
    full_history = {k: list(v) for k, v in history_phase1.history.items()}

    # ── Step 4: Phase 2 Fine-Tuning (Optional) ─────────────────────────
    if fine_tune:
        logger.info(f"\n[PHASE 3] Fine-tuning top layers for {fine_tune_epochs} epochs...")
        model = unfreeze_top_layers(model, num_layers_to_unfreeze=30, learning_rate=1e-5)

        # Reset callbacks for fine-tuning
        ft_callbacks = create_callbacks()

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=ft_callbacks,
            verbose=1,
        )

        # Append Phase 2 history
        for key in full_history:
            if key in history_phase2.history:
                full_history[key].extend(history_phase2.history[key])

    # ── Step 5: Save Final Model & History ──────────────────────────────
    logger.info("\n[PHASE 4] Saving model and training history...")
    model.save(str(FINAL_MODEL_PATH))
    logger.info(f"  Final model saved to: {FINAL_MODEL_PATH}")

    # Convert numpy types for JSON serialization
    serializable_history = {}
    for key, values in full_history.items():
        serializable_history[key] = [float(v) for v in values]

    with open(str(HISTORY_PATH), "w") as f:
        json.dump(serializable_history, f, indent=2)
    logger.info(f"  Training history saved to: {HISTORY_PATH}")

    # ── Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    best_val_acc = max(full_history.get("val_accuracy", [0]))
    best_val_loss = min(full_history.get("val_loss", [float("inf")]))

    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total training time    : {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Total epochs trained   : {len(full_history['loss'])}")
    logger.info(f"  Best val accuracy      : {best_val_acc:.4f}")
    logger.info(f"  Best val loss          : {best_val_loss:.4f}")
    logger.info(f"  Train samples          : {train_size}")
    logger.info(f"  Validation samples     : {val_size}")
    logger.info("=" * 60)

    return full_history


# =========================================================================
#  3. CLI INTERFACE
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask Detection Pro G11 — Training Script"
    )
    parser.add_argument(
        "--data_dir", type=str, default=str(RAW_DATA_DIR),
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--no_fine_tune", action="store_true",
        help="Skip fine-tuning phase.",
    )
    parser.add_argument(
        "--fine_tune_epochs", type=int, default=15,
        help="Number of fine-tuning epochs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set memory growth for GPU (if available)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    history = train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fine_tune=not args.no_fine_tune,
        fine_tune_epochs=args.fine_tune_epochs,
    )
