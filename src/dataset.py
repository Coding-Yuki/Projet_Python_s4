"""
==========================================================================
  Mask Detection Pro G11 — Data Engineering & Augmentation Pipeline
==========================================================================
  Handles dataset loading, image verification, cleaning, augmentation,
  stratified splitting, and high-performance tf.data pipeline creation.

  Key features:
    - Automatic detection and removal of corrupted images
    - Stratified train/validation split
    - Creative augmentation with lighting variation simulation
    - tf.data.Dataset with prefetch for GPU/CPU optimization

  Author  : G11 Team
  Project : PFE / Master — Face Mask Detection
==========================================================================
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split

from src.config import (
    IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED,
    RAW_DATA_DIR, CLEAN_DATA_DIR, SUPPORTED_EXTENSIONS,
    LABEL_MAP, DATA_DIR,
)

# ── Logger Setup ─────────────────────────────────────────────────────────
# NOTE: Do NOT call logging.basicConfig here; main.py configures the root logger.
logger = logging.getLogger("DataPipeline")


# =========================================================================
#  1. IMAGE VERIFICATION & DATASET CLEANING
# =========================================================================

def verify_image(file_path: str) -> bool:
    """
    Verify that an image file is valid and not corrupted.

    Args:
        file_path: Absolute path to the image file.

    Returns:
        True if image is valid, False otherwise.
    """
    try:
        raw = tf.io.read_file(file_path)
        img = tf.image.decode_image(raw, channels=3)
        # Check minimum size (at least 10x10)
        if img.shape[0] < 10 or img.shape[1] < 10:
            return False
        return True
    except Exception:
        return False


def clean_dataset(data_dir: str) -> Dict[str, int]:
    """
    Scan dataset directory and remove corrupted / invalid images.

    Args:
        data_dir: Root directory containing class subdirectories.

    Returns:
        Dictionary with cleaning statistics.
    """
    stats = {"total_scanned": 0, "removed": 0, "valid": 0}
    data_path = Path(data_dir)

    logger.info(f"Scanning dataset at: {data_path}")

    for img_path in data_path.rglob("*"):
        if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            stats["total_scanned"] += 1
            if not verify_image(str(img_path)):
                logger.warning(f"  [CORRUPT] Removing: {img_path.name}")
                img_path.unlink()
                stats["removed"] += 1
            else:
                stats["valid"] += 1

    logger.info(
        f"Cleaning complete: {stats['valid']} valid, "
        f"{stats['removed']} removed out of {stats['total_scanned']} scanned."
    )
    return stats


# =========================================================================
#  2. DATASET LOADING & STRATIFIED SPLIT
# =========================================================================

def load_file_paths_and_labels(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Walk the dataset directory and collect image paths with labels.

    Expected structure:
        data_dir/
            with_mask/       -> label 1
            without_mask/    -> label 0

    Args:
        data_dir: Root directory containing class subdirectories.

    Returns:
        Tuple of (file_paths, labels).
    """
    file_paths: List[str] = []
    labels: List[int] = []
    data_path = Path(data_dir)

    for class_name, label in LABEL_MAP.items():
        class_dir = data_path / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        class_images = [
            str(p) for p in class_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        file_paths.extend(class_images)
        labels.extend([label] * len(class_images))
        logger.info(f"  Class '{class_name}' (label={label}): {len(class_images)} images")

    logger.info(f"Total images loaded: {len(file_paths)}")
    return file_paths, labels


def stratified_split(
    file_paths: List[str],
    labels: List[int],
    val_split: float = VALIDATION_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Perform stratified train/validation split.

    Args:
        file_paths: List of image file paths.
        labels: Corresponding labels.
        val_split: Fraction for validation (default 0.20).
        seed: Random seed for reproducibility.

    Returns:
        train_paths, val_paths, train_labels, val_labels
    """
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels,
        test_size=val_split,
        random_state=seed,
        stratify=labels,
    )

    logger.info(
        f"Split complete: {len(train_paths)} train, {len(val_paths)} validation"
    )

    # Log class balance
    for split_name, split_labels in [("Train", train_labels), ("Val", val_labels)]:
        unique, counts = np.unique(split_labels, return_counts=True)
        balance = dict(zip(unique, counts))
        logger.info(f"  {split_name} balance: {balance}")

    return train_paths, val_paths, train_labels, val_labels


# =========================================================================
#  3. DATA AUGMENTATION
# =========================================================================

def create_augmentation_layer() -> tf.keras.Sequential:
    """
    Build a Keras data augmentation pipeline.

    Includes:
        - Rotation (+/- 15 degrees)
        - Horizontal flip
        - Zoom (up to 10%)
        - Translation (up to 5%)
        - Brightness variation (simulating day/night/neon lighting)
        - Contrast variation (camera quality simulation)

    Returns:
        tf.keras.Sequential augmentation model.
    """
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(
            factor=0.08,                      # ~15 degrees
            fill_mode="nearest",
        ),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
        ),
        tf.keras.layers.RandomBrightness(
            factor=0.2,                       # Simulate lighting variations
        ),
        tf.keras.layers.RandomContrast(
            factor=0.15,                      # Simulate camera quality
        ),
    ], name="data_augmentation")

    logger.info("Data augmentation pipeline created.")
    return augmentation


# =========================================================================
#  4. TF.DATA PIPELINE
# =========================================================================

def _parse_image(file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load and preprocess a single image for MobileNetV2.

    Applies:
        - JPEG/PNG decoding
        - Resize to IMG_SIZE x IMG_SIZE
        - MobileNetV2 preprocessing (scale to [-1, 1])
    """
    raw = tf.io.read_file(file_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    # MobileNetV2 expects [-1, 1] range
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label


def create_tf_dataset(
    file_paths: List[str],
    labels: List[int],
    batch_size: int = BATCH_SIZE,
    augment: bool = False,
    shuffle: bool = True,
    cache: bool = True,
) -> tf.data.Dataset:
    """
    Create an optimized tf.data.Dataset pipeline.

    Args:
        file_paths: List of image file paths.
        labels: Corresponding integer labels.
        batch_size: Batch size for training/evaluation.
        augment: Whether to apply data augmentation.
        shuffle: Whether to shuffle the dataset.
        cache: Whether to cache the dataset in memory.

    Returns:
        Optimized tf.data.Dataset.
    """
    # Create base dataset
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # Shuffle before processing for better randomization
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(file_paths),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    # Map parsing function (parallelized)
    ds = ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache after parsing (before augmentation)
    if cache:
        ds = ds.cache()

    # Batch
    ds = ds.batch(batch_size)

    # Apply augmentation on batched data for efficiency
    if augment:
        augmentation_layer = create_augmentation_layer()
        ds = ds.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Prefetch for pipeline optimization
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


# =========================================================================
#  5. MAIN PIPELINE BUILDER
# =========================================================================

def build_data_pipeline(
    data_dir: str = None,
    clean: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Complete data pipeline: clean -> load -> split -> create tf.data.

    Args:
        data_dir: Path to the dataset root directory.
        clean: Whether to run the cleaning step.

    Returns:
        train_ds, val_ds, train_size, val_size
    """
    if data_dir is None:
        data_dir = str(RAW_DATA_DIR)

    logger.info("=" * 60)
    logger.info("  BUILDING DATA PIPELINE")
    logger.info("=" * 60)

    # Step 1: Clean dataset
    if clean:
        logger.info("[Step 1/4] Cleaning dataset...")
        clean_stats = clean_dataset(data_dir)
    else:
        logger.info("[Step 1/4] Skipping cleaning step.")

    # Step 2: Load file paths and labels
    logger.info("[Step 2/4] Loading file paths and labels...")
    file_paths, labels = load_file_paths_and_labels(data_dir)

    if len(file_paths) == 0:
        raise ValueError(
            f"No images found in {data_dir}. "
            f"Expected subdirectories: {list(LABEL_MAP.keys())}"
        )

    # Step 3: Stratified split
    logger.info("[Step 3/4] Performing stratified split...")
    train_paths, val_paths, train_labels, val_labels = stratified_split(
        file_paths, labels
    )

    # Step 4: Create tf.data pipelines
    logger.info("[Step 4/4] Creating tf.data pipelines...")
    train_ds = create_tf_dataset(
        train_paths, train_labels,
        augment=True, shuffle=True,
    )
    val_ds = create_tf_dataset(
        val_paths, val_labels,
        augment=False, shuffle=False,
    )

    logger.info("Data pipeline ready.")
    logger.info(f"  Train batches : {tf.data.experimental.cardinality(train_ds)}")
    logger.info(f"  Val batches   : {tf.data.experimental.cardinality(val_ds)}")
    logger.info("=" * 60)

    return train_ds, val_ds, len(train_paths), len(val_paths)


# =========================================================================
#  6. DATASET STATISTICS
# =========================================================================

def dataset_summary(data_dir: str = None) -> Dict:
    """
    Generate a summary report of the dataset distribution.

    Returns:
        Dictionary with dataset statistics.
    """
    if data_dir is None:
        data_dir = str(RAW_DATA_DIR)

    file_paths, labels = load_file_paths_and_labels(data_dir)
    labels_np = np.array(labels)

    summary = {
        "total_images": len(file_paths),
        "class_distribution": {},
        "balance_ratio": 0.0,
    }

    unique, counts = np.unique(labels_np, return_counts=True)
    for cls, count in zip(unique, counts):
        summary["class_distribution"][int(cls)] = int(count)

    if len(counts) == 2:
        summary["balance_ratio"] = round(min(counts) / max(counts), 3)

    logger.info(f"Dataset Summary: {json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    # Run as standalone to test the pipeline
    print("Testing data pipeline...")
    summary = dataset_summary()
    print(f"\nDataset summary:\n{json.dumps(summary, indent=2)}")
