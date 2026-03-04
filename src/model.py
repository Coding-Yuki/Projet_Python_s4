"""
==========================================================================
  Mask Detection Pro G11 — Model Architecture
==========================================================================
  Builds a lightweight CNN using Transfer Learning with MobileNetV2.

  Architecture:
    - Backbone     : MobileNetV2 (ImageNet weights, frozen convolutions)
    - Pooling      : GlobalAveragePooling2D
    - Dense Head   : 128 units with Swish activation
    - Regularize   : Dropout (0.4)
    - Output       : Dense(1, sigmoid) for binary classification

  The model is designed for Edge AI deployment (CPU / Mobile).

  Author  : G11 Team
  Project : PFE / Master — Face Mask Detection
==========================================================================
"""

import logging
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from typing import Optional, Tuple

from src.config import (
    IMG_SIZE, BACKBONE, DENSE_UNITS, DROPOUT_RATE,
    LEARNING_RATE, ACTIVATION_HIDDEN, ACTIVATION_OUTPUT,
)

# ── Logger Setup ─────────────────────────────────────────────────────────
logger = logging.getLogger("ModelBuilder")


# =========================================================================
#  1. MODEL CONSTRUCTION
# =========================================================================

def build_mobilenetv2_model(
    input_shape: Tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, 3),
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True,
) -> Model:
    """
    Build and compile the MobileNetV2-based face mask detector.

    Architecture:
        Input (224x224x3)
            -> MobileNetV2 Backbone (frozen)
            -> GlobalAveragePooling2D
            -> Dense(128, swish)
            -> Dropout(0.4)
            -> Dense(1, sigmoid)

    Args:
        input_shape: Input image dimensions (H, W, C).
        dense_units: Number of units in the dense layer.
        dropout_rate: Dropout rate for regularization.
        learning_rate: Initial learning rate for Adam optimizer.
        freeze_backbone: Whether to freeze backbone layers.

    Returns:
        Compiled Keras Model.
    """
    logger.info("=" * 60)
    logger.info(f"  Building {BACKBONE} Model")
    logger.info("=" * 60)

    # ── Step 1: Load MobileNetV2 backbone ───────────────────────────────
    logger.info("[1/4] Loading MobileNetV2 backbone (ImageNet weights)...")
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,                     # Remove classification head
        weights="imagenet",                    # Pre-trained on ImageNet
    )

    # ── Step 2: Freeze backbone layers ──────────────────────────────────
    if freeze_backbone:
        backbone.trainable = False
        logger.info("[2/4] Backbone layers FROZEN (transfer learning mode).")
    else:
        backbone.trainable = True
        logger.info("[2/4] Backbone layers UNFROZEN (fine-tuning mode).")

    # ── Step 3: Build custom classification head ────────────────────────
    logger.info("[3/4] Constructing custom classification head...")

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = backbone(inputs, training=False)       # Always in inference mode for BN

    # Global Average Pooling — reduces spatial dimensions
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dense layer with Swish activation (better gradient flow than ReLU)
    x = layers.Dense(
        dense_units,
        activation=ACTIVATION_HIDDEN,
        name="dense_features",
    )(x)

    # Dropout for regularization
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Binary output (sigmoid: 0 = No Mask, 1 = Mask)
    outputs = layers.Dense(
        1,
        activation=ACTIVATION_OUTPUT,
        name="output_prediction",
    )(x)

    # ── Step 4: Assemble and compile ────────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name="MaskDetector_MobileNetV2")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    logger.info("[4/4] Model compiled successfully.")
    _print_model_summary(model, backbone)

    return model


# =========================================================================
#  2. MODEL SUMMARY & DIAGNOSTICS
# =========================================================================

def _print_model_summary(model: Model, backbone: Model) -> None:
    """Print detailed model summary with parameter counts."""
    print("\n" + "=" * 60)
    print("  MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)

    model.summary()

    total_params     = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable    = total_params - trainable_params

    print("\n" + "-" * 60)
    print(f"  Total parameters       : {total_params:>12,}")
    print(f"  Trainable parameters   : {trainable_params:>12,}")
    print(f"  Non-trainable params   : {non_trainable:>12,}")
    print(f"  Backbone layers        : {len(backbone.layers):>12}")
    print(f"  Model size (est.)      : ~{total_params * 4 / 1e6:.1f} MB (FP32)")
    print("-" * 60 + "\n")


def get_model_info(model: Model) -> dict:
    """
    Extract model metadata for reporting.

    Returns:
        Dictionary with model parameters and architecture info.
    """
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )

    return {
        "backbone": BACKBONE,
        "input_shape": f"{IMG_SIZE}x{IMG_SIZE}x3",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "dense_units": DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "activation_hidden": ACTIVATION_HIDDEN,
        "activation_output": ACTIVATION_OUTPUT,
        "learning_rate": LEARNING_RATE,
        "estimated_size_mb": round(total_params * 4 / 1e6, 2),
    }


# =========================================================================
#  3. FINE-TUNING UTILITIES
# =========================================================================

def unfreeze_top_layers(
    model: Model,
    num_layers_to_unfreeze: int = 30,
    learning_rate: float = 1e-5,
) -> Model:
    """
    Unfreeze the top N layers of the backbone for fine-tuning.

    This is typically done after initial training with a frozen backbone
    to further improve accuracy with a lower learning rate.

    Args:
        model: The compiled model.
        num_layers_to_unfreeze: Number of top backbone layers to unfreeze.
        learning_rate: Lower learning rate for fine-tuning.

    Returns:
        Recompiled model with unfrozen layers.
    """
    # Access the backbone (second layer after Input)
    backbone = model.layers[1]
    backbone.trainable = True

    # Freeze all layers except the top N
    for layer in backbone.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    unfrozen = sum(1 for l in backbone.layers if l.trainable)
    logger.info(
        f"Fine-tuning mode: {unfrozen} backbone layers unfrozen, "
        f"lr={learning_rate}"
    )

    return model


# =========================================================================
#  4. CUSTOM CNN BASELINE (FOR COMPARISON)
# =========================================================================

def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, 3),
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """
    Build a simple custom CNN for baseline comparison.

    Architecture:
        Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool ->
        Conv2D(128) -> MaxPool -> Flatten -> Dense(128) -> Output

    Args:
        input_shape: Input image dimensions.
        learning_rate: Learning rate for compilation.

    Returns:
        Compiled baseline CNN model.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="CustomCNN_Baseline")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Custom CNN baseline model built.")
    return model


if __name__ == "__main__":
    # Test model construction
    print("Building MobileNetV2 model...")
    model = build_mobilenetv2_model()
    info = get_model_info(model)
    print(f"\nModel Info: {info}")
