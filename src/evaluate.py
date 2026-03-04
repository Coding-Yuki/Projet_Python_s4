"""
==========================================================================
  Mask Detection Pro G11 — Evaluation & Visualization
==========================================================================
  Comprehensive model evaluation with publication-quality visualizations.

  Features:
    - Training curves (Loss & Accuracy) with Seaborn dark grid style
    - Confusion matrix heatmap with annotations
    - Precision, Recall, F1-Score computation
    - Classification report with detailed analysis
    - Overfitting analysis and commentary

  All plots are saved as high-DPI PNGs suitable for academic reports.

  Author  : G11 Team
  Project : PFE / Master — Face Mask Detection
==========================================================================
"""

import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.config import (
    BEST_MODEL_PATH, FINAL_MODEL_PATH, HISTORY_PATH,
    PLOTS_DIR, PLOT_DPI, PLOT_STYLE,
    FIGURE_SIZE_TRAINING, FIGURE_SIZE_CM,
    CLASS_NAMES, RAW_DATA_DIR, IMG_SIZE, BATCH_SIZE,
)
from src.dataset import build_data_pipeline

# ── Logger ───────────────────────────────────────────────────────────────
logger = logging.getLogger("Evaluator")


# =========================================================================
#  1. LOAD TRAINING HISTORY
# =========================================================================

def load_history(history_path: str = None) -> Dict:
    """
    Load training history from JSON file.

    Args:
        history_path: Path to history JSON file.

    Returns:
        Dictionary containing training metrics per epoch.
    """
    if history_path is None:
        history_path = str(HISTORY_PATH)

    with open(history_path, "r") as f:
        history = json.load(f)

    logger.info(f"Loaded training history: {len(history.get('loss', []))} epochs")
    return history


# =========================================================================
#  2. TRAINING CURVES
# =========================================================================

def plot_training_curves(
    history: Dict,
    save_path: str = None,
) -> None:
    """
    Plot Loss and Accuracy curves with magazine-quality styling.

    Generates a side-by-side plot:
        - Left:  Training Loss vs Validation Loss
        - Right: Training Accuracy vs Validation Accuracy

    Uses Seaborn darkgrid style for a modern, publication-ready look.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("dark_background")

    if save_path is None:
        save_path = str(PLOTS_DIR / "training_curves.png")

    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_TRAINING)
    fig.patch.set_facecolor("#1a1a2e")

    # ── Color Palette ───────────────────────────────────────────────────
    train_color = "#00ff88"       # Neon green
    val_color   = "#ff4466"       # Coral red
    grid_color  = "#2a2a4a"

    for ax in [ax1, ax2]:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaaaaa")
        ax.spines["bottom"].set_color(grid_color)
        ax.spines["left"].set_color(grid_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color=grid_color)

    # ── Loss Plot ───────────────────────────────────────────────────────
    ax1.plot(epochs, history["loss"], color=train_color, linewidth=2,
             label="Train Loss", marker="o", markersize=3, alpha=0.9)
    ax1.plot(epochs, history["val_loss"], color=val_color, linewidth=2,
             label="Val Loss", marker="s", markersize=3, alpha=0.9)
    ax1.fill_between(epochs, history["loss"], history["val_loss"],
                     alpha=0.08, color=val_color)
    ax1.set_title("Loss", fontsize=16, fontweight="bold", color="#ffffff", pad=15)
    ax1.set_xlabel("Epoch", fontsize=12, color="#aaaaaa")
    ax1.set_ylabel("Loss", fontsize=12, color="#aaaaaa")
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.3)

    # ── Accuracy Plot ───────────────────────────────────────────────────
    ax2.plot(epochs, history["accuracy"], color=train_color, linewidth=2,
             label="Train Accuracy", marker="o", markersize=3, alpha=0.9)
    ax2.plot(epochs, history["val_accuracy"], color=val_color, linewidth=2,
             label="Val Accuracy", marker="s", markersize=3, alpha=0.9)
    ax2.fill_between(epochs, history["accuracy"], history["val_accuracy"],
                     alpha=0.08, color=val_color)
    ax2.set_title("Accuracy", fontsize=16, fontweight="bold", color="#ffffff", pad=15)
    ax2.set_xlabel("Epoch", fontsize=12, color="#aaaaaa")
    ax2.set_ylabel("Accuracy", fontsize=12, color="#aaaaaa")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.3)

    fig.suptitle(
        "MASK DETECTION PRO G11 — Training Metrics",
        fontsize=18, fontweight="bold", color="#00ffcc", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    logger.info(f"Training curves saved to: {save_path}")


# =========================================================================
#  3. CONFUSION MATRIX
# =========================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
) -> np.ndarray:
    """
    Plot a stylized confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels (binary).
        save_path: Path to save the plot.

    Returns:
        Confusion matrix array.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    if save_path is None:
        save_path = str(PLOTS_DIR / "confusion_matrix.png")

    cm = confusion_matrix(y_true, y_pred)
    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_CM)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Custom colormap (dark blue -> neon cyan)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ["#1a1a2e", "#003344", "#006677", "#00ccaa", "#00ff88"]
    cmap = LinearSegmentedColormap.from_list("neon", colors_list)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "#1a1a2e" if cm[i, j] > thresh else "#ffffff"
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    fontsize=22, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=13, color="#aaaaaa")
    ax.set_yticklabels(labels, fontsize=13, color="#aaaaaa")
    ax.set_xlabel("Predicted Label", fontsize=14, color="#aaaaaa", labelpad=10)
    ax.set_ylabel("True Label", fontsize=14, color="#aaaaaa", labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold",
                 color="#00ffcc", pad=15)
    ax.tick_params(colors="#aaaaaa")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    logger.info(f"Confusion matrix saved to: {save_path}")
    return cm


# =========================================================================
#  4. CLASSIFICATION METRICS
# =========================================================================

def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """
    Compute Precision, Recall, F1-Score and generate analysis.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with classification metrics and analysis commentary.
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        accuracy_score, classification_report,
    )

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    accuracy  = accuracy_score(y_true, y_pred)

    report_str = classification_report(
        y_true, y_pred,
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        digits=4,
    )

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "classification_report": report_str,
    }

    # Print formatted report
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_str)
    print("-" * 60)
    print(f"  Overall Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (Mask)  : {precision:.4f}")
    print(f"  Recall (Mask)     : {recall:.4f}")
    print(f"  F1-Score (Mask)   : {f1:.4f}")
    print("=" * 60)

    return metrics


# =========================================================================
#  5. OVERFITTING ANALYSIS
# =========================================================================

def analyze_overfitting(history: Dict) -> str:
    """
    Perform an analytical assessment of overfitting indicators.

    Analyzes:
        - Train/Val accuracy gap
        - Train/Val loss divergence
        - Convergence behavior

    Returns:
        Detailed commentary string suitable for academic reporting.
    """
    train_acc = history["accuracy"]
    val_acc   = history["val_accuracy"]
    train_loss = history["loss"]
    val_loss   = history["val_loss"]

    # Final epoch metrics
    final_train_acc = train_acc[-1]
    final_val_acc   = val_acc[-1]
    final_train_loss = train_loss[-1]
    final_val_loss   = val_loss[-1]

    # Gaps
    acc_gap  = final_train_acc - final_val_acc
    loss_gap = final_val_loss - final_train_loss

    # Best metrics
    best_val_acc  = max(val_acc)
    best_val_loss = min(val_loss)
    best_epoch_acc  = val_acc.index(best_val_acc) + 1
    best_epoch_loss = val_loss.index(best_val_loss) + 1

    # Build analysis
    analysis = []
    analysis.append("=" * 60)
    analysis.append("  OVERFITTING ANALYSIS & TRAINING COMMENTARY")
    analysis.append("=" * 60)
    analysis.append("")
    analysis.append(f"  Total Epochs Trained     : {len(train_acc)}")
    analysis.append(f"  Best Val Accuracy        : {best_val_acc:.4f} (Epoch {best_epoch_acc})")
    analysis.append(f"  Best Val Loss            : {best_val_loss:.4f} (Epoch {best_epoch_loss})")
    analysis.append(f"  Final Train Accuracy     : {final_train_acc:.4f}")
    analysis.append(f"  Final Val Accuracy       : {final_val_acc:.4f}")
    analysis.append(f"  Accuracy Gap (Train-Val) : {acc_gap:.4f}")
    analysis.append(f"  Loss Gap (Val-Train)     : {loss_gap:.4f}")
    analysis.append("")

    # Overfitting assessment
    if acc_gap < 0.02:
        analysis.append("  [EXCELLENT] Minimal overfitting detected.")
        analysis.append("  The model generalizes well — train/val metrics are closely aligned.")
    elif acc_gap < 0.05:
        analysis.append("  [GOOD] Slight overfitting observed but within acceptable range.")
        analysis.append("  The dropout and early stopping are effectively regularizing the model.")
    elif acc_gap < 0.10:
        analysis.append("  [WARNING] Moderate overfitting detected.")
        analysis.append("  Consider: increasing dropout, adding L2 regularization,")
        analysis.append("  or using more aggressive data augmentation.")
    else:
        analysis.append("  [CRITICAL] Significant overfitting detected.")
        analysis.append("  Recommendations:")
        analysis.append("    - Increase dropout rate (current: 0.4 -> try 0.5-0.6)")
        analysis.append("    - Apply stronger data augmentation")
        analysis.append("    - Add L2 weight regularization")
        analysis.append("    - Consider reducing model capacity")
        analysis.append("    - Ensure dataset is large and balanced enough")

    analysis.append("")

    # Convergence assessment
    if best_val_acc >= 0.95:
        analysis.append("  [TARGET MET] Validation accuracy >= 95% achieved.")
    else:
        analysis.append(f"  [TARGET MISSED] Val accuracy {best_val_acc:.2%} < 95% target.")
        analysis.append("  Consider: fine-tuning more backbone layers or expanding the dataset.")

    analysis.append("")
    analysis.append("=" * 60)

    commentary = "\n".join(analysis)
    print(commentary)

    # Save analysis to file
    analysis_path = PLOTS_DIR / "overfitting_analysis.txt"
    with open(str(analysis_path), "w") as f:
        f.write(commentary)

    return commentary


# =========================================================================
#  6. FULL EVALUATION PIPELINE
# =========================================================================

def evaluate(
    model_path: str = None,
    data_dir: str = None,
    history_path: str = None,
) -> Dict:
    """
    Run the complete evaluation pipeline.

    Steps:
        1. Load model and training history
        2. Build validation dataset
        3. Generate predictions
        4. Plot training curves
        5. Plot confusion matrix
        6. Compute classification metrics
        7. Analyze overfitting

    Returns:
        Dictionary with all metrics and analysis.
    """
    if model_path is None:
        # Try best model first, fallback to final
        if BEST_MODEL_PATH.exists():
            model_path = str(BEST_MODEL_PATH)
        else:
            model_path = str(FINAL_MODEL_PATH)

    if data_dir is None:
        data_dir = str(RAW_DATA_DIR)

    logger.info("=" * 60)
    logger.info("  MASK DETECTION PRO G11 — EVALUATION")
    logger.info("=" * 60)

    # ── Step 1: Load model ──────────────────────────────────────────────
    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # ── Step 2: Build validation data ───────────────────────────────────
    logger.info("Building validation dataset...")
    _, val_ds, _, val_size = build_data_pipeline(data_dir=data_dir, clean=False)

    # ── Step 3: Generate predictions ────────────────────────────────────
    logger.info("Generating predictions on validation set...")
    y_true = []
    y_prob = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_prob.extend(preds.flatten())
        y_true.extend(labels.numpy().flatten())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    # ── Step 4: Training curves ─────────────────────────────────────────
    if history_path is None and HISTORY_PATH.exists():
        history = load_history()
        plot_training_curves(history)
        analysis = analyze_overfitting(history)
    else:
        history = None
        analysis = "No training history available."

    # ── Step 5: Confusion matrix ────────────────────────────────────────
    cm = plot_confusion_matrix(y_true, y_pred)

    # ── Step 6: Classification metrics ──────────────────────────────────
    metrics = compute_classification_report(y_true, y_pred)

    # ── Step 7: Model evaluation on val_ds ──────────────────────────────
    logger.info("Running model.evaluate on validation set...")
    eval_results = model.evaluate(val_ds, verbose=1)

    results = {
        "model_path": model_path,
        "val_size": val_size,
        "eval_loss": float(eval_results[0]),
        "eval_accuracy": float(eval_results[1]),
        **metrics,
        "confusion_matrix": cm.tolist(),
    }

    # Save results
    results_path = PLOTS_DIR / "evaluation_results.json"
    with open(str(results_path), "w") as f:
        json.dump({k: v for k, v in results.items()
                    if k != "classification_report"}, f, indent=2)

    logger.info(f"\nEvaluation results saved to: {results_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate mask detection model")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
    args = parser.parse_args()

    results = evaluate(model_path=args.model, data_dir=args.data_dir)
