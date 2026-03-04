"""
==========================================================================
  Mask Detection Pro G11 — TFLite Export & Optimization
==========================================================================
  Converts the trained Keras model to TensorFlow Lite format with
  multiple optimization strategies for Edge AI deployment.

  Optimizations:
    - Dynamic Range Quantization (default, ~4x size reduction)
    - Full Integer Quantization (int8, maximum compression)
    - Float16 Quantization (balanced size/accuracy)

  Generates a comparison report:
    Original Model vs TFLite (size, inference speed).

  Author  : G11 Team
  Project : PFE / Master — Face Mask Detection
==========================================================================
"""

import os
import time
import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.config import (
    BEST_MODEL_PATH, FINAL_MODEL_PATH,
    TFLITE_MODEL_PATH, TFLITE_QUANT_PATH,
    MODELS_DIR, PLOTS_DIR, IMG_SIZE,
    TARGET_MODEL_SIZE_MB,
)

# ── Logger ───────────────────────────────────────────────────────────────
logger = logging.getLogger("TFLiteExporter")


# =========================================================================
#  1. TFLITE CONVERSION
# =========================================================================

def convert_to_tflite(
    model_path: str = None,
    output_path: str = None,
    quantize: str = "dynamic",
    representative_data: Optional[np.ndarray] = None,
) -> str:
    """
    Convert a Keras model to TensorFlow Lite format.

    Quantization options:
        - "none"    : No quantization (FP32)
        - "dynamic" : Dynamic range quantization (~4x compression)
        - "float16" : Float16 quantization (~2x compression)
        - "int8"    : Full integer quantization (max compression)

    Args:
        model_path: Path to the Keras model file.
        output_path: Output path for the TFLite model.
        quantize: Quantization strategy.
        representative_data: Sample data for int8 calibration.

    Returns:
        Path to the saved TFLite model.
    """
    if model_path is None:
        model_path = str(BEST_MODEL_PATH) if BEST_MODEL_PATH.exists() \
                     else str(FINAL_MODEL_PATH)

    if output_path is None:
        if quantize in ("dynamic", "int8"):
            output_path = str(TFLITE_QUANT_PATH)
        else:
            output_path = str(TFLITE_MODEL_PATH)

    logger.info(f"Converting model: {model_path}")
    logger.info(f"Quantization: {quantize}")

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply quantization
    if quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        logger.info("Applied Dynamic Range Quantization.")

    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        logger.info("Applied Float16 Quantization.")

    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for calibration
        if representative_data is None:
            logger.info("Generating synthetic representative dataset for int8 calibration...")
            representative_data = np.random.randn(100, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

        def representative_dataset():
            for i in range(min(100, len(representative_data))):
                yield [representative_data[i:i+1]]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        logger.info("Applied Full Integer (int8) Quantization.")

    elif quantize == "none":
        logger.info("No quantization applied (FP32).")

    else:
        raise ValueError(f"Unknown quantization: {quantize}")

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"TFLite model saved: {output_path} ({file_size_mb:.2f} MB)")

    return output_path


# =========================================================================
#  2. INFERENCE BENCHMARKING
# =========================================================================

def benchmark_inference(
    model_path: str,
    num_runs: int = 100,
    input_shape: Tuple[int, ...] = (1, IMG_SIZE, IMG_SIZE, 3),
) -> Dict:
    """
    Benchmark inference speed of a TFLite model.

    Args:
        model_path: Path to the TFLite model.
        num_runs: Number of inference iterations.
        input_shape: Shape of the input tensor.

    Returns:
        Dictionary with timing statistics.
    """
    # Load interpreter
    interpreter = tf.lite.Interpreter(
        model_path=model_path,
        num_threads=4,
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input data
    input_dtype = input_details[0]["dtype"]
    if input_dtype == np.uint8:
        input_data = np.random.randint(0, 255, size=input_shape).astype(np.uint8)
    else:
        input_data = np.random.randn(*input_shape).astype(np.float32)

    # Warm-up
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    stats = {
        "model_path": model_path,
        "num_runs": num_runs,
        "mean_ms": round(np.mean(times), 2),
        "median_ms": round(np.median(times), 2),
        "min_ms": round(np.min(times), 2),
        "max_ms": round(np.max(times), 2),
        "std_ms": round(np.std(times), 2),
        "fps_estimate": round(1000 / np.mean(times), 1),
    }

    logger.info(f"Inference benchmark: {stats['mean_ms']:.2f}ms avg, "
                f"~{stats['fps_estimate']} FPS")
    return stats


# =========================================================================
#  3. COMPARISON REPORT
# =========================================================================

def generate_comparison_report(
    original_model_path: str = None,
    tflite_paths: dict = None,
) -> Dict:
    """
    Generate a detailed comparison report between original and TFLite models.

    Compares:
        - File size (MB)
        - Inference speed (ms)
        - Estimated FPS
        - Compression ratio

    Args:
        original_model_path: Path to the original Keras model.
        tflite_paths: Dict of {name: path} for TFLite variants.

    Returns:
        Comparison report dictionary.
    """
    if original_model_path is None:
        original_model_path = str(BEST_MODEL_PATH) if BEST_MODEL_PATH.exists() \
                              else str(FINAL_MODEL_PATH)

    if tflite_paths is None:
        tflite_paths = {}
        if TFLITE_MODEL_PATH.exists():
            tflite_paths["TFLite (FP32)"] = str(TFLITE_MODEL_PATH)
        if TFLITE_QUANT_PATH.exists():
            tflite_paths["TFLite (Quantized)"] = str(TFLITE_QUANT_PATH)

    report = {"models": []}

    # Original model size
    original_size_mb = os.path.getsize(original_model_path) / (1024 * 1024)
    report["models"].append({
        "name": "Original (Keras)",
        "size_mb": round(original_size_mb, 2),
        "inference_ms": "N/A (use TFLite for benchmarking)",
        "compression": "1.0x (baseline)",
    })

    # TFLite variants
    for name, path in tflite_paths.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            bench = benchmark_inference(path)
            compression = original_size_mb / max(size_mb, 0.01)

            report["models"].append({
                "name": name,
                "size_mb": round(size_mb, 2),
                "inference_ms": bench["mean_ms"],
                "fps_estimate": bench["fps_estimate"],
                "compression": f"{compression:.1f}x",
                "meets_size_target": size_mb < TARGET_MODEL_SIZE_MB,
            })

    # Print report
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON REPORT")
    print("=" * 70)
    print(f"  {'Model':<25} {'Size (MB)':<12} {'Inference (ms)':<16} {'Compression':<12}")
    print("-" * 70)
    for m in report["models"]:
        inference = m.get("inference_ms", "N/A")
        if isinstance(inference, float):
            inference = f"{inference:.2f}"
        print(f"  {m['name']:<25} {m['size_mb']:<12} {inference:<16} {m['compression']:<12}")
    print("=" * 70)

    # Save report
    report_path = PLOTS_DIR / "model_comparison.json"
    with open(str(report_path), "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Comparison report saved to: {report_path}")

    return report


# =========================================================================
#  4. FULL EXPORT PIPELINE
# =========================================================================

def export_all(
    model_path: str = None,
) -> Dict:
    """
    Run the complete export pipeline:
        1. Convert to TFLite (no quantization)
        2. Convert to TFLite (dynamic range quantization)
        3. Attempt int8 quantization
        4. Generate comparison report

    Args:
        model_path: Path to the Keras model.

    Returns:
        Comparison report.
    """
    if model_path is None:
        model_path = str(BEST_MODEL_PATH) if BEST_MODEL_PATH.exists() \
                     else str(FINAL_MODEL_PATH)

    logger.info("=" * 60)
    logger.info("  TFLITE EXPORT PIPELINE")
    logger.info("=" * 60)

    tflite_paths = {}

    # ── FP32 (no quantization) ──────────────────────────────────────────
    logger.info("\n[1/3] Exporting FP32 TFLite model...")
    fp32_path = str(MODELS_DIR / "mask_detector_fp32.tflite")
    convert_to_tflite(model_path, fp32_path, quantize="none")
    tflite_paths["TFLite (FP32)"] = fp32_path

    # ── Dynamic Range Quantization ──────────────────────────────────────
    logger.info("\n[2/3] Exporting Dynamic Range Quantized model...")
    dynamic_path = str(MODELS_DIR / "mask_detector_dynamic.tflite")
    convert_to_tflite(model_path, dynamic_path, quantize="dynamic")
    tflite_paths["TFLite (Dynamic)"] = dynamic_path

    # ── Int8 Quantization ───────────────────────────────────────────────
    logger.info("\n[3/3] Attempting Int8 Quantization...")
    try:
        int8_path = str(TFLITE_QUANT_PATH)
        convert_to_tflite(model_path, int8_path, quantize="int8")
        tflite_paths["TFLite (Int8)"] = int8_path
    except Exception as e:
        logger.warning(f"Int8 quantization failed: {e}")
        logger.info("Falling back to dynamic quantization only.")

    # ── Generate Report ─────────────────────────────────────────────────
    logger.info("\nGenerating comparison report...")
    report = generate_comparison_report(model_path, tflite_paths)

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export model to TFLite")
    parser.add_argument("--model", type=str, default=None, help="Keras model path")
    parser.add_argument("--quantize", type=str, default="dynamic",
                        choices=["none", "dynamic", "float16", "int8"],
                        help="Quantization strategy")
    args = parser.parse_args()

    if args.model:
        convert_to_tflite(args.model, quantize=args.quantize)
    else:
        export_all()
