import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Main")


def main():
    parser = argparse.ArgumentParser(
        description="Mask Detection Pro G11 — Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  info       Show project configuration
  train      Train the MobileNetV2 model
  evaluate   Evaluate model and generate visualizations
  export     Convert model to TFLite format
  run        Launch real-time webcam detection
  all        Run full pipeline (train + evaluate + export)

Examples:
  python main.py train --epochs 30
  python main.py evaluate --model models/best_mask_detector.keras
  python main.py export --quantize int8
  python main.py run --camera 0 --gradcam
  python main.py all --data_dir ./data/raw
        """,
    )

    parser.add_argument(
        "command",
        choices=["info", "train", "evaluate", "export", "run", "all"],
        help="Command to execute.",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_fine_tune", action="store_true")
    parser.add_argument("--quantize", type=str, default="dynamic")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--gradcam", action="store_true")

    args = parser.parse_args()

    # ── Route Commands ──────────────────────────────────────────────────
    if args.command == "info":
        from src.config import print_config
        print_config()

    elif args.command == "train":
        from src.train import train
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            fine_tune=not args.no_fine_tune,
        )

    elif args.command == "evaluate":
        from src.evaluate import evaluate
        evaluate(model_path=args.model, data_dir=args.data_dir)

    elif args.command == "export":
        from src.export import export_all, convert_to_tflite
        if args.model:
            convert_to_tflite(args.model, quantize=args.quantize)
        else:
            export_all()

    elif args.command == "run":
        from app.realtime_app import run_realtime
        run_realtime(
            model_path=args.model,
            camera_id=args.camera,
            show_gradcam=args.gradcam,
        )

    elif args.command == "all":
        logger.info("=" * 60)
        logger.info("  FULL PIPELINE EXECUTION")
        logger.info("=" * 60)

        # Step 1: Train
        logger.info("\n[1/3] TRAINING...")
        from src.train import train
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            fine_tune=not args.no_fine_tune,
        )

        # Step 2: Evaluate
        logger.info("\n[2/3] EVALUATING...")
        from src.evaluate import evaluate
        evaluate(model_path=args.model, data_dir=args.data_dir)

        # Step 3: Export
        logger.info("\n[3/3] EXPORTING TO TFLITE...")
        from src.export import export_all
        export_all(model_path=args.model)

        logger.info("\n" + "=" * 60)
        logger.info("  PIPELINE COMPLETE!")
        logger.info("  Run 'python main.py run' to start real-time detection.")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
