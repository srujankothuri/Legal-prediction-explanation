#!/usr/bin/env python3
"""
Run the complete training + evaluation pipeline for all models.

This is the single script to run on Lightning AI to train everything.

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --models xlnet roberta
    python scripts/run_full_pipeline.py --skip-training --evaluate-only
"""

import argparse
import subprocess
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger

logger = get_logger("full_pipeline")

ALL_MODELS = ["xlnet", "roberta", "bert", "distilbert"]


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command and log the result."""
    logger.info(f"{'─' * 50}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {cmd}")
    logger.info(f"{'─' * 50}")

    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start

    if result.returncode == 0:
        logger.info(f"✅ {description} — completed in {elapsed:.1f}s")
        return True
    else:
        logger.error(f"❌ {description} — FAILED (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run full training pipeline")
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        help=f"Models to train (default: {ALL_MODELS})"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip encoder fine-tuning (use existing weights)"
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip embedding generation (use existing .npy files)"
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Only run evaluation on existing models"
    )
    parser.add_argument(
        "--build-vectordb", action="store_true",
        help="Also build the FAISS vector database"
    )
    args = parser.parse_args()

    total_start = time.time()

    logger.info("=" * 60)
    logger.info("FULL TRAINING PIPELINE")
    logger.info(f"Models: {args.models}")
    logger.info("=" * 60)

    results = {}

    if not args.evaluate_only:
        for model in args.models:
            logger.info(f"\n{'╔' + '═' * 50 + '╗'}")
            logger.info(f"  Training: {model.upper()}")
            logger.info(f"{'╚' + '═' * 50 + '╝'}\n")

            config = f"configs/models/{model}_bigru.yaml"
            model_results = {"encoder": False, "embeddings": False, "classifier": False}

            # Step 1: Fine-tune encoder
            if not args.skip_training:
                success = run_command(
                    f"python scripts/train_encoder.py --config {config}",
                    f"{model.upper()} encoder fine-tuning",
                )
                model_results["encoder"] = success
                if not success:
                    logger.warning(f"Skipping remaining steps for {model}")
                    results[model] = model_results
                    continue
            else:
                logger.info(f"Skipping encoder training for {model}")
                model_results["encoder"] = True

            # Step 2: Generate embeddings
            if not args.skip_embeddings:
                success = run_command(
                    f"python scripts/generate_embeddings.py --config {config}",
                    f"{model.upper()} embedding generation",
                )
                model_results["embeddings"] = success
                if not success:
                    logger.warning(f"Skipping classifier training for {model}")
                    results[model] = model_results
                    continue
            else:
                logger.info(f"Skipping embedding generation for {model}")
                model_results["embeddings"] = True

            # Step 3: Train HAN classifier
            success = run_command(
                f"python scripts/train_classifier.py --config {config}",
                f"{model.upper()} HAN classifier training",
            )
            model_results["classifier"] = success

            results[model] = model_results

    # Step 4: Evaluate all
    logger.info(f"\n{'═' * 60}")
    logger.info("EVALUATION")
    logger.info(f"{'═' * 60}\n")

    models_arg = " ".join(args.models)
    run_command(
        f"python scripts/evaluate.py --models {models_arg}",
        "Model evaluation + comparison",
    )

    # Optional: Build vector DB
    if args.build_vectordb:
        run_command(
            "python scripts/build_vector_db.py",
            "FAISS vector database build",
        )

    # Summary
    total_elapsed = time.time() - total_start
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)

    logger.info(f"\n{'═' * 60}")
    logger.info(f"PIPELINE COMPLETE — Total time: {hours}h {minutes}m")
    logger.info(f"{'═' * 60}")

    if results:
        logger.info("\nResults Summary:")
        for model, r in results.items():
            status = "✅" if all(r.values()) else "⚠️"
            logger.info(f"  {status} {model.upper()}: {r}")

    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review results: cat logs/evaluation_results.json")
    logger.info(f"  2. Push to GitHub: git add -A && git commit && git push")
    logger.info(f"  3. Run the app: make app")


if __name__ == "__main__":
    main()