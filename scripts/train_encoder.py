#!/usr/bin/env python3
"""
Fine-tune a transformer encoder on the ILDC dataset.

Usage:
    python scripts/train_encoder.py --config configs/models/xlnet_bigru.yaml
    python scripts/train_encoder.py --config configs/models/roberta_bigru.yaml

    # Or via Makefile:
    make train-encoder MODEL=xlnet
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, merge_configs
from src.utils.logger import get_logger
from src.utils.device import log_device_info
from src.data.dataset import ILDCDataset
from src.training.trainer import EncoderTrainer

logger = get_logger("train_encoder")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune transformer encoder on ILDC")
    parser.add_argument(
        "--config", required=True,
        help="Path to model config YAML (e.g., configs/models/xlnet_bigru.yaml)"
    )
    parser.add_argument(
        "--training-config", default="configs/training.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--data", default=None,
        help="Override dataset path"
    )
    args = parser.parse_args()

    # Load configs
    model_config = load_config(args.config)
    training_config = load_config(args.training_config)

    encoder_name = model_config["encoder"]["name"]
    logger.info(f"{'=' * 60}")
    logger.info(f"Fine-tuning encoder: {encoder_name}")
    logger.info(f"{'=' * 60}")

    # Log device info
    log_device_info()

    # Load dataset
    data_path = args.data or training_config["data"]["dataset_path"]
    dataset = ILDCDataset(data_path)
    logger.info(f"\n{dataset.summary()}")

    train_df = dataset.get_split("train")
    val_df = dataset.get_split("dev")

    # Train
    trainer = EncoderTrainer(model_config)
    trainer.train(train_df, val_df)

    # Save
    trainer.save()

    logger.info(f"{'=' * 60}")
    logger.info(f"Training complete for: {encoder_name}")
    logger.info(f"Model saved to: trained_models/{encoder_name}_finetuned/")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
