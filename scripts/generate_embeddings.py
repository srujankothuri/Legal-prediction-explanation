#!/usr/bin/env python3
"""
Generate embeddings from a fine-tuned encoder for HAN classifier training.

Reads the fine-tuned model from trained_models/{encoder}_finetuned/,
processes each document into chunk embeddings, and saves as .npy files.

Usage:
    python scripts/generate_embeddings.py --config configs/models/xlnet_bigru.yaml
    python scripts/generate_embeddings.py --config configs/models/roberta_bigru.yaml

    # Or via Makefile:
    make generate-embeddings MODEL=xlnet
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.device import log_device_info
from src.data.dataset import ILDCDataset
from src.training.embeddings import EmbeddingGenerator

logger = get_logger("generate_embeddings")


def main():
    parser = argparse.ArgumentParser(description="Generate document embeddings")
    parser.add_argument(
        "--config", required=True,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--training-config", default="configs/training.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--splits", default="train,dev,test",
        help="Comma-separated splits to process (default: train,dev,test)"
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
    logger.info(f"Generating embeddings for: {encoder_name}")
    logger.info(f"{'=' * 60}")

    log_device_info()

    # Load dataset
    data_path = args.data or training_config["data"]["dataset_path"]
    dataset = ILDCDataset(data_path)

    # Output directory
    output_dir = os.path.join(
        training_config["paths"]["embeddings_dir"],
        encoder_name,
    )

    # Generate embeddings
    generator = EmbeddingGenerator(model_config)

    splits = [s.strip() for s in args.splits.split(",")]
    for split in splits:
        logger.info(f"\nProcessing split: {split}")
        generator.generate_for_dataset(dataset, split, output_dir)

    logger.info(f"{'=' * 60}")
    logger.info(f"All embeddings saved to: {output_dir}/")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()