#!/usr/bin/env python3
"""
Train the HAN classifier on pre-computed embeddings.

Reads embeddings from data/embeddings/{encoder}/ and trains
the BiGRU + Attention classifier, saving weights as .h5.

Usage:
    python scripts/train_classifier.py --config configs/models/xlnet_bigru.yaml
    python scripts/train_classifier.py --config configs/models/roberta_bigru.yaml

    # Or via Makefile:
    make train-classifier MODEL=xlnet
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.dataset import ILDCDataset
from src.models.han_classifier import HANClassifier
from src.training.callbacks import (
    create_batch_generator,
    get_steps_per_epoch,
    get_training_callbacks,
)

logger = get_logger("train_classifier")


def load_embeddings(embeddings_dir: str, encoder_name: str, split: str) -> np.ndarray:
    """
    Load pre-computed embeddings for a given split.
    Handles single file or multi-part files (e.g., train_1.npy, train_2.npy).

    Args:
        embeddings_dir: Base embeddings directory
        encoder_name: Encoder name (xlnet, roberta, etc.)
        split: Data split (train, dev, test)

    Returns:
        Array of document embeddings
    """
    emb_dir = os.path.join(embeddings_dir, encoder_name)

    # Try single file first
    single_path = os.path.join(emb_dir, f"{encoder_name}_{split}.npy")
    if os.path.exists(single_path):
        logger.info(f"Loading embeddings: {single_path}")
        return np.load(single_path, allow_pickle=True)

    # Try multi-part files (train_1.npy, train_2.npy, ...)
    parts = []
    part_num = 1
    while True:
        part_path = os.path.join(emb_dir, f"{encoder_name}_{split}_{part_num}.npy")
        if os.path.exists(part_path):
            logger.info(f"Loading embeddings part: {part_path}")
            parts.append(np.load(part_path, allow_pickle=True))
            part_num += 1
        else:
            break

    if parts:
        combined = np.concatenate(parts)
        logger.info(f"Loaded {len(parts)} parts → {len(combined)} documents")
        return combined

    raise FileNotFoundError(
        f"No embeddings found for {encoder_name}/{split} in {emb_dir}. "
        f"Run: python scripts/generate_embeddings.py --config configs/models/{encoder_name}_bigru.yaml"
    )


def main():
    parser = argparse.ArgumentParser(description="Train HAN classifier on embeddings")
    parser.add_argument(
        "--config", required=True,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--training-config", default="configs/training.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--data", default=None,
        help="Override dataset path (for labels)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs"
    )
    args = parser.parse_args()

    # Load configs
    model_config = load_config(args.config)
    training_config = load_config(args.training_config)

    encoder_name = model_config["encoder"]["name"]
    classifier_cfg = model_config["classifier"]

    logger.info(f"{'=' * 60}")
    logger.info(f"Training HAN classifier for: {encoder_name}")
    logger.info(f"{'=' * 60}")

    # Load dataset (for labels)
    data_path = args.data or training_config["data"]["dataset_path"]
    dataset = ILDCDataset(data_path)

    # Load pre-computed embeddings
    embeddings_dir = training_config["paths"]["embeddings_dir"]

    x_train = load_embeddings(embeddings_dir, encoder_name, "train")
    x_dev = load_embeddings(embeddings_dir, encoder_name, "dev")

    y_train = dataset.get_labels("train")
    y_dev = dataset.get_labels("dev")

    logger.info(f"Train: {len(x_train)} docs, Dev: {len(x_dev)} docs")

    # Validate alignment
    assert len(x_train) == len(y_train), (
        f"Embedding/label mismatch: {len(x_train)} embeddings vs {len(y_train)} labels"
    )
    assert len(x_dev) == len(y_dev), (
        f"Embedding/label mismatch: {len(x_dev)} embeddings vs {len(y_dev)} labels"
    )

    # Build model
    han = HANClassifier(model_config)
    model = han.build()
    model.summary()

    # Create generators
    batch_size = classifier_cfg.get("batch_size", 32)
    epochs = args.epochs or classifier_cfg.get("epochs", 1)
    num_features = model_config["encoder"]["embedding_dim"]

    train_gen = create_batch_generator(x_train, y_train, batch_size, num_features)
    val_gen = create_batch_generator(x_dev, y_dev, batch_size, num_features)

    train_steps = get_steps_per_epoch(len(x_train), batch_size)
    val_steps = get_steps_per_epoch(len(x_dev), batch_size)

    callbacks = get_training_callbacks()

    logger.info(
        f"Training: epochs={epochs}, batch_size={batch_size}, "
        f"train_steps={train_steps}, val_steps={val_steps}"
    )

    # Train
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    # Save model
    models_dir = training_config["paths"]["trained_models_dir"]
    save_path = os.path.join(models_dir, f"han_{encoder_name}.h5")
    os.makedirs(models_dir, exist_ok=True)
    model.save(save_path)

    logger.info(f"{'=' * 60}")
    logger.info(f"HAN classifier saved to: {save_path}")
    logger.info(f"Final train acc: {history.history['acc'][-1]:.4f}")
    if 'val_acc' in history.history:
        logger.info(f"Final val acc:   {history.history['val_acc'][-1]:.4f}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()