#!/usr/bin/env python3
"""
Evaluate trained models on the ILDC test set and generate comparison tables.

Loads each trained HAN classifier, runs predictions on test embeddings,
and outputs a side-by-side comparison of all models.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --models xlnet roberta
    python scripts/evaluate.py --split dev

    # Or via Makefile:
    make evaluate
"""

import argparse
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.dataset import ILDCDataset
from src.models.han_classifier import HANClassifier
from src.training.callbacks import (
    create_test_generator,
    get_test_steps,
)
from src.evaluation.metrics import (
    compute_metrics,
    compute_metrics_from_probabilities,
    format_metrics_table,
    format_comparison_table,
)

logger = get_logger("evaluate")

AVAILABLE_MODELS = ["xlnet", "roberta", "bert", "distilbert"]


def load_test_embeddings(embeddings_dir: str, encoder_name: str, split: str):
    """Load embeddings for evaluation."""
    emb_dir = os.path.join(embeddings_dir, encoder_name)

    # Try single file
    single_path = os.path.join(emb_dir, f"{encoder_name}_{split}.npy")
    if os.path.exists(single_path):
        return np.load(single_path, allow_pickle=True)

    # Try multi-part
    parts = []
    part_num = 1
    while True:
        part_path = os.path.join(emb_dir, f"{encoder_name}_{split}_{part_num}.npy")
        if os.path.exists(part_path):
            parts.append(np.load(part_path, allow_pickle=True))
            part_num += 1
        else:
            break

    if parts:
        return np.concatenate(parts)

    return None


def evaluate_model(
    encoder_name: str,
    model_config,
    training_config,
    dataset: ILDCDataset,
    split: str,
) -> dict:
    """
    Evaluate a single model on a dataset split.

    Returns:
        Metrics dictionary with model_name added, or None if model not found
    """
    # Check if HAN weights exist
    models_dir = training_config["paths"]["trained_models_dir"]
    han_path = os.path.join(models_dir, f"han_{encoder_name}.h5")

    if not os.path.exists(han_path):
        logger.warning(f"HAN model not found: {han_path} — skipping {encoder_name}")
        return None

    # Check if embeddings exist
    embeddings_dir = training_config["paths"]["embeddings_dir"]
    x_test = load_test_embeddings(embeddings_dir, encoder_name, split)

    if x_test is None:
        logger.warning(f"Embeddings not found for {encoder_name}/{split} — skipping")
        return None

    y_test = dataset.get_labels(split)

    # Validate alignment
    if len(x_test) != len(y_test):
        logger.error(
            f"Embedding/label mismatch for {encoder_name}: "
            f"{len(x_test)} embeddings vs {len(y_test)} labels"
        )
        return None

    logger.info(f"Evaluating {encoder_name} on {split}: {len(x_test)} documents")

    # Build and load model
    han = HANClassifier(model_config, weights_path=han_path)
    model = han.build()

    # Generate predictions
    num_features = model_config["encoder"]["embedding_dim"]
    batch_size = model_config["classifier"].get("batch_size", 32)

    test_gen = create_test_generator(x_test, y_test, batch_size, num_features)
    test_steps = get_test_steps(len(x_test), batch_size)

    # Get probabilities
    preds = model.predict(test_gen, steps=test_steps)

    # Trim predictions to actual test size (last batch may overflow)
    preds = preds[:len(y_test)]

    # Compute metrics
    y_pred = (preds > 0.5).astype(int).flatten().tolist()
    metrics = compute_metrics(y_test, y_pred)
    metrics["model_name"] = f"{encoder_name.upper()} + BiGRU-HAN"

    # Log results
    table = format_metrics_table(metrics, metrics["model_name"])
    logger.info(f"\n{table}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to evaluate (default: all). Options: {AVAILABLE_MODELS}"
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--training-config", default="configs/training.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--output", default="logs/evaluation_results.json",
        help="Path to save results JSON"
    )
    args = parser.parse_args()

    training_config = load_config(args.training_config)

    # Load dataset for labels
    data_path = training_config["data"]["dataset_path"]
    dataset = ILDCDataset(data_path)

    models_to_eval = args.models or AVAILABLE_MODELS

    logger.info(f"{'=' * 60}")
    logger.info(f"Evaluating models: {models_to_eval}")
    logger.info(f"Split: {args.split}")
    logger.info(f"{'=' * 60}")

    # Evaluate each model
    all_results = []
    for encoder_name in models_to_eval:
        config_path = f"configs/models/{encoder_name}_bigru.yaml"

        if not os.path.exists(config_path):
            logger.warning(f"Config not found: {config_path} — skipping {encoder_name}")
            continue

        model_config = load_config(config_path)
        result = evaluate_model(
            encoder_name, model_config, training_config, dataset, args.split
        )

        if result:
            all_results.append(result)

    # Print comparison table
    if len(all_results) > 1:
        comparison = format_comparison_table(all_results)
        logger.info(f"\n{comparison}")
    elif len(all_results) == 0:
        logger.warning("No models were successfully evaluated.")
        return

    # Save results to JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Convert numpy types for JSON serialization
    serializable_results = []
    for r in all_results:
        result_copy = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                result_copy[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                result_copy[k] = float(v)
            else:
                result_copy[k] = v
        serializable_results.append(result_copy)

    with open(args.output, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()