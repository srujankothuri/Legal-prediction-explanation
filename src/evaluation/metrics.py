"""
Evaluation metrics for binary classification.

Computes macro/micro precision, recall, F1, accuracy,
confusion matrix, and per-class statistics.

Usage:
    from src.evaluation.metrics import compute_metrics, format_metrics_table

    metrics = compute_metrics(y_true, y_pred)
    print(format_metrics_table(metrics))
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

LABEL_NAMES = {0: "Rejected", 1: "Accepted"}


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, Any]:
    """
    Compute comprehensive binary classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Dictionary containing all metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        # Overall
        "accuracy": accuracy_score(y_true, y_pred),
        "total_samples": len(y_true),

        # Macro (unweighted average across classes)
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),

        # Micro (aggregate TP, FP, FN across classes)
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),

        # Per-class
        "precision_rejected": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_rejected": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_rejected": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_accepted": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_accepted": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_accepted": f1_score(y_true, y_pred, pos_label=1, zero_division=0),

        # Confusion matrix
        "confusion_matrix": cm,
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
    }

    logger.info(
        f"Metrics computed: accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"micro_f1={metrics['micro_f1']:.4f}"
    )

    return metrics


def compute_metrics_from_probabilities(
    y_true: List[int],
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute metrics from prediction probabilities.

    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities (for positive class)
        threshold: Classification threshold

    Returns:
        Dictionary containing all metrics + threshold info
    """
    y_pred = (np.array(y_probs) > threshold).astype(int).tolist()
    metrics = compute_metrics(y_true, y_pred)
    metrics["threshold"] = threshold
    metrics["mean_probability"] = float(np.mean(y_probs))
    metrics["std_probability"] = float(np.std(y_probs))
    return metrics


def format_metrics_table(metrics: Dict[str, Any], model_name: str = "") -> str:
    """
    Format metrics as a readable table string.

    Args:
        metrics: Metrics dictionary from compute_metrics()
        model_name: Optional model name for the header

    Returns:
        Formatted string
    """
    header = f"Evaluation Results: {model_name}" if model_name else "Evaluation Results"
    separator = "=" * 55

    cm = metrics["confusion_matrix"]

    lines = [
        separator,
        header,
        separator,
        "",
        f"  Accuracy:          {metrics['accuracy']:.4f}  ({metrics['total_samples']} samples)",
        "",
        "  Macro Metrics:",
        f"    Precision:       {metrics['macro_precision']:.4f}",
        f"    Recall:          {metrics['macro_recall']:.4f}",
        f"    F1-Score:        {metrics['macro_f1']:.4f}",
        "",
        "  Micro Metrics:",
        f"    Precision:       {metrics['micro_precision']:.4f}",
        f"    Recall:          {metrics['micro_recall']:.4f}",
        f"    F1-Score:        {metrics['micro_f1']:.4f}",
        "",
        "  Per-Class:",
        f"    Rejected  — P: {metrics['precision_rejected']:.4f}  "
        f"R: {metrics['recall_rejected']:.4f}  F1: {metrics['f1_rejected']:.4f}",
        f"    Accepted  — P: {metrics['precision_accepted']:.4f}  "
        f"R: {metrics['recall_accepted']:.4f}  F1: {metrics['f1_accepted']:.4f}",
        "",
        "  Confusion Matrix:",
        f"                  Pred Rejected  Pred Accepted",
        f"    True Rejected     {cm[0][0]:>5}          {cm[0][1]:>5}",
        f"    True Accepted     {cm[1][0]:>5}          {cm[1][1]:>5}",
        "",
        separator,
    ]

    return "\n".join(lines)


def format_comparison_table(results: List[Dict[str, Any]]) -> str:
    """
    Format a comparison table across multiple models.

    Args:
        results: List of dicts, each with 'model_name' and metrics keys

    Returns:
        Formatted comparison table string
    """
    separator = "=" * 80

    lines = [
        separator,
        "Model Comparison",
        separator,
        "",
        f"  {'Model':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Macro-P':>10} {'Macro-R':>10}",
        f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}",
    ]

    for result in results:
        name = result.get("model_name", "Unknown")
        lines.append(
            f"  {name:<25} "
            f"{result['accuracy']:>10.4f} "
            f"{result['macro_f1']:>10.4f} "
            f"{result['macro_precision']:>10.4f} "
            f"{result['macro_recall']:>10.4f}"
        )

    lines.extend(["", separator])
    return "\n".join(lines)