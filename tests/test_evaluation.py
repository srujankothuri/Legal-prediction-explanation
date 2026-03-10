"""
Tests for evaluation metrics.

Run: python -m pytest tests/test_evaluation.py -v
"""

import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_metrics,
    compute_metrics_from_probabilities,
    format_metrics_table,
    format_comparison_table,
)


class TestComputeMetrics:

    def test_perfect_predictions(self):
        """Perfect predictions yield accuracy = 1.0."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 0, 1, 1, 0, 1]

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["micro_f1"] == 1.0

    def test_worst_predictions(self):
        """Completely wrong predictions yield accuracy = 0.0."""
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0

    def test_mixed_predictions(self):
        """Partial accuracy is computed correctly."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.5
        assert metrics["total_samples"] == 4

    def test_confusion_matrix_values(self):
        """Confusion matrix components are correct."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1, 0]

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["true_negatives"] == 2   # correct rejections
        assert metrics["false_positives"] == 1   # wrong acceptances
        assert metrics["false_negatives"] == 1   # wrong rejections
        assert metrics["true_positives"] == 2    # correct acceptances

    def test_all_metric_keys_present(self):
        """All expected metric keys are in the output."""
        metrics = compute_metrics([0, 1], [0, 1])

        expected_keys = [
            "accuracy", "total_samples",
            "macro_precision", "macro_recall", "macro_f1",
            "micro_precision", "micro_recall", "micro_f1",
            "precision_rejected", "recall_rejected", "f1_rejected",
            "precision_accepted", "recall_accepted", "f1_accepted",
            "confusion_matrix",
            "true_negatives", "false_positives",
            "false_negatives", "true_positives",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"


class TestProbabilityMetrics:

    def test_threshold_default(self):
        """Default threshold of 0.5 is applied correctly."""
        y_true = [0, 0, 1, 1]
        y_probs = [0.2, 0.4, 0.6, 0.8]

        metrics = compute_metrics_from_probabilities(y_true, y_probs)

        assert metrics["accuracy"] == 1.0
        assert metrics["threshold"] == 0.5

    def test_custom_threshold(self):
        """Custom threshold changes predictions."""
        y_true = [0, 0, 1, 1]
        y_probs = [0.2, 0.4, 0.6, 0.8]

        # With threshold=0.3, probs [0.4, 0.6, 0.8] become positive
        metrics = compute_metrics_from_probabilities(y_true, y_probs, threshold=0.3)

        # 0.2→0 (correct), 0.4→1 (wrong), 0.6→1 (correct), 0.8→1 (correct)
        assert metrics["accuracy"] == 0.75

    def test_probability_stats(self):
        """Mean and std of probabilities are computed."""
        y_true = [0, 1]
        y_probs = [0.3, 0.7]

        metrics = compute_metrics_from_probabilities(y_true, y_probs)

        assert metrics["mean_probability"] == pytest.approx(0.5)
        assert metrics["std_probability"] > 0


class TestFormatting:

    def test_format_metrics_table_runs(self):
        """Table formatting doesn't crash."""
        metrics = compute_metrics([0, 0, 1, 1], [0, 1, 1, 0])
        table = format_metrics_table(metrics, "Test Model")

        assert "Test Model" in table
        assert "Accuracy" in table
        assert "Confusion Matrix" in table

    def test_format_comparison_table(self):
        """Comparison table with multiple models."""
        results = [
            {
                "model_name": "XLNet",
                "accuracy": 0.78,
                "macro_f1": 0.76,
                "macro_precision": 0.77,
                "macro_recall": 0.75,
            },
            {
                "model_name": "RoBERTa",
                "accuracy": 0.81,
                "macro_f1": 0.80,
                "macro_precision": 0.79,
                "macro_recall": 0.81,
            },
        ]

        table = format_comparison_table(results)

        assert "XLNet" in table
        assert "RoBERTa" in table
        assert "Model Comparison" in table

    def test_format_empty_comparison(self):
        """Empty comparison table doesn't crash."""
        table = format_comparison_table([])
        assert "Model Comparison" in table