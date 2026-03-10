"""
Tests for data loading and preprocessing modules.

Run: python -m pytest tests/test_data.py -v
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data.dataset import ILDCDataset


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal sample CSV for testing."""
    data = {
        "text": [
            "The appellant filed a case regarding property dispute in the supreme court.",
            "The respondent argued that the lower court judgment was correct.",
            "The court examined the evidence presented by both parties carefully.",
            "After reviewing all arguments the court decided to accept the appeal.",
            "The bench dismissed the petition citing lack of merit in arguments.",
            "The judgment was reserved and later pronounced in open court.",
        ],
        "label": [1, 0, 1, 1, 0, 0],
        "split": ["train", "train", "train", "dev", "dev", "test"],
    }
    csv_path = tmp_path / "test_dataset.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return str(csv_path)


# ── Dataset Loading Tests ───────────────────────────────────────────────────

class TestILDCDataset:

    def test_load_valid_csv(self, sample_csv):
        """Dataset loads successfully from valid CSV."""
        dataset = ILDCDataset(sample_csv)
        assert len(dataset) == 6

    def test_file_not_found(self):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            ILDCDataset("nonexistent_file.csv")

    def test_missing_columns(self, tmp_path):
        """Raises ValueError if required columns are missing."""
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1, 2]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            ILDCDataset(str(csv_path))

    def test_get_split_train(self, sample_csv):
        """Returns correct number of train documents."""
        dataset = ILDCDataset(sample_csv)
        train_df = dataset.get_split("train")
        assert len(train_df) == 3

    def test_get_split_dev(self, sample_csv):
        """Returns correct number of dev documents."""
        dataset = ILDCDataset(sample_csv)
        dev_df = dataset.get_split("dev")
        assert len(dev_df) == 2

    def test_get_split_test(self, sample_csv):
        """Returns correct number of test documents."""
        dataset = ILDCDataset(sample_csv)
        test_df = dataset.get_split("test")
        assert len(test_df) == 1

    def test_invalid_split_raises(self, sample_csv):
        """Raises ValueError for invalid split name."""
        dataset = ILDCDataset(sample_csv)
        with pytest.raises(ValueError, match="Invalid split"):
            dataset.get_split("validation")

    def test_get_texts_and_labels(self, sample_csv):
        """Returns parallel lists of texts and labels."""
        dataset = ILDCDataset(sample_csv)
        texts, labels = dataset.get_texts_and_labels("train")
        assert len(texts) == 3
        assert len(labels) == 3
        assert all(isinstance(t, str) for t in texts)
        assert all(l in [0, 1] for l in labels)

    def test_summary_runs(self, sample_csv):
        """Summary generation doesn't crash."""
        dataset = ILDCDataset(sample_csv)
        summary = dataset.summary()
        assert "Total documents: 6" in summary
        assert "train" in summary

    def test_repr(self, sample_csv):
        """String representation is informative."""
        dataset = ILDCDataset(sample_csv)
        assert "n=6" in repr(dataset)