"""
ILDC (Indian Legal Documents Corpus) dataset loader.

Handles loading CSV data, splitting into train/dev/test,
and providing clean interfaces for the training pipeline.

Usage:
    from src.data.dataset import ILDCDataset

    dataset = ILDCDataset("data/raw/single_ILDC.csv")
    train_df = dataset.get_split("train")
    print(dataset.summary())
"""

import os
import pandas as pd
from typing import Optional, Tuple, List

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


class ILDCDataset:
    """
    Loader for the ILDC dataset.

    Expected CSV format:
        - text: Full case document text
        - label: Binary label (0 = rejected, 1 = accepted)
        - split: One of 'train', 'dev', 'test'

    Optional columns:
        - name: Case identifier
        - headnote: Case headnote/summary
    """

    REQUIRED_COLUMNS = ["text", "label", "split"]
    VALID_SPLITS = ["train", "dev", "test"]
    LABEL_MAP = {0: "rejected", 1: "accepted"}

    def __init__(self, csv_path: str):
        """
        Load and validate the ILDC dataset.

        Args:
            csv_path: Path to the CSV file

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        logger.info(f"Loading dataset from {csv_path}")
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        self._validate()
        logger.info(
            f"Dataset loaded: {len(self.df)} documents "
            f"({self._split_counts_str()})"
        )

    def _validate(self):
        """Validate that dataset has required columns and valid values."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(
                f"Dataset missing required columns: {missing}. "
                f"Found: {list(self.df.columns)}"
            )

        # Validate splits
        invalid_splits = set(self.df["split"].unique()) - set(self.VALID_SPLITS)
        if invalid_splits:
            logger.warning(f"Unexpected split values found: {invalid_splits}")

        # Validate labels
        unique_labels = self.df["label"].unique()
        if not set(unique_labels).issubset({0, 1}):
            logger.warning(f"Unexpected label values: {unique_labels}")

        # Check for nulls in critical columns
        null_counts = self.df[self.REQUIRED_COLUMNS].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")

    def _split_counts_str(self) -> str:
        """Format split counts as a readable string."""
        counts = self.df["split"].value_counts()
        parts = [f"{split}: {counts.get(split, 0)}" for split in self.VALID_SPLITS]
        return ", ".join(parts)

    def get_split(self, split: str) -> pd.DataFrame:
        """
        Get a specific data split.

        Args:
            split: One of 'train', 'dev', 'test'

        Returns:
            DataFrame filtered to the requested split

        Raises:
            ValueError: If split name is invalid
        """
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {self.VALID_SPLITS}"
            )

        split_df = self.df[self.df["split"] == split].reset_index(drop=True)
        logger.debug(f"Split '{split}': {len(split_df)} documents")
        return split_df

    def get_texts(self, split: str) -> List[str]:
        """Get list of document texts for a split."""
        return self.get_split(split)["text"].tolist()

    def get_labels(self, split: str) -> List[int]:
        """Get list of labels for a split."""
        return self.get_split(split)["label"].tolist()

    def get_texts_and_labels(self, split: str) -> Tuple[List[str], List[int]]:
        """Get texts and labels as parallel lists."""
        split_df = self.get_split(split)
        return split_df["text"].tolist(), split_df["label"].tolist()

    def summary(self) -> str:
        """Generate a summary report of the dataset."""
        lines = [
            f"ILDC Dataset: {self.csv_path}",
            f"Total documents: {len(self.df)}",
            "",
            "Split distribution:",
        ]

        for split in self.VALID_SPLITS:
            split_df = self.df[self.df["split"] == split]
            if len(split_df) == 0:
                continue

            label_counts = split_df["label"].value_counts()
            accepted = label_counts.get(1, 0)
            rejected = label_counts.get(0, 0)
            total = len(split_df)
            balance = accepted / total * 100 if total > 0 else 0

            lines.append(
                f"  {split:>5}: {total:>6} docs "
                f"(accepted: {accepted}, rejected: {rejected}, "
                f"balance: {balance:.1f}% accepted)"
            )

        # Text length statistics
        text_lengths = self.df["text"].str.split().str.len()
        lines.extend([
            "",
            "Text length (words):",
            f"  Mean:   {text_lengths.mean():.0f}",
            f"  Median: {text_lengths.median():.0f}",
            f"  Min:    {text_lengths.min():.0f}",
            f"  Max:    {text_lengths.max():.0f}",
        ])

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"ILDCDataset({self.csv_path}, n={len(self.df)})"


def load_dataset_from_config(config_path: str = "configs/training.yaml") -> ILDCDataset:
    """
    Load dataset using path from training config.

    Args:
        config_path: Path to training config YAML

    Returns:
        ILDCDataset instance
    """
    cfg = load_config(config_path)
    dataset_path = cfg["data"]["dataset_path"]
    return ILDCDataset(dataset_path)