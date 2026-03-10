"""
Training callbacks and data generators for HAN classifier training.

Provides:
- Batch generators for variable-length embedding sequences
- Keras callbacks for learning rate scheduling

Usage:
    from src.training.callbacks import create_batch_generator, get_training_callbacks

    train_gen = create_batch_generator(x_train, y_train, batch_size=32)
    callbacks = get_training_callbacks()
"""

import numpy as np
from typing import List, Tuple

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_batch_generator(
    x_data: np.ndarray,
    y_data: List[int],
    batch_size: int,
    num_features: int = 3072,
    mask_value: float = -99.0,
):
    """
    Create a batch generator for variable-length embedding sequences.

    Each document has a different number of chunks, so batches are padded
    to the max sequence length within each batch using mask_value.

    Args:
        x_data: Array of document embeddings (object array, variable inner shapes)
        y_data: List of labels
        batch_size: Batch size
        num_features: Embedding dimension
        mask_value: Value used for padding (must match Masking layer)

    Yields:
        Tuple of (x_batch, y_batch)
        x_batch shape: (batch_size, max_timesteps_in_batch, num_features)
        y_batch shape: (batch_size, 1)
    """
    num_sequences = len(x_data)
    batches_per_epoch = num_sequences // batch_size

    if batches_per_epoch == 0:
        logger.warning(
            f"batch_size ({batch_size}) >= dataset size ({num_sequences}). "
            f"Using batch_size=1."
        )
        batch_size = 1
        batches_per_epoch = num_sequences

    logger.info(
        f"Generator created: {num_sequences} sequences, "
        f"batch_size={batch_size}, "
        f"batches_per_epoch={batches_per_epoch}"
    )

    while True:
        for b in range(batches_per_epoch):
            # Find max timesteps in this batch
            batch_start = b * batch_size
            batch_end = batch_start + batch_size
            batch_slice = x_data[batch_start:batch_end]

            timesteps = max(len(doc) for doc in batch_slice)

            # Initialize padded batch
            x_batch = np.full(
                (batch_size, timesteps, num_features),
                mask_value,
                dtype=np.float32,
            )
            y_batch = np.zeros((batch_size, 1), dtype=np.float32)

            # Fill in actual values
            for i in range(batch_size):
                idx = batch_start + i
                doc_emb = x_data[idx]
                x_batch[i, :len(doc_emb), :] = doc_emb
                y_batch[i] = y_data[idx]

            yield x_batch, y_batch


def create_test_generator(
    x_data: np.ndarray,
    y_data: List[int],
    batch_size: int = 32,
    num_features: int = 3072,
    mask_value: float = -99.0,
):
    """
    Create a test/evaluation batch generator.

    Handles the last batch which may be smaller than batch_size.

    Args:
        x_data: Array of document embeddings
        y_data: List of labels
        batch_size: Batch size
        num_features: Embedding dimension
        mask_value: Padding value

    Yields:
        Tuple of (x_batch, y_batch)
    """
    num_sequences = len(x_data)

    if num_sequences % batch_size == 0:
        total_batches = num_sequences // batch_size
    else:
        total_batches = (num_sequences // batch_size) + 1

    while True:
        for b in range(total_batches):
            batch_start = b * batch_size

            # Handle last batch (may be smaller)
            if b == total_batches - 1:
                actual_size = num_sequences - batch_start
            else:
                actual_size = batch_size

            batch_end = batch_start + actual_size
            batch_slice = x_data[batch_start:batch_end]

            timesteps = max(len(doc) for doc in batch_slice)

            x_batch = np.full(
                (actual_size, timesteps, num_features),
                mask_value,
                dtype=np.float32,
            )
            y_batch = np.zeros((actual_size, 1), dtype=np.float32)

            for i in range(actual_size):
                idx = batch_start + i
                doc_emb = x_data[idx]
                x_batch[i, :len(doc_emb), :] = doc_emb
                y_batch[i] = y_data[idx]

            yield x_batch, y_batch


def get_steps_per_epoch(num_sequences: int, batch_size: int) -> int:
    """Calculate number of batches per epoch."""
    return max(1, num_sequences // batch_size)


def get_test_steps(num_sequences: int, batch_size: int) -> int:
    """Calculate number of test batches (including partial last batch)."""
    if num_sequences % batch_size == 0:
        return num_sequences // batch_size
    return (num_sequences // batch_size) + 1


def get_training_callbacks():
    """
    Create standard training callbacks.

    Returns:
        List of Keras callbacks
    """
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_acc",
            factor=0.95,
            patience=2,
            verbose=1,
            mode="auto",
            min_delta=0.01,
            cooldown=0,
            min_lr=0,
        ),
    ]

    logger.info(f"Training callbacks: {[type(c).__name__ for c in callbacks]}")
    return callbacks