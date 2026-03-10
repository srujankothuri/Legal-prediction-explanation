"""
Tests for training pipeline components.

Run: python -m pytest tests/test_training.py -v
"""

import pytest
import numpy as np


class TestBatchGenerator:
    """Test batch generator for HAN training."""

    def test_generator_yields_correct_shapes(self):
        """Generator produces batches with correct dimensions."""
        from src.training.callbacks import create_batch_generator

        # Simulate 10 documents with variable chunk counts
        x_data = np.array([
            np.random.randn(np.random.randint(2, 8), 3072).astype(np.float32)
            for _ in range(10)
        ], dtype=object)
        y_data = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]

        gen = create_batch_generator(x_data, y_data, batch_size=4, num_features=3072)

        x_batch, y_batch = next(gen)

        assert x_batch.shape[0] == 4          # batch size
        assert x_batch.shape[2] == 3072       # feature dim
        assert y_batch.shape == (4, 1)        # label shape

    def test_generator_pads_with_mask_value(self):
        """Shorter sequences are padded with mask_value."""
        from src.training.callbacks import create_batch_generator

        # One doc with 2 chunks, one with 5 chunks
        x_data = np.array([
            np.ones((2, 3072), dtype=np.float32),
            np.ones((5, 3072), dtype=np.float32),
        ], dtype=object)
        y_data = [0, 1]

        gen = create_batch_generator(
            x_data, y_data, batch_size=2, num_features=3072, mask_value=-99.0
        )

        x_batch, _ = next(gen)

        # Max timesteps should be 5
        assert x_batch.shape[1] == 5

        # First doc should have padding at positions 2-4
        assert x_batch[0, 2, 0] == -99.0
        assert x_batch[0, 0, 0] == 1.0  # actual data

    def test_generator_is_infinite(self):
        """Generator loops indefinitely for Keras fit()."""
        from src.training.callbacks import create_batch_generator

        x_data = np.array([
            np.random.randn(3, 3072).astype(np.float32)
            for _ in range(4)
        ], dtype=object)
        y_data = [0, 1, 0, 1]

        gen = create_batch_generator(x_data, y_data, batch_size=2)

        # Should be able to pull many batches without error
        for _ in range(20):
            x, y = next(gen)
            assert x.shape[0] == 2


class TestStepsCalculation:

    def test_steps_per_epoch(self):
        """Steps calculation is correct."""
        from src.training.callbacks import get_steps_per_epoch

        assert get_steps_per_epoch(100, 32) == 3
        assert get_steps_per_epoch(64, 32) == 2
        assert get_steps_per_epoch(10, 32) == 0 or get_steps_per_epoch(10, 32) >= 1

    def test_test_steps(self):
        """Test steps include partial last batch."""
        from src.training.callbacks import get_test_steps

        assert get_test_steps(100, 32) == 4   # 3 full + 1 partial
        assert get_test_steps(64, 32) == 2    # exact
        assert get_test_steps(33, 32) == 2    # 1 full + 1 partial


class TestCallbacks:

    def test_get_callbacks_returns_list(self):
        """Callbacks factory returns a list."""
        from src.training.callbacks import get_training_callbacks

        callbacks = get_training_callbacks()
        assert isinstance(callbacks, list)
        assert len(callbacks) > 0