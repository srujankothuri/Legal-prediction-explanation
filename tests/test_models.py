"""
Tests for model architectures.

Run: python -m pytest tests/test_models.py -v
"""

import pytest
import numpy as np


# ── Attention Layer Tests ───────────────────────────────────────────────────

class TestAttentionLayer:

    def test_output_shape_without_coefficients(self):
        """Attention layer produces correct output shape."""
        from src.models.attention import AttentionLayer
        import tensorflow as tf

        layer = AttentionLayer(attention_dim=100, return_coefficients=False)
        # Simulate input: batch=2, timesteps=5, features=200
        dummy_input = tf.random.normal((2, 5, 200))
        output = layer(dummy_input)

        assert output.shape == (2, 200)

    def test_output_shape_with_coefficients(self):
        """Attention layer returns both output and coefficients."""
        from src.models.attention import AttentionLayer
        import tensorflow as tf

        layer = AttentionLayer(attention_dim=100, return_coefficients=True)
        dummy_input = tf.random.normal((2, 5, 200))
        weighted_sum, coefficients = layer(dummy_input)

        assert weighted_sum.shape == (2, 200)
        assert coefficients.shape == (2, 5, 1)

    def test_attention_weights_sum_to_one(self):
        """Attention coefficients should approximately sum to 1."""
        from src.models.attention import AttentionLayer
        import tensorflow as tf

        layer = AttentionLayer(attention_dim=50, return_coefficients=True)
        dummy_input = tf.random.normal((1, 10, 100))
        _, coefficients = layer(dummy_input)

        # Squeeze and check sum
        weights = coefficients.numpy().squeeze()
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_serialization(self):
        """Layer config can be serialized and deserialized."""
        from src.models.attention import AttentionLayer

        layer = AttentionLayer(attention_dim=300, return_coefficients=True)
        config = layer.get_config()

        assert config["attention_dim"] == 300
        assert config["return_coefficients"] is True

    def test_different_attention_dims(self):
        """Layer works with various attention dimensions."""
        from src.models.attention import AttentionLayer
        import tensorflow as tf

        for dim in [50, 100, 200, 400]:
            layer = AttentionLayer(attention_dim=dim)
            output = layer(tf.random.normal((1, 3, 128)))
            assert output.shape == (1, 128)


# ── HAN Classifier Tests ───────────────────────────────────────────────────

class TestHANClassifier:

    @pytest.fixture
    def default_config(self):
        """Minimal config for testing."""
        return {
            "encoder": {"embedding_dim": 3072},
            "classifier": {
                "gru_units": 32,        # small for fast tests
                "gru_layers": 2,
                "attention_dim": 64,
                "dropout": 0.3,
                "dense_units": 16,
                "mask_value": -99.0,
            },
        }

    def test_build_model(self, default_config):
        """Model builds without errors."""
        from src.models.han_classifier import HANClassifier

        han = HANClassifier(default_config)
        model = han.build()

        assert model is not None
        assert model.output_shape == (None, 1)

    def test_predict_document(self, default_config):
        """Prediction returns bool and probability."""
        from src.models.han_classifier import HANClassifier

        han = HANClassifier(default_config)
        han.build()

        # Simulate 5 chunks of 3072-dim embeddings
        fake_embeddings = np.random.randn(5, 3072).astype(np.float32)
        prediction, probability = han.predict_document(fake_embeddings)

        assert isinstance(prediction, bool)
        assert 0.0 <= probability <= 1.0

    def test_predict_with_occlusion(self, default_config):
        """Occlusion scoring returns correct number of scores."""
        from src.models.han_classifier import HANClassifier

        han = HANClassifier(default_config)
        han.build()

        n_chunks = 4
        fake_embeddings = np.random.randn(n_chunks, 3072).astype(np.float32)
        prediction, prob, scores = han.predict_with_occlusion(fake_embeddings)

        assert isinstance(prediction, bool)
        assert len(scores) == n_chunks

    def test_predict_without_build_raises(self, default_config):
        """Calling predict before build() raises RuntimeError."""
        from src.models.han_classifier import HANClassifier

        han = HANClassifier(default_config)
        fake_embeddings = np.random.randn(3, 3072).astype(np.float32)

        with pytest.raises(RuntimeError, match="not built"):
            han.predict_document(fake_embeddings)

    def test_single_chunk_input(self, default_config):
        """Model handles single-chunk documents."""
        from src.models.han_classifier import HANClassifier

        han = HANClassifier(default_config)
        han.build()

        single_chunk = np.random.randn(1, 3072).astype(np.float32)
        prediction, prob = han.predict_document(single_chunk)

        assert isinstance(prediction, bool)


# ── Encoder Registry Tests ──────────────────────────────────────────────────

class TestEncoderRegistry:

    def test_supported_encoders(self):
        """All expected encoders are registered."""
        from src.models.encoders import get_supported_encoders

        supported = get_supported_encoders()
        assert "xlnet" in supported
        assert "roberta" in supported
        assert "bert" in supported
        assert "distilbert" in supported

    def test_unsupported_encoder_raises(self):
        """Loading unsupported encoder raises ValueError."""
        from src.models.encoders import load_encoder

        with pytest.raises(ValueError, match="Unsupported encoder"):
            load_encoder("gpt4", "gpt4-turbo")