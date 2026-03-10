"""
Tests for inference pipeline components.

Run: python -m pytest tests/test_inference.py -v
"""

import pytest
import numpy as np


class TestPredictionResult:

    def test_dataclass_creation(self):
        """PredictionResult can be created with all fields."""
        from src.inference.predictor import PredictionResult

        result = PredictionResult(
            prediction=True,
            label="accepted",
            probability=0.85,
            confidence=0.85,
            num_chunks=5,
            embedding_dim=3072,
        )

        assert result.prediction is True
        assert result.label == "accepted"
        assert result.probability == 0.85

    def test_to_dict(self):
        """PredictionResult can be converted to dictionary."""
        from src.inference.predictor import PredictionResult

        result = PredictionResult(
            prediction=False,
            label="rejected",
            probability=0.3,
            confidence=0.7,
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["prediction"] is False
        assert d["label"] == "rejected"
        assert d["confidence"] == 0.7

    def test_default_values(self):
        """Default values are set correctly."""
        from src.inference.predictor import PredictionResult

        result = PredictionResult(
            prediction=True,
            label="accepted",
            probability=0.9,
            confidence=0.9,
        )

        assert result.explanation == ""
        assert result.summary == ""
        assert result.chunk_scores == []
        assert result.processing_time_seconds == 0.0


class TestSummarizer:

    def test_cosine_similarity_matrix(self):
        """Similarity matrix is correct shape and symmetric."""
        from src.inference.summarizer import _cosine_similarity_matrix

        embeddings = np.random.randn(5, 768).astype(np.float32)
        sim_matrix = _cosine_similarity_matrix(embeddings)

        assert sim_matrix.shape == (5, 5)
        # Diagonal should be ~1.0 (self-similarity)
        np.testing.assert_allclose(np.diag(sim_matrix), 1.0, atol=1e-5)
        # Should be symmetric
        np.testing.assert_allclose(sim_matrix, sim_matrix.T, atol=1e-5)

    def test_cosine_similarity_with_zero_vector(self):
        """Handles zero vectors without crashing."""
        from src.inference.summarizer import _cosine_similarity_matrix

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # zero vector
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        sim_matrix = _cosine_similarity_matrix(embeddings)

        assert sim_matrix.shape == (3, 3)
        # Should not contain NaN
        assert not np.any(np.isnan(sim_matrix))

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors have zero similarity."""
        from src.inference.summarizer import _cosine_similarity_matrix

        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        sim_matrix = _cosine_similarity_matrix(embeddings)

        assert abs(sim_matrix[0][1]) < 1e-5  # orthogonal → 0 similarity


class TestExplainer:

    def test_sentence_markers(self):
        """Sentence markers are built correctly."""
        from src.inference.explainer import ExplanationGenerator

        # We just need the method, not a full model
        eg = ExplanationGenerator.__new__(ExplanationGenerator)

        tokenized_sents = [
            ["The", "court", "decided"],
            ["Appeal", "was", "filed"],
        ]

        markers = eg._build_sentence_markers(tokenized_sents)

        assert markers[0] == [1, 0, 0]      # first token marked as sentence 1
        assert markers[1] == [2, 0, 0]      # first token marked as sentence 2

    def test_detokenize(self):
        """XLNet detokenization produces readable text."""
        from src.inference.explainer import ExplanationGenerator

        eg = ExplanationGenerator.__new__(ExplanationGenerator)

        tokens = ["▁The", "▁court", "▁decided", "▁the", "▁case"]
        result = eg._detokenize(tokens)

        assert "court" in result
        assert "decided" in result

    def test_chunk_with_markers(self):
        """Chunking preserves token-marker alignment."""
        from src.inference.explainer import ExplanationGenerator

        eg = ExplanationGenerator.__new__(ExplanationGenerator)

        tokens = list(range(20))
        markers = list(range(20))

        chunked_toks, chunked_marks = eg._chunk_with_markers(
            tokens, markers, window=10, stride=8
        )

        # Both should have same number of chunks
        assert len(chunked_toks) == len(chunked_marks)

        # Each chunk should have matching lengths
        for t, m in zip(chunked_toks, chunked_marks):
            assert len(t) == len(m)