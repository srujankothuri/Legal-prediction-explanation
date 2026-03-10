"""
Tests for Streamlit app components.

Run: python -m pytest tests/test_app.py -v
"""

import pytest


class TestModelSelector:

    def test_models_dict_has_all_encoders(self):
        """All four encoder types are in the selector."""
        from app.components.model_selector import MODELS

        assert "xlnet" in MODELS
        assert "roberta" in MODELS
        assert "bert" in MODELS
        assert "distilbert" in MODELS

    def test_models_display_names(self):
        """Display names include encoder and classifier info."""
        from app.components.model_selector import MODELS

        for key, name in MODELS.items():
            assert "BiGRU-HAN" in name


class TestResultDisplay:

    def test_prediction_result_accepted(self):
        """Result display handles accepted prediction."""
        from src.inference.predictor import PredictionResult

        result = PredictionResult(
            prediction=True,
            label="accepted",
            probability=0.85,
            confidence=0.85,
            num_chunks=10,
            embedding_dim=3072,
            encoder_name="xlnet",
            processing_time_seconds=5.2,
        )

        assert result.label == "accepted"
        assert result.confidence == 0.85

    def test_prediction_result_rejected(self):
        """Result display handles rejected prediction."""
        from src.inference.predictor import PredictionResult

        result = PredictionResult(
            prediction=False,
            label="rejected",
            probability=0.25,
            confidence=0.75,
            num_chunks=8,
            embedding_dim=3072,
            encoder_name="roberta",
            processing_time_seconds=4.8,
        )

        assert result.label == "rejected"
        assert result.confidence == 0.75

    def test_prediction_result_with_explanation(self):
        """Result handles explanation and chunk scores."""
        from src.inference.predictor import PredictionResult

        result = PredictionResult(
            prediction=True,
            label="accepted",
            probability=0.9,
            confidence=0.9,
            explanation="The court found that the evidence was sufficient.",
            chunk_scores=[0.1, 0.3, -0.05, 0.2],
            summary="Appeal was accepted based on evidence review.",
        )

        assert len(result.explanation) > 0
        assert len(result.chunk_scores) == 4
        assert len(result.summary) > 0