"""
Hierarchical Attention Network (HAN) classifier.

Takes pre-computed transformer chunk embeddings and classifies documents
using stacked Bidirectional GRUs with hierarchical attention.

Architecture:
    Input (n_chunks, embedding_dim) → Masking → 3x BiGRU → Attention → Dropout → Dense → Sigmoid

Usage:
    from src.models.han_classifier import HANClassifier

    # Build model
    han = HANClassifier(config)
    model = han.build()
    model.summary()

    # Load pre-trained weights
    han = HANClassifier(config, weights_path="trained_models/han_xlnet.h5")
    model = han.build()

    # Predict
    prediction, probability = han.predict_document(embeddings)

    # Predict with occlusion scores (for explainability)
    prediction, probability, chunk_scores = han.predict_with_occlusion(embeddings)
"""

import os
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras import layers

from src.models.attention import AttentionLayer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HANClassifier:
    """
    Hierarchical Attention Network for document-level binary classification.

    Takes variable-length sequences of chunk embeddings (from a fine-tuned
    transformer) and produces a binary prediction with attention coefficients.
    """

    def __init__(self, config: Dict[str, Any], weights_path: Optional[str] = None):
        """
        Args:
            config: Classifier config dict with keys:
                - gru_units: Hidden units per GRU direction (default: 200)
                - gru_layers: Number of stacked BiGRU layers (default: 3)
                - attention_dim: Attention projection dimension (default: 400)
                - dropout: Dropout rate after attention (default: 0.5)
                - dense_units: Units in pre-output dense layer (default: 64)
                - mask_value: Padding mask value (default: -99.0)
            weights_path: Optional path to pre-trained .h5 weights
        """
        # Extract config with defaults
        classifier_cfg = config.get("classifier", config)
        encoder_cfg = config.get("encoder", {})

        self.embedding_dim = encoder_cfg.get("embedding_dim", 3072)
        self.gru_units = classifier_cfg.get("gru_units", 200)
        self.gru_layers = classifier_cfg.get("gru_layers", 3)
        self.attention_dim = classifier_cfg.get("attention_dim", 400)
        self.dropout_rate = classifier_cfg.get("dropout", 0.5)
        self.dense_units = classifier_cfg.get("dense_units", 64)
        self.mask_value = classifier_cfg.get("mask_value", -99.0)
        self.optimizer = classifier_cfg.get("optimizer", "adam")
        self.loss = classifier_cfg.get("loss", "binary_crossentropy")

        self.weights_path = weights_path
        self.model = None

        logger.info(
            f"HANClassifier initialized: "
            f"emb_dim={self.embedding_dim}, "
            f"gru={self.gru_units}x{self.gru_layers}, "
            f"attn_dim={self.attention_dim}"
        )

    def build(self) -> Model:
        """
        Build the HAN model architecture.

        Returns:
            Compiled Keras Model
        """
        # Input: variable-length sequence of chunk embeddings
        text_input = Input(
            shape=(None, self.embedding_dim),
            dtype="float32",
            name="chunk_embeddings",
        )

        # Masking: ignore padded chunks (filled with mask_value)
        x = layers.Masking(mask_value=self.mask_value)(text_input)

        # Stacked Bidirectional GRU layers
        for i in range(self.gru_layers):
            x = Bidirectional(
                GRU(self.gru_units, return_sequences=True),
                name=f"bigru_{i + 1}",
            )(x)

        # Hierarchical Attention with coefficients for explainability
        attention_output, attention_coefficients = AttentionLayer(
            attention_dim=self.attention_dim,
            return_coefficients=True,
            name="attention",
        )(x)

        # Classification head
        x = Dropout(self.dropout_rate, name="dropout")(attention_output)
        x = Dense(self.dense_units, activation="relu", name="dense")(x)
        output = Dense(1, activation="sigmoid", name="prediction")(x)

        model = Model(inputs=text_input, outputs=output, name="HAN_Classifier")

        # Compile
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=["acc"],
        )

        # Load weights if provided
        if self.weights_path:
            if os.path.exists(self.weights_path):
                model.load_weights(self.weights_path)
                logger.info(f"Loaded HAN weights from: {self.weights_path}")
            else:
                logger.warning(f"Weights file not found: {self.weights_path}")

        param_count = model.count_params()
        logger.info(f"HAN model built: {param_count / 1e6:.2f}M parameters")

        self.model = model
        return model

    def predict_document(
        self, embeddings: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Predict judgment for a single document.

        Args:
            embeddings: Document chunk embeddings, shape (n_chunks, embedding_dim)

        Returns:
            Tuple of (prediction_bool, probability)
            prediction_bool: True = accepted, False = rejected
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")

        x = self._prepare_single_input(embeddings)
        prob = float(self.model.predict(x, verbose=0)[0][0])
        prediction = prob > 0.5

        label_str = "accepted" if prediction else "rejected"
        logger.info(f"Prediction: {label_str} (probability: {prob:.4f})")

        return prediction, prob

    def predict_with_occlusion(
        self, embeddings: np.ndarray
    ) -> Tuple[bool, float, List[float]]:
        """
        Predict with occlusion-based chunk importance scoring.

        For each chunk, masks it with zeros and measures how much the
        prediction confidence drops. Higher drop = more important chunk.

        Args:
            embeddings: Document chunk embeddings, shape (n_chunks, embedding_dim)

        Returns:
            Tuple of (prediction, probability, chunk_scores)
            chunk_scores: List of importance scores per chunk
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")

        # Get base prediction
        prediction, base_prob = self.predict_document(embeddings)

        # Score each chunk by occlusion
        chunk_scores = []
        n_chunks = len(embeddings)

        logger.info(f"Computing occlusion scores for {n_chunks} chunks...")

        for j in range(n_chunks):
            # Create occluded version: replace chunk j with zeros
            occluded = np.copy(embeddings)
            occluded[j] = np.zeros(embeddings[j].shape, dtype=np.float32)

            # Get prediction without this chunk
            x = self._prepare_single_input(occluded)
            occ_prob = float(self.model.predict(x, verbose=0)[0][0])

            # Score = drop in confidence when chunk is removed
            if prediction:
                score = base_prob - occ_prob  # positive = chunk supports acceptance
            else:
                score = occ_prob - base_prob  # positive = chunk supports rejection

            chunk_scores.append(score)

        logger.info(
            f"Occlusion scores computed: "
            f"max={max(chunk_scores):.4f}, "
            f"min={min(chunk_scores):.4f}, "
            f"mean={np.mean(chunk_scores):.4f}"
        )

        return prediction, base_prob, chunk_scores

    def _prepare_single_input(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Prepare a single document's embeddings for model input.
        Adds batch dimension and handles variable-length padding.

        Args:
            embeddings: Shape (n_chunks, embedding_dim)

        Returns:
            Batched input array, shape (1, n_chunks, embedding_dim)
        """
        embeddings = np.array(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Pad feature dimension if needed (e.g., 768 → 3072)
        if embeddings.shape[-1] < self.embedding_dim:
            padding_width = self.embedding_dim - embeddings.shape[-1]
            embeddings = np.pad(
                embeddings, ((0, 0), (0, padding_width)), mode="constant"
            )

        # Add batch dimension
        return np.expand_dims(embeddings, axis=0)

    def save(self, save_path: str):
        """
        Save model weights.

        Args:
            save_path: Path to save .h5 file
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        logger.info(f"HAN model saved to: {save_path}")

    def summary(self):
        """Print model architecture summary."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")
        self.model.summary()