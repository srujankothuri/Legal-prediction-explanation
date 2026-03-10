"""
Hierarchical Attention Layer for document classification.

Implements the attention mechanism from:
    "Hierarchical Attention Networks for Document Classification"
    Yang et al., 2016 — https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

The attention layer learns to assign importance weights to each chunk/sentence
representation, producing a single document-level vector via weighted sum.

Usage:
    from src.models.attention import AttentionLayer

    # In a Keras functional model:
    weighted_output = AttentionLayer(attention_dim=400)(gru_output)

    # With attention coefficients (for explainability):
    weighted_output, coefficients = AttentionLayer(
        attention_dim=400, return_coefficients=True
    )(gru_output)
"""

import tensorflow as tf
from tensorflow.keras import layers, initializers

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttentionLayer(layers.Layer):
    """
    Hierarchical Attention Layer.

    Computes attention weights over a sequence of vectors and returns
    their weighted sum as a fixed-length document representation.

    Architecture:
        u_it = tanh(W * h_it + b)         # project to attention space
        a_it = softmax(u_it^T * u)        # compute attention weights
        s = Σ (a_it * h_it)               # weighted sum

    Where:
        h_it: input hidden states (from BiGRU)
        W: weight matrix (input_dim → attention_dim)
        b: bias vector (attention_dim)
        u: context vector (attention_dim → 1)

    Args:
        attention_dim: Dimensionality of the attention projection space
        return_coefficients: If True, returns [weighted_sum, attention_weights]
    """

    def __init__(self, attention_dim=200, return_coefficients=False, **kwargs):
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Create trainable weights.

        Args:
            input_shape: (batch_size, timesteps, features)
        """
        assert len(input_shape) == 3, (
            f"AttentionLayer expects 3D input (batch, timesteps, features), "
            f"got shape: {input_shape}"
        )

        input_dim = input_shape[-1]

        # W: projects hidden states to attention space
        self.W = self.add_weight(
            name="W",
            shape=(input_dim, self.attention_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # b: bias for the projection
        self.b = self.add_weight(
            name="b",
            shape=(self.attention_dim,),
            initializer="zeros",
            trainable=True,
        )

        # u: context vector — learned query for "what is important"
        self.u = self.add_weight(
            name="u",
            shape=(self.attention_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        logger.debug(
            f"AttentionLayer built: input_dim={input_dim}, "
            f"attention_dim={self.attention_dim}, "
            f"return_coefficients={self.return_coefficients}"
        )

        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, inputs, input_mask=None):
        """Attention layer consumes the mask — downstream layers don't need it."""
        return None

    def call(self, hit, mask=None):
        """
        Forward pass.

        Args:
            hit: Input tensor of shape (batch, timesteps, features)
                 Typically output of a BiGRU layer.
            mask: Optional boolean mask of shape (batch, timesteps)

        Returns:
            If return_coefficients=False:
                weighted_sum: (batch, features)
            If return_coefficients=True:
                [weighted_sum, attention_weights]: [(batch, features), (batch, timesteps, 1)]
        """
        # Step 1: Project to attention space
        # uit = tanh(hit @ W + b)
        uit = tf.tanh(tf.add(tf.matmul(hit, self.W), self.b))

        # Step 2: Compute alignment scores
        # ait = uit @ u → (batch, timesteps, 1) → squeeze → (batch, timesteps)
        ait = tf.squeeze(tf.matmul(uit, self.u), axis=-1)

        # Step 3: Stable softmax with optional masking
        ait = tf.exp(ait)

        if mask is not None:
            ait *= tf.cast(mask, tf.float32)

        # Normalize to get attention weights (sum to 1 across timesteps)
        ait /= tf.cast(
            tf.reduce_sum(ait, axis=1, keepdims=True) + tf.keras.backend.epsilon(),
            tf.float32,
        )

        # Step 4: Expand dims for broadcasting and compute weighted sum
        ait_expanded = tf.expand_dims(ait, axis=-1)  # (batch, timesteps, 1)
        weighted_input = hit * ait_expanded  # element-wise multiply
        weighted_sum = tf.reduce_sum(weighted_input, axis=1)  # (batch, features)

        if self.return_coefficients:
            return [weighted_sum, ait_expanded]
        return weighted_sum

    def compute_output_shape(self, input_shape):
        """Compute output shape for Keras model building."""
        if self.return_coefficients:
            return [
                (input_shape[0], input_shape[-1]),      # weighted_sum
                (input_shape[0], input_shape[1], 1),    # attention_weights
            ]
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        """Serialize layer config for model saving/loading."""
        config = super(AttentionLayer, self).get_config()
        config.update({
            "attention_dim": self.attention_dim,
            "return_coefficients": self.return_coefficients,
        })
        return config