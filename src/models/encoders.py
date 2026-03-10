"""
Transformer encoder loading and management.

Provides a unified interface to load any supported encoder
(XLNet, RoBERTa, BERT, DistilBERT) for fine-tuning or inference.

Usage:
    from src.models.encoders import load_encoder

    model, tokenizer = load_encoder("xlnet", "xlnet-base-cased")
    model, tokenizer = load_encoder("roberta", "roberta-base")
    model, tokenizer = load_encoder_from_config(config)
"""

import os
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import (
    XLNetForSequenceClassification, XLNetTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    BertForSequenceClassification, BertTokenizer,
    DistilBertForSequenceClassification, DistilBertTokenizer,
    PreTrainedModel, PreTrainedTokenizer,
)

from src.utils.logger import get_logger
from src.utils.device import get_device

logger = get_logger(__name__)


# ── Supported Model Registry ───────────────────────────────────────────────

ENCODER_REGISTRY = {
    "xlnet": {
        "model_class": XLNetForSequenceClassification,
        "tokenizer_class": XLNetTokenizer,
        "default_pretrained": "xlnet-base-cased",
    },
    "roberta": {
        "model_class": RobertaForSequenceClassification,
        "tokenizer_class": RobertaTokenizer,
        "default_pretrained": "roberta-base",
    },
    "bert": {
        "model_class": BertForSequenceClassification,
        "tokenizer_class": BertTokenizer,
        "default_pretrained": "bert-base-uncased",
    },
    "distilbert": {
        "model_class": DistilBertForSequenceClassification,
        "tokenizer_class": DistilBertTokenizer,
        "default_pretrained": "distilbert-base-uncased",
    },
}


def get_supported_encoders():
    """Return list of supported encoder names."""
    return list(ENCODER_REGISTRY.keys())


def load_encoder(
    encoder_name: str,
    pretrained_path: str,
    num_labels: int = 2,
    output_hidden_states: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a transformer encoder model and tokenizer.

    Args:
        encoder_name: One of 'xlnet', 'roberta', 'bert', 'distilbert'
        pretrained_path: HuggingFace model name or local directory path
        num_labels: Number of classification labels (2 for binary)
        output_hidden_states: Whether to return all hidden layer outputs
        device: Target device (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If encoder_name is not supported
    """
    if encoder_name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unsupported encoder: '{encoder_name}'. "
            f"Supported: {get_supported_encoders()}"
        )

    registry = ENCODER_REGISTRY[encoder_name]
    model_class = registry["model_class"]
    tokenizer_class = registry["tokenizer_class"]

    if device is None:
        device = get_device()

    logger.info(f"Loading encoder: {encoder_name} from '{pretrained_path}'")

    # Load tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_path)
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Load model
    model = model_class.from_pretrained(
        pretrained_path,
        num_labels=num_labels,
        output_hidden_states=output_hidden_states,
    )
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model loaded: {param_count / 1e6:.1f}M params "
        f"({trainable_count / 1e6:.1f}M trainable) → {device}"
    )

    return model, tokenizer


def load_encoder_from_config(
    config: Dict[str, Any],
    for_inference: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load encoder using a config dictionary.

    For training: loads from HuggingFace pretrained weights.
    For inference: loads from local fine-tuned directory.

    Args:
        config: Config dict with 'encoder' section
        for_inference: If True, loads from trained_models/ directory
        device: Target device (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    encoder_cfg = config["encoder"]
    encoder_name = encoder_cfg["name"]

    if for_inference:
        # Load from local fine-tuned model directory
        model_dir = os.path.join("trained_models", f"{encoder_name}_finetuned")
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Fine-tuned model not found at: {model_dir}. "
                f"Run training first with: make train-encoder MODEL={encoder_name}"
            )
        pretrained_path = model_dir
        output_hidden = True  # need hidden states for embedding extraction
    else:
        # Load from HuggingFace for fine-tuning
        pretrained_path = encoder_cfg["pretrained"]
        output_hidden = False

    return load_encoder(
        encoder_name=encoder_name,
        pretrained_path=pretrained_path,
        output_hidden_states=output_hidden,
        device=device,
    )


def save_encoder(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    encoder_name: str,
    output_dir: Optional[str] = None,
):
    """
    Save fine-tuned encoder model and tokenizer.

    Args:
        model: Fine-tuned model to save
        tokenizer: Tokenizer to save
        encoder_name: Name for directory (e.g., 'xlnet')
        output_dir: Override output directory
    """
    if output_dir is None:
        output_dir = os.path.join("trained_models", f"{encoder_name}_finetuned")

    os.makedirs(output_dir, exist_ok=True)

    # Handle DataParallel wrapper
    model_to_save = model.module if hasattr(model, "module") else model

    # Ensure contiguous tensors (required for some models like XLNet)
    for name, param in model_to_save.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Encoder saved to: {output_dir}")