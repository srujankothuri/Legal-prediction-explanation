"""
Unified encoder fine-tuning trainer.

Handles the training loop for fine-tuning any supported transformer encoder
on the ILDC binary classification task.

Usage:
    from src.training.trainer import EncoderTrainer

    trainer = EncoderTrainer(config)
    trainer.train(train_df, val_df)
    trainer.save()
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.models.encoders import load_encoder, save_encoder
from src.data.preprocessing import DocumentPreprocessor
from src.utils.logger import get_logger
from src.utils.device import get_device

logger = get_logger(__name__)


class EncoderTrainer:
    """
    Fine-tunes a transformer encoder on ILDC for binary classification.

    Pipeline:
        1. Preprocess documents into overlapping chunks
        2. Each chunk inherits the document label
        3. Fine-tune encoder with AdamW + linear warmup schedule
        4. Validate periodically during training
        5. Save best model
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full model config with 'encoder' and 'fine_tuning' sections
        """
        self.config = config
        self.encoder_cfg = config["encoder"]
        self.ft_cfg = config["fine_tuning"]

        self.encoder_name = self.encoder_cfg["name"]
        self.device = get_device()

        # Load model and tokenizer from HuggingFace pretrained
        self.model, self.tokenizer = load_encoder(
            encoder_name=self.encoder_name,
            pretrained_path=self.encoder_cfg["pretrained"],
            num_labels=2,
            output_hidden_states=False,
            device=self.device,
        )

        # Preprocessor for tokenization and chunking
        self.preprocessor = DocumentPreprocessor(self.tokenizer, self.encoder_cfg)

        # Training state
        self.train_loss_history = []
        self.val_accuracy_history = []

        logger.info(f"EncoderTrainer initialized for: {self.encoder_name}")

    def train(self, train_df, val_df=None):
        """
        Fine-tune the encoder on training data.

        Args:
            train_df: Training DataFrame with 'text' and 'label' columns
            val_df: Optional validation DataFrame
        """
        # Step 1: Preprocess training data
        logger.info("Preprocessing training data...")
        train_ids, train_masks, train_labels = (
            self.preprocessor.process_dataset_for_finetuning(train_df)
        )
        logger.info(f"Training chunks: {len(train_ids)}")

        # Convert to tensors
        train_inputs = torch.tensor(np.array(train_ids), dtype=torch.long)
        train_masks_t = torch.tensor(np.array(train_masks), dtype=torch.long)
        train_labels_t = torch.tensor(train_labels, dtype=torch.long)

        # DataLoader
        batch_size = self.ft_cfg["batch_size"]
        train_data = TensorDataset(train_inputs, train_masks_t, train_labels_t)
        train_loader = DataLoader(
            train_data, sampler=RandomSampler(train_data), batch_size=batch_size
        )

        # Prepare validation if provided
        val_loader = None
        if val_df is not None:
            val_loader = self._prepare_validation_loader(val_df)

        # Optimizer and scheduler
        epochs = self.ft_cfg["epochs"]
        total_steps = len(train_loader) * epochs

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.ft_cfg["learning_rate"],
            
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.ft_cfg["warmup_steps"],
            num_training_steps=total_steps,
        )

        # Seed for reproducibility
        seed = self.ft_cfg.get("seed", 21)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Training loop
        logger.info(
            f"Starting training: {epochs} epochs, "
            f"{len(train_loader)} batches/epoch, "
            f"batch_size={batch_size}"
        )

        for epoch in range(epochs):
            epoch_start = time.time()
            logger.info(f"{'=' * 20} Epoch {epoch + 1}/{epochs} {'=' * 20}")

            epoch_loss = self._train_epoch(
                train_loader, optimizer, scheduler, val_loader, epoch
            )

            elapsed = time.time() - epoch_start
            self.train_loss_history.append(epoch_loss)
            logger.info(
                f"Epoch {epoch + 1} complete: "
                f"avg_loss={epoch_loss:.4f}, "
                f"time={elapsed:.1f}s"
            )

        logger.info("Training complete!")

    def _train_epoch(self, train_loader, optimizer, scheduler, val_loader, epoch):
        """Run one training epoch with fp16 mixed precision."""
        self.model.train()
        total_loss = 0
        log_interval = self.ft_cfg.get("log_every_n_steps", 100)
        val_interval = self.ft_cfg.get("validate_every_n_steps", 1000)

        # Mixed precision scaler
        scaler = GradScaler("cuda")

        for step, batch in enumerate(train_loader):
            # Log progress
            if step > 0 and step % log_interval == 0:
                avg_loss = total_loss / step
                logger.info(
                    f"  Step {step}/{len(train_loader)} — "
                    f"avg_loss: {avg_loss:.4f}"
                )

            # Forward pass
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            self.model.zero_grad()

            with autocast("cuda"):
                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                loss = outputs[0]

            total_loss += loss.item()

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.ft_cfg["max_grad_norm"],
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Periodic validation
            if val_loader and step > 0 and step % val_interval == 0:
                val_acc = self._validate(val_loader)
                self.val_accuracy_history.append(val_acc)
                logger.info(f"  Validation accuracy: {val_acc:.4f}")
                self.model.train()  # back to training mode

        return total_loss / len(train_loader)

    def _prepare_validation_loader(self, val_df):
        """Prepare validation DataLoader using truncated single-sequence inputs."""
        logger.info("Preprocessing validation data...")

        all_ids, all_masks, all_labels = [], [], []
        for i in range(len(val_df)):
            text = val_df["text"].iloc[i]
            label = val_df["label"].iloc[i]

            input_ids, att_mask = self.preprocessor.process_single_for_truncated_input(text)
            all_ids.append(input_ids[0])
            all_masks.append(att_mask[0])
            all_labels.append(label)

        val_inputs = torch.tensor(np.array(all_ids), dtype=torch.long)
        val_masks = torch.tensor(np.array(all_masks), dtype=torch.long)
        val_labels = torch.tensor(all_labels, dtype=torch.long)

        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_loader = DataLoader(
            val_data, sampler=RandomSampler(val_data), batch_size=self.ft_cfg["batch_size"]
        )

        logger.info(f"Validation sequences: {len(all_ids)}")
        return val_loader

    def _validate(self, val_loader):
        """Run validation and return accuracy."""
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                )

                logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                preds = torch.argmax(logits, dim=1)
                correct += (preds == b_labels).sum().item()
                total += len(b_labels)

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def save(self, output_dir: Optional[str] = None):
        """Save fine-tuned model and tokenizer."""
        save_encoder(self.model, self.tokenizer, self.encoder_name, output_dir)