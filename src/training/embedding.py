"""
Embedding generation from fine-tuned transformer encoders.

Takes a fine-tuned encoder and generates document-level embeddings by:
1. Tokenizing and chunking the document
2. Passing each chunk through the encoder
3. Extracting and concatenating the last N hidden layer outputs
4. Saving as .npy files for HAN training

Usage:
    from src.training.embeddings import EmbeddingGenerator

    generator = EmbeddingGenerator(config)
    embeddings = generator.generate_for_text(text)
    generator.generate_for_dataset(dataset, split="train", output_dir="data/embeddings/xlnet")
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List
from tqdm import tqdm

import torch

from src.models.encoders import load_encoder_from_config
from src.data.preprocessing import DocumentPreprocessor
from src.utils.logger import get_logger
from src.utils.device import get_device

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generates document embeddings from a fine-tuned transformer encoder.

    For each document:
        1. Tokenize and split into overlapping chunks
        2. Feed each chunk through the encoder
        3. Concatenate last N hidden layers → 768 * N dimensional vector per chunk
        4. Document representation = array of chunk vectors (variable length)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full model config with 'encoder' section
        """
        self.config = config
        self.encoder_cfg = config["encoder"]
        self.encoder_name = self.encoder_cfg["name"]
        self.n_layers_concat = self.encoder_cfg.get("num_layers_concat", 4)
        self.embedding_dim = self.encoder_cfg.get("embedding_dim", 3072)
        self.device = get_device()

        # Load fine-tuned model
        logger.info(f"Loading fine-tuned {self.encoder_name} for embedding generation...")
        self.model, self.tokenizer = load_encoder_from_config(
            config, for_inference=True, device=self.device
        )
        self.model.eval()

        # Preprocessor
        self.preprocessor = DocumentPreprocessor(self.tokenizer, self.encoder_cfg)

        logger.info(
            f"EmbeddingGenerator ready: {self.encoder_name}, "
            f"concat_layers={self.n_layers_concat}, "
            f"embedding_dim={self.embedding_dim}"
        )

    def _extract_chunk_embedding(self, input_id: np.ndarray, att_mask: np.ndarray) -> np.ndarray:
        """
        Extract concatenated hidden states from the last N layers for one chunk.

        Args:
            input_id: Token IDs, shape (max_seq_length,)
            att_mask: Attention mask, shape (max_seq_length,)

        Returns:
            Embedding vector, shape (embedding_dim,) — e.g., (3072,) for 4 layers
        """
        input_ids = torch.tensor(input_id, dtype=torch.long).unsqueeze(0).to(self.device)
        att_masks = torch.tensor(att_mask, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                token_type_ids=None,
                attention_mask=att_masks,
            )

        # Extract hidden states
        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_size)
        total_layers = len(hidden_states)
        n_layers = min(self.n_layers_concat, total_layers)

        # Concatenate last N layers, take the last token's representation
        layer_vecs = [hidden_states[-(i + 1)][0][-1] for i in range(n_layers)]
        embedding = torch.cat(layer_vecs, dim=0).detach().cpu().numpy()

        return embedding

    def generate_for_text(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a single document text.

        Args:
            text: Raw document text

        Returns:
            Array of chunk embeddings, shape (n_chunks, embedding_dim)
        """
        # Tokenize and chunk
        input_ids, att_masks = self.preprocessor.process_document(text)

        # Extract embedding for each chunk
        chunk_embeddings = []
        for i in range(len(input_ids)):
            emb = self._extract_chunk_embedding(input_ids[i], att_masks[i])
            chunk_embeddings.append(emb)

        return np.array(chunk_embeddings)

    def generate_for_dataset(
        self,
        dataset,
        split: str,
        output_dir: Optional[str] = None,
        batch_label: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate embeddings for all documents in a dataset split.

        Args:
            dataset: ILDCDataset instance
            split: One of 'train', 'dev', 'test'
            output_dir: Directory to save .npy file (None = don't save)
            batch_label: Optional suffix for the output filename

        Returns:
            Array of document embeddings (object array, variable-length inner arrays)
        """
        texts = dataset.get_texts(split)
        logger.info(f"Generating embeddings for {len(texts)} {split} documents...")

        all_doc_embeddings = []

        for i in tqdm(range(len(texts)), desc=f"Embeddings ({split})"):
            doc_emb = self.generate_for_text(texts[i])
            all_doc_embeddings.append(doc_emb)

            if (i + 1) % 100 == 0:
                logger.info(
                    f"  Progress: {i + 1}/{len(texts)} docs, "
                    f"latest shape: {doc_emb.shape}"
                )

        all_doc_embeddings = np.array(all_doc_embeddings, dtype=object)

        # Save to disk
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            suffix = f"_{batch_label}" if batch_label else ""
            filename = f"{self.encoder_name}_{split}{suffix}.npy"
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, all_doc_embeddings)
            logger.info(f"Embeddings saved to: {save_path}")

        logger.info(
            f"Embedding generation complete for {split}: "
            f"{len(all_doc_embeddings)} documents"
        )

        return all_doc_embeddings

    def generate_all_splits(self, dataset, output_dir: str):
        """
        Generate and save embeddings for all dataset splits.

        Args:
            dataset: ILDCDataset instance
            output_dir: Directory to save .npy files
        """
        for split in ["train", "dev", "test"]:
            self.generate_for_dataset(dataset, split, output_dir)

        logger.info(f"All embeddings saved to: {output_dir}")