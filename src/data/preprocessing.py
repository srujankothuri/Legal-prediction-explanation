"""
Document preprocessing pipeline for transformer encoders.

Handles:
- Tokenization of long legal documents
- Splitting into overlapping chunks (sliding window)
- Attention mask generation
- Padding to fixed sequence length

Works with any HuggingFace tokenizer (XLNet, RoBERTa, BERT, DistilBERT).

Usage:
    from src.data.preprocessing import DocumentPreprocessor

    preprocessor = DocumentPreprocessor(tokenizer, config)
    chunks, masks = preprocessor.process_document(text)
    all_ids, all_masks, all_labels = preprocessor.process_dataset_for_finetuning(df)
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from keras.preprocessing.sequence import pad_sequences

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentPreprocessor:
    """
    Preprocesses legal documents for transformer encoder input.

    Long documents are split into overlapping chunks of `chunk_window` tokens
    with a stride of `chunk_stride`. Each chunk gets special tokens appended
    and is padded to `max_seq_length`.
    """

    def __init__(self, tokenizer, config: Dict[str, Any]):
        """
        Args:
            tokenizer: HuggingFace tokenizer instance
            config: Encoder config dict with keys:
                - max_tokens: Max tokens to keep per document (truncation)
                - chunk_window: Tokens per chunk before special tokens
                - chunk_stride: Stride between chunks
                - max_seq_length: Final padded length (typically 512)
                - name: Encoder name for model-specific logic
        """
        self.tokenizer = tokenizer
        self.max_tokens = config.get("max_tokens", 10000)
        self.chunk_window = config.get("chunk_window", 510)
        self.chunk_stride = config.get("chunk_stride", 410)
        self.max_seq_length = config.get("max_seq_length", 512)
        self.encoder_name = config.get("name", "xlnet")

        # Get special tokens
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.pad_token = tokenizer.pad_token

        logger.debug(
            f"DocumentPreprocessor initialized: "
            f"encoder={self.encoder_name}, "
            f"window={self.chunk_window}, "
            f"stride={self.chunk_stride}, "
            f"max_seq_len={self.max_seq_length}"
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text and truncate to max_tokens.
        Keeps the LAST max_tokens (recency bias for legal documents).

        Args:
            text: Raw document text

        Returns:
            List of token strings
        """
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) > self.max_tokens:
            logger.debug(
                f"Truncating document from {len(tokens)} to {self.max_tokens} tokens"
            )
            tokens = tokens[-self.max_tokens:]

        return tokens

    def create_chunks(self, tokens: List[str]) -> List[List[str]]:
        """
        Split token list into overlapping chunks using sliding window.

        Args:
            tokens: List of token strings

        Returns:
            List of token chunks
        """
        chunks = []
        left = 0
        right = self.chunk_window

        while left < len(tokens):
            chunk = tokens[left:min(right, len(tokens))]
            chunks.append(chunk)
            left += self.chunk_stride
            right += self.chunk_stride

        logger.debug(f"Created {len(chunks)} chunks from {len(tokens)} tokens")
        return chunks

    def encode_chunks(self, chunks: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add special tokens, convert to IDs, pad, and create attention masks.

        Args:
            chunks: List of token chunks

        Returns:
            Tuple of (input_ids, attention_masks) as numpy arrays
            Both have shape (n_chunks, max_seq_length)
        """
        encoded_chunks = []

        for chunk in chunks:
            # Add special tokens: [chunk tokens] + [SEP] + [CLS]
            chunk_with_special = chunk + [self.sep_token] + [self.cls_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(chunk_with_special)
            encoded_chunks.append(token_ids)

        # Pad all chunks to max_seq_length
        padded = pad_sequences(
            encoded_chunks,
            maxlen=self.max_seq_length,
            value=0,
            dtype="long",
            padding="pre"
        )

        # Create attention masks (1 for real tokens, 0 for padding)
        attention_masks = np.array(
            [[int(token_id > 0) for token_id in seq] for seq in padded]
        )

        return padded, attention_masks

    def process_document(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline for a single document.

        Args:
            text: Raw document text

        Returns:
            Tuple of (input_ids, attention_masks)
            Shape: (n_chunks, max_seq_length)
        """
        tokens = self.tokenize(text)
        chunks = self.create_chunks(tokens)
        input_ids, attention_masks = self.encode_chunks(chunks)
        return input_ids, attention_masks

    def process_dataset_for_finetuning(
        self, df, text_col: str = "text", label_col: str = "label"
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Process all documents for encoder fine-tuning.
        Each chunk inherits the document-level label.

        Args:
            df: DataFrame with text and label columns
            text_col: Name of text column
            label_col: Name of label column

        Returns:
            Tuple of (all_input_ids, all_attention_masks, all_labels)
            Flattened across all documents and chunks.
        """
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        total = len(df)
        for i in range(total):
            if i % 500 == 0 and i > 0:
                logger.info(f"Preprocessing document {i}/{total}")

            text = df[text_col].iloc[i]
            label = df[label_col].iloc[i]

            input_ids, attention_masks = self.process_document(text)

            # Each chunk gets the document-level label
            for j in range(len(input_ids)):
                all_input_ids.append(input_ids[j])
                all_attention_masks.append(attention_masks[j])
                all_labels.append(label)

        logger.info(
            f"Preprocessing complete: {total} documents → "
            f"{len(all_input_ids)} chunks"
        )

        return all_input_ids, all_attention_masks, all_labels

    def process_single_for_truncated_input(
        self, text: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single document taking only the last 510 tokens.
        Used for validation during fine-tuning (no chunking).

        Args:
            text: Raw document text

        Returns:
            Tuple of (input_ids, attention_mask) — single sequence
            Shape: (1, max_seq_length)
        """
        tokens = self.tokenizer.tokenize(text)

        # Take last 510 tokens
        if len(tokens) > self.chunk_window:
            tokens = tokens[-self.chunk_window:]

        tokens = tokens + [self.sep_token] + [self.cls_token]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        padded = pad_sequences(
            [token_ids],
            maxlen=self.max_seq_length,
            value=0,
            dtype="long",
            padding="pre",
            truncating="pre"
        )

        attention_mask = np.array(
            [[int(token_id > 0) for token_id in padded[0]]]
        )

        return padded, attention_mask