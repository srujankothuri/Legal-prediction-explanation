"""
Occlusion-based sentence-level explanation generator.

For each important chunk, masks individual sentences and measures the drop
in prediction confidence to identify the most influential sentences.

Usage:
    from src.inference.explainer import ExplanationGenerator

    explainer = ExplanationGenerator(model, tokenizer, device)
    explanation = explainer.generate(text, chunk_scores, predicted_label)
"""

import itertools
import numpy as np
from typing import List, Optional

import torch
import nltk
from keras.preprocessing.sequence import pad_sequences

from src.utils.logger import get_logger

logger = get_logger(__name__)

# NLTK sentence tokenizer (lazy-loaded to avoid import-time failures)
_SENT_TOKENIZER = None


def _get_sent_tokenizer():
    """Lazy-load NLTK sentence tokenizer."""
    global _SENT_TOKENIZER
    if _SENT_TOKENIZER is None:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            logger.warning("Failed to download punkt_tab via NLTK downloader")
        try:
            _SENT_TOKENIZER = nltk.data.load("tokenizers/punkt_tab/english.pickle")
        except LookupError:
            try:
                _SENT_TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")
            except LookupError:
                logger.warning("NLTK punkt tokenizer not found, using fallback split")
                _SENT_TOKENIZER = None
    return _SENT_TOKENIZER


class ExplanationGenerator:
    """
    Generates sentence-level explanations for judgment predictions.

    Method: For each positively-scored chunk, omit each sentence one at a time
    and measure how much the encoder's prediction score drops. Sentences whose
    removal causes the largest drop are the most important.
    """

    def __init__(self, model, tokenizer, device):
        """
        Args:
            model: Fine-tuned transformer encoder (with output_hidden_states=True)
            tokenizer: Corresponding tokenizer
            device: torch device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        text: str,
        chunk_scores: List[float],
        predicted_label: int,
        top_k_ratio: float = 0.1,
        max_tokens: int = 10000,
    ) -> str:
        """
        Generate explanation text from the most important sentences.

        Args:
            text: Full document text
            chunk_scores: Per-chunk importance scores from HAN occlusion
            predicted_label: 0 (rejected) or 1 (accepted)
            top_k_ratio: Fraction of top sentences to extract per chunk
            max_tokens: Max tokens to consider from document

        Returns:
            Explanation string (concatenated important sentences)
        """
        logger.info(
            f"Generating explanation: {len(chunk_scores)} chunks, "
            f"label={predicted_label}, top_k_ratio={top_k_ratio}"
        )

        # Sentence tokenization
        sent_tokenizer = _get_sent_tokenizer()
        if sent_tokenizer:
            sentences = sent_tokenizer.tokenize(text)
        else:
            # Fallback: split on period + space
            sentences = [s.strip() for s in text.split(". ") if s.strip()]
        logger.debug(f"Document has {len(sentences)} sentences")

        # Tokenize each sentence with XLNet tokenizer
        tokenized_sents = [self.tokenizer.tokenize(s) for s in sentences]
        marked_sents = self._build_sentence_markers(tokenized_sents)

        # Flatten tokens and markers
        all_tokens = list(itertools.chain.from_iterable(tokenized_sents))
        all_markers = list(itertools.chain.from_iterable(marked_sents))

        # Truncate to last max_tokens
        if len(all_tokens) > max_tokens:
            all_tokens = all_tokens[-max_tokens:]
            all_markers = all_markers[-max_tokens:]

        # Create overlapping chunks (matching embedding generation)
        chunked_tokens, chunked_markers = self._chunk_with_markers(all_tokens, all_markers)

        # Special tokens
        CLS = self.tokenizer.cls_token
        SEP = self.tokenizer.sep_token
        PAD = self.tokenizer.pad_token

        explanation_sentences = []

        for chunk_num, score in enumerate(chunk_scores):
            # Skip chunks that don't exist or have negative scores
            if chunk_num >= len(chunked_markers) or score < 0:
                continue

            # Mark sentence boundaries in this chunk
            cm = chunked_markers[chunk_num]
            if chunk_num == 0:
                if len(cm) > 0:
                    cm[0] = -777
                if len(cm) > 1:
                    cm[-1] = 777
            else:
                if len(cm) < 101:
                    continue
                cm[100] = -777
                cm[-1] = 777

            # Count sentences in chunk
            n_sents = sum(1 for m in cm if m not in (0, -777, 777))
            if n_sents <= 0:
                continue

            top_k = max(1, int(top_k_ratio * n_sents))

            # Find sentence boundaries
            boundaries = self._find_sentence_boundaries(cm)

            # Get original score for this chunk
            orig_encoded = self.tokenizer.convert_tokens_to_ids(
                chunked_tokens[chunk_num] + [SEP] + [CLS]
            )
            orig_logits = self._get_logits(orig_encoded)
            orig_score = float(orig_logits[0][predicted_label])

            # Score each sentence by omission
            sent_scores = {}
            for start, end in boundaries:
                if start == -1000:
                    start = 0
                length = end - start + 1
                if length <= 0:
                    continue

                # Replace sentence tokens with PAD
                pad_tokens = [PAD] * length
                left = chunked_tokens[chunk_num][:start]
                right = chunked_tokens[chunk_num][end + 1:]
                masked_seq = left + pad_tokens + right + [SEP] + [CLS]

                encoded = self.tokenizer.convert_tokens_to_ids(masked_seq)
                logits = self._get_logits(encoded)
                masked_score = float(logits[0][predicted_label])

                # Importance = drop in confidence when sentence removed
                if masked_score > orig_score:
                    importance = -(masked_score - orig_score)
                else:
                    importance = orig_score - masked_score

                # Normalize by sentence length
                importance_norm = importance / length
                sentence_text = self._detokenize(
                    chunked_tokens[chunk_num][start:end + 1]
                )
                sent_scores[sentence_text] = importance_norm

            # Take top-k sentences from this chunk
            sorted_sents = sorted(sent_scores.items(), key=lambda x: x[1], reverse=True)
            for sent_text, _ in sorted_sents[:top_k]:
                explanation_sentences.append(sent_text)

        explanation = " ".join(explanation_sentences)
        logger.info(f"Explanation generated: {len(explanation_sentences)} sentences")
        return explanation

    def _get_logits(self, encoded_ids: List[int]) -> torch.Tensor:
        """Get model logits for an encoded sequence."""
        padded = pad_sequences(
            [encoded_ids], maxlen=512, value=0, dtype="long", padding="pre"
        )
        masks = [[int(tok > 0) for tok in padded[0]]]

        input_ids = torch.tensor(padded[0], dtype=torch.long).unsqueeze(0).to(self.device)
        att_masks = torch.tensor(masks[0], dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, token_type_ids=None, attention_mask=att_masks
            )

        return outputs.logits

    def _build_sentence_markers(self, tokenized_sents):
        """Create marker arrays: first token of each sentence gets sentence number."""
        markers = []
        for sent_num, tokens in enumerate(tokenized_sents, 1):
            sent_markers = [sent_num if i == 0 else 0 for i in range(len(tokens))]
            markers.append(sent_markers)
        return markers

    def _chunk_with_markers(self, all_toks, markers, window=510, stride=410):
        """Split tokens and markers into overlapping chunks."""
        chunked_toks, chunked_markers = [], []
        left, right = 0, window
        while left < len(all_toks):
            chunked_toks.append(all_toks[left:min(right, len(all_toks))])
            chunked_markers.append(markers[left:min(right, len(markers))])
            left += stride
            right += stride
        return chunked_toks, chunked_markers

    def _find_sentence_boundaries(self, marks):
        """Find (start, end) index pairs for each sentence in a chunk."""
        pairs = []
        st = -1000
        for i, mark in enumerate(marks):
            if mark == -777:
                st = i
            elif mark not in (-777, 777, 0):
                pairs.append((st, i - 1))
                st = i
            elif mark == 777:
                pairs.append((st, i))
        return pairs

    def _detokenize(self, tokens):
        """Convert XLNet/SentencePiece tokens back to readable text."""
        if not tokens:
            return ""
        tokens = list(tokens)  # copy
        tokens[0] = "▁" + tokens[0]
        words = []
        word = ""
        for tok in tokens:
            if tok.startswith("▁"):
                if word:
                    words.append(word)
                word = tok[1:]
            else:
                word += tok
        if word:
            words.append(word)
        return " ".join(words)