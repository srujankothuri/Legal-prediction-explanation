"""
Extractive summarization using InLegalBERT.

Selects the most representative sentences by computing pairwise
cosine similarity of sentence embeddings and picking those with
the highest total similarity (most central sentences).

Usage:
    from src.inference.summarizer import generate_extractive_summary

    from transformers import AutoTokenizer, AutoModel
    model = AutoModel.from_pretrained("law-ai/InLegalBERT")
    tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")

    summary = generate_extractive_summary(model, tokenizer, text)
"""

import numpy as np
from typing import Optional

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_extractive_summary(
    model,
    tokenizer,
    text: str,
    percentage: float = 0.08,
    max_sentences: int = 50,
    min_sentence_words: int = 5,
) -> str:
    """
    Generate extractive summary using InLegalBERT sentence embeddings.

    Method:
        1. Split text into sentences
        2. Compute embeddings for each sentence using InLegalBERT
        3. Build pairwise cosine similarity matrix
        4. Select top-N sentences with highest total similarity (most central)
        5. Return sentences in original order

    Args:
        model: InLegalBERT model (AutoModel)
        tokenizer: InLegalBERT tokenizer
        text: Full document text
        percentage: Fraction of sentences to include (default 8%)
        max_sentences: Cap on number of summary sentences
        min_sentence_words: Skip sentences shorter than this

    Returns:
        Summary string
    """
    device = next(model.parameters()).device

    # Split into sentences
    sentences = text.split(". ")

    # Filter out very short sentences
    valid_sentences = [
        (i, s) for i, s in enumerate(sentences)
        if len(s.split()) >= min_sentence_words
    ]

    if not valid_sentences:
        logger.warning("No valid sentences found for summarization")
        return text[:500] + "..."

    # Calculate number of sentences for summary
    num_summary = max(1, min(int(len(valid_sentences) * percentage), max_sentences))

    if len(valid_sentences) <= num_summary:
        logger.info("Document too short for summarization, returning full text")
        return text

    logger.info(
        f"Summarizing: {len(valid_sentences)} sentences → {num_summary} "
        f"({percentage * 100:.0f}%)"
    )

    # Compute sentence embeddings
    indices, sent_texts = zip(*valid_sentences)
    embeddings = _compute_sentence_embeddings(model, tokenizer, sent_texts, device)

    # Compute pairwise cosine similarity
    similarity_matrix = _cosine_similarity_matrix(embeddings)

    # Score each sentence by total similarity (centrality)
    centrality_scores = similarity_matrix.sum(axis=1)

    # Select top-N most central sentences
    top_sentence_positions = np.argsort(-centrality_scores)[:num_summary]

    # Sort by original document order (preserves narrative flow)
    top_sentence_positions.sort()

    # Build summary
    summary_sentences = [sent_texts[pos] for pos in top_sentence_positions]
    summary = ". ".join(summary_sentences)

    if not summary.endswith("."):
        summary += "."

    logger.info(f"Summary generated: {len(summary_sentences)} sentences, {len(summary.split())} words")
    return summary


def _compute_sentence_embeddings(model, tokenizer, sentences, device) -> np.ndarray:
    """
    Compute InLegalBERT embeddings for a list of sentences.

    Uses the [CLS] token pooler output as the sentence representation.

    Args:
        model: InLegalBERT model
        tokenizer: InLegalBERT tokenizer
        sentences: List of sentence strings
        device: torch device

    Returns:
        Array of shape (n_sentences, hidden_size)
    """
    embeddings = []

    for sentence in sentences:
        encoded = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        # Use pooler_output (CLS token representation)
        emb = output.pooler_output.squeeze(0).cpu().numpy()
        embeddings.append(emb)

    return np.array(embeddings)


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Array of shape (n, hidden_size)

    Returns:
        Similarity matrix of shape (n, n)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normalized = embeddings / norms

    # Pairwise cosine similarity
    similarity = normalized @ normalized.T

    return similarity