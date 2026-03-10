"""
End-to-end prediction pipeline.

Orchestrates the full flow:
    PDF/Text → XLNet Embeddings → HAN Prediction → Explanation → Summary

Usage:
    from src.inference.predictor import JudgmentPredictor

    predictor = JudgmentPredictor("xlnet", config)
    result = predictor.predict_from_text(text)
    result = predictor.predict_from_pdf("path/to/judgment.pdf")
"""

import os
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction output."""

    # Core prediction
    prediction: bool                     # True = accepted, False = rejected
    label: str                           # "accepted" or "rejected"
    probability: float                   # sigmoid output (0 to 1)
    confidence: float                    # max(prob, 1-prob)

    # Embeddings info
    num_chunks: int = 0
    embedding_dim: int = 0

    # Explanation
    explanation: str = ""
    chunk_scores: list = field(default_factory=list)

    # Summary
    summary: str = ""

    # Metadata
    encoder_name: str = ""
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "prediction": self.prediction,
            "label": self.label,
            "probability": self.probability,
            "confidence": self.confidence,
            "num_chunks": self.num_chunks,
            "embedding_dim": self.embedding_dim,
            "explanation": self.explanation,
            "chunk_scores": self.chunk_scores,
            "summary": self.summary,
            "encoder_name": self.encoder_name,
            "processing_time_seconds": self.processing_time_seconds,
        }


class JudgmentPredictor:
    """
    End-to-end judgment prediction pipeline.

    Combines:
        1. Transformer encoder (embedding generation)
        2. HAN classifier (document-level prediction)
        3. Occlusion explainer (sentence-level importance)
        4. InLegalBERT summarizer (extractive summary)
    """

    def __init__(
        self,
        encoder_name: str,
        model_config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        enable_explanation: bool = True,
        enable_summary: bool = True,
    ):
        """
        Args:
            encoder_name: One of 'xlnet', 'roberta', 'bert', 'distilbert'
            model_config: Pre-loaded config dict (takes priority)
            config_path: Path to model config YAML
            enable_explanation: Whether to generate explanations
            enable_summary: Whether to generate summaries
        """
        self.encoder_name = encoder_name
        self.enable_explanation = enable_explanation
        self.enable_summary = enable_summary

        # Load config
        if model_config:
            self.config = model_config
        elif config_path:
            self.config = load_config(config_path)
        else:
            self.config = load_config(f"configs/models/{encoder_name}_bigru.yaml")

        # Lazy-loaded components (loaded on first use)
        self._encoder_model = None
        self._tokenizer = None
        self._device = None
        self._han_classifier = None
        self._embedding_generator = None
        self._explainer = None
        self._summarizer_model = None
        self._summarizer_tokenizer = None

        logger.info(
            f"JudgmentPredictor initialized: encoder={encoder_name}, "
            f"explanation={enable_explanation}, summary={enable_summary}"
        )

    def _load_encoder(self):
        """Lazy-load the transformer encoder."""
        if self._encoder_model is not None:
            return

        logger.info(f"Loading {self.encoder_name} encoder...")
        from src.training.embeddings import EmbeddingGenerator

        self._embedding_generator = EmbeddingGenerator(self.config)
        self._encoder_model = self._embedding_generator.model
        self._tokenizer = self._embedding_generator.tokenizer
        self._device = self._embedding_generator.device

        logger.info("Encoder loaded successfully")

    def _load_han(self):
        """Lazy-load the HAN classifier."""
        if self._han_classifier is not None:
            return

        logger.info("Loading HAN classifier...")
        from src.models.han_classifier import HANClassifier

        han_path = os.path.join(
            "trained_models", f"han_{self.encoder_name}.h5"
        )

        if not os.path.exists(han_path):
            raise FileNotFoundError(
                f"HAN model not found: {han_path}. "
                f"Train it first with: make train-classifier MODEL={self.encoder_name}"
            )

        self._han_classifier = HANClassifier(self.config, weights_path=han_path)
        self._han_classifier.build()

        logger.info("HAN classifier loaded successfully")

    def _load_summarizer(self):
        """Lazy-load InLegalBERT summarizer."""
        if self._summarizer_model is not None:
            return

        logger.info("Loading InLegalBERT summarizer...")
        from transformers import AutoTokenizer, AutoModel

        self._summarizer_tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
        self._summarizer_model = AutoModel.from_pretrained("law-ai/InLegalBERT")
        self._summarizer_model.eval()

        logger.info("Summarizer loaded successfully")

    def predict_from_text(
        self,
        text: str,
        generate_explanation: bool = None,
        generate_summary: bool = None,
    ) -> PredictionResult:
        """
        Run full prediction pipeline on raw text.

        Args:
            text: Raw document text
            generate_explanation: Override self.enable_explanation
            generate_summary: Override self.enable_summary

        Returns:
            PredictionResult with all outputs
        """
        start_time = time.time()
        do_explain = generate_explanation if generate_explanation is not None else self.enable_explanation
        do_summary = generate_summary if generate_summary is not None else self.enable_summary

        word_count = len(text.split())
        logger.info(f"Starting prediction pipeline: {word_count} words")

        # ── Step 1: Generate embeddings ─────────────────────────────────────
        logger.info("Step 1/4: Generating embeddings...")
        self._load_encoder()
        embeddings = self._embedding_generator.generate_for_text(text)
        logger.info(f"  Generated {embeddings.shape[0]} chunks × {embeddings.shape[1]}-dim")

        # ── Step 2: HAN prediction ──────────────────────────────────────────
        logger.info("Step 2/4: Running HAN prediction...")
        self._load_han()

        if do_explain:
            prediction, probability, chunk_scores = (
                self._han_classifier.predict_with_occlusion(embeddings)
            )
        else:
            prediction, probability = self._han_classifier.predict_document(embeddings)
            chunk_scores = []

        label = "accepted" if prediction else "rejected"
        confidence = probability if prediction else (1 - probability)
        logger.info(f"  Prediction: {label} (prob={probability:.4f}, conf={confidence:.4f})")

        # ── Step 3: Explanation ─────────────────────────────────────────────
        explanation = ""
        if do_explain and chunk_scores:
            logger.info("Step 3/4: Generating explanation...")
            try:
                from src.inference.explainer import ExplanationGenerator

                if self._explainer is None:
                    self._explainer = ExplanationGenerator(
                        self._encoder_model,
                        self._tokenizer,
                        self._device,
                    )

                pred_label = 1 if prediction else 0
                explanation = self._explainer.generate(text, chunk_scores, pred_label)
                logger.info(f"  Explanation: {len(explanation.split())} words")
            except Exception as e:
                logger.warning(f"  Explanation generation failed: {e}")
        else:
            logger.info("Step 3/4: Explanation skipped")

        # ── Step 4: Summary ─────────────────────────────────────────────────
        summary = ""
        if do_summary:
            logger.info("Step 4/4: Generating summary...")
            try:
                self._load_summarizer()
                from src.inference.summarizer import generate_extractive_summary

                summary = generate_extractive_summary(
                    self._summarizer_model,
                    self._summarizer_tokenizer,
                    text,
                )
                logger.info(f"  Summary: {len(summary.split())} words")
            except Exception as e:
                logger.warning(f"  Summary generation failed: {e}")
        else:
            logger.info("Step 4/4: Summary skipped")

        # ── Build result ────────────────────────────────────────────────────
        elapsed = time.time() - start_time

        result = PredictionResult(
            prediction=prediction,
            label=label,
            probability=probability,
            confidence=confidence,
            num_chunks=embeddings.shape[0],
            embedding_dim=embeddings.shape[1],
            explanation=explanation,
            chunk_scores=chunk_scores,
            summary=summary,
            encoder_name=self.encoder_name,
            processing_time_seconds=round(elapsed, 2),
        )

        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        return result

    def predict_from_pdf(self, pdf_path: str, **kwargs) -> PredictionResult:
        """
        Run prediction pipeline on a PDF file.

        Args:
            pdf_path: Path to the judgment PDF
            **kwargs: Passed to predict_from_text()

        Returns:
            PredictionResult
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        from src.data.pdf_extractor import PDFExtractor

        extractor = PDFExtractor(pdf_path)
        text = extractor.get_text_for_prediction()

        return self.predict_from_text(text, **kwargs)