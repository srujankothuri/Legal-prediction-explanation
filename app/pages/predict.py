"""Prediction page — upload PDF, get judgment prediction + explanation + summary."""

import os
import tempfile
import streamlit as st

from app.components.model_selector import render_model_selector
from app.components.result_display import display_prediction_result

from src.utils.logger import get_logger

logger = get_logger("page.predict")


# ── Cached model loading ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_predictor(encoder_name: str):
    """Load and cache the prediction pipeline for a given encoder."""
    from src.utils.config import load_config
    from src.inference.predictor import JudgmentPredictor

    config_path = f"configs/models/{encoder_name}_bigru.yaml"
    config = load_config(config_path)

    return JudgmentPredictor(
        encoder_name=encoder_name,
        model_config=config,
        enable_explanation=True,
        enable_summary=True,
    )


# ── Page ──────────────────────────────────────────────────────────────────────

def render():
    st.markdown(
        '<p class="main-header">📄 Judgment Prediction</p>',
        unsafe_allow_html=True,
    )

    st.write(
        "Upload a legal judgment PDF to predict the case outcome. "
        "The system will analyze the document and provide a prediction "
        "with explanations and a summary."
    )

    # Model selection
    encoder_name = render_model_selector()

    st.markdown("---")

    # Input method tabs
    tab_pdf, tab_text = st.tabs(["📁 Upload PDF", "📝 Paste Text"])

    input_text = None

    with tab_pdf:
        uploaded_file = st.file_uploader(
            "Upload a legal judgment PDF",
            type=["pdf"],
            help="Supports Indian Supreme Court judgments from Indian Kanoon",
        )

        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("Extracting text from PDF..."):
                from src.data.pdf_extractor import PDFExtractor

                try:
                    extractor = PDFExtractor(tmp_path)
                    input_text = extractor.get_text_for_prediction()

                    word_count = len(input_text.split())
                    st.success(f"Extracted {word_count:,} words from PDF")

                    with st.expander("Preview extracted text", expanded=False):
                        st.text(input_text[:3000] + ("..." if len(input_text) > 3000 else ""))

                except Exception as e:
                    st.error(f"Failed to extract text: {e}")
                    logger.error(f"PDF extraction failed: {e}")

            os.unlink(tmp_path)

    with tab_text:
        pasted_text = st.text_area(
            "Paste legal document text",
            height=300,
            placeholder="Paste the full text of a legal judgment here...",
        )
        if pasted_text and len(pasted_text.strip()) > 100:
            input_text = pasted_text.strip()

    # Run prediction
    if input_text:
        st.markdown("---")

        # Options
        col1, col2, col3 = st.columns(3)
        with col1:
            do_explain = st.checkbox("Generate explanation", value=True)
        with col2:
            do_summary = st.checkbox("Generate summary", value=True)
        with col3:
            st.write("")  # spacer

        if st.button("🔮 Predict Judgment", type="primary", use_container_width=True):
            # Check model availability
            models_dir = "trained_models"
            han_path = os.path.join(models_dir, f"han_{encoder_name}.h5")
            encoder_dir = os.path.join(models_dir, f"{encoder_name}_finetuned")

            if not os.path.exists(encoder_dir):
                st.error(
                    f"Fine-tuned {encoder_name.upper()} model not found at `{encoder_dir}/`. "
                    f"Train it first with: `make train-encoder MODEL={encoder_name}`"
                )
                return

            if not os.path.exists(han_path):
                st.error(
                    f"HAN classifier not found at `{han_path}`. "
                    f"Train it first with: `make train-classifier MODEL={encoder_name}`"
                )
                return

            # Run pipeline
            progress_bar = st.progress(0, text="Loading model...")

            try:
                progress_bar.progress(10, text="Loading prediction pipeline...")
                predictor = load_predictor(encoder_name)

                progress_bar.progress(30, text="Generating embeddings (Level 1)...")

                result = predictor.predict_from_text(
                    input_text,
                    generate_explanation=do_explain,
                    generate_summary=do_summary,
                )

                progress_bar.progress(100, text="Complete!")
                progress_bar.empty()

                # Display results
                display_prediction_result(result)

            except Exception as e:
                progress_bar.empty()
                st.error(f"Prediction failed: {e}")
                logger.error(f"Prediction pipeline error: {e}", exc_info=True)
    else:
        st.info("Upload a PDF or paste text to get started.")