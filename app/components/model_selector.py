import os
import streamlit as st

MODELS = {
    "xlnet": "XLNet + BiGRU-HAN",
    "roberta": "RoBERTa + BiGRU-HAN",
    "bert": "BERT + BiGRU-HAN",
    "distilbert": "DistilBERT + BiGRU-HAN",
}


def render_model_selector() -> str:
    """
    Render a model selection dropdown.
    Shows availability status for each model.

    Returns:
        Selected encoder name (e.g., 'xlnet')
    """
    # Check which models are trained
    available = {}
    for key, display_name in MODELS.items():
        encoder_dir = os.path.join("trained_models", f"{key}_finetuned")
        han_path = os.path.join("trained_models", f"han_{key}.h5")
        is_ready = os.path.exists(encoder_dir) and os.path.exists(han_path)
        status = "✅" if is_ready else "⏳"
        available[key] = f"{status} {display_name}"

    # Dropdown
    selected_display = st.selectbox(
        "Select Model:",
        list(available.values()),
        help="Choose which encoder + classifier combination to use for prediction",
    )

    # Map back to encoder name
    for key, display in available.items():
        if display == selected_display:
            return key

    return "xlnet"  # fallback
