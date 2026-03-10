"""Model comparison page — side-by-side evaluation results across all trained models."""

import os
import json
import streamlit as st
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("page.compare")

RESULTS_PATH = "logs/evaluation_results.json"


def render():
    st.markdown(
        '<p class="main-header">📊 Model Comparison</p>',
        unsafe_allow_html=True,
    )

    st.write(
        "Compare evaluation metrics across all trained encoder + HAN classifier combinations. "
        "Run `make evaluate` to generate results."
    )

    st.markdown("---")

    # Check for results
    if not os.path.exists(RESULTS_PATH):
        st.warning(
            f"No evaluation results found at `{RESULTS_PATH}`. "
            f"Run evaluation first:\n\n```bash\nmake evaluate\n```"
        )
        _show_placeholder()
        return

    # Load results
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    if not results:
        st.warning("Evaluation results file is empty.")
        return

    # Main comparison table
    st.markdown("### Overall Performance")

    table_data = []
    for r in results:
        table_data.append({
            "Model": r.get("model_name", "Unknown"),
            "Accuracy": f"{r['accuracy']:.4f}",
            "Macro F1": f"{r['macro_f1']:.4f}",
            "Macro Precision": f"{r['macro_precision']:.4f}",
            "Macro Recall": f"{r['macro_recall']:.4f}",
            "Micro F1": f"{r['micro_f1']:.4f}",
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart comparison
    st.markdown("### Visual Comparison")

    chart_data = pd.DataFrame({
        "Model": [r.get("model_name", "?") for r in results],
        "Accuracy": [r["accuracy"] for r in results],
        "Macro F1": [r["macro_f1"] for r in results],
        "Macro Precision": [r["macro_precision"] for r in results],
        "Macro Recall": [r["macro_recall"] for r in results],
    })

    metric_choice = st.selectbox(
        "Select metric to visualize:",
        ["Accuracy", "Macro F1", "Macro Precision", "Macro Recall"],
    )

    st.bar_chart(chart_data.set_index("Model")[metric_choice])

    # Per-model details
    st.markdown("---")
    st.markdown("### Detailed Per-Model Results")

    for r in results:
        model_name = r.get("model_name", "Unknown")

        with st.expander(f"📋 {model_name}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Accuracy", f"{r['accuracy']:.4f}")
            with col2:
                st.metric("Macro F1", f"{r['macro_f1']:.4f}")
            with col3:
                st.metric("Macro Precision", f"{r['macro_precision']:.4f}")
            with col4:
                st.metric("Macro Recall", f"{r['macro_recall']:.4f}")

            # Confusion matrix
            cm = r.get("confusion_matrix")
            if cm:
                st.markdown("**Confusion Matrix:**")
                cm_df = pd.DataFrame(
                    cm,
                    index=["True Rejected", "True Accepted"],
                    columns=["Pred Rejected", "Pred Accepted"],
                )
                st.dataframe(cm_df, use_container_width=False)

            # Per-class metrics
            st.markdown("**Per-Class Metrics:**")
            class_df = pd.DataFrame({
                "Class": ["Rejected", "Accepted"],
                "Precision": [
                    f"{r.get('precision_rejected', 0):.4f}",
                    f"{r.get('precision_accepted', 0):.4f}",
                ],
                "Recall": [
                    f"{r.get('recall_rejected', 0):.4f}",
                    f"{r.get('recall_accepted', 0):.4f}",
                ],
                "F1": [
                    f"{r.get('f1_rejected', 0):.4f}",
                    f"{r.get('f1_accepted', 0):.4f}",
                ],
            })
            st.dataframe(class_df, use_container_width=False, hide_index=True)


def _show_placeholder():
    """Show placeholder table with expected model results."""
    st.markdown("### Expected Models (after training)")

    placeholder = pd.DataFrame({
        "Model": [
            "XLNET + BiGRU-HAN",
            "ROBERTA + BiGRU-HAN",
            "BERT + BiGRU-HAN",
            "DISTILBERT + BiGRU-HAN",
        ],
        "Status": ["🔜 Pending"] * 4,
        "Accuracy": ["—"] * 4,
        "Macro F1": ["—"] * 4,
    })

    st.dataframe(placeholder, use_container_width=True, hide_index=True)