"""Home page — project overview and feature highlights."""

import os
import streamlit as st


def render():
    st.markdown(
        '<p class="main-header">⚖️ Legal Judgment Prediction & Explanation</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">'
        'AI-powered Indian Supreme Court judgment prediction with explainability'
        '</p>',
        unsafe_allow_html=True,
    )

    # Banner image
    image_path = "assets/chatbot_legal.jpeg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

    st.markdown("---")

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Predict Judgment")
        st.write(
            "Upload a legal PDF and get an AI prediction of the case outcome "
            "(accepted/rejected) with sentence-level explanations highlighting "
            "the key reasoning behind the prediction."
        )

    with col2:
        st.markdown("### 🧠 How It Works")
        st.write(
            "Fine-tuned transformer encoders (XLNet, RoBERTa, BERT) generate "
            "chunk embeddings → Hierarchical Attention Network aggregates them → "
            "Occlusion analysis identifies the most influential sentences."
        )

    with col3:
        st.markdown("### 💬 Legal Chatbot")
        st.write(
            "Ask questions about Indian law — IPC, Constitution, and BNS — "
            "powered by retrieval-augmented generation with Mistral-7B and a "
            "FAISS vector store built from legal documents."
        )

    st.markdown("---")

    # Architecture diagram
    st.markdown("### Architecture")
    st.code(
        "PDF Upload\n"
        "  → Text Extraction (PyPDF2)\n"
        "  → Transformer Embeddings (Level 1: last-4-layer concat → 3072-dim)\n"
        "  → HAN Prediction (Level 2: 3x BiGRU + Hierarchical Attention)\n"
        "  → Occlusion Explanation (Level 3: sentence-level importance)\n"
        "  → InLegalBERT Extractive Summary\n"
        "  → Streamlit UI + Legal Chatbot (FAISS + Mistral-7B)",
        language=None,
    )

    # Tech stack
    st.markdown("### Tech Stack")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "**Encoders:** XLNet, RoBERTa, BERT, DistilBERT\n\n"
            "**Classifier:** 3-layer BiGRU + Hierarchical Attention\n\n"
            "**Explanation:** Occlusion-based sentence scoring\n\n"
            "**Summary:** InLegalBERT extractive"
        )

    with col2:
        st.markdown(
            "**Chatbot:** FAISS + LangChain + Mistral-7B\n\n"
            "**Dataset:** ILDC (Indian Legal Documents Corpus)\n\n"
            "**Framework:** PyTorch + TensorFlow/Keras\n\n"
            "**UI:** Streamlit"
        )