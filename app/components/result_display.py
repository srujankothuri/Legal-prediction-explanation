import streamlit as st


def display_prediction_result(result):
    """
    Display a PredictionResult in a formatted layout.

    Args:
        result: PredictionResult instance from the predictor
    """
    st.markdown("---")

    # ── Prediction Banner ─────────────────────────────────────────────────
    st.markdown("### 🎯 Prediction Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if result.prediction:
            st.success(f"**ACCEPTED**")
        else:
            st.error(f"**REJECTED**")

    with col2:
        st.metric("Confidence", f"{result.confidence:.1%}")

    with col3:
        st.metric("Chunks Processed", result.num_chunks)

    with col4:
        st.metric("Processing Time", f"{result.processing_time_seconds:.1f}s")

    # ── Explanation ────────────────────────────────────────────────────────
    if result.explanation:
        st.markdown("---")
        st.markdown("### 📝 Key Sentences Supporting Prediction")
        st.info(
            f"The following sentences were identified as most influential "
            f"in the model's {'acceptance' if result.prediction else 'rejection'} decision:"
        )
        st.write(result.explanation)

        # Chunk importance visualization
        if result.chunk_scores:
            with st.expander("📊 Chunk Importance Scores", expanded=False):
                import pandas as pd

                chart_data = pd.DataFrame({
                    "Chunk": [f"Chunk {i+1}" for i in range(len(result.chunk_scores))],
                    "Importance": result.chunk_scores,
                })
                st.bar_chart(chart_data.set_index("Chunk"))

    # ── Summary ────────────────────────────────────────────────────────────
    if result.summary:
        st.markdown("---")
        st.markdown("### 📋 Document Summary")
        st.write(result.summary)

    # ── Technical Details ──────────────────────────────────────────────────
    with st.expander("🔧 Technical Details", expanded=False):
        st.write(f"**Encoder:** {result.encoder_name}")
        st.write(f"**Embedding dimension:** {result.embedding_dim}")
        st.write(f"**Number of chunks:** {result.num_chunks}")
        st.write(f"**Raw probability:** {result.probability:.6f}")
        st.write(f"**Prediction threshold:** 0.5")
        st.write(f"**Total processing time:** {result.processing_time_seconds:.2f}s")