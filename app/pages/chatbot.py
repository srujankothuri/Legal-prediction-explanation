"""Chatbot page — RAG-powered legal Q&A interface."""

import os
import time
import streamlit as st

from src.utils.logger import get_logger

logger = get_logger("page.chatbot")


# ── Cached chatbot loading ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_chatbot():
    """Load and cache the legal chatbot."""
    from src.chatbot.rag_chain import LegalChatbot
    chatbot = LegalChatbot()
    return chatbot


# ── Page ──────────────────────────────────────────────────────────────────────

def render():
    st.markdown(
        '<p class="main-header">💬 Legal Chatbot</p>',
        unsafe_allow_html=True,
    )

    # Banner image
    image_path = "assets/chatbot_legal.jpeg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

    st.write(
        "Ask questions about Indian law — IPC, Constitution, and Bharatiya Nyaya Sanhita. "
        "Powered by retrieval-augmented generation."
    )

    # Check availability
    chatbot = load_chatbot()
    status = chatbot.get_status()

    if not status["api_key_set"]:
        st.error(
            "**TOGETHER_API_KEY not set.** "
            "Add it to your `.env` file:\n\n"
            "```\nTOGETHER_API_KEY=your_key_here\n```"
        )
        return

    if not status["vector_db_exists"]:
        st.error(
            f"**Vector database not found** at `{status['vector_db_path']}`. "
            "Build it first:\n\n"
            "```bash\nmake build-vectordb\n```"
        )
        return

    st.markdown("---")

    # Initialize session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "chatbot_initialized" not in st.session_state:
        st.session_state.chatbot_initialized = False

    # Lazy initialize the chain
    if not st.session_state.chatbot_initialized:
        with st.spinner("Initializing chatbot..."):
            try:
                chatbot.initialize()
                st.session_state.chatbot_initialized = True
            except Exception as e:
                st.error(f"Failed to initialize chatbot: {e}")
                logger.error(f"Chatbot init failed: {e}", exc_info=True)
                return

    # Display chat history
    for msg in st.session_state.chat_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask your legal question here...")

    if user_input:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = chatbot.ask(user_input)
                except Exception as e:
                    answer = f"Sorry, I encountered an error: {e}"
                    logger.error(f"Chatbot error: {e}", exc_info=True)

            # Streaming-style display
            disclaimer = "⚠️ **_Note: Information provided may be inaccurate. Consult a legal professional._**\n\n"
            full_response = disclaimer
            placeholder = st.empty()

            for chunk in answer:
                full_response += chunk
                time.sleep(0.01)
                placeholder.markdown(full_response + " ▌")

            placeholder.markdown(full_response)

        # Save assistant message
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    # Reset button
    def reset_chat():
        st.session_state.chat_messages = []
        chatbot.reset()
        logger.info("Chat history reset")

    st.sidebar.button("🗑️ Reset Chat", on_click=reset_chat)

    # Example questions
    with st.sidebar.expander("💡 Example Questions"):
        examples = [
            "What does the right to freedom of speech mean?",
            "Explain IPC Section 302",
            "What is the difference between IPC and BNS?",
            "Can you explain Article 21 of the Indian Constitution?",
            "What are the grounds for divorce under Hindu Marriage Act?",
        ]
        for ex in examples:
            st.write(f"• {ex}")