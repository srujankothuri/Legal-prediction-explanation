"""
Tests for chatbot components.

Run: python -m pytest tests/test_chatbot.py -v
"""

import os
import pytest


class TestLegalChatbot:

    def test_chatbot_creation(self):
        """Chatbot can be created with custom config."""
        from src.chatbot.rag_chain import LegalChatbot

        config = {
            "vector_db_path": "vector_db",
            "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
            "retriever_k": 4,
            "memory_window": 2,
        }

        chatbot = LegalChatbot(config)
        assert chatbot._initialized is False

    def test_chatbot_not_available_without_api_key(self):
        """Chatbot reports unavailable when API key is missing."""
        from src.chatbot.rag_chain import LegalChatbot

        # Temporarily remove API key
        original_key = os.environ.pop("TOGETHER_API_KEY", None)

        try:
            config = {"vector_db_path": "nonexistent_path"}
            chatbot = LegalChatbot(config)
            assert chatbot.is_available is False
        finally:
            if original_key:
                os.environ["TOGETHER_API_KEY"] = original_key

    def test_chatbot_status(self):
        """Status check returns expected keys."""
        from src.chatbot.rag_chain import LegalChatbot

        config = {"vector_db_path": "vector_db"}
        chatbot = LegalChatbot(config)
        status = chatbot.get_status()

        assert "api_key_set" in status
        assert "vector_db_exists" in status
        assert "initialized" in status
        assert status["initialized"] is False

    def test_prompt_template_has_required_variables(self):
        """Prompt template contains all required placeholders."""
        from src.chatbot.rag_chain import LEGAL_PROMPT_TEMPLATE

        assert "{context}" in LEGAL_PROMPT_TEMPLATE
        assert "{chat_history}" in LEGAL_PROMPT_TEMPLATE
        assert "{question}" in LEGAL_PROMPT_TEMPLATE


class TestVectorStore:

    def test_load_nonexistent_raises(self):
        """Loading nonexistent vector store raises FileNotFoundError."""
        from src.chatbot.vector_store import load_vector_store

        with pytest.raises(FileNotFoundError, match="not found"):
            load_vector_store("totally_fake_path_12345")