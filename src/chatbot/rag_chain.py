"""
RAG (Retrieval-Augmented Generation) chain for legal Q&A chatbot.

Combines FAISS retrieval with Mistral-7B (via Together AI) for
context-aware answers about Indian law.

Usage:
    from src.chatbot.rag_chain import LegalChatbot

    chatbot = LegalChatbot()
    answer = chatbot.ask("What is IPC Section 302?")
    chatbot.reset()
"""

import os
from typing import Optional, Dict, Any

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

LEGAL_PROMPT_TEMPLATE = """<s>[INST] You are a highly knowledgeable and professional legal chatbot \
trained on the Indian Penal Code (IPC), Indian Constitution, and Bharatiya Nyaya Sanhita (BNS). \
Your role is to assist users by answering legal questions accurately, concisely, and professionally.

Guidelines:
1. For general users: use simple, clear language. Explain legal terms when used. Provide examples.
2. For legal professionals: use precise terminology. Cite relevant sections and articles.
3. Always specify the source (IPC section, Constitution article, BNS chapter).
4. If a question requires clarification, politely ask for more details.
5. If unsure, suggest consulting a qualified legal professional.
6. Do not provide speculative advice or personal opinions.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER: </s>[INST]"""


class LegalChatbot:
    """
    RAG-powered legal chatbot for Indian law Q&A.

    Uses:
        - FAISS vector store for relevant document retrieval
        - Together AI API for LLM inference (Mistral-7B)
        - LangChain ConversationalRetrievalChain for memory
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: App config dict. If None, loads from configs/app.yaml.
        """
        if config is None:
            config = load_config("configs/app.yaml")

        self.config = config.get("chatbot", config)
        self._chain = None
        self._memory = None
        self._initialized = False

        logger.info("LegalChatbot created (lazy initialization)")

    def initialize(self):
        """
        Initialize all components (retriever, LLM, chain).
        Call this explicitly or it auto-initializes on first ask().
        """
        if self._initialized:
            return

        logger.info("Initializing LegalChatbot...")

        # Load retriever
        from src.chatbot.vector_store import load_vector_store

        db_path = self.config.get("vector_db_path", "vector_db")
        search_k = self.config.get("retriever_k", 4)
        retriever = load_vector_store(db_path, search_k=search_k)

        # Load LLM
        api_key = os.getenv("TOGETHER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY not set. "
                "Add it to your .env file: TOGETHER_API_KEY=your_key_here"
            )

        from langchain_together import Together

        llm_model = self.config.get("llm_model", "mistralai/Mistral-7B-Instruct-v0.2")
        llm_temp = self.config.get("llm_temperature", 0.5)
        llm_max_tokens = self.config.get("llm_max_tokens", 1024)

        llm = Together(
            model=llm_model,
            temperature=llm_temp,
            max_tokens=llm_max_tokens,
            together_api_key=api_key,
        )

        logger.info(f"LLM loaded: {llm_model}")

        # Build prompt
        from langchain.prompts import PromptTemplate

        prompt = PromptTemplate(
            template=LEGAL_PROMPT_TEMPLATE,
            input_variables=["context", "question", "chat_history"],
        )

        # Build memory
        from langchain.memory import ConversationBufferWindowMemory

        memory_k = self.config.get("memory_window", 2)
        self._memory = ConversationBufferWindowMemory(
            k=memory_k,
            memory_key="chat_history",
            return_messages=True,
        )

        # Build chain
        from langchain.chains import ConversationalRetrievalChain

        self._chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=self._memory,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": prompt},
        )

        self._initialized = True
        logger.info("LegalChatbot fully initialized")

    def ask(self, question: str) -> str:
        """
        Ask a legal question and get a response.

        Args:
            question: User's legal question

        Returns:
            Answer string from the LLM
        """
        if not self._initialized:
            self.initialize()

        logger.info(f"Question: {question[:100]}...")

        result = self._chain.invoke(input=question)
        answer = result.get("answer", "")

        logger.info(f"Answer length: {len(answer)} chars")
        return answer

    def reset(self):
        """Clear conversation history."""
        if self._memory:
            self._memory.clear()
            logger.info("Conversation history cleared")

    @property
    def is_available(self) -> bool:
        """Check if all dependencies are available for the chatbot."""
        # Check API key
        if not os.getenv("TOGETHER_API_KEY"):
            return False

        # Check vector store
        db_path = self.config.get("vector_db_path", "vector_db")
        if not os.path.exists(db_path):
            return False

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get chatbot component status for debugging."""
        db_path = self.config.get("vector_db_path", "vector_db")
        return {
            "api_key_set": bool(os.getenv("TOGETHER_API_KEY")),
            "vector_db_exists": os.path.exists(db_path),
            "vector_db_path": db_path,
            "initialized": self._initialized,
            "llm_model": self.config.get("llm_model", "not configured"),
        }