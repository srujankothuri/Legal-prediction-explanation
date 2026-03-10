"""
FAISS vector store builder for legal document retrieval.

Splits ILDC documents into chunks, embeds them using nomic-embed-text,
and builds a FAISS index for similarity-based retrieval.

Usage:
    from src.chatbot.vector_store import build_vector_store, load_vector_store

    build_vector_store("data/raw/single_ildc.csv", "vector_db")
    retriever = load_vector_store("vector_db")
"""

import os
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

# Default embedding model for RAG
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
DEFAULT_EMBEDDING_KWARGS = {
    "trust_remote_code": True,
    "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7",
}


def _get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load the embedding model for vectorization."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info(f"Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=DEFAULT_EMBEDDING_KWARGS,
    )


def build_vector_store(
    data_path: str,
    output_dir: str = "vector_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 500,
    text_column: str = "text",
):
    """
    Build a FAISS vector store from a CSV dataset.

    Args:
        data_path: Path to CSV file with legal documents
        output_dir: Directory to save FAISS index
        chunk_size: Characters per text chunk
        chunk_overlap: Overlap between consecutive chunks
        batch_size: Documents to process per batch (memory management)
        text_column: Name of the text column in CSV
    """
    import pandas as pd
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    logger.info(f"Building vector store from: {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    texts = df[text_column].dropna().tolist()
    logger.info(f"Loaded {len(texts)} documents")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks = []
    for text in texts:
        chunks = splitter.split_text(str(text))
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} text chunks")

    # Load embeddings
    embeddings = _get_embeddings()

    # Build FAISS index in batches
    logger.info("Building FAISS index...")
    db = None

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]

        if db is None:
            db = FAISS.from_texts(batch, embeddings)
        else:
            batch_db = FAISS.from_texts(batch, embeddings)
            db.merge_from(batch_db)

        processed = min(i + batch_size, len(all_chunks))
        logger.info(f"  Indexed {processed}/{len(all_chunks)} chunks")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    db.save_local(output_dir)
    logger.info(f"Vector store saved to: {output_dir}/")

    return db


def load_vector_store(
    db_path: str = "vector_db",
    search_type: str = "similarity",
    search_k: int = 4,
):
    """
    Load a saved FAISS vector store and return a retriever.

    Args:
        db_path: Path to saved FAISS index directory
        search_type: Retrieval strategy ("similarity" or "mmr")
        search_k: Number of documents to retrieve

    Returns:
        LangChain retriever instance

    Raises:
        FileNotFoundError: If vector store doesn't exist
    """
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Vector store not found at: {db_path}. "
            f"Build it first with: python scripts/build_vector_db.py"
        )

    logger.info(f"Loading vector store from: {db_path}")
    embeddings = _get_embeddings()

    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs={"k": search_k},
    )

    logger.info(f"Vector store loaded: search_type={search_type}, k={search_k}")
    return retriever