#!/usr/bin/env python3
"""
Build FAISS vector database for the legal chatbot.

Processes the ILDC dataset into chunks, embeds them, and saves
a FAISS index for retrieval-augmented generation.

Usage:
    python scripts/build_vector_db.py
    python scripts/build_vector_db.py --data data/raw/single_ildc.csv --output vector_db

    # Or via Makefile:
    make build-vectordb
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.chatbot.vector_store import build_vector_store

logger = get_logger("build_vector_db")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector database")
    parser.add_argument(
        "--data", default=None,
        help="Path to dataset CSV (default: from training.yaml)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for FAISS index (default: from app.yaml)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Characters per text chunk (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200,
        help="Overlap between chunks (default: 200)"
    )
    args = parser.parse_args()

    # Resolve paths from configs
    training_config = load_config("configs/training.yaml")
    app_config = load_config("configs/app.yaml")

    data_path = args.data or training_config["data"]["dataset_path"]
    output_dir = args.output or app_config["chatbot"]["vector_db_path"]

    logger.info(f"{'=' * 60}")
    logger.info(f"Building FAISS Vector Database")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Data source:  {data_path}")
    logger.info(f"  Output dir:   {output_dir}")
    logger.info(f"  Chunk size:   {args.chunk_size}")
    logger.info(f"  Chunk overlap: {args.chunk_overlap}")

    build_vector_store(
        data_path=data_path,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    logger.info(f"{'=' * 60}")
    logger.info(f"Vector database built successfully!")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()