"""
CLI script to ingest a directory of documents into ChromaDB.

Usage:
    python scripts/ingest_docs.py --source data/raw --strategy recursive
    python scripts/ingest_docs.py --source data/raw --strategy fixed --chunk-size 256
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from app.core.ingestion import load_directory
from app.core.chunking import chunk_documents, ChunkStrategy
from app.core.retrieval import get_vector_store, add_documents, collection_stats
from app.core.config import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument(
        "--source", type=str, default="data/raw",
        help="Directory containing source documents."
    )
    parser.add_argument(
        "--strategy", type=str, default="recursive",
        choices=["fixed", "recursive", "token", "semantic"],
        help="Chunking strategy to use."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=settings.chunk_size,
        help="Target chunk size in characters."
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=settings.chunk_overlap,
        help="Overlap between consecutive chunks."
    )
    parser.add_argument(
        "--collection", type=str, default=settings.chroma_collection_name,
        help="ChromaDB collection name."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading documents from: {args.source}")
    documents = load_directory(args.source)

    if not documents:
        logger.error("No documents found. Check the source directory.")
        sys.exit(1)

    logger.info(f"Chunking with strategy: {args.strategy}")
    chunks = chunk_documents(
        documents,
        strategy=ChunkStrategy(args.strategy),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    logger.info(f"Indexing {len(chunks)} chunks into collection: {args.collection}")
    store = get_vector_store(collection_name=args.collection)
    add_documents(store, chunks)

    stats = collection_stats(store)
    logger.info(f"Done. Collection stats: {stats}")


if __name__ == "__main__":
    main()
