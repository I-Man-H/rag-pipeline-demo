"""
Embedding model factory and comparison utilities.

Supports:
  - OpenAI embeddings  (text-embedding-3-small, text-embedding-3-large)
  - HuggingFace / sentence-transformers (local, no API key required)

The factory pattern means the rest of the codebase stays model-agnostic:
swap the embedding model in .env without changing any other code.
"""

from enum import Enum
from typing import List, Dict, Any
import time

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

from app.core.config import settings


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


# Curated list of models available for comparison
EMBEDDING_MODELS = {
    EmbeddingProvider.OPENAI: [
        "text-embedding-3-small",
        "text-embedding-3-large",
    ],
    EmbeddingProvider.HUGGINGFACE: [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5",
    ],
}


def get_embeddings(model_name: str = None):
    """
    Return a LangChain-compatible embeddings object.

    Automatically detects whether the model name refers to an OpenAI
    or HuggingFace model.

    Args:
        model_name: Model identifier string. Defaults to settings value.

    Returns:
        Embeddings object compatible with ChromaDB and LangChain retrievers.
    """
    model_name = model_name or settings.embedding_model

    if model_name.startswith("text-embedding"):
        logger.info(f"Using OpenAI embeddings: {model_name}")
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.openai_api_key,
        )
    else:
        logger.info(f"Using HuggingFace embeddings: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def compare_embedding_models(
    texts: List[str],
    query: str,
    model_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Embed the same texts and query with multiple models and compare results.

    Measures encoding speed, embedding dimensionality, and cosine similarity
    rankings — useful for choosing the right model for a new domain.

    Args:
        texts:       Corpus of text snippets to embed.
        query:       Query to rank against the corpus.
        model_names: List of model identifiers to compare.

    Returns:
        Dictionary with per-model metrics and similarity rankings.
    """
    if model_names is None:
        model_names = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-small-en-v1.5",
        ]

    results = {}

    for model_name in model_names:
        logger.info(f"Benchmarking: {model_name}")
        embedder = get_embeddings(model_name)

        start = time.time()
        doc_embeddings = embedder.embed_documents(texts)
        query_embedding = embedder.embed_query(query)
        elapsed = time.time() - start

        doc_arr = np.array(doc_embeddings)
        q_arr = np.array(query_embedding)

        # Cosine similarity
        similarities = np.dot(doc_arr, q_arr) / (
            np.linalg.norm(doc_arr, axis=1) * np.linalg.norm(q_arr) + 1e-10
        )
        ranked_indices = np.argsort(similarities)[::-1]

        results[model_name] = {
            "embedding_dim": len(doc_embeddings[0]),
            "encode_time_s": round(elapsed, 3),
            "top_matches": [
                {
                    "rank": rank + 1,
                    "text": texts[idx][:120],
                    "score": round(float(similarities[idx]), 4),
                }
                for rank, idx in enumerate(ranked_indices[:3])
            ],
        }

    return results
