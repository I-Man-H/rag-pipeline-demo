"""
Vector store management and retrieval.

Wraps ChromaDB with LangChain to provide:
  - Persistent vector store creation and loading
  - Similarity search (cosine) and MMR (diversity-aware) retrieval
  - Metadata filtering
  - Collection management utilities
"""

from typing import List, Optional, Dict, Any

import chromadb
from langchain_chroma import Chroma
from langchain.schema import Document
from loguru import logger

from app.core.config import settings
from app.core.embeddings import get_embeddings


def get_vector_store(
    collection_name: str = None,
    embedding_model: str = None,
    persist_dir: str = None,
) -> Chroma:
    """
    Load or create a persistent ChromaDB vector store.

    Args:
        collection_name: ChromaDB collection name.
        embedding_model:  Embedding model identifier.
        persist_dir:      Directory for persistent storage.

    Returns:
        LangChain Chroma vector store instance.
    """
    embeddings = get_embeddings(embedding_model)

    store = Chroma(
        collection_name=collection_name or settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir or settings.chroma_persist_dir,
    )
    logger.info(
        f"Vector store ready: '{collection_name or settings.chroma_collection_name}' "
        f"at {persist_dir or settings.chroma_persist_dir}"
    )
    return store


def add_documents(
    vector_store: Chroma,
    documents: List[Document],
    batch_size: int = 100,
) -> None:
    """
    Add documents to the vector store in batches.

    Args:
        vector_store: Target Chroma instance.
        documents:    Chunked LangChain Documents.
        batch_size:   Number of documents per embedding batch.
    """
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(batch)
        logger.info(
            f"Indexed batch {i // batch_size + 1}: "
            f"docs {i}–{min(i + batch_size, len(documents))}"
        )
    logger.info(f"Indexing complete: {len(documents)} chunks added.")


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = None,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Standard cosine similarity retrieval.

    Args:
        vector_store:     Chroma instance.
        query:            User query string.
        k:                Number of results to return.
        filter_metadata:  Optional ChromaDB metadata filter dict.

    Returns:
        List of most similar Document chunks.
    """
    k = k or settings.retrieval_top_k
    results = vector_store.similarity_search(
        query=query,
        k=k,
        filter=filter_metadata,
    )
    logger.debug(f"Similarity search: '{query[:60]}...' → {len(results)} results")
    return results


def mmr_search(
    vector_store: Chroma,
    query: str,
    k: int = None,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Maximum Marginal Relevance retrieval.

    Balances relevance and diversity — reduces redundant chunks
    when multiple passages cover the same content.

    Args:
        vector_store:    Chroma instance.
        query:           User query string.
        k:               Number of results to return.
        fetch_k:         Candidate pool size before MMR re-ranking.
        lambda_mult:     0.0 = max diversity, 1.0 = max relevance.
        filter_metadata: Optional ChromaDB metadata filter dict.

    Returns:
        List of diverse, relevant Document chunks.
    """
    k = k or settings.retrieval_top_k
    results = vector_store.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        filter=filter_metadata,
    )
    logger.debug(f"MMR search: '{query[:60]}...' → {len(results)} results")
    return results


def get_retriever(
    vector_store: Chroma,
    search_type: str = "similarity",
    k: int = None,
    **kwargs,
):
    """
    Return a LangChain retriever for use in chains.

    Args:
        vector_store: Chroma instance.
        search_type:  'similarity' or 'mmr'.
        k:            Number of documents to retrieve.

    Returns:
        LangChain BaseRetriever instance.
    """
    k = k or settings.retrieval_top_k
    search_kwargs = {"k": k, **kwargs}

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def collection_stats(vector_store: Chroma) -> Dict[str, Any]:
    """Return document count and metadata summary for the collection."""
    collection = vector_store._collection
    count = collection.count()
    return {
        "collection_name": collection.name,
        "document_count": count,
    }
