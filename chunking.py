"""
Chunking strategies for document splitting.

Implements and compares four strategies:
  1. Fixed-size         — simple, fast, ignores structure
  2. Recursive          — respects paragraph/sentence boundaries (default)
  3. Semantic           — groups sentences by embedding similarity
  4. Token-based        — splits on token count (LLM-aware)

Each strategy preserves source metadata and adds chunk-level metadata.
"""

from enum import Enum
from typing import List, Dict, Any

from langchain.schema import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from loguru import logger


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    TOKEN = "token"


def chunk_documents(
    documents: List[Document],
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embeddings=None,
) -> List[Document]:
    """
    Split documents into chunks using the specified strategy.

    Args:
        documents:     List of raw LangChain Documents.
        strategy:      Chunking strategy to apply.
        chunk_size:    Target chunk size (chars or tokens depending on strategy).
        chunk_overlap: Overlap between consecutive chunks.
        embeddings:    Required only for SemanticChunker strategy.

    Returns:
        List of chunked Documents with enriched metadata.
    """
    splitter = _build_splitter(strategy, chunk_size, chunk_overlap, embeddings)
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_strategy"] = strategy.value
        chunk.metadata["chunk_size_config"] = chunk_size
        chunk.metadata["chunk_overlap_config"] = chunk_overlap
        chunk.metadata["char_count"] = len(chunk.page_content)

    logger.info(
        f"Chunking complete: {len(documents)} docs → "
        f"{len(chunks)} chunks using '{strategy.value}' strategy"
    )
    return chunks


def _build_splitter(
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    embeddings=None,
):
    if strategy == ChunkStrategy.FIXED:
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
        )

    elif strategy == ChunkStrategy.RECURSIVE:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    elif strategy == ChunkStrategy.TOKEN:
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif strategy == ChunkStrategy.SEMANTIC:
        if embeddings is None:
            raise ValueError(
                "An embeddings model must be provided for semantic chunking."
            )
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
        )

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def compare_chunking_strategies(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embeddings=None,
) -> Dict[str, Any]:
    """
    Run all strategies on the same documents and return comparison statistics.
    Useful for notebooks and evaluation scripts.
    """
    results = {}
    strategies = [
        ChunkStrategy.FIXED,
        ChunkStrategy.RECURSIVE,
        ChunkStrategy.TOKEN,
    ]
    if embeddings is not None:
        strategies.append(ChunkStrategy.SEMANTIC)

    for strategy in strategies:
        chunks = chunk_documents(
            documents, strategy, chunk_size, chunk_overlap, embeddings
        )
        char_counts = [len(c.page_content) for c in chunks]
        results[strategy.value] = {
            "num_chunks": len(chunks),
            "avg_chars": round(sum(char_counts) / len(char_counts), 1),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "chunks": chunks,
        }

    return results
