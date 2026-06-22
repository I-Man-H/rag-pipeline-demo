"""Tests for chunking strategies."""

import pytest
from langchain.schema import Document
from app.core.chunking import chunk_documents, ChunkStrategy, compare_chunking_strategies


SAMPLE_DOCS = [
    Document(
        page_content="This is the first paragraph.\n\nThis is the second paragraph with more content. "
                     "It contains multiple sentences to test splitting behaviour.",
        metadata={"source": "test.txt", "page": 1},
    ),
    Document(
        page_content="Another document with different content. " * 20,
        metadata={"source": "test2.txt", "page": 1},
    ),
]


@pytest.mark.parametrize("strategy", [
    ChunkStrategy.FIXED,
    ChunkStrategy.RECURSIVE,
    ChunkStrategy.TOKEN,
])
def test_chunking_returns_documents(strategy):
    chunks = chunk_documents(SAMPLE_DOCS, strategy=strategy, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 0
    assert all(isinstance(c, Document) for c in chunks)


def test_chunks_have_metadata():
    chunks = chunk_documents(SAMPLE_DOCS, strategy=ChunkStrategy.RECURSIVE)
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata
        assert "chunk_strategy" in chunk.metadata
        assert "char_count" in chunk.metadata


def test_chunk_overlap_creates_more_chunks():
    chunks_no_overlap = chunk_documents(
        SAMPLE_DOCS, strategy=ChunkStrategy.RECURSIVE,
        chunk_size=100, chunk_overlap=0
    )
    chunks_with_overlap = chunk_documents(
        SAMPLE_DOCS, strategy=ChunkStrategy.RECURSIVE,
        chunk_size=100, chunk_overlap=50
    )
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)


def test_compare_strategies_returns_all_keys():
    results = compare_chunking_strategies(SAMPLE_DOCS, chunk_size=200, chunk_overlap=20)
    assert "fixed" in results
    assert "recursive" in results
    assert "token" in results
    for key, stats in results.items():
        assert "num_chunks" in stats
        assert "avg_chars" in stats


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        from app.core.chunking import _build_splitter
        _build_splitter("invalid_strategy", 512, 64, None)
