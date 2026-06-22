"""Tests for retrieval evaluation metrics."""

import pytest
from app.core.evaluation import evaluate_retrieval, build_retrieval_report


QUERIES = ["What is RAG?", "How does ChromaDB work?", "Explain chunking."]
GROUND_TRUTH = [["rag_intro.pdf"], ["chromadb_docs.pdf"], ["chunking_guide.pdf"]]
RETRIEVED_PERFECT = [["rag_intro.pdf", "other.pdf"], ["chromadb_docs.pdf"], ["chunking_guide.pdf"]]
RETRIEVED_NONE = [["irrelevant.pdf"], ["irrelevant.pdf"], ["irrelevant.pdf"]]


def test_perfect_retrieval_scores():
    metrics = evaluate_retrieval(QUERIES, GROUND_TRUTH, RETRIEVED_PERFECT, k=5)
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0


def test_zero_retrieval_scores():
    metrics = evaluate_retrieval(QUERIES, GROUND_TRUTH, RETRIEVED_NONE, k=5)
    assert metrics["hit_rate"] == 0.0
    assert metrics["mrr"] == 0.0


def test_metrics_keys_present():
    metrics = evaluate_retrieval(QUERIES, GROUND_TRUTH, RETRIEVED_PERFECT, k=5)
    assert "hit_rate" in metrics
    assert "mrr" in metrics
    assert "precision_at_5" in metrics
    assert "num_queries" in metrics


def test_report_dataframe_shape():
    df = build_retrieval_report(QUERIES, GROUND_TRUTH, RETRIEVED_PERFECT, k=5)
    assert len(df) == len(QUERIES)
    assert "query" in df.columns
    assert "hit" in df.columns
    assert "reciprocal_rank" in df.columns
