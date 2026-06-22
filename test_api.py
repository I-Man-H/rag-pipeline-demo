"""Integration tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    with patch("app.api.routes.get_vector_store") as mock_store:
        mock_store.return_value._collection.count.return_value = 42
        mock_store.return_value._collection.name = "test"
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_query_invalid_search_type():
    response = client.post("/api/v1/query", json={
        "question": "What is RAG?",
        "search_type": "invalid_type",
    })
    assert response.status_code == 400


def test_query_too_short():
    response = client.post("/api/v1/query", json={"question": "Hi"})
    assert response.status_code == 422


def test_query_valid_request_structure():
    """Verify the request schema validates correctly without hitting the DB."""
    with patch("app.api.routes.build_rag_chain") as mock_chain:
        mock_result = MagicMock()
        mock_result.invoke.return_value = {
            "result": "RAG stands for Retrieval-Augmented Generation.",
            "source_documents": [],
        }
        mock_chain.return_value = mock_result

        response = client.post("/api/v1/query", json={
            "question": "What is retrieval-augmented generation?",
            "k": 3,
            "search_type": "similarity",
        })

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
