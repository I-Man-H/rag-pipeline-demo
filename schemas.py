"""
Pydantic request and response schemas for the FastAPI endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_indexed: int
    collection_name: str
    message: str


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to answer using the indexed documents.",
        example="What are the main findings of the study?",
    )
    k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve.",
    )
    search_type: Optional[str] = Field(
        default="similarity",
        description="Retrieval strategy: 'similarity' or 'mmr'.",
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="ChromaDB collection to query. Defaults to config value.",
    )


class SourceDocument(BaseModel):
    source: str
    page: Any
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    num_sources: int
    question: str
    search_type: str
    k: int


# ---------------------------------------------------------------------------
# Chat (conversational)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message in the conversation.",
    )
    session_id: Optional[str] = Field(
        default="default",
        description="Session identifier for multi-turn conversations.",
    )


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    session_id: str


# ---------------------------------------------------------------------------
# Health & Stats
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    collection_stats: Dict[str, Any]
