"""
FastAPI route definitions.

Endpoints:
  GET  /health              — liveness check + collection stats
  POST /ingest              — upload and index documents
  POST /query               — single-turn RAG query
  POST /chat                — multi-turn conversational RAG
  GET  /collections         — list available collections
  DELETE /collections/{name} — delete a collection
"""

import os
import tempfile
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from loguru import logger

from app.api.schemas import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SourceDocument,
)
from app.core.config import settings
from app.core.ingestion import load_document
from app.core.chunking import chunk_documents, ChunkStrategy
from app.core.retrieval import (
    get_vector_store,
    add_documents,
    collection_stats,
)
from app.models.rag_chain import build_rag_chain, build_conversational_chain, format_response

router = APIRouter()

# In-memory session store for conversational chains
# In production, replace with Redis or a persistent session store
_conversation_chains = {}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness check. Returns API status and vector store statistics."""
    try:
        store = get_vector_store()
        stats = collection_stats(store)
    except Exception as e:
        logger.warning(f"Could not fetch collection stats: {e}")
        stats = {"error": str(e)}

    return HealthResponse(
        status="ok",
        version="1.0.0",
        collection_stats=stats,
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = Form(default=None),
    chunk_strategy: str = Form(default="recursive"),
    chunk_size: int = Form(default=512),
    chunk_overlap: int = Form(default=64),
):
    """
    Upload one or more documents and index them into ChromaDB.

    Supported formats: PDF, DOCX, TXT, MD.
    """
    collection = collection_name or settings.chroma_collection_name
    all_chunks = []
    total_pages = 0

    for upload in files:
        suffix = os.path.splitext(upload.filename)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await upload.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            docs = load_document(tmp_path)
            # Restore original filename in metadata
            for doc in docs:
                doc.metadata["source"] = upload.filename

            strategy = ChunkStrategy(chunk_strategy)
            chunks = chunk_documents(
                docs,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(chunks)
            total_pages += len(docs)
            logger.info(f"Processed '{upload.filename}': {len(chunks)} chunks")

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process '{upload.filename}': {str(e)}",
            )
        finally:
            os.unlink(tmp_path)

    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No content could be extracted from the uploaded files.",
        )

    store = get_vector_store(collection_name=collection)
    add_documents(store, all_chunks)

    return IngestResponse(
        status="success",
        documents_loaded=total_pages,
        chunks_indexed=len(all_chunks),
        collection_name=collection,
        message=f"Successfully indexed {len(all_chunks)} chunks from {len(files)} file(s).",
    )


# ---------------------------------------------------------------------------
# Query (single-turn)
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Single-turn RAG query. Returns an answer grounded in indexed documents
    with source citations.
    """
    if request.search_type not in ("similarity", "mmr"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="search_type must be 'similarity' or 'mmr'.",
        )

    try:
        chain = build_rag_chain(
            collection_name=request.collection_name,
            search_type=request.search_type,
            k=request.k,
        )
        result = chain.invoke({"query": request.question})
        formatted = format_response(result)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )

    return QueryResponse(
        answer=formatted["answer"],
        sources=[SourceDocument(**s) for s in formatted["sources"]],
        num_sources=formatted["num_sources"],
        question=request.question,
        search_type=request.search_type,
        k=request.k,
    )


# ---------------------------------------------------------------------------
# Chat (multi-turn)
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse, tags=["RAG"])
async def chat(request: ChatRequest):
    """
    Multi-turn conversational RAG. Maintains chat history per session_id.
    Follow-up questions can reference prior exchanges.
    """
    session_id = request.session_id or "default"

    if session_id not in _conversation_chains:
        _conversation_chains[session_id] = build_conversational_chain()
        logger.info(f"New conversation session: {session_id}")

    chain = _conversation_chains[session_id]

    try:
        result = chain.invoke({"question": request.message})
        formatted = format_response(result)
    except Exception as e:
        logger.error(f"Chat failed (session={session_id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}",
        )

    return ChatResponse(
        answer=formatted["answer"],
        sources=[SourceDocument(**s) for s in formatted["sources"]],
        session_id=session_id,
    )
