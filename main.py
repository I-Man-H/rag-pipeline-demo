"""
FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router

app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "Production-style Retrieval-Augmented Generation API. "
        "Supports document ingestion, chunking strategies, embedding comparison, "
        "similarity and MMR retrieval, and conversational QA."
    ),
    version="1.0.0",
    contact={
        "name": "Iman Hosseini",
        "url": "https://github.com/I-Man-H",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    logger.info("RAG Pipeline API starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("RAG Pipeline API shutting down.")
