# rag-pipeline-demo

> **Production-style Retrieval-Augmented Generation (RAG) pipeline with LangChain, ChromaDB, and FastAPI**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat-square)](https://langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6F61?style=flat-square)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-pytest-green?style=flat-square&logo=pytest)](https://pytest.org/)

---

## Overview

This repository demonstrates a production-ready RAG pipeline built with:

- **LangChain** for orchestration and chain assembly
- **ChromaDB** as the persistent vector store
- **FastAPI** for a documented REST API with Swagger UI
- **OpenAI / HuggingFace** embeddings (configurable via `.env`)
- **RAGAS** for end-to-end RAG quality evaluation

The pipeline covers the full lifecycle: document ingestion → chunking → embedding → retrieval → generation → evaluation.

---

## Architecture

```
Documents (PDF/DOCX/TXT/MD)
        │
        ▼
┌─────────────────┐
│   Ingestion     │  load_document() / load_directory()
│   Pipeline      │  Supports PDF, DOCX, TXT, Markdown
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking     │  Fixed / Recursive / Token / Semantic
│   Strategies    │  Configurable size and overlap
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │  OpenAI text-embedding-3-small
│     Layer       │  or HuggingFace (local, no API key)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    ChromaDB     │  Persistent vector store
│  Vector Store   │  Cosine similarity indexing
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
Similarity   MMR
 Search    Search   ← configurable per-request
    └────┬────┘
         │
         ▼
┌─────────────────┐
│   RAG Chain     │  RetrievalQA / ConversationalRetrievalChain
│   (LangChain)   │  Grounded prompt template
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI       │  REST endpoints with Pydantic validation
│     API         │  Single-turn query + multi-turn chat
└─────────────────┘
```

---

## Features

- **Four chunking strategies** — Fixed, Recursive (default), Token-based, and Semantic — with a comparison utility to evaluate trade-offs on your own corpus
- **Two retrieval modes** — Cosine similarity search and Maximum Marginal Relevance (MMR) for diversity-aware retrieval
- **Pluggable embeddings** — swap between OpenAI and HuggingFace models via a single environment variable
- **Conversational RAG** — multi-turn chat endpoint with rolling memory window
- **Source citations** — every response includes the source document and excerpt
- **Retrieval evaluation** — Hit Rate, MRR, and Precision@k metrics against a labelled test set
- **RAGAS integration** — Faithfulness, Answer Relevancy, and Context Recall scoring
- **Fully Dockerised** — single `docker-compose up` deployment
- **OpenAPI docs** — auto-generated Swagger UI at `/docs`

---

## Project Structure

```
rag-pipeline-demo/
├── app/
│   ├── api/
│   │   ├── routes.py          # FastAPI endpoints (ingest, query, chat, health)
│   │   └── schemas.py         # Pydantic request/response models
│   ├── core/
│   │   ├── config.py          # Pydantic Settings — all config from .env
│   │   ├── ingestion.py       # Document loaders (PDF, DOCX, TXT, MD)
│   │   ├── chunking.py        # Four chunking strategies + comparison utility
│   │   ├── embeddings.py      # Embedding factory + model benchmarking
│   │   ├── retrieval.py       # ChromaDB vector store + similarity/MMR search
│   │   └── evaluation.py      # Retrieval metrics (HitRate, MRR) + RAGAS
│   ├── models/
│   │   └── rag_chain.py       # RetrievalQA and ConversationalRetrievalChain
│   └── main.py                # FastAPI app entrypoint
├── scripts/
│   ├── ingest_docs.py         # CLI: batch ingest a directory
│   ├── evaluate_retrieval.py  # CLI: evaluate retrieval against labelled set
│   └── compare_embeddings.py  # CLI: benchmark embedding models
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_api.py
├── notebooks/
│   ├── 01_chunking_strategies.ipynb   # Visualise chunk size distributions
│   ├── 02_embedding_comparison.ipynb  # Compare embedding models
│   └── 03_retrieval_evaluation.ipynb  # End-to-end RAGAS evaluation
├── data/
│   └── raw/                   # Place source documents here
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/I-Man-H/rag-pipeline-demo.git
cd rag-pipeline-demo
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ingest documents

Place PDF, DOCX, or TXT files in `data/raw/`, then run:

```bash
python scripts/ingest_docs.py --source data/raw --strategy recursive
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

Interactive docs available at: **http://localhost:8000/docs**

### 5. Run with Docker

```bash
docker-compose up --build
```

---

## API Reference

### `POST /api/v1/ingest`

Upload and index one or more documents.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "files=@report.pdf" \
  -F "chunk_strategy=recursive" \
  -F "chunk_size=512"
```

### `POST /api/v1/query`

Single-turn RAG query with source citations.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "k": 5, "search_type": "mmr"}'
```

**Response:**
```json
{
  "answer": "The main findings indicate...",
  "sources": [
    {
      "source": "report.pdf",
      "page": 3,
      "excerpt": "The study concludes that..."
    }
  ],
  "num_sources": 2,
  "question": "What are the main findings?",
  "search_type": "mmr",
  "k": 5
}
```

### `POST /api/v1/chat`

Multi-turn conversational RAG with session memory.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarise the methodology.", "session_id": "user-123"}'
```

### `GET /api/v1/health`

Liveness check with vector store statistics.

---

## Chunking Strategies

| Strategy | Description | Best for |
|---|---|---|
| `fixed` | Splits at a fixed character count | Uniform corpora, fast baseline |
| `recursive` | Respects paragraph/sentence boundaries | General documents (default) |
| `token` | Splits on token count (LLM-aware) | Precise context window control |
| `semantic` | Groups sentences by embedding similarity | Conceptually dense documents |

Run `notebooks/01_chunking_strategies.ipynb` to visualise chunk size distributions across strategies on your own corpus.

---

## Retrieval Modes

| Mode | Description | When to use |
|---|---|---|
| `similarity` | Standard cosine similarity top-k | Fast; most use cases |
| `mmr` | Maximum Marginal Relevance | When retrieved chunks are repetitive |

---

## Evaluation

### Retrieval metrics (no LLM required)

```bash
python scripts/evaluate_retrieval.py \
  --test-file data/eval_set.json \
  --k 5 \
  --output results/retrieval_report.csv
```

| Metric | Description |
|---|---|
| Hit Rate | Fraction of queries where a correct chunk appears in top-k |
| MRR | Mean Reciprocal Rank of the first correct result |
| Precision@k | Fraction of top-k results that are relevant |

### RAG quality metrics (RAGAS)

See `notebooks/03_retrieval_evaluation.ipynb` for end-to-end RAGAS scoring:

| Metric | Description |
|---|---|
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer Relevancy | Does the answer address the question? |
| Context Recall | Does the context cover the ground truth answer? |

---

## Configuration

All settings are managed via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for OpenAI embeddings and GPT models |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Set to a HuggingFace model name for local embeddings |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | `512` | Default chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `RETRIEVAL_TOP_K` | `5` | Default number of chunks to retrieve |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB persistence directory |

To run without an OpenAI API key, set:
```
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=  # configure a local LLM via Ollama or LM Studio
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Contact

**Iman Hosseini** — PhD Candidate, University of Canberra
[LinkedIn](https://www.linkedin.com/in/i-man-hosseini/) · [Google Scholar](https://scholar.google.com/citations?user=ZBlw7J0AAAAJ) · [GitHub](https://github.com/I-Man-H)
