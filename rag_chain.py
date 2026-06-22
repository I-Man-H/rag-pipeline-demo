"""
RAG chain assembly.

Builds a LangChain RetrievalQA chain that:
  1. Retrieves relevant chunks from ChromaDB
  2. Formats them into a structured prompt
  3. Generates a grounded answer via the LLM
  4. Returns the answer with source citations

Two chain types are provided:
  - build_rag_chain:        Simple RetrievalQA (fast)
  - build_conversational_chain: Conversational RAG with chat history
"""

from typing import List, Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from loguru import logger

from app.core.config import settings
from app.core.retrieval import get_retriever, get_vector_store


RAG_PROMPT_TEMPLATE = """You are a precise and helpful assistant. Answer the question
using ONLY the information provided in the context below.

If the context does not contain enough information to answer the question,
say "I do not have enough information in the provided documents to answer this."
Do not make up or infer information not present in the context.

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def get_llm(model: str = None, temperature: float = None) -> ChatOpenAI:
    """Instantiate the LLM with settings from config."""
    return ChatOpenAI(
        model=model or settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        openai_api_key=settings.openai_api_key,
    )


def build_rag_chain(
    collection_name: str = None,
    search_type: str = "similarity",
    k: int = None,
    model: str = None,
    return_source_documents: bool = True,
) -> RetrievalQA:
    """
    Build a standard RetrievalQA chain.

    Args:
        collection_name:        ChromaDB collection to query.
        search_type:            'similarity' or 'mmr'.
        k:                      Number of chunks to retrieve.
        model:                  LLM model name override.
        return_source_documents: Include source chunks in response.

    Returns:
        Configured RetrievalQA chain.
    """
    vector_store = get_vector_store(collection_name=collection_name)
    retriever = get_retriever(vector_store, search_type=search_type, k=k)
    llm = get_llm(model=model)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=return_source_documents,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    logger.info(
        f"RAG chain ready: {search_type} retrieval, "
        f"k={k or settings.retrieval_top_k}, model={model or settings.llm_model}"
    )
    return chain


def build_conversational_chain(
    collection_name: str = None,
    k: int = None,
    model: str = None,
    memory_window: int = 5,
) -> ConversationalRetrievalChain:
    """
    Build a conversational RAG chain with rolling message memory.

    Maintains the last `memory_window` exchanges so follow-up
    questions can reference prior context.

    Args:
        collection_name: ChromaDB collection to query.
        k:               Number of chunks to retrieve.
        model:           LLM model name override.
        memory_window:   Number of past exchanges to retain.

    Returns:
        Configured ConversationalRetrievalChain.
    """
    vector_store = get_vector_store(collection_name=collection_name)
    retriever = get_retriever(vector_store, search_type="mmr", k=k)
    llm = get_llm(model=model)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=memory_window,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )

    logger.info("Conversational RAG chain ready.")
    return chain


def format_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a chain result into a clean API-ready response.

    Extracts the answer and deduplicates source citations.
    """
    answer = result.get("result") or result.get("answer", "")
    source_docs: List[Document] = result.get("source_documents", [])

    sources = []
    seen = set()
    for doc in source_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        key = f"{source}:{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": source,
                "page": page,
                "excerpt": doc.page_content[:200].strip(),
            })

    return {
        "answer": answer.strip(),
        "sources": sources,
        "num_sources": len(sources),
    }
