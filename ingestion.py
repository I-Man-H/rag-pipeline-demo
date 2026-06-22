"""
Document ingestion pipeline.

Supports PDF, DOCX, TXT, and Markdown files.
Tracks metadata (source, page, file type) through the pipeline.
"""

import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    DirectoryLoader,
)
from loguru import logger


SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_document(file_path: str) -> List[Document]:
    """Load a single document and attach source metadata."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    loader_cls = SUPPORTED_EXTENSIONS[ext]
    loader = loader_cls(str(path))
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "source": path.name,
            "file_path": str(path.resolve()),
            "file_type": ext.lstrip("."),
        })

    logger.info(f"Loaded {len(docs)} page(s) from {path.name}")
    return docs


def load_directory(directory: str, glob: str = "**/*") -> List[Document]:
    """
    Recursively load all supported documents from a directory.

    Args:
        directory: Path to directory containing documents.
        glob: Glob pattern for file matching.

    Returns:
        List of LangChain Document objects with metadata.
    """
    all_docs: List[Document] = []
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for ext, loader_cls in SUPPORTED_EXTENSIONS.items():
        pattern = f"**/*{ext}"
        files = list(directory_path.glob(pattern))

        for file_path in files:
            try:
                docs = load_document(str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

    logger.info(
        f"Ingestion complete: {len(all_docs)} total pages from {directory}"
    )
    return all_docs
