"""PDF ingestion, chunking, and FAISS index management."""

import logging
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.state import state
from app.services.retrieval import build_retriever

logger = logging.getLogger("docmind")


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=list(settings.TEXT_SPLITTER_SEPARATORS),
    )


def load_pdf_documents(pdf_path: str) -> list:
    """Load PDF pages and enforce demo page limit."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    page_count = len(documents)
    logger.info("PDF loaded: %s pages", page_count)

    if page_count > settings.MAX_PAGES:
        raise ValueError(
            f"This demo supports PDFs up to {settings.MAX_PAGES} pages."
        )
    return documents


def process_pdf_and_update_index(
    pdf_path: str,
    replace: bool = True,
    filename: str | None = None,
) -> tuple[int, int]:
    """Process PDF file and update the in-memory and on-disk FAISS index."""
    if filename:
        state.last_indexed_filename = filename

    logger.info("Processing PDF: %s (mode=%s)", pdf_path, "REPLACE" if replace else "ADD")

    documents = load_pdf_documents(pdf_path)
    text_splitter = create_text_splitter()
    chunks = text_splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["file_name"] = state.last_indexed_filename
        chunk.metadata["chunk_id"] = chunk_index

    logger.info("Document split into %s chunks", len(chunks))

    if state.vector_store is None or replace:
        if replace and state.vector_store is not None:
            logger.info("Replacing existing FAISS index")
        else:
            logger.info("Creating new FAISS index")
        state.vector_store = FAISS.from_documents(chunks, state.embeddings_model)
    else:
        logger.info("Adding documents to existing FAISS index")
        state.vector_store.add_documents(chunks)

    state.vector_store.save_local(settings.VECTOR_STORE_PATH)
    logger.info("Index saved to '%s'", settings.VECTOR_STORE_PATH)

    state.retriever = build_retriever(state.vector_store)
    state.index_cleared = False
    state.index_pages = len(documents)
    state.index_chunks = len(chunks)

    return len(chunks), len(documents)


def clear_vector_index() -> None:
    """Reset in-memory index and prevent stale disk reload after clear."""
    state.vector_store = None
    state.retriever = None
    state.index_cleared = True
    state.reset_index_metadata()

    removed = False
    if os.path.exists(settings.VECTOR_STORE_PATH):
        shutil.rmtree(settings.VECTOR_STORE_PATH, ignore_errors=True)
        removed = not os.path.exists(settings.VECTOR_STORE_PATH)

    if removed:
        logger.info("FAISS index removed from disk: %s", settings.VECTOR_STORE_PATH)
    else:
        logger.info(
            "FAISS index cleared in memory (disk folder may remain on this host): %s",
            settings.VECTOR_STORE_PATH,
        )
