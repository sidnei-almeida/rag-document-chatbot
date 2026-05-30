"""PDF ingestion, chunking, and FAISS index management."""

import logging
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.state import state
from app.services.conversation import (
    record_indexed_document,
    validate_can_add_document,
)
from app.services.retrieval import build_retriever

logger = logging.getLogger("docmind")


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=list(settings.TEXT_SPLITTER_SEPARATORS),
    )


def load_pdf_documents(pdf_path: str) -> list:
    """Load PDF pages and enforce demo page limit for a single file."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    page_count = len(documents)
    logger.info("PDF loaded: %s pages", page_count)

    if page_count > settings.MAX_PAGES:
        raise ValueError(
            f"This demo supports PDFs up to {settings.MAX_PAGES} pages per file."
        )
    return documents


def _assign_chunk_metadata(chunks: list, file_name: str) -> None:
    """Assign stable chunk ids and file names within the current conversation."""
    for chunk in chunks:
        chunk.metadata["file_name"] = file_name
        chunk.metadata["chunk_id"] = state.next_chunk_id
        if state.conversation_id:
            chunk.metadata["conversation_id"] = state.conversation_id
        state.next_chunk_id += 1


def process_pdf_and_update_index(
    pdf_path: str,
    *,
    replace: bool = True,
    filename: str | None = None,
) -> tuple[int, int]:
    """
    Process PDF and update the in-memory and on-disk FAISS index.

    When replace=True, builds a new index from this file only (used for the first
    document in a conversation). When replace=False, appends to the current index.
    """
    file_name = filename or os.path.basename(pdf_path)

    logger.info(
        "Processing PDF: %s (mode=%s, conversation=%s)",
        file_name,
        "REPLACE" if replace else "ADD",
        state.conversation_id,
    )

    documents = load_pdf_documents(pdf_path)
    page_count = len(documents)

    if not replace:
        validate_can_add_document(page_count)

    text_splitter = create_text_splitter()
    chunks = text_splitter.split_documents(documents)
    _assign_chunk_metadata(chunks, file_name)

    logger.info("Document split into %s chunks", len(chunks))

    if state.vector_store is None or replace:
        if replace and state.vector_store is not None:
            logger.info("Replacing FAISS index for current conversation")
        else:
            logger.info("Creating new FAISS index")
        state.vector_store = FAISS.from_documents(chunks, state.embeddings_model)
    else:
        logger.info("Adding document to conversation index (%s)", state.conversation_id)
        state.vector_store.add_documents(chunks)

    state.vector_store.save_local(settings.VECTOR_STORE_PATH)
    logger.info("Index saved to '%s'", settings.VECTOR_STORE_PATH)

    state.retriever = build_retriever(state.vector_store)
    state.index_cleared = False
    record_indexed_document(file_name, page_count, len(chunks))

    return len(chunks), page_count


def clear_vector_index() -> None:
    """Reset in-memory index and prevent stale disk reload after clear."""
    from app.services.conversation import end_conversation

    state.vector_store = None
    state.retriever = None
    state.index_cleared = True
    state.reset_index_metadata()
    end_conversation()

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
