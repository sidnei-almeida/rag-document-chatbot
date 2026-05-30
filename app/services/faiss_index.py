"""FAISS index discovery, loading, and optional bootstrap from bundled documents."""

import logging
import os
from pathlib import Path

from app.core.config import settings
from app.core.state import state
from app.services.retrieval import build_retriever

logger = logging.getLogger("docmind")


def get_index_directory() -> Path:
    """Absolute path to the FAISS index folder."""
    return Path(settings.VECTOR_STORE_PATH).resolve()


def get_index_faiss_path() -> Path:
    """Path to the main FAISS index file (LangChain default: index.faiss)."""
    return get_index_directory() / "index.faiss"


def faiss_index_exists() -> bool:
    """Return True only when index.faiss is present (directory alone is not enough)."""
    return get_index_faiss_path().is_file()


def try_load_faiss_from_disk() -> bool:
    """
    Load FAISS index from disk when index.faiss exists.

    Never raises — returns False and leaves RAG disabled on failure.
    """
    if state.index_cleared:
        logger.info("Index was cleared; skipping FAISS reload from disk")
        return False

    index_dir = get_index_directory()
    index_file = get_index_faiss_path()

    if not index_file.is_file():
        logger.info("FAISS index not found at %s", index_file)
        return False

    if state.embeddings_model is None:
        logger.warning("Embeddings model not ready; cannot load FAISS index yet")
        return False

    try:
        from langchain_community.vectorstores import FAISS

        logger.info("FAISS index found, loading...")
        state.vector_store = FAISS.load_local(
            str(index_dir),
            state.embeddings_model,
            allow_dangerous_deserialization=True,
        )
        state.retriever = build_retriever(state.vector_store)
        logger.info("FAISS index loaded successfully")
        return True
    except Exception as exc:
        logger.warning("Could not load FAISS index from '%s': %s", index_dir, exc)
        state.vector_store = None
        state.retriever = None
        return False


def log_rag_disabled() -> None:
    """Log that RAG endpoints need upload or load-sample."""
    logger.info("RAG disabled until an index is created or uploaded")


def maybe_bootstrap_index_from_bundled_documents() -> bool:
    """
    When index.faiss is missing, optionally build an index from the bundled sample PDF.

    Returns True when an index is available after this step.
    """
    if not settings.AUTO_LOAD_SAMPLE_ON_STARTUP:
        return False

    try:
        from app.services.sample_loader import resolve_sample_pdf_path, load_sample_document

        sample_path = resolve_sample_pdf_path()
    except FileNotFoundError:
        logger.info("No bundled sample PDF available for automatic index creation")
        return False

    if state.embeddings_model is None:
        return False

    try:
        logger.info("No FAISS index on disk; building from bundled sample: %s", sample_path)
        load_sample_document()
        return state.is_index_ready()
    except Exception as exc:
        logger.warning("Automatic sample index build failed: %s", exc)
        state.vector_store = None
        state.retriever = None
        return False
