"""Application startup: embeddings, FAISS, and Groq LLM."""

import logging
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.state import state
from app.services.faiss_index import (
    faiss_index_exists,
    log_rag_disabled,
    maybe_bootstrap_index_from_bundled_documents,
    try_load_faiss_from_disk,
)

logger = logging.getLogger("docmind")


def initialize_models() -> None:
    """Load embeddings, optional FAISS index, and Groq LLM. Never fails on missing index."""
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured. Set the GROQ_API_KEY environment variable.")

    logger.info("Loading embeddings model: %s", settings.EMBEDDING_MODEL_NAME)
    state.embeddings_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    if faiss_index_exists():
        loaded = try_load_faiss_from_disk()
        if not loaded:
            log_rag_disabled()
    else:
        bootstrapped = maybe_bootstrap_index_from_bundled_documents()
        if not bootstrapped:
            log_rag_disabled()

    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
    logger.info(
        "Connecting to Groq model=%s temperature=%s max_tokens=%s",
        settings.GROQ_MODEL,
        settings.GROQ_TEMPERATURE,
        settings.GROQ_MAX_TOKENS,
    )
    try:
        state.llm = ChatGroq(
            model_name=settings.GROQ_MODEL,
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
        )
        logger.info("Groq LLM configured successfully")
    except Exception as exc:
        logger.error("Failed to configure Groq LLM: %s", exc)
        state.llm = None
        raise

    logger.info(
        "Startup complete — api_ready=%s llm_ready=%s index_ready=%s index_path=%s",
        state.is_api_ready(),
        state.is_llm_ready(),
        state.is_index_ready(),
        state.index_path_display(),
    )
