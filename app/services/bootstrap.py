"""Application startup: embeddings, FAISS, and Groq LLM."""

import logging
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.state import state
from app.services.retrieval import build_retriever

logger = logging.getLogger("docmind")


def initialize_models() -> None:
    """Load embeddings, optional FAISS index, and Groq LLM."""
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured. Set the GROQ_API_KEY environment variable.")

    logger.info("Loading embeddings model: %s", settings.EMBEDDING_MODEL_NAME)
    state.embeddings_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    if state.index_cleared:
        logger.info("Index was cleared; skipping automatic FAISS reload")
        state.vector_store = None
        state.retriever = None
    elif os.path.exists(settings.VECTOR_STORE_PATH):
        try:
            from langchain_community.vectorstores import FAISS

            logger.info("Loading FAISS index from '%s'", settings.VECTOR_STORE_PATH)
            state.vector_store = FAISS.load_local(
                settings.VECTOR_STORE_PATH,
                state.embeddings_model,
                allow_dangerous_deserialization=True,
            )
            state.retriever = build_retriever(state.vector_store)
            logger.info("Vector database loaded successfully")
        except Exception as exc:
            logger.warning("Could not load FAISS index: %s", exc)
            state.vector_store = None
            state.retriever = None
    else:
        logger.info("No FAISS index on disk; waiting for PDF upload")

    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
    logger.info(
        "Connecting to Groq model=%s temperature=%s max_tokens=%s",
        settings.GROQ_MODEL,
        settings.GROQ_TEMPERATURE,
        settings.GROQ_MAX_TOKENS,
    )
    state.llm = ChatGroq(
        model_name=settings.GROQ_MODEL,
        temperature=settings.GROQ_TEMPERATURE,
        max_tokens=settings.GROQ_MAX_TOKENS,
    )
    logger.info("Groq LLM configured successfully")
