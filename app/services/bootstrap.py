"""Application startup: embeddings and Groq LLM (workspace indexes load on demand)."""

import logging
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.state import state

logger = logging.getLogger("docmind")


def _maybe_bootstrap_sample_workspace() -> None:
    if not settings.AUTO_LOAD_SAMPLE_ON_STARTUP:
        return
    from app.services.workspace.registry import get_default_workspace_id

    if get_default_workspace_id():
        logger.info("Sample workspace already present; skipping auto-load")
        return
    try:
        from app.services.sample_loader import load_sample_workspace

        load_sample_workspace()
    except FileNotFoundError:
        logger.info("No bundled sample PDF for workspace bootstrap")
    except Exception as exc:
        logger.warning("Sample workspace bootstrap failed: %s", exc)


def initialize_models() -> None:
    """Load embeddings and Groq LLM. Workspace FAISS indexes are loaded per request."""
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured. Set the GROQ_API_KEY environment variable.")

    logger.info("Loading embeddings model: %s", settings.EMBEDDING_MODEL_NAME)
    state.embeddings_model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    _maybe_bootstrap_sample_workspace()

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
        "Startup complete — api_ready=%s llm_ready=%s",
        state.is_api_ready(),
        state.is_llm_ready(),
    )
