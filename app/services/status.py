"""Health and status payload builders."""

from app.core.config import settings
from app.core.state import state
from app.services.retrieval import ensure_retriever_ready


def api_status_label() -> str:
    """Return ok when LLM is ready, initializing otherwise."""
    return "ok" if state.llm is not None else "initializing"


def build_health_payload() -> dict:
    """Technical health check for monitors and frontend boot."""
    ensure_retriever_ready()
    return {
        "status": api_status_label(),
        "index_ready": state.is_index_ready(),
        "model": settings.GROQ_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "retrieval": settings.retrieval_dict(),
        "limits": settings.limits_dict(),
    }


def build_status_payload() -> dict:
    """Document/index state for demo UI."""
    ensure_retriever_ready()
    return {
        "status": api_status_label(),
        "index_ready": state.is_index_ready(),
        "document_loaded": state.document_loaded(),
        "file_name": state.last_indexed_filename if state.document_loaded() else None,
        "pages": state.index_pages if state.document_loaded() else None,
        "chunks": state.index_chunks if state.document_loaded() else None,
        "model": settings.GROQ_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "retrieval": {
            "type": state.retriever_type,
            "k": settings.RETRIEVAL_K,
        },
        "limits": settings.limits_dict(),
    }
