"""Health and status payload builders."""

from app.core.config import settings
from app.core.state import state


def api_status_label() -> str:
    """Return ok when core API finished startup."""
    return "ok" if state.is_api_ready() else "initializing"


def build_health_payload() -> dict:
    """Technical health check for monitors and frontend boot."""
    return {
        "status": api_status_label(),
        "api_ready": state.is_api_ready(),
        "llm_ready": state.is_llm_ready(),
        "index_ready": state.is_index_ready(),
        "index_path": state.index_path_display(),
        "model": settings.GROQ_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "retrieval": settings.retrieval_dict(),
        "limits": settings.limits_dict(),
    }


def build_status_payload() -> dict:
    """Document/index state for demo UI."""
    return {
        "status": api_status_label(),
        "api_ready": state.is_api_ready(),
        "llm_ready": state.is_llm_ready(),
        "index_ready": state.is_index_ready(),
        "index_path": state.index_path_display(),
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
