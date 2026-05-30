"""Guards for API routes."""

from fastapi import HTTPException

from app.core.state import state
from app.services.workspace.cache import workspace_index_ready
from app.services.workspace.service import workspace_exists

RAG_UNAVAILABLE_DETAIL = (
    "RAG is disabled for this workspace. Upload PDFs via POST /workspaces/upload first."
)


def require_api_ready() -> None:
    if state.embeddings_model is None:
        raise HTTPException(status_code=503, detail="API is still initializing.")


def require_llm_ready() -> None:
    require_api_ready()
    if state.llm is None:
        raise HTTPException(status_code=503, detail="LLM is not configured. Set GROQ_API_KEY.")


def require_workspace_ready(workspace_id: str) -> None:
    """Raise 404/503 when workspace or its FAISS index is unavailable."""
    require_llm_ready()
    if not workspace_exists(workspace_id):
        raise HTTPException(
            status_code=404,
            detail=f"Workspace '{workspace_id}' not found.",
        )
    if not workspace_index_ready(workspace_id):
        raise HTTPException(
            status_code=503,
            detail=RAG_UNAVAILABLE_DETAIL,
        )
