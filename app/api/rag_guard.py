"""Guards for RAG-dependent API routes."""

from fastapi import HTTPException

from app.core.state import state

RAG_UNAVAILABLE_DETAIL = (
    "RAG is disabled until a document index is ready. "
    "Use POST /demo/load-sample or POST /upload to index a PDF first."
)


def require_api_ready() -> None:
    if state.embeddings_model is None:
        raise HTTPException(status_code=503, detail="API is still initializing.")


def require_llm_ready() -> None:
    require_api_ready()
    if state.llm is None:
        raise HTTPException(status_code=503, detail="LLM is not configured. Set GROQ_API_KEY.")


def require_rag_index() -> None:
    """Raise 503 when document retrieval is not available."""
    from app.services.retrieval import ensure_retriever_ready

    require_llm_ready()
    ensure_retriever_ready()
    if not state.is_index_ready():
        raise HTTPException(status_code=503, detail=RAG_UNAVAILABLE_DETAIL)
