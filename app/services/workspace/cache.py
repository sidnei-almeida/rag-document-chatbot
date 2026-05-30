"""Per-workspace FAISS vector store and retriever caches."""

import logging
from typing import Any

from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.core.state import state
from app.services.workspace.paths import faiss_dir, faiss_index_file

logger = logging.getLogger("docmind")

vectorstore_cache: dict[str, FAISS] = {}
retriever_cache: dict[str, Any] = {}
retriever_type_cache: dict[str, str] = {}


def invalidate_workspace_cache(workspace_id: str) -> None:
    vectorstore_cache.pop(workspace_id, None)
    retriever_cache.pop(workspace_id, None)
    retriever_type_cache.pop(workspace_id, None)
    logger.info("Invalidated in-memory cache for workspace %s", workspace_id)


def clear_all_caches() -> None:
    vectorstore_cache.clear()
    retriever_cache.clear()
    retriever_type_cache.clear()


def _build_retriever(store: FAISS, workspace_id: str):
    try:
        retriever_type_cache[workspace_id] = "mmr"
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.RETRIEVAL_K,
                "fetch_k": settings.RETRIEVAL_FETCH_K,
                "lambda_mult": settings.RETRIEVAL_LAMBDA,
            },
        )
    except Exception as exc:
        logger.warning(
            "MMR retriever unavailable for %s, falling back to similarity: %s",
            workspace_id,
            exc,
        )
        retriever_type_cache[workspace_id] = "similarity"
        return store.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})


def load_vectorstore_for_workspace(workspace_id: str) -> FAISS:
    if workspace_id in vectorstore_cache:
        return vectorstore_cache[workspace_id]

    if state.embeddings_model is None:
        raise RuntimeError("Embeddings model is not initialized.")

    index_file = faiss_index_file(workspace_id)
    if not index_file.is_file():
        raise FileNotFoundError(
            f"FAISS index not found for workspace '{workspace_id}'."
        )

    store = FAISS.load_local(
        str(faiss_dir(workspace_id)),
        state.embeddings_model,
        allow_dangerous_deserialization=True,
    )
    vectorstore_cache[workspace_id] = store
    logger.info("Loaded FAISS index for workspace %s", workspace_id)
    return store


def get_retriever_for_workspace(workspace_id: str):
    if workspace_id in retriever_cache:
        return retriever_cache[workspace_id]

    store = load_vectorstore_for_workspace(workspace_id)
    retriever = _build_retriever(store, workspace_id)
    retriever_cache[workspace_id] = retriever
    return retriever


def workspace_index_ready(workspace_id: str) -> bool:
    return faiss_index_file(workspace_id).is_file()
