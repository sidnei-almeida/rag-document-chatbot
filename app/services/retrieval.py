"""Document retrieval, source formatting, and confidence heuristics."""

import logging
import os
import re
from typing import Any, Literal, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import settings
from app.core.state import state

logger = logging.getLogger("docmind")


def build_retriever(store: FAISS):
    """Build a retriever with MMR search, falling back to similarity search."""
    try:
        state.retriever_type = "mmr"
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.RETRIEVAL_K,
                "fetch_k": settings.RETRIEVAL_FETCH_K,
                "lambda_mult": settings.RETRIEVAL_LAMBDA,
            },
        )
    except Exception as exc:
        logger.warning("MMR retriever unavailable, falling back to similarity: %s", exc)
        state.retriever_type = "similarity"
        return store.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})


def ensure_retriever_ready() -> None:
    """Load retriever from disk when available and index has not been cleared."""
    if state.index_cleared:
        return
    if state.retriever is not None:
        return
    if state.embeddings_model is None:
        logger.warning("Embeddings model not initialized; cannot load retriever yet")
        return
    if not os.path.exists(settings.VECTOR_STORE_PATH):
        return

    try:
        logger.info("Loading FAISS index from '%s'", settings.VECTOR_STORE_PATH)
        state.vector_store = FAISS.load_local(
            settings.VECTOR_STORE_PATH,
            state.embeddings_model,
            allow_dangerous_deserialization=True,
        )
        state.retriever = build_retriever(state.vector_store)
        logger.info("Retriever initialized from saved index")
    except Exception as exc:
        logger.warning("Failed to load retriever from disk: %s", exc)
        state.vector_store = None
        state.retriever = None


def retrieve_documents(question: str) -> tuple[list[Document], Optional[list[float]]]:
    """Retrieve documents, preferring similarity scores when available."""
    if state.vector_store is not None:
        try:
            results = state.vector_store.similarity_search_with_score(
                question, k=settings.RETRIEVAL_K
            )
            docs = [doc for doc, _ in results]
            scores = [float(score) for _, score in results]
            logger.info("Retrieved %s chunks via similarity_search_with_score", len(docs))
            return docs, scores
        except Exception as exc:
            logger.warning("similarity_search_with_score failed, using retriever: %s", exc)

    if state.retriever is not None:
        docs = state.retriever.invoke(question)
        logger.info("Retrieved %s chunks via retriever fallback", len(docs))
        return docs, None

    return [], None


def clean_preview(text: str, max_length: int | None = None) -> str:
    """Normalize whitespace and truncate chunk text for API previews."""
    limit = max_length or settings.PREVIEW_MAX_LENGTH
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def extract_page_index(metadata: dict) -> int:
    """Return 0-indexed page number from document metadata."""
    raw = metadata.get("page", metadata.get("page_number", 0))
    if isinstance(raw, int):
        return max(0, raw)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def extract_file_name(metadata: dict) -> str:
    """Resolve a human-readable file name from chunk metadata."""
    for key in ("file_name", "filename", "source"):
        value = metadata.get(key)
        if not value:
            continue
        if key == "source":
            return os.path.basename(str(value))
        return str(value)
    return state.last_indexed_filename


def format_sources(
    docs: list[Document],
    scores: Optional[list[float]] = None,
) -> list[dict[str, Any]]:
    """Format retrieved documents into structured source objects."""
    sources: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        content = doc.page_content.strip()
        if not content:
            continue

        metadata = doc.metadata or {}
        page_index = extract_page_index(metadata)
        chunk_id = metadata.get("chunk_id", metadata.get("id", idx))

        score: Optional[float] = None
        if scores is not None and idx < len(scores):
            score = round(float(scores[idx]), 4)

        sources.append(
            {
                "chunk_id": chunk_id,
                "page": page_index + 1,
                "page_index": page_index,
                "file_name": extract_file_name(metadata),
                "preview": clean_preview(content),
                "score": score,
            }
        )
    return sources


def filter_useful_documents_with_scores(
    docs: list[Document],
    scores: Optional[list[float]] = None,
) -> tuple[list[Document], Optional[list[float]]]:
    """Keep non-empty chunks and align similarity scores when present."""
    useful_docs: list[Document] = []
    useful_scores: list[float] = []
    has_scores = scores is not None

    for idx, doc in enumerate(docs):
        if not doc.page_content.strip():
            continue
        useful_docs.append(doc)
        if has_scores and scores is not None and idx < len(scores):
            useful_scores.append(scores[idx])

    if has_scores and useful_scores:
        return useful_docs, useful_scores
    return useful_docs, None


def format_context_from_documents(docs: list[Document]) -> str:
    """Build LLM context with humanized page numbers (1-indexed)."""
    context_parts = []
    for doc in docs:
        page_index = extract_page_index(doc.metadata or {})
        page_display = page_index + 1
        content = doc.page_content.strip()
        if content:
            context_parts.append(f"[Page {page_display}]\n{content}")
    return "\n\n---\n\n".join(context_parts)


def compute_confidence(
    useful_chunk_count: int,
    has_context: bool,
    scores: Optional[list[float]] = None,
) -> tuple[Literal["high", "medium", "low"], str]:
    """Heuristic confidence based on retrieved evidence."""
    _ = scores
    if not has_context or useful_chunk_count == 0:
        return "low", "No reliable document context was found."
    if useful_chunk_count >= 4:
        return "high", f"Answer based on {useful_chunk_count} retrieved passages."
    if useful_chunk_count >= 2:
        return "medium", f"Answer based on {useful_chunk_count} retrieved passages."
    return "low", "Only 1 relevant passage was found."


def build_response_metadata(chunks_used: int) -> dict[str, Any]:
    return {
        "model": settings.GROQ_MODEL,
        "retrieval_k": settings.RETRIEVAL_K,
        "chunks_used": chunks_used,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
    }


def build_ask_response(
    answer: str,
    sources: list[dict[str, Any]],
    confidence_label: Literal["high", "medium", "low"],
    confidence_reason: str,
    chunks_used: int = 0,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "sources": sources,
        "confidence": {
            "label": confidence_label,
            "reason": confidence_reason,
        },
        "metadata": build_response_metadata(chunks_used),
    }
