"""Workspace-scoped document retrieval and source formatting."""

import logging
import re
import time
from typing import Any, Literal, Optional

from langchain_core.documents import Document

from app.core.config import settings
from app.services.workspace.cache import get_retriever_for_workspace, load_vectorstore_for_workspace

logger = logging.getLogger("docmind")


def _filter_docs_for_workspace(workspace_id: str, docs: list[Document]) -> list[Document]:
    """Drop chunks whose metadata workspace_id does not match (safety guard)."""
    filtered: list[Document] = []
    for doc in docs:
        meta_ws = (doc.metadata or {}).get("workspace_id")
        if meta_ws and meta_ws != workspace_id:
            logger.warning(
                "Discarding retrieved chunk from workspace %s (requested %s)",
                meta_ws,
                workspace_id,
            )
            continue
        filtered.append(doc)
    return filtered


def retrieve_documents_for_workspace(
    workspace_id: str,
    question: str,
) -> tuple[list[Document], Optional[list[float]]]:
    """Retrieve chunks only from the given workspace FAISS index."""
    store = load_vectorstore_for_workspace(workspace_id)
    try:
        results = store.similarity_search_with_score(question, k=settings.RETRIEVAL_K)
        pairs: list[tuple[Document, float]] = []
        for doc, score in results:
            meta_ws = (doc.metadata or {}).get("workspace_id")
            if meta_ws and meta_ws != workspace_id:
                logger.warning(
                    "Discarding retrieved chunk from workspace %s (requested %s)",
                    meta_ws,
                    workspace_id,
                )
                continue
            pairs.append((doc, float(score)))
        docs = [doc for doc, _ in pairs]
        scores = [score for _, score in pairs] if pairs else None
        logger.info(
            "Workspace %s: retrieved %s chunks via similarity_search_with_score",
            workspace_id,
            len(docs),
        )
        return docs, scores
    except Exception as exc:
        logger.warning(
            "Workspace %s: similarity_search_with_score failed (%s), using retriever",
            workspace_id,
            exc,
        )

    retriever = get_retriever_for_workspace(workspace_id)
    docs = _filter_docs_for_workspace(workspace_id, retriever.invoke(question))
    logger.info("Workspace %s: retrieved %s chunks via retriever", workspace_id, len(docs))
    return docs, None


def clean_preview(text: str, max_length: int | None = None) -> str:
    limit = max_length or settings.PREVIEW_MAX_LENGTH
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _metadata_filename(metadata: dict) -> str:
    return str(metadata.get("filename") or metadata.get("file_name") or "document.pdf")


def extract_page_index(metadata: dict) -> int:
    raw = metadata.get("page", metadata.get("page_number", 0))
    if isinstance(raw, int):
        return max(0, raw)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def format_sources_for_workspace(
    workspace_id: str,
    docs: list[Document],
    scores: Optional[list[float]] = None,
) -> list[dict[str, Any]]:
    """Format sources; discard any chunk whose workspace_id does not match."""
    sources: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        content = doc.page_content.strip()
        if not content:
            continue

        metadata = doc.metadata or {}
        chunk_workspace = metadata.get("workspace_id")
        if chunk_workspace and chunk_workspace != workspace_id:
            logger.warning(
                "Discarding chunk from workspace %s (requested %s)",
                chunk_workspace,
                workspace_id,
            )
            continue

        page_index = extract_page_index(metadata)
        chunk_id = metadata.get("chunk_id", f"chunk_{idx:03d}")
        filename = _metadata_filename(metadata)
        document_id = metadata.get("document_id", "unknown")

        score: Optional[float] = None
        if scores is not None and idx < len(scores):
            score = round(float(scores[idx]), 4)

        sources.append(
            {
                "workspace_id": workspace_id,
                "document_id": document_id,
                "filename": filename,
                "page": page_index + 1,
                "chunk_id": chunk_id,
                "score": score,
                "preview": clean_preview(content),
            }
        )
    return sources


def filter_useful_documents_with_scores(
    docs: list[Document],
    scores: Optional[list[float]] = None,
) -> tuple[list[Document], Optional[list[float]]]:
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
    context_parts = []
    for doc in docs:
        metadata = doc.metadata or {}
        page_index = extract_page_index(metadata)
        filename = _metadata_filename(metadata)
        content = doc.page_content.strip()
        if content:
            context_parts.append(
                f"[{filename} — Page {page_index + 1}]\n{content}"
            )
    return "\n\n---\n\n".join(context_parts)


def compute_confidence(
    useful_chunk_count: int,
    has_context: bool,
    scores: Optional[list[float]] = None,
) -> tuple[Literal["high", "medium", "low"], str]:
    _ = scores
    if not has_context or useful_chunk_count == 0:
        return "low", "No reliable document context was found."
    if useful_chunk_count >= 4:
        return "high", f"Answer based on {useful_chunk_count} retrieved passages."
    if useful_chunk_count >= 2:
        return "medium", f"Answer based on {useful_chunk_count} retrieved passages."
    return "low", "Only 1 relevant passage was found."


def build_ask_response(
    *,
    workspace_id: str,
    answer: str,
    sources: list[dict[str, Any]],
    confidence_label: Literal["high", "medium", "low"],
    retrieval_used: bool,
    latency_ms: int,
    confidence_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "workspace_id": workspace_id,
        "sources": sources,
        "confidence": confidence_label,
        "confidence_reason": confidence_reason,
        "retrieval_used": retrieval_used,
        "latency_ms": latency_ms,
        "model": settings.GROQ_MODEL,
        "metadata": {
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "retrieval_k": settings.RETRIEVAL_K,
            "chunks_used": len(sources),
        },
    }
