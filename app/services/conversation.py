"""One conversation = one isolated FAISS index (one or more PDFs in the same batch)."""

import logging
import uuid
from dataclasses import dataclass

from app.core.config import settings
from app.core.state import state
logger = logging.getLogger("docmind")


@dataclass
class IndexedDocument:
    file_name: str
    pages: int
    chunks: int


def start_new_conversation(*, clear_index: bool = True) -> str:
    """
    Begin a fresh conversation: new id, empty document list, optional index wipe.

    The next indexed PDF(s) belong only to this conversation.
    """
    if clear_index:
        from app.services.ingestion import clear_vector_index

        clear_vector_index()

    conversation_id = str(uuid.uuid4())
    state.conversation_id = conversation_id
    state.indexed_documents = []
    state.next_chunk_id = 0
    logger.info("Started new conversation %s", conversation_id)
    return conversation_id


def ensure_active_conversation() -> str:
    """Return current conversation id, creating one if the index exists but id was missing."""
    if state.conversation_id:
        return state.conversation_id
    conversation_id = str(uuid.uuid4())
    state.conversation_id = conversation_id
    logger.info("Assigned conversation id %s to existing index", conversation_id)
    return conversation_id


def record_indexed_document(file_name: str, pages: int, chunks: int) -> None:
    """Track a PDF that was added to the current conversation."""
    doc = IndexedDocument(file_name=file_name, pages=pages, chunks=chunks)
    state.indexed_documents.append(doc)
    state.last_indexed_filename = file_name
    state.index_pages = sum(d.pages for d in state.indexed_documents)
    state.index_chunks = sum(d.chunks for d in state.indexed_documents)


def conversation_document_count() -> int:
    return len(state.indexed_documents)


def total_pages_in_conversation() -> int:
    return sum(d.pages for d in state.indexed_documents)


def validate_can_add_document(page_count: int) -> None:
    """Enforce per-conversation limits before indexing another PDF."""
    if conversation_document_count() >= settings.MAX_DOCUMENTS_PER_SESSION:
        raise ValueError(
            f"This conversation already has {settings.MAX_DOCUMENTS_PER_SESSION} documents. "
            "Start a new conversation (new_session=true) to upload a different set."
        )
    if total_pages_in_conversation() + page_count > settings.MAX_PAGES:
        raise ValueError(
            f"Adding this PDF would exceed the demo limit of {settings.MAX_PAGES} pages "
            f"per conversation ({total_pages_in_conversation()} pages already indexed)."
        )


def end_conversation() -> None:
    """Clear conversation metadata (called from clear_vector_index)."""
    state.conversation_id = None
    state.indexed_documents = []
    state.next_chunk_id = 0


def conversation_info_dict() -> dict | None:
    if not state.conversation_id:
        return None
    documents = [
        {"file_name": d.file_name, "pages": d.pages, "chunks": d.chunks}
        for d in state.indexed_documents
    ]
    return {
        "conversation_id": state.conversation_id,
        "document_count": len(documents),
        "documents": documents,
        "total_pages": state.index_pages,
        "total_chunks": state.index_chunks,
    }


def validate_conversation_id(client_conversation_id: str | None) -> None:
    """Reject /ask when the client targets a stale conversation."""
    if not client_conversation_id:
        return
    if not state.conversation_id:
        raise ValueError(
            "No active conversation on the server. Upload documents with new_session=true first."
        )
    if client_conversation_id != state.conversation_id:
        raise ValueError(
            "This conversation is no longer active. A new document set was indexed. "
            "Use the latest conversation_id from /upload or /status."
        )
