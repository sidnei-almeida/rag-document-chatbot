"""Build a workspace FAISS index from one or more PDF uploads."""

import logging
import shutil
from datetime import datetime, timezone
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.state import state
from app.services.workspace.ids import format_chunk_id, new_document_id, new_workspace_id
from app.services.workspace.io import ensure_dir, write_json
from app.services.workspace.paths import (
    document_chunks_path,
    document_metadata_path,
    document_source_pdf,
    document_dir,
    faiss_dir,
    workspace_dir,
    workspace_json_path,
)
from app.services.workspace.registry import register_workspace

logger = logging.getLogger("docmind")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=list(settings.TEXT_SPLITTER_SEPARATORS),
    )


def _workspace_title(filenames: list[str]) -> str:
    """Friendly sidebar title: single filename or 'first.pdf + N files'."""
    if len(filenames) == 1:
        return filenames[0]
    joined = " + ".join(filenames)
    if len(joined) <= 80:
        return joined
    extra = len(filenames) - 1
    return f"{filenames[0]} + {extra} files"


def _process_single_pdf(
    workspace_id: str,
    document_id: str,
    filename: str,
    pdf_bytes: bytes,
    chunk_sequence_start: int,
) -> tuple[dict[str, Any], list[Document], int]:
    doc_dir = document_dir(workspace_id, document_id)
    ensure_dir(doc_dir)
    pdf_path = document_source_pdf(workspace_id, document_id)
    pdf_path.write_bytes(pdf_bytes)

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    page_count = len(pages)
    if page_count > settings.MAX_PAGES_PER_FILE:
        raise ValueError(
            f"'{filename}' has {page_count} pages; limit is {settings.MAX_PAGES_PER_FILE} per file."
        )

    splitter = create_text_splitter()
    chunks = splitter.split_documents(pages)

    if not chunks or not any(c.page_content.strip() for c in chunks):
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "Upload a text-based PDF (not a scanned image-only file)."
        )

    chunk_records: list[dict[str, Any]] = []
    langchain_docs: list[Document] = []
    sequence = chunk_sequence_start

    for chunk in chunks:
        page_index = chunk.metadata.get("page", 0)
        if not isinstance(page_index, int):
            try:
                page_index = int(page_index)
            except (TypeError, ValueError):
                page_index = 0

        chunk_id = format_chunk_id(sequence)
        chunk.metadata = {
            "workspace_id": workspace_id,
            "document_id": document_id,
            "filename": filename,
            "page": page_index,
            "chunk_id": chunk_id,
        }
        langchain_docs.append(chunk)
        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "workspace_id": workspace_id,
                "document_id": document_id,
                "filename": filename,
                "page": page_index + 1,
                "text_preview": chunk.page_content[:200],
            }
        )
        sequence += 1

    metadata = {
        "document_id": document_id,
        "workspace_id": workspace_id,
        "filename": filename,
        "pages": page_count,
        "chunks": len(chunks),
        "status": "ready",
    }
    write_json(document_metadata_path(workspace_id, document_id), metadata)
    write_json(document_chunks_path(workspace_id, document_id), chunk_records)

    doc_summary = {
        "document_id": document_id,
        "filename": filename,
        "pages": page_count,
        "chunks": len(chunks),
        "status": "ready",
    }
    return doc_summary, langchain_docs, sequence


def create_workspace_from_uploads(
    uploads: list[tuple[str, bytes]],
    *,
    source: str = "upload",
) -> dict[str, Any]:
    """
    Create an isolated workspace with one FAISS index for all PDFs in this upload batch.

    uploads: list of (filename, pdf_bytes)
    """
    if not uploads:
        raise ValueError("At least one PDF file is required.")
    if len(uploads) > settings.MAX_FILES_PER_WORKSPACE:
        raise ValueError(
            f"A workspace can include at most {settings.MAX_FILES_PER_WORKSPACE} PDF files."
        )
    if state.embeddings_model is None:
        raise RuntimeError("Embeddings model is not initialized.")

    workspace_id = new_workspace_id()
    ensure_dir(workspace_dir(workspace_id))
    ensure_dir(faiss_dir(workspace_id))

    filenames = [name for name, _ in uploads]
    total_pages = 0
    documents_meta: list[dict[str, Any]] = []
    all_langchain_docs: list[Document] = []
    chunk_sequence = 0

    for filename, pdf_bytes in uploads:
        document_id = new_document_id()
        doc_summary, docs, chunk_sequence = _process_single_pdf(
            workspace_id,
            document_id,
            filename,
            pdf_bytes,
            chunk_sequence,
        )
        total_pages += doc_summary["pages"]
        if total_pages > settings.MAX_TOTAL_PAGES:
            shutil.rmtree(workspace_dir(workspace_id), ignore_errors=True)
            raise ValueError(
                f"Total pages ({total_pages}) exceed limit of {settings.MAX_TOTAL_PAGES} per workspace."
            )
        documents_meta.append(doc_summary)
        all_langchain_docs.extend(docs)

    if not all_langchain_docs:
        shutil.rmtree(workspace_dir(workspace_id), ignore_errors=True)
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    vector_store = FAISS.from_documents(all_langchain_docs, state.embeddings_model)
    vector_store.save_local(str(faiss_dir(workspace_id)))

    total_chunks = len(all_langchain_docs)
    now = _utc_now()
    workspace_data = {
        "workspace_id": workspace_id,
        "title": _workspace_title(filenames),
        "status": "ready",
        "index_ready": True,
        "created_at": now,
        "updated_at": now,
        "documents": documents_meta,
        "document_count": len(documents_meta),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "vector_store": "FAISS",
        "source": source,
    }
    write_json(workspace_json_path(workspace_id), workspace_data)
    register_workspace(workspace_data)

    from app.services.workspace.cache import invalidate_workspace_cache

    invalidate_workspace_cache(workspace_id)

    logger.info(
        "Created workspace %s with %s document(s), %s chunks",
        workspace_id,
        len(documents_meta),
        total_chunks,
    )
    return {
        **workspace_data,
        "message": "Workspace uploaded and indexed successfully",
    }
