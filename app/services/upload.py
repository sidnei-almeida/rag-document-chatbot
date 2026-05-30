"""Upload orchestration: new conversations vs adding to the current one."""

import logging
import os
import tempfile
from typing import BinaryIO

from fastapi import UploadFile

from app.api.validators import validate_pdf_upload
from app.core.config import settings
from app.core.state import state
from app.services.conversation import (
    conversation_info_dict,
    ensure_active_conversation,
    start_new_conversation,
)
from app.services.ingestion import process_pdf_and_update_index

logger = logging.getLogger("docmind")


def _save_upload_to_temp(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(content)
        return tmp_file.name


def ingest_uploaded_pdf(
    *,
    filename: str,
    content: bytes,
    new_session: bool,
) -> dict:
    """
    Index one PDF. new_session=True discards any previous conversation/index.
    new_session=False appends to the active conversation.
    """
    if new_session:
        conversation_id = start_new_conversation(clear_index=True)
        replace = True
        logger.info("New conversation %s — indexing %s", conversation_id, filename)
    else:
        if not state.is_index_ready() and state.index_cleared:
            raise ValueError(
                "No active conversation. Upload with new_session=true to start a new document set, "
                "or use POST /upload/batch for several PDFs in one conversation."
            )
        conversation_id = ensure_active_conversation()
        replace = False
        logger.info(
            "Adding %s to conversation %s (%s document(s) already indexed)",
            filename,
            conversation_id,
            len(state.indexed_documents),
        )

    tmp_path = _save_upload_to_temp(content)
    try:
        chunks_count, pages_count = process_pdf_and_update_index(
            tmp_path,
            replace=replace,
            filename=filename,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    info = conversation_info_dict()
    return {
        "message": "PDF processed successfully",
        "filename": filename,
        "pages": pages_count,
        "chunks": chunks_count,
        "status": "ready",
        "new_session": new_session,
        "conversation_id": conversation_id,
        "conversation": info,
    }


async def ingest_upload_file(file: UploadFile, *, new_session: bool) -> dict:
    content = await file.read()
    validate_pdf_upload(file, content)
    logger.info(
        "Received PDF upload: %s (%s bytes, new_session=%s)",
        file.filename,
        len(content),
        new_session,
    )
    return ingest_uploaded_pdf(
        filename=file.filename or "document.pdf",
        content=content,
        new_session=new_session,
    )


async def ingest_upload_batch(files: list[UploadFile]) -> dict:
    """Index multiple PDFs in a single new conversation (replaces any previous index)."""
    if not files:
        raise ValueError("At least one PDF file is required.")
    if len(files) > settings.MAX_DOCUMENTS_PER_SESSION:
        raise ValueError(
            f"This demo allows up to {settings.MAX_DOCUMENTS_PER_SESSION} PDFs per conversation."
        )

    conversation_id = start_new_conversation(clear_index=True)
    logger.info(
        "Batch upload: new conversation %s with %s file(s)",
        conversation_id,
        len(files),
    )

    results: list[dict] = []
    for index, file in enumerate(files):
        content = await file.read()
        validate_pdf_upload(file, content)
        filename = file.filename or f"document-{index + 1}.pdf"
        tmp_path = _save_upload_to_temp(content)
        try:
            chunks_count, pages_count = process_pdf_and_update_index(
                tmp_path,
                replace=index == 0,
                filename=filename,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        results.append(
            {
                "filename": filename,
                "pages": pages_count,
                "chunks": chunks_count,
            }
        )

    info = conversation_info_dict()
    return {
        "message": f"{len(results)} PDF(s) indexed in one conversation",
        "status": "ready",
        "new_session": True,
        "conversation_id": conversation_id,
        "conversation": info,
        "files": results,
    }
