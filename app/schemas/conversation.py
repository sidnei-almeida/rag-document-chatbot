"""Schemas for document conversation / session isolation."""

from pydantic import BaseModel, Field


class IndexedDocumentInfo(BaseModel):
    file_name: str
    pages: int
    chunks: int


class ConversationInfo(BaseModel):
    conversation_id: str
    document_count: int
    documents: list[IndexedDocumentInfo]
    total_pages: int
    total_chunks: int
