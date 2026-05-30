"""Schemas for demo/sample document endpoints."""

from pydantic import BaseModel


class SampleDocumentInfo(BaseModel):
    document_id: str | None = None
    filename: str
    pages: int
    chunks: int


class SampleLoadResponse(BaseModel):
    status: str
    workspace_id: str
    document: SampleDocumentInfo
    index_ready: bool
    conversation_id: str | None = None
