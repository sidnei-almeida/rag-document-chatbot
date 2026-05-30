"""Pydantic schemas for workspace endpoints."""

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    status: str = "ready"


class WorkspaceUploadResponse(BaseModel):
    workspace_id: str
    title: str
    status: str
    index_ready: bool
    document_count: int
    total_pages: int
    total_chunks: int
    documents: list[DocumentSummary]
    message: str


class WorkspaceListItem(BaseModel):
    workspace_id: str
    title: str | None = None
    status: str | None = None
    index_ready: bool = False
    document_count: int = 0
    total_pages: int = 0
    total_chunks: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class WorkspaceListResponse(BaseModel):
    workspaces: list[WorkspaceListItem]


class WorkspaceDetailResponse(BaseModel):
    workspace_id: str
    title: str
    status: str
    index_ready: bool
    created_at: str | None = None
    updated_at: str | None = None
    documents: list[DocumentSummary]
    document_count: int
    total_pages: int
    total_chunks: int
    embedding_model: str | None = None
    vector_store: str | None = None
    source: str | None = None
