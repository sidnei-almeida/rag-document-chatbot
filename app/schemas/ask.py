"""Pydantic schemas for /ask endpoint."""

from typing import Literal

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str
    workspace_id: str | None = Field(
        default=None,
        description="Workspace to query (required for RAG; uses default if omitted)",
    )
    conversation_id: str | None = Field(
        default=None,
        deprecated=True,
        description="Legacy alias for workspace_id (do not use document_id to ask questions)",
    )


class SourceItem(BaseModel):
    workspace_id: str
    document_id: str
    filename: str
    page: int
    chunk_id: str
    score: float | None = None
    preview: str


class ConfidenceInfo(BaseModel):
    """Legacy nested confidence (optional for older clients)."""

    label: Literal["high", "medium", "low"]
    reason: str


class AskMetadata(BaseModel):
    embedding_model: str
    retrieval_k: int
    chunks_used: int


class AskResponse(BaseModel):
    answer: str
    workspace_id: str
    sources: list[SourceItem]
    confidence: Literal["high", "medium", "low"]
    confidence_reason: str | None = None
    retrieval_used: bool
    latency_ms: int
    model: str
    metadata: AskMetadata
