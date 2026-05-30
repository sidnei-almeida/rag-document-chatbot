"""Pydantic schemas for /ask endpoint."""

from typing import Literal

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    chunk_id: str | int
    page: int = Field(..., description="1-indexed page number for display")
    page_index: int = Field(..., description="0-indexed page number from metadata")
    file_name: str
    preview: str
    score: float | None = None


class ConfidenceInfo(BaseModel):
    label: Literal["high", "medium", "low"]
    reason: str


class AskMetadata(BaseModel):
    model: str
    retrieval_k: int
    chunks_used: int
    embedding_model: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: ConfidenceInfo
    metadata: AskMetadata
