"""Schemas for demo/sample document endpoints."""

from pydantic import BaseModel


class SampleDocumentInfo(BaseModel):
    file_name: str
    pages: int
    chunks: int


class SampleLoadResponse(BaseModel):
    status: str
    document: SampleDocumentInfo
    index_ready: bool
