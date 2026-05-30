"""Unit tests for request validators."""

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.validators import validate_pdf_upload, validate_question
from app.core.config import settings


def test_validate_question_empty():
    with pytest.raises(HTTPException) as exc_info:
        validate_question("   ")
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.detail["error"].lower()


def test_validate_question_too_long():
    with pytest.raises(HTTPException) as exc_info:
        validate_question("x" * (settings.MAX_QUESTION_LENGTH + 1))
    assert exc_info.value.status_code == 400
    assert "too long" in exc_info.value.detail["error"].lower()


def test_validate_question_strips_and_returns():
    assert validate_question("  What is RAG?  ") == "What is RAG?"


def test_validate_pdf_upload_rejects_non_pdf():
    upload = UploadFile(filename="notes.txt", file=None)
    with pytest.raises(HTTPException) as exc_info:
        validate_pdf_upload(upload, b"hello")
    assert exc_info.value.status_code == 400
    assert "PDF" in exc_info.value.detail["error"]


def test_validate_pdf_upload_rejects_oversized_file():
    upload = UploadFile(filename="big.pdf", file=None)
    oversized = b"x" * (settings.max_file_size_bytes() + 1)
    with pytest.raises(HTTPException) as exc_info:
        validate_pdf_upload(upload, oversized)
    assert exc_info.value.status_code == 413
    assert "exceeds" in exc_info.value.detail["error"].lower()


def test_validate_pdf_upload_accepts_small_pdf():
    upload = UploadFile(filename="ok.pdf", file=None)
    validate_pdf_upload(upload, b"%PDF-1.4 fake content")
