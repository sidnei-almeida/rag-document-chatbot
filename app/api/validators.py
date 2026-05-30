"""Request validation helpers."""

from fastapi import HTTPException, UploadFile

from app.core.config import settings


def validate_question(question: str) -> str:
    """Validate and return stripped question text."""
    stripped = question.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > settings.MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="Question is too long for this demo.",
        )
    return stripped


def validate_pdf_upload(file: UploadFile, content: bytes) -> None:
    """Validate uploaded PDF extension and size."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if len(content) > settings.max_file_size_bytes():
        raise HTTPException(
            status_code=400,
            detail="PDF too large for this public demo.",
        )
