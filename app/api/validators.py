"""Request validation helpers."""

from fastapi import HTTPException, UploadFile

from app.api.errors import api_error
from app.core.config import settings

PDF_ONLY_MESSAGE = "Only PDF files are supported right now."


def validate_question(question: str) -> str:
    stripped = question.strip()
    if not stripped:
        raise api_error(400, "Question cannot be empty.")
    if len(question) > settings.MAX_QUESTION_LENGTH:
        raise api_error(400, "Question is too long for this demo.")
    return stripped


def validate_pdf_upload(file: UploadFile, content: bytes) -> None:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise api_error(400, PDF_ONLY_MESSAGE)

    if len(content) > settings.max_file_size_bytes():
        raise HTTPException(
            status_code=413,
            detail={
                "error": (
                    f"File '{filename}' exceeds the maximum size of "
                    f"{settings.MAX_FILE_SIZE_MB} MB."
                )
            },
        )


async def read_and_validate_upload_files(
    files: list[UploadFile] | None,
) -> list[tuple[str, bytes]]:
    """Validate multipart `files` and return (filename, bytes) pairs."""
    if not files:
        raise api_error(400, "At least one PDF file is required in 'files'.")

    if len(files) > settings.MAX_FILES_PER_WORKSPACE:
        raise api_error(
            400,
            f"A workspace can include at most {settings.MAX_FILES_PER_WORKSPACE} PDF files.",
        )

    uploads: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        validate_pdf_upload(upload, content)
        uploads.append((upload.filename or "document.pdf", content))
    return uploads
