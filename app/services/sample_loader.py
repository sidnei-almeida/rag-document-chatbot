"""Load bundled sample PDFs for public demo."""

import logging
import os
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger("docmind")

# Project root (repository root, parent of `app/`)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_sample_pdf_path() -> Path:
    """Return absolute path to the configured sample PDF."""
    sample_path = PROJECT_ROOT / settings.SAMPLE_DOCUMENTS_DIR / settings.SAMPLE_DOCUMENT_FILENAME
    if not sample_path.is_file():
        raise FileNotFoundError(
            f"Sample document not found: {sample_path}. "
            "Ensure sample_documents/ is deployed with the API."
        )
    return sample_path


def get_sample_display_name() -> str:
    """File name exposed in API responses and chunk metadata."""
    return settings.SAMPLE_DOCUMENT_FILENAME


def load_sample_document() -> tuple[int, int, str]:
    """
    Process the bundled sample PDF through the standard ingestion pipeline.

    Returns:
        (chunks_count, pages_count, file_name)
    """
    from app.services.ingestion import process_pdf_and_update_index

    sample_path = resolve_sample_pdf_path()
    file_name = get_sample_display_name()

    logger.info("Loading sample document from %s", sample_path)
    chunks_count, pages_count = process_pdf_and_update_index(
        str(sample_path),
        replace=True,
        filename=file_name,
    )
    return chunks_count, pages_count, file_name
