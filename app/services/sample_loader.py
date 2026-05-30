"""Load bundled sample PDF as a workspace (demo)."""

import logging
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger("docmind")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_sample_pdf_path() -> Path:
    sample_path = PROJECT_ROOT / settings.SAMPLE_DOCUMENTS_DIR / settings.SAMPLE_DOCUMENT_FILENAME
    if not sample_path.is_file():
        raise FileNotFoundError(
            f"Sample document not found: {sample_path}. "
            "Ensure sample_documents/ is deployed with the API."
        )
    return sample_path


def get_sample_display_name() -> str:
    return settings.SAMPLE_DOCUMENT_FILENAME


def load_sample_workspace() -> dict:
    """Create a demo workspace from the bundled sample PDF."""
    from app.services.workspace.ingestion import create_workspace_from_uploads

    sample_path = resolve_sample_pdf_path()
    file_name = get_sample_display_name()
    logger.info("Creating sample workspace from %s", sample_path)
    return create_workspace_from_uploads(
        [(file_name, sample_path.read_bytes())],
        source="sample",
    )
