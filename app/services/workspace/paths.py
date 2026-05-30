"""Filesystem paths for workspace storage."""

from pathlib import Path

from app.core.config import settings


def workspaces_root() -> Path:
    return Path(settings.WORKSPACE_STORAGE_ROOT).resolve()


def registry_path() -> Path:
    return workspaces_root() / "registry.json"


def workspace_dir(workspace_id: str) -> Path:
    return workspaces_root() / workspace_id


def workspace_json_path(workspace_id: str) -> Path:
    return workspace_dir(workspace_id) / "workspace.json"


def documents_root(workspace_id: str) -> Path:
    return workspace_dir(workspace_id) / "documents"


def document_dir(workspace_id: str, document_id: str) -> Path:
    return documents_root(workspace_id) / document_id


def document_source_pdf(workspace_id: str, document_id: str) -> Path:
    return document_dir(workspace_id, document_id) / "source.pdf"


def document_metadata_path(workspace_id: str, document_id: str) -> Path:
    return document_dir(workspace_id, document_id) / "metadata.json"


def document_chunks_path(workspace_id: str, document_id: str) -> Path:
    return document_dir(workspace_id, document_id) / "chunks.json"


def faiss_dir(workspace_id: str) -> Path:
    return workspace_dir(workspace_id) / "faiss_index"


def faiss_index_file(workspace_id: str) -> Path:
    return faiss_dir(workspace_id) / "index.faiss"
