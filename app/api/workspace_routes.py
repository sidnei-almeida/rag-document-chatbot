"""Workspace upload and management routes."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.errors import api_error
from app.api.rag_guard import require_api_ready
from app.api.validators import read_and_validate_upload_files
from app.schemas.workspace import WorkspaceDetailResponse, WorkspaceListResponse, WorkspaceUploadResponse
from app.services.workspace.ingestion import create_workspace_from_uploads
from app.services.workspace.service import delete_workspace, get_workspace, list_workspaces

logger = logging.getLogger("docmind")
router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("/upload", response_model=WorkspaceUploadResponse)
async def workspaces_upload(
    files: list[UploadFile] = File(
        ...,
        description="One or more PDF files — use this field even for a single PDF",
    ),
):
    """
    Primary upload endpoint. Each request creates a **new** isolated workspace.

    ```bash
    curl -X POST http://localhost:7860/workspaces/upload -F "files=@doc.pdf"
    ```
    """
    require_api_ready()
    try:
        uploads = await read_and_validate_upload_files(files)
        result = create_workspace_from_uploads(uploads, source="upload")
        return WorkspaceUploadResponse(**result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise api_error(400, str(exc)) from exc
    except Exception:
        logger.exception("Workspace upload failed")
        raise api_error(500, "Unable to index the workspace right now. Please try again later.") from None


@router.get("", response_model=WorkspaceListResponse)
def workspaces_list():
    return WorkspaceListResponse(**list_workspaces())


@router.get("/{workspace_id}", response_model=WorkspaceDetailResponse)
def workspaces_get(workspace_id: str):
    try:
        return WorkspaceDetailResponse(**get_workspace(workspace_id))
    except FileNotFoundError as exc:
        raise api_error(404, str(exc)) from exc


@router.delete("/{workspace_id}")
def workspaces_delete(workspace_id: str):
    try:
        delete_workspace(workspace_id)
        return {
            "status": "deleted",
            "workspace_id": workspace_id,
            "message": "Workspace and all documents removed.",
        }
    except FileNotFoundError as exc:
        raise api_error(404, str(exc)) from exc
    except Exception:
        logger.exception("Failed to delete workspace %s", workspace_id)
        raise api_error(500, "Unable to delete workspace.") from None
