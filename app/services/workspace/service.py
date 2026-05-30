"""Workspace CRUD and resolution helpers."""

import logging
import shutil
from typing import Any

from app.services.workspace.cache import (
    clear_all_caches,
    invalidate_workspace_cache,
    workspace_index_ready,
)
from app.services.workspace.io import read_json
from app.services.workspace.paths import workspace_dir, workspace_json_path, workspaces_root
from app.services.workspace.registry import (
    get_default_workspace_id,
    list_registry_summaries,
    load_registry,
    unregister_workspace,
)

logger = logging.getLogger("docmind")


def workspace_exists(workspace_id: str) -> bool:
    return workspace_json_path(workspace_id).is_file()


def get_workspace(workspace_id: str) -> dict[str, Any]:
    if not workspace_exists(workspace_id):
        raise FileNotFoundError(f"Workspace '{workspace_id}' not found.")
    data = read_json(workspace_json_path(workspace_id), default={})
    data["index_ready"] = workspace_index_ready(workspace_id)
    return data


def list_workspaces() -> dict[str, Any]:
    summaries = list_registry_summaries()
    for item in summaries:
        item["index_ready"] = workspace_index_ready(item["workspace_id"])
    return {"workspaces": summaries}


def delete_workspace(workspace_id: str) -> None:
    if not workspace_exists(workspace_id):
        raise FileNotFoundError(f"Workspace '{workspace_id}' not found.")

    invalidate_workspace_cache(workspace_id)
    shutil.rmtree(workspace_dir(workspace_id), ignore_errors=True)
    unregister_workspace(workspace_id)
    logger.info("Deleted workspace %s", workspace_id)


def delete_all_workspaces() -> int:
    clear_all_caches()
    root = workspaces_root()
    count = 0
    if root.is_dir():
        for entry in list(root.iterdir()):
            if entry.is_dir() and (entry / "workspace.json").is_file():
                shutil.rmtree(entry, ignore_errors=True)
                count += 1
    registry = load_registry()
    registry["workspaces"] = {}
    registry["default_workspace_id"] = None
    from app.services.workspace.registry import save_registry

    save_registry(registry)
    logger.info("Deleted %s workspace(s)", count)
    return count


def resolve_workspace_id(
    *,
    workspace_id: str | None = None,
    conversation_id: str | None = None,
) -> str:
    """
    Resolve the workspace for /ask.

    Questions are always scoped to a workspace (never document_id).
    Priority: workspace_id > conversation_id (legacy alias) > default_workspace_id.
    """
    if workspace_id:
        if not workspace_exists(workspace_id):
            raise FileNotFoundError(f"Workspace '{workspace_id}' not found.")
        return workspace_id

    if conversation_id:
        if workspace_exists(conversation_id):
            return conversation_id
        raise FileNotFoundError(f"Workspace '{conversation_id}' not found.")

    default_id = get_default_workspace_id()
    if default_id:
        return default_id

    raise ValueError(
        "No workspace_id provided. Upload one or more PDFs first."
    )
