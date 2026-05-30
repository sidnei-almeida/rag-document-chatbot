"""Global workspace registry (registry.json)."""

from datetime import datetime, timezone
from typing import Any

from app.services.workspace.io import read_json, write_json
from app.services.workspace.paths import registry_path, workspaces_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_registry() -> dict[str, Any]:
    return {"version": 1, "default_workspace_id": None, "workspaces": {}}


def load_registry() -> dict[str, Any]:
    workspaces_root().mkdir(parents=True, exist_ok=True)
    data = read_json(registry_path(), default=None)
    if not data:
        registry = _empty_registry()
        write_json(registry_path(), registry)
        return registry
    data.setdefault("version", 1)
    data.setdefault("workspaces", {})
    return data


def save_registry(registry: dict[str, Any]) -> None:
    write_json(registry_path(), registry)


def registry_summary_entry(workspace: dict[str, Any]) -> dict[str, Any]:
    return {
        "workspace_id": workspace["workspace_id"],
        "title": workspace.get("title"),
        "status": workspace.get("status"),
        "index_ready": workspace.get("index_ready", False),
        "document_count": workspace.get("document_count", len(workspace.get("documents", []))),
        "total_pages": workspace.get("total_pages", 0),
        "total_chunks": workspace.get("total_chunks", 0),
        "created_at": workspace.get("created_at"),
        "updated_at": workspace.get("updated_at"),
    }


def register_workspace(workspace: dict[str, Any]) -> None:
    registry = load_registry()
    workspace_id = workspace["workspace_id"]
    registry["workspaces"][workspace_id] = registry_summary_entry(workspace)
    registry["default_workspace_id"] = workspace_id
    registry["updated_at"] = _utc_now()
    save_registry(registry)


def unregister_workspace(workspace_id: str) -> None:
    registry = load_registry()
    registry["workspaces"].pop(workspace_id, None)
    if registry.get("default_workspace_id") == workspace_id:
        remaining = list(registry["workspaces"].keys())
        registry["default_workspace_id"] = remaining[-1] if remaining else None
    save_registry(registry)


def get_default_workspace_id() -> str | None:
    registry = load_registry()
    default_id = registry.get("default_workspace_id")
    if default_id and default_id in registry.get("workspaces", {}):
        return default_id
    return None


def list_registry_summaries() -> list[dict[str, Any]]:
    registry = load_registry()
    return list(registry.get("workspaces", {}).values())


def find_workspace_for_document(document_id: str) -> str | None:
    """Scan workspace.json files to map document_id -> workspace_id."""
    root = workspaces_root()
    if not root.is_dir():
        return None
    for workspace_path in root.iterdir():
        if not workspace_path.is_dir():
            continue
        workspace_json = workspace_path / "workspace.json"
        if not workspace_json.is_file():
            continue
        data = read_json(workspace_json, default={})
        for doc in data.get("documents", []):
            if doc.get("document_id") == document_id:
                return data.get("workspace_id")
    return None
