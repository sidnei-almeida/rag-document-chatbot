"""Health and status payload builders."""

from app.core.config import settings
from app.core.state import state
from app.services.workspace.cache import workspace_index_ready
from app.services.workspace.paths import workspaces_root
from app.services.workspace.registry import get_default_workspace_id, list_registry_summaries


def _storage_ready() -> bool:
    root = workspaces_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def api_status_label() -> str:
    return "ok" if state.is_api_ready() else "initializing"


def build_health_payload() -> dict:
    default_ws = get_default_workspace_id()
    workspace_count = len(list_registry_summaries())
    documents_ready = bool(
        default_ws and workspace_index_ready(default_ws)
    )

    return {
        "status": api_status_label(),
        "api_ready": state.is_api_ready(),
        "llm_ready": state.is_llm_ready(),
        "embeddings_ready": state.embeddings_model is not None,
        "storage_ready": _storage_ready(),
        "documents_ready": documents_ready,
        "index_ready": documents_ready,
        "workspace_count": workspace_count,
        "default_workspace_id": default_ws,
        "storage_root": str(workspaces_root()),
        "model": settings.GROQ_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "retrieval": settings.retrieval_dict(),
        "limits": settings.limits_dict(),
    }


def build_status_payload() -> dict:
    health = build_health_payload()
    return {
        "status": health["status"],
        "api_ready": health["api_ready"],
        "llm_ready": health["llm_ready"],
        "embeddings_ready": health["embeddings_ready"],
        "storage_ready": health["storage_ready"],
        "documents_ready": health["documents_ready"],
        "index_ready": health["index_ready"],
        "workspace_count": health["workspace_count"],
        "default_workspace_id": health["default_workspace_id"],
        "storage_root": health["storage_root"],
        "model": health["model"],
        "embedding_model": health["embedding_model"],
        "retrieval": health["retrieval"],
        "limits": health["limits"],
    }
