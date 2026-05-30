"""Shared pytest fixtures — no real Groq or Hugging Face calls."""

import os
import tempfile
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-fake-key-for-pytest")
os.environ.setdefault("LOG_LEVEL", "WARNING")
_tmp_storage = tempfile.mkdtemp(prefix="docmind_ws_test_")
os.environ["WORKSPACE_STORAGE_ROOT"] = os.path.join(_tmp_storage, "workspaces")


def _fake_initialize_models() -> None:
    from app.core.state import state

    state.llm = MagicMock()
    state.llm.invoke.return_value = MagicMock(content="Mocked LLM response for tests.")
    state.embeddings_model = MagicMock()


@asynccontextmanager
async def _test_lifespan(_app):
    _fake_initialize_models()
    yield


@pytest.fixture(autouse=True)
def reset_state_and_storage():
    from app.services.workspace.cache import clear_all_caches
    from app.services.workspace.service import delete_all_workspaces

    _fake_initialize_models()
    clear_all_caches()
    try:
        delete_all_workspaces()
    except Exception:
        pass
    yield
    clear_all_caches()
    try:
        delete_all_workspaces()
    except Exception:
        pass


def mock_workspace_index_pipeline(monkeypatch):
    """Avoid real PDF parsing and FAISS in unit tests."""
    from langchain_core.documents import Document

    from app.services.workspace.ids import format_chunk_id
    from app.services.workspace.service import workspace_exists

    def fake_process(workspace_id, document_id, filename, pdf_bytes, chunk_sequence_start):
        docs = []
        for i in range(3):
            seq = chunk_sequence_start + i
            docs.append(
                Document(
                    page_content=f"Content of {filename} chunk {i}",
                    metadata={
                        "workspace_id": workspace_id,
                        "document_id": document_id,
                        "filename": filename,
                        "page": 0,
                        "chunk_id": format_chunk_id(seq),
                    },
                )
            )
        summary = {
            "document_id": document_id,
            "filename": filename,
            "pages": 2,
            "chunks": 3,
            "status": "ready",
        }
        return summary, docs, chunk_sequence_start + 3

    mock_store = MagicMock()
    monkeypatch.setattr(
        "app.services.workspace.ingestion._process_single_pdf",
        fake_process,
    )
    monkeypatch.setattr(
        "app.services.workspace.ingestion.FAISS.from_documents",
        lambda docs, emb: mock_store,
    )
    monkeypatch.setattr(
        "langchain_community.vectorstores.FAISS.load_local",
        lambda *_a, **_k: mock_store,
    )
    index_ready = lambda ws_id: workspace_exists(ws_id)
    monkeypatch.setattr("app.services.workspace.cache.workspace_index_ready", index_ready)
    monkeypatch.setattr("app.api.rag_guard.workspace_index_ready", index_ready)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import create_app

    with TestClient(create_app(lifespan_handler=_test_lifespan)) as test_client:
        yield test_client
