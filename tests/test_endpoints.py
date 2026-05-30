"""API endpoint tests with mocked LLM and workspace ingestion."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.core.config import settings
from app.core.state import state
from tests.conftest import mock_workspace_index_pipeline


def test_health_returns_expected_fields(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "initializing")
    assert data["api_ready"] is True
    assert data["llm_ready"] is True
    assert data["embeddings_ready"] is True
    assert data["storage_ready"] is True
    assert data["documents_ready"] is False
    assert data["workspace_count"] == 0
    assert data["default_workspace_id"] is None
    assert data["model"] == settings.GROQ_MODEL


def test_status_returns_workspace_fields(client):
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "default_workspace_id" in data
    assert "workspace_count" in data


def test_clear_all_workspaces(client, monkeypatch):
    mock_workspace_index_pipeline(monkeypatch)
    client.post(
        "/workspaces/upload",
        files=[("files", ("a.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    response = client.delete("/clear")
    assert response.status_code == 200
    assert response.json()["workspaces_removed"] >= 1


def test_ask_empty_question_returns_400(client):
    response = client.post("/ask", json={"question": "   ", "workspace_id": "ws_x"})
    assert response.status_code == 400


def test_ask_question_too_long_returns_400(client):
    response = client.post(
        "/ask",
        json={
            "question": "a" * (settings.MAX_QUESTION_LENGTH + 1),
            "workspace_id": "ws_x",
        },
    )
    assert response.status_code == 400


def test_ask_general_question_without_workspace(client):
    response = client.post("/ask", json={"question": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert data["retrieval_used"] is False


def test_ask_no_evidence_does_not_call_llm(client, monkeypatch):
    mock_workspace_index_pipeline(monkeypatch)
    up = client.post(
        "/workspaces/upload",
        files=[("files", ("q.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    ws_id = up.json()["workspace_id"]

    with patch(
        "app.api.routes.retrieve_documents_for_workspace",
        lambda _ws, _q: ([], None),
    ):
        response = client.post(
            "/ask",
            json={"workspace_id": ws_id, "question": "What is the secret code?"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == settings.NO_EVIDENCE_MESSAGE
    assert data["sources"] == []
    assert data["confidence"] == "low"
    state.llm.invoke.assert_not_called()


def test_ask_with_retrieved_context(client, monkeypatch):
    mock_workspace_index_pipeline(monkeypatch)
    up = client.post(
        "/workspaces/upload",
        files=[("files", ("spec.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    ws_id = up.json()["workspace_id"]

    docs = [
        Document(
            page_content="FastAPI and FAISS stack.",
            metadata={
                "workspace_id": ws_id,
                "document_id": "doc_1",
                "filename": "spec.pdf",
                "page": 0,
                "chunk_id": "chunk_000",
            },
        ),
    ]
    with patch(
        "app.api.routes.retrieve_documents_for_workspace",
        lambda _ws, _q: (docs, [0.3]),
    ):
        response = client.post(
            "/ask",
            json={"workspace_id": ws_id, "question": "What stack is used?"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["workspace_id"] == ws_id
    assert data["sources"][0]["filename"] == "spec.pdf"
    assert data["retrieval_used"] is True
    state.llm.invoke.assert_called_once()


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/upload",
        files={"file": ("readme.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
