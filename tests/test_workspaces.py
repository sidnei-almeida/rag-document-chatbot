"""Workspace isolation and upload flows."""

from unittest.mock import patch

from langchain_core.documents import Document

from app.services.workspace.ingestion import _workspace_title
from app.services.workspace.service import get_workspace, list_workspaces
from tests.conftest import mock_workspace_index_pipeline


def test_workspace_title_single_file():
    assert _workspace_title(["report.pdf"]) == "report.pdf"


def test_workspace_title_multiple_short():
    assert _workspace_title(["a.pdf", "b.pdf"]) == "a.pdf + b.pdf"


def test_workspace_title_multiple_long_uses_count():
    names = [
        "very-long-contract-filename-v1.pdf",
        "very-long-appendix-filename-v2.pdf",
        "very-long-invoice-filename-v3.pdf",
    ]
    title = _workspace_title(names)
    assert title == "very-long-contract-filename-v1.pdf + 2 files"


def test_workspaces_upload_single_pdf(client, monkeypatch):
    mock_workspace_index_pipeline(monkeypatch)
    response = client.post(
        "/workspaces/upload",
        files=[("files", ("doc1.pdf", b"%PDF-1.4 test", "application/pdf"))],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["workspace_id"].startswith("ws_")
    assert data["document_count"] == 1
    assert data["index_ready"] is True


def test_workspaces_upload_three_pdfs_one_workspace(client, monkeypatch):
    mock_workspace_index_pipeline(monkeypatch)
    response = client.post(
        "/workspaces/upload",
        files=[
            ("files", ("doc1.pdf", b"%PDF-1.4 a", "application/pdf")),
            ("files", ("doc2.pdf", b"%PDF-1.4 b", "application/pdf")),
            ("files", ("doc3.pdf", b"%PDF-1.4 c", "application/pdf")),
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["document_count"] == 3
    ws_id = data["workspace_id"]
    filenames = {d["filename"] for d in data["documents"]}
    assert filenames == {"doc1.pdf", "doc2.pdf", "doc3.pdf"}

    detail = get_workspace(ws_id)
    assert detail["total_chunks"] == 9


def test_workspace_isolation_between_uploads(client, monkeypatch):
    mock_workspace_index_pipeline(monkeypatch)

    r1 = client.post(
        "/workspaces/upload",
        files=[("files", ("financial.pdf", b"%PDF-1.4 fin", "application/pdf"))],
    )
    r2 = client.post(
        "/workspaces/upload",
        files=[("files", ("medical.pdf", b"%PDF-1.4 med", "application/pdf"))],
    )
    ws_a = r1.json()["workspace_id"]
    ws_b = r2.json()["workspace_id"]
    assert ws_a != ws_b

    with patch(
        "app.api.routes.retrieve_documents_for_workspace",
    ) as mock_retrieve:
        mock_retrieve.return_value = (
            [
                Document(
                    page_content="financial topic",
                    metadata={
                        "workspace_id": ws_a,
                        "document_id": "doc_x",
                        "filename": "financial.pdf",
                        "page": 0,
                        "chunk_id": "chunk_000",
                    },
                )
            ],
            [0.2],
        )
        ask_a = client.post(
            "/ask",
            json={"workspace_id": ws_a, "question": "What is the topic?"},
        )
    assert ask_a.status_code == 200
    assert ask_a.json()["workspace_id"] == ws_a
    assert ask_a.json()["sources"][0]["filename"] == "financial.pdf"
    assert ask_a.json()["sources"][0]["workspace_id"] == ws_a

    with patch(
        "app.api.routes.retrieve_documents_for_workspace",
    ) as mock_retrieve:
        mock_retrieve.return_value = (
            [
                Document(
                    page_content="medical topic",
                    metadata={
                        "workspace_id": ws_b,
                        "document_id": "doc_y",
                        "filename": "medical.pdf",
                        "page": 0,
                        "chunk_id": "chunk_000",
                    },
                )
            ],
            [0.3],
        )
        ask_b = client.post(
            "/ask",
            json={"workspace_id": ws_b, "question": "What is the topic?"},
        )
    assert ask_b.json()["sources"][0]["filename"] == "medical.pdf"
    assert ask_b.json()["sources"][0]["workspace_id"] == ws_b


def test_delete_workspace_then_ask_404(client, monkeypatch):
    mock_workspace_index_pipeline(monkeypatch)
    up = client.post(
        "/workspaces/upload",
        files=[("files", ("x.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    ws_id = up.json()["workspace_id"]
    assert client.delete(f"/workspaces/{ws_id}").status_code == 200
    ask = client.post(
        "/ask",
        json={"workspace_id": ws_id, "question": "What is the main topic?"},
    )
    assert ask.status_code == 404
    listed = list_workspaces()["workspaces"]
    assert all(w["workspace_id"] != ws_id for w in listed)


def test_ask_without_workspace_id_returns_400(client):
    response = client.post("/ask", json={"question": "What is this?"})
    assert response.status_code == 400
    assert "workspace_id" in response.json()["error"].lower()


def test_legacy_upload_returns_workspace_id(client, monkeypatch):
    mock_workspace_index_pipeline(monkeypatch)
    response = client.post(
        "/upload",
        files={"file": ("legacy.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["workspace_id"].startswith("ws_")
