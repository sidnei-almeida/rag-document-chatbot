"""Tests for sample document demo endpoint."""

from pathlib import Path
from unittest.mock import MagicMock


def test_load_sample_returns_expected_shape(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.load_sample_workspace",
        lambda: {
            "workspace_id": "ws_sample123",
            "index_ready": True,
            "documents": [
                {
                    "document_id": "doc_sample1",
                    "filename": "ai-document-intelligence-report.pdf",
                    "pages": 3,
                    "chunks": 8,
                    "status": "ready",
                }
            ],
        },
    )

    response = client.post("/demo/load-sample")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "sample_loaded"
    assert data["workspace_id"] == "ws_sample123"
    assert data["index_ready"] is True
    assert data["document"]["filename"] == "ai-document-intelligence-report.pdf"
    assert data["document"]["pages"] == 3
    assert data["document"]["chunks"] == 8


def test_load_sample_missing_file_returns_404(client, monkeypatch):
    def _missing():
        raise FileNotFoundError("not found")

    monkeypatch.setattr("app.api.routes.load_sample_workspace", _missing)
    response = client.post("/demo/load-sample")
    assert response.status_code == 404
    assert "not found" in response.json()["error"].lower()


def test_resolve_sample_pdf_path_finds_bundled_file():
    from app.services.sample_loader import resolve_sample_pdf_path

    path = resolve_sample_pdf_path()
    assert path.is_file()
    assert path.suffix.lower() == ".pdf"
    assert path.parent.name == "sample_documents"
