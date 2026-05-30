"""Tests for sample document demo endpoint."""

from pathlib import Path
from unittest.mock import MagicMock

from app.core.state import state


def test_load_sample_returns_expected_shape(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.load_sample_document",
        lambda: (8, 3, "ai-document-intelligence-report.pdf"),
    )
    state.retriever = MagicMock()
    state.vector_store = MagicMock()

    response = client.post("/demo/load-sample")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "sample_loaded"
    assert data["index_ready"] is True
    assert data["document"]["file_name"] == "ai-document-intelligence-report.pdf"
    assert data["document"]["pages"] == 3
    assert data["document"]["chunks"] == 8


def test_load_sample_missing_file_returns_404(client, monkeypatch):
    def _missing():
        raise FileNotFoundError("not found")

    monkeypatch.setattr("app.api.routes.load_sample_document", _missing)
    response = client.post("/demo/load-sample")
    assert response.status_code == 404
    assert "not available" in response.json()["detail"].lower()


def test_resolve_sample_pdf_path_finds_bundled_file():
    from app.services.sample_loader import resolve_sample_pdf_path

    path = resolve_sample_pdf_path()
    assert path.is_file()
    assert path.suffix.lower() == ".pdf"
    assert path.parent.name == "sample_documents"
