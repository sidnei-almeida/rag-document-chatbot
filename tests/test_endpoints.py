"""API endpoint tests with mocked LLM and ingestion."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.core.config import settings
from app.core.state import state


def test_health_returns_expected_fields(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "initializing")
    assert data["api_ready"] is True
    assert data["llm_ready"] is True
    assert "index_ready" in data
    assert "index_path" in data
    assert data["index_path"].endswith("index.faiss")
    assert data["model"] == settings.GROQ_MODEL
    assert data["embedding_model"] == settings.EMBEDDING_MODEL_NAME
    assert data["retrieval"]["k"] == settings.RETRIEVAL_K
    assert data["limits"]["max_file_size_mb"] == settings.MAX_FILE_SIZE_MB
    assert data["limits"]["max_pages"] == settings.MAX_PAGES


def test_status_returns_document_fields(client):
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "index_ready" in data
    assert "document_loaded" in data
    assert "limits" in data
    assert data["model"] == settings.GROQ_MODEL


def test_clear_resets_index_state(client):
    state.retriever = MagicMock()
    state.vector_store = MagicMock()
    state.index_pages = 10
    state.index_chunks = 50
    state.index_cleared = False

    response = client.delete("/clear")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cleared"
    assert data["index_ready"] is False
    assert state.retriever is None
    assert state.vector_store is None
    assert state.index_cleared is True
    assert state.index_pages == 0


def test_ask_empty_question_returns_400(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ask_question_too_long_returns_400(client):
    response = client.post("/ask", json={"question": "a" * (settings.MAX_QUESTION_LENGTH + 1)})
    assert response.status_code == 400
    assert "too long" in response.json()["detail"].lower()


def test_ask_without_index_returns_503(client):
    state.retriever = None
    state.vector_store = None
    response = client.post("/ask", json={"question": "What is the main topic?"})
    assert response.status_code == 503
    assert "rag" in response.json()["detail"].lower()
    state.llm.invoke.assert_not_called()


def test_ask_general_question_without_index_uses_mock_llm(client):
    state.retriever = None
    state.vector_store = None
    response = client.post("/ask", json={"question": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert data["sources"] == []
    state.llm.invoke.assert_called_once()


def test_ask_no_evidence_does_not_call_llm(client, monkeypatch):
    state.retriever = MagicMock()
    state.vector_store = MagicMock()
    monkeypatch.setattr(
        "app.api.routes.retrieve_documents",
        lambda _question: ([], None),
    )

    response = client.post("/ask", json={"question": "What is the secret code?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == settings.NO_EVIDENCE_MESSAGE
    assert data["sources"] == []
    assert data["confidence"]["label"] == "low"
    state.llm.invoke.assert_not_called()


def test_ask_with_retrieved_context_returns_structured_response(client, monkeypatch):
    state.retriever = MagicMock()
    state.vector_store = MagicMock()
    docs = [
        Document(
            page_content="The project uses FastAPI and FAISS.",
            metadata={"page": 1, "file_name": "spec.pdf", "chunk_id": 0},
        ),
        Document(
            page_content="Groq provides the LLM layer.",
            metadata={"page": 2, "file_name": "spec.pdf", "chunk_id": 1},
        ),
    ]
    monkeypatch.setattr(
        "app.api.routes.retrieve_documents",
        lambda _question: (docs, [0.3, 0.5]),
    )

    response = client.post("/ask", json={"question": "What stack is used?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mocked LLM response for tests."
    assert len(data["sources"]) == 2
    assert data["sources"][0]["page"] == 2
    assert data["sources"][0]["page_index"] == 1
    assert "preview" in data["sources"][0]
    assert data["confidence"]["label"] in ("high", "medium", "low")
    assert data["metadata"]["chunks_used"] == 2
    state.llm.invoke.assert_called_once()


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/upload",
        files={"file": ("readme.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_success_with_mocked_processing(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.process_pdf_and_update_index",
        lambda *_args, **_kwargs: (12, 3),
    )
    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 test content", "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["pages"] == 3
    assert data["chunks"] == 12
    assert data["filename"] == "sample.pdf"
