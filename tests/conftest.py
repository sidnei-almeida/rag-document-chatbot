"""Shared pytest fixtures — no real Groq or Hugging Face calls."""

import os
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-fake-key-for-pytest")
os.environ.setdefault("LOG_LEVEL", "WARNING")


def _fake_initialize_models() -> None:
    from app.core.state import state

    state.llm = MagicMock()
    state.llm.invoke.return_value = MagicMock(content="Mocked LLM response for tests.")
    state.embeddings_model = MagicMock()
    state.vector_store = None
    state.retriever = None
    state.index_cleared = False
    state.index_pages = 0
    state.index_chunks = 0
    state.last_indexed_filename = "document.pdf"
    state.retriever_type = "mmr"


@asynccontextmanager
async def _test_lifespan(_app):
    _fake_initialize_models()
    yield


@pytest.fixture(autouse=True)
def reset_state():
    """Reset in-memory state before and after each test."""
    _fake_initialize_models()
    yield
    from app.core.state import state

    state.vector_store = None
    state.retriever = None
    state.llm = MagicMock()
    state.llm.invoke.return_value = MagicMock(content="Mocked LLM response for tests.")
    state.embeddings_model = MagicMock()
    state.index_cleared = False
    state.index_pages = 0
    state.index_chunks = 0


@pytest.fixture
def client():
    """FastAPI test client; startup uses fake models (no Groq / embeddings)."""
    from fastapi.testclient import TestClient
    from app.main import create_app

    with TestClient(create_app(lifespan_handler=_test_lifespan)) as test_client:
        yield test_client
