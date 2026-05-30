"""FAISS index path and existence checks."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.core.state import state
from app.services.faiss_index import (
    faiss_index_exists,
    get_index_faiss_path,
    try_load_faiss_from_disk,
)


def test_get_index_faiss_path_uses_vector_store_path():
    path = get_index_faiss_path()
    assert path.name == "index.faiss"
    assert path.parent == Path(settings.VECTOR_STORE_PATH).resolve()


def _patch_index_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.services.faiss_index.get_index_directory",
        lambda: tmp_path,
    )


def test_faiss_index_exists_false_when_file_missing(tmp_path, monkeypatch):
    _patch_index_dir(monkeypatch, tmp_path)
    assert faiss_index_exists() is False


def test_faiss_index_exists_true_when_index_file_present(tmp_path, monkeypatch):
    _patch_index_dir(monkeypatch, tmp_path)
    (tmp_path / "index.faiss").write_bytes(b"fake")
    assert faiss_index_exists() is True


def test_try_load_faiss_returns_false_without_index_file(tmp_path, monkeypatch):
    _patch_index_dir(monkeypatch, tmp_path)
    state.embeddings_model = MagicMock()
    state.index_cleared = False
    assert try_load_faiss_from_disk() is False
    assert state.vector_store is None


def test_try_load_faiss_loads_when_file_present(tmp_path, monkeypatch):
    _patch_index_dir(monkeypatch, tmp_path)
    (tmp_path / "index.faiss").write_bytes(b"x")
    state.embeddings_model = MagicMock()
    state.index_cleared = False
    mock_store = MagicMock()
    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_store):
        with patch("app.services.faiss_index.build_retriever", return_value=MagicMock()):
            assert try_load_faiss_from_disk() is True
    assert state.vector_store is mock_store
